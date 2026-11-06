"""Real optimizer continuation and honest learned-policy checkpoint metadata."""

import copy
import json
import random

import numpy as np
import pytest
import torch

from sleep_repro import runner
from sleep_repro.arc import ArcDataset, Pair, Task
from sleep_repro.engine import CausalEngine
from sleep_repro.runtime import ExperimentConfig, seed_all, train_adapter


def small_config():
    return ExperimentConfig.from_dict(
        {
            "model": "tiny",
            "device": "cpu",
            "dtype": "float32",
            "mode": "self_edit",
            "rank": 2,
            "capacity": 4,
            "policy_rank": 2,
            "inner_rank": 2,
            "outer_lr": 0.01,
            "max_length": 256,
            "max_new_tokens": 2,
            "train_dreams": 2,
            "top_k": 1,
            "random_k": 1,
            "attempts": 5,
        }
    )


def policy_state(cfg):
    model, tokenizer = runner.build_model(cfg)
    # Stochastic actual attention during SFT makes RNG restoration affect the
    # resumed weights, rather than merely checking a post-load random draw.
    for layer in model.model.layers:
        layer.self_attn.attention_dropout = 0.2
    optimizer = torch.optim.AdamW(train_adapter(model, "policy"), lr=cfg.outer_lr)
    return model, optimizer, CausalEngine(model, tokenizer, cfg, "cpu")


def random_samples():
    return random.random(), np.random.random(), torch.rand(5)


def assert_random_samples_equal(actual, expected):
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for observed, wanted in zip(actual, expected, strict=True):
            assert_state_equal(observed, wanted)
    else:
        assert actual == expected


def test_real_policy_sft_checkpoint_resume_matches_uninterrupted_weights_optimizer_and_rng(
    tmp_path,
):
    cfg = small_config()
    first_examples = [("ARC input:", "[[1,2]]"), ("Another input:", "[[3]]")]
    second_examples = [("Next input:", "[[4,5]]"), ("Last input:", "[[6]]")]

    seed_all(73)
    uninterrupted, uninterrupted_optimizer, full_engine = policy_state(cfg)
    initial = copy.deepcopy(uninterrupted.state_dict())
    first_loss = full_engine.sft(
        first_examples, adapter="policy", optimizer=uninterrupted_optimizer
    )
    between_samples = random_samples()
    second_loss = full_engine.sft(
        second_examples, adapter="policy", optimizer=uninterrupted_optimizer
    )
    final_samples = random_samples()
    expected_weights = copy.deepcopy(uninterrupted.state_dict())
    expected_optimizer = copy.deepcopy(uninterrupted_optimizer.state_dict())
    assert any(
        not torch.equal(initial[name], value)
        for name, value in expected_weights.items()
    )
    assert expected_optimizer["state"]
    assert all(
        int(state["step"]) == 2 for state in expected_optimizer["state"].values()
    )

    seed_all(73)
    interrupted, interrupted_optimizer, interrupted_engine = policy_state(cfg)
    assert (
        interrupted_engine.sft(
            first_examples, adapter="policy", optimizer=interrupted_optimizer
        )
        == first_loss
    )
    assert_random_samples_equal(random_samples(), between_samples)
    checkpoint = tmp_path / "after-first-update.pt"
    runner.save_checkpoint(
        checkpoint, interrupted, interrupted_optimizer, cfg, 1, outer_updates=1
    )

    # Rebuilding consumes a different RNG stream and creates fresh parameter
    # identities; loading must restore both moments and the saved random state.
    seed_all(999)
    resumed, resumed_optimizer, resumed_engine = policy_state(cfg)
    assert resumed_optimizer.state == {}
    progress = runner.load_checkpoint(checkpoint, resumed, resumed_optimizer, cfg)
    assert progress == {"next_episode": 1, "outer_updates": 1}
    assert (
        resumed_engine.sft(
            second_examples, adapter="policy", optimizer=resumed_optimizer
        )
        == second_loss
    )
    assert_state_equal(resumed.state_dict(), expected_weights)
    assert_state_equal(resumed_optimizer.state_dict(), expected_optimizer)
    assert_random_samples_equal(random_samples(), final_samples)


def test_all_invalid_training_dreams_keep_loaded_evaluation_policy_untrained(
    tmp_path, monkeypatch
):
    cfg = small_config()
    demonstration_pairs = (Pair(((1,),), ((2,),)), Pair(((3,),), ((4,),)))
    train_task = Task("synthetic-train", demonstration_pairs, (((5,),),))
    eval_task = Task("synthetic-eval", demonstration_pairs, (((6,),),))
    fixtures = {
        "train": ArcDataset(
            (train_task,), {train_task.task_id: (((7,),),)}, {"fixture": "train"}
        ),
        "eval": ArcDataset(
            (eval_task,), {eval_task.task_id: (((8,),),)}, {"fixture": "eval"}
        ),
    }
    monkeypatch.setattr(runner, "dataset_for", lambda config, split: fixtures[split])
    actual_build = runner.build_model

    def build_tiny(config):
        assert config.model == "tiny" and config.dtype == "float32"
        return actual_build(config)

    monkeypatch.setattr(runner, "build_model", build_tiny)
    monkeypatch.setattr(
        CausalEngine, "generate_directive", lambda self, task: "invalid JSON directive"
    )

    def forbidden_sft(*args, **kwargs):
        pytest.fail(
            "Invalid directives must not trigger inner adaptation or outer policy SFT"
        )

    monkeypatch.setattr(CausalEngine, "sft", forbidden_sft)

    training_output = tmp_path / "train"
    checkpoint = runner.train(cfg, training_output)
    training_summary = json.loads(
        (training_output / "training_summary.json").read_text()
    )
    assert training_summary["outer_updates"] == 0
    assert training_summary["policy_trained"] is False
    event = json.loads((training_output / "training.jsonl").read_text())
    assert len(event["invalid"]) == cfg.train_dreams
    assert event["accepted_count"] == 0 and event["outer_losses"] == []
    assert checkpoint.is_file()

    evaluation_output = tmp_path / "evaluation"
    summary = runner.evaluate(cfg, evaluation_output, checkpoint=checkpoint)
    assert summary["outer_updates"] == 0
    assert summary["policy_trained"] is False
    assert summary["total_attempts"] == 5
    assert summary["successful_attempts"] == 0
    provenance = json.loads((evaluation_output / "provenance.json").read_text())
    assert provenance["checkpoint"] == str(checkpoint)
    assert (
        json.loads((evaluation_output / "summary.json").read_text())["policy_trained"]
        is False
    )
