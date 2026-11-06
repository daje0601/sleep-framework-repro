import json

import pytest
import torch

from sleep_repro.arc import ArcDataset, Pair, Task
from sleep_repro.runtime import ExperimentConfig


def fixture_data(cfg, split):
    if split != "train":
        raise AssertionError("preflight must not load final evaluation tasks")
    task = Task(
        "unit-training-fixture",
        (Pair(((1,),), ((1,),)), Pair(((2,),), ((2,),))),
        (((3,),),),
    )
    return ArcDataset((task,), {}, {"fixture": "synthetic-unit-test"})


def config():
    return ExperimentConfig.from_dict(
        dict(
            model="tiny",
            device="cpu",
            dtype="float32",
            rank=2,
            capacity=3,
            policy_rank=2,
            inner_rank=2,
            wake_steps=2,
            consolidation_steps=1,
            max_length=2048,
            max_new_tokens=2,
            train_dreams=2,
            top_k=1,
            random_k=1,
            inner_lr=0.01,
            outer_lr=0.01,
        )
    )


def test_cpu_preflight_executes_real_gradients_and_checkpoint_without_gpu_claim(
    tmp_path, monkeypatch
):
    from sleep_repro import preflight as module

    torch.set_num_threads(1)
    monkeypatch.setattr(module, "dataset_for", fixture_data)
    report = module.preflight(config(), tmp_path / "preflight")
    assert report["passed"] is True
    assert report["gpu_validation"] is False
    assert report["paper_result_reproduced"] is False
    assert report["inner_parameter_delta_l1"] > 0
    assert report["outer_parameter_delta_l1"] > 0
    assert report["memory_base_delta_l1"] > 0
    assert report["checkpoint_resume_verified"] is True
    assert report["fixture_update"]["accepted_policy_update"] is False
    assert torch.load(tmp_path / "preflight" / "fixture_checkpoint.pt", weights_only=True)["purpose"] == "diagnostic"
    assert report["task_id"] == "unit-training-fixture"
    assert json.loads((tmp_path / "preflight" / "preflight.json").read_text())["passed"]


def test_preflight_refuses_unchanged_adapters_instead_of_reporting_success(
    tmp_path, monkeypatch
):
    from sleep_repro import preflight as module

    monkeypatch.setattr(module, "dataset_for", fixture_data)
    monkeypatch.setattr(module.CausalEngine, "sft", lambda *a, **k: [1.0])
    with pytest.raises(RuntimeError, match="inner.*delta"):
        module.preflight(config(), tmp_path / "preflight")
    assert not (tmp_path / "preflight" / "preflight.json").exists()
