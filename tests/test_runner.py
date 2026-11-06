import torch
from sleep_repro.runner import (
    save_checkpoint,
    load_checkpoint,
    build_model,
    wake_episode,
)
from sleep_repro.runtime import ExperimentConfig, seed_all
from sleep_repro.arc import Task, Pair


def small_config():
    return ExperimentConfig.from_dict(
        dict(
            model="tiny",
            device="cpu",
            dtype="float32",
            rank=2,
            capacity=4,
            policy_rank=2,
            inner_rank=2,
            wake_steps=2,
            consolidation_steps=1,
            max_length=2048,
            max_new_tokens=2,
            train_dreams=2,
            top_k=1,
            random_k=1,
        )
    )


def test_checkpoint_continuation_and_wrong_config(tmp_path):
    import pytest

    cfg = small_config()
    seed_all(7)
    model, tok = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, opt, cfg, 3)
    expected = torch.rand(3)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    progress = load_checkpoint(path, model, opt, cfg)
    assert progress["next_episode"] == 3
    assert progress["outer_updates"] == 0
    torch.testing.assert_close(torch.rand(3), expected)
    bad = ExperimentConfig.from_dict({**cfg.to_dict(), "seed": 99})
    with pytest.raises(ValueError, match="config"):
        load_checkpoint(path, model, opt, bad)


def test_checkpoint_binds_protocol_and_records_real_updates(tmp_path):
    import pytest

    cfg = small_config()
    model, _ = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, opt, cfg, 1, outer_updates=2)
    assert (
        load_checkpoint(path, model, None, cfg, evaluation=True)["outer_updates"] == 2
    )
    state = torch.load(path, weights_only=True)
    state["protocol_fingerprint"] = "changed-dataset-or-implementation"
    torch.save(state, path)
    with pytest.raises(ValueError, match="protocol"):
        load_checkpoint(path, model, opt, cfg)


def test_dataset_registry_is_required_and_hash_verified(tmp_path):
    import json
    import pytest
    from importlib.resources import files
    from sleep_repro.runner import dataset_for

    cfg = ExperimentConfig.from_dict(
        {**small_config().to_dict(), "data_dir": str(tmp_path)}
    )
    (tmp_path / "manifest.json").write_text('{"files": {}}')
    with pytest.raises(ValueError, match="provenance"):
        dataset_for(cfg, "train")
    registry = json.loads(
        files("sleep_repro").joinpath("arc_data_manifest.json").read_text()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    name: {k: info[k] for k in ("url", "sha256")}
                    for name, info in registry["files"].items()
                }
            }
        )
    )
    with pytest.raises(ValueError, match="hash"):
        dataset_for(cfg, "train")


def test_wake_episode_does_not_modify_policy():
    cfg = small_config()
    seed_all(1)
    model, tok = build_model(cfg)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    task = Task(
        "synthetic", (Pair(((1,),), ((1,),)), Pair(((2,),), ((2,),))), (((3,),),)
    )
    episode, events = wake_episode(model, tok, cfg, task)
    assert len(events) >= 2
    for k, v in before.items():
        torch.testing.assert_close(model.state_dict()[k], v)
    assert any("consolidation" == e["event"] for e in events)


def test_attention_backend_is_explicit_and_preserves_masked_forward():
    import copy
    from sleep_repro.runtime import encode_batch, train_adapter

    cfg = ExperimentConfig.from_dict(
        {**small_config().to_dict(), "attention_backend": "sdpa"}
    )
    model, tok = build_model(cfg)
    assert model.config._attn_implementation == "sdpa"
    eager = copy.deepcopy(model)
    eager.set_attn_implementation("eager")
    batch = encode_batch(tok, [("p:", "1"), ("longer:", "234")], "cpu", cfg.max_length)
    losses = []
    grads = []
    for instance in (model, eager):
        parameters = train_adapter(instance, "inner")
        output = instance(**batch, use_cache=False)
        losses.append(output.loss.detach())
        grads.append(torch.autograd.grad(output.loss, parameters))
    torch.testing.assert_close(losses[0], losses[1], atol=1e-6, rtol=1e-5)
    for left, right in zip(grads[0], grads[1], strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-4)
