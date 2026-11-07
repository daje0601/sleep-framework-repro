import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from sleep_repro import checkpoint_migration, runner
from sleep_repro.runtime import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_resume", ROOT / "scripts/prepare_resume.py"
)
resume_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume_script)


def make_interrupted_run(tmp_path, monkeypatch):
    parent = tmp_path / checkpoint_migration.PARENT_RUN
    new = tmp_path / "new-run"
    (parent / "train").mkdir(parents=True)
    new.mkdir()
    experiment = json.loads((ROOT / "configs/arc_sleep_v2.json").read_text())
    cfg = ExperimentConfig.from_dict(experiment)
    (new / "config.json").write_text(json.dumps(experiment))
    (parent / "job.json").write_text(
        json.dumps(
            {
                "stage": "train",
                "run_id": parent.name,
                "source_commit": "old",
            }
        )
    )
    (new / "job.json").write_text(
        json.dumps(
            {
                "stage": "train",
                "resume_run_id": parent.name,
                "source_commit": "new",
            }
        )
    )
    provenance = parent / "train" / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "source_commit": "old",
                "protocol_fingerprint": checkpoint_migration.SOURCE_PROTOCOL,
            }
        )
    )
    record = {
        "episode": 0,
        "task_id": "44f52bb0",
        "generated": 60,
        "accepted": [],
        "accepted_count": 0,
        "episode_outer_optimizer_steps": 0,
        "outer_updates": 0,
        "outer_losses": [],
    }
    prefix = json.dumps(record) + "\n"
    # An interruption between logging a subsequent episode and saving its checkpoint.
    (parent / "train" / "training.jsonl").write_text(prefix + '{"episode":1}\n')
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = parent / "train" / "checkpoint.pt"
    monkeypatch.setattr(
        runner, "protocol_fingerprint", lambda: checkpoint_migration.SOURCE_PROTOCOL
    )
    runner.save_checkpoint(path, model, optimizer, cfg, 1)
    weights = model.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(
        runner, "protocol_fingerprint", lambda: checkpoint_migration.TARGET_PROTOCOL
    )
    monkeypatch.setattr(
        checkpoint_migration, "PARENT_CHECKPOINT_BYTES", path.stat().st_size
    )
    monkeypatch.setattr(
        checkpoint_migration,
        "PARENT_PROVENANCE",
        sha256(provenance.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        checkpoint_migration, "FIRST_EPISODE", sha256(prefix.encode()).hexdigest()
    )
    return new, cfg, weights, rng, prefix


def test_train_only_migration_restores_state_and_committed_prefix(
    tmp_path, monkeypatch
):
    new, cfg, weights, rng, prefix = make_interrupted_run(tmp_path, monkeypatch)
    path = resume_script.prepare_resume(new)
    output = new / "train"
    assert (output / "training.jsonl").read_text() == prefix
    assert len((path.parent / "training.jsonl").read_text().splitlines()) == 2
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    for kwargs in (
        {},
        {"evaluation": True, "resume_output": output},
        {"allow_diagnostic": True, "resume_output": output},
    ):
        with pytest.raises(ValueError, match="protocol mismatch"):
            runner.load_checkpoint(path, model, optimizer, cfg, **kwargs)
    progress = runner.load_checkpoint(path, model, optimizer, cfg, resume_output=output)
    assert progress == {"next_episode": 1, "outer_updates": 0}
    assert torch.equal(model.weight, weights)
    assert torch.equal(torch.get_rng_state(), rng)
    assert optimizer.state == {}
    lineage = json.loads((output / "checkpoint_migration.json").read_text())
    assert lineage["completed_task_ids"] == ["44f52bb0"]
    assert (
        lineage["parent_protocol_fingerprint"] == checkpoint_migration.SOURCE_PROTOCOL
    )
    # A different target revision and a modified evidence log must each be rejected.
    monkeypatch.setattr(runner, "protocol_fingerprint", lambda: "another-protocol")
    with pytest.raises(ValueError, match="protocol mismatch"):
        runner.load_checkpoint(path, model, optimizer, cfg, resume_output=output)
    monkeypatch.setattr(
        runner, "protocol_fingerprint", lambda: checkpoint_migration.TARGET_PROTOCOL
    )
    (output / "training.jsonl").write_text(prefix + "\n")
    with pytest.raises(ValueError, match="evidence has changed"):
        runner.load_checkpoint(path, model, optimizer, cfg, resume_output=output)


def test_refuse_bad_parent_episode_before_copy(tmp_path, monkeypatch):
    new, _, _, _, prefix = make_interrupted_run(tmp_path, monkeypatch)
    parent_history = tmp_path / checkpoint_migration.PARENT_RUN / "train/training.jsonl"
    parent_history.write_text(prefix.replace("44f52bb0", "4be741c5"))
    with pytest.raises(ValueError, match="schedule or counts"):
        resume_script.prepare_resume(new)
    assert not (new / "train/training.jsonl").exists()
