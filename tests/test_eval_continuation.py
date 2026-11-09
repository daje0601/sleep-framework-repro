import contextlib
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sleep_repro import engine, runner
from sleep_repro.arc import ArcDataset, DreamDirective, EVALUATION_IDS, Pair, Task
from sleep_repro.runtime import ExperimentConfig


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/continue_evaluation.py"
spec = importlib.util.spec_from_file_location("continue_evaluation", SCRIPT)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    cfg = ExperimentConfig.from_dict(json.loads((SCRIPT.parent.parent / "configs/arc_sleep_v3.json").read_text()))
    tasks = tuple(
        Task(task_id, (Pair([[0]], [[0]]), Pair([[1]], [[1]])),
             tuple([[[0]]] * (2 if index == 4 else 1)))
        for index, task_id in enumerate(EVALUATION_IDS)
    )
    data = ArcDataset(tasks, {t.task_id: tuple([[[0]]] * len(t.test_inputs)) for t in tasks},
                      {"challenges_sha256": "a" * 64, "solutions_sha256": "b" * 64})
    current = {"seed": None, "source": "1" * 40}
    calls = []

    def seed(value):
        current["seed"] = value
        calls.append(value)

    class FakeEngine:
        def __init__(self, model, tokenizer, config, device):
            self.config = config

        def generate_directive(self, task):
            return DreamDirective(rotations=(current["seed"] % 4,)).to_json()

        def sft(self, examples, **kwargs):
            return [(current["seed"] % 100) / 100]

        def predict(self, task):
            result = tuple(((current["seed"] % 2,),) for _ in task.test_inputs)
            self.last_prediction_texts = [json.dumps(grid) for grid in result]
            return result

    def load(path, policy, optimizer, config, **kwargs):
        assert kwargs == {"evaluation": True}
        assert config == cfg
        return {"next_episode": 11, "outer_updates": 48}

    def metadata(config, policy, dataset):
        return {"source_commit": current["source"], "config": config.to_dict(),
                "protocol_fingerprint": runner.protocol_fingerprint(), "data_hashes": dict(dataset.file_hashes)}

    monkeypatch.setattr(runner, "seed_all", seed)
    monkeypatch.setattr(runner, "build_model", lambda c: (object(), object()))
    monkeypatch.setattr(runner, "load_checkpoint", load)
    monkeypatch.setattr(runner, "dataset_for", lambda c, split: data)
    monkeypatch.setattr(runner, "wake_episode", lambda p, t, c, task: (object(), [{"seed": current["seed"]}]))
    monkeypatch.setattr(runner, "dream_routing", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(runner, "evidence", metadata)
    monkeypatch.setattr(engine, "CausalEngine", FakeEngine)
    monkeypatch.setattr(continuation, "CausalEngine", FakeEngine)
    mount = tmp_path / "persistent"
    checkpoint = mount / "train/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    parent = mount / "eval-parent"
    parent.mkdir()
    full_summary = runner.evaluate(cfg, parent / "evaluate", checkpoint)
    full_rows = (parent / "evaluate/attempts.jsonl").read_bytes().splitlines(keepends=True)
    parent_job = {"stage": "evaluate", "org": "research", "team": "arc", "run_id": parent.name,
                  "job_name": "parent-job", "resource_spec": "resourcespec-a100x1", "image": "image-sha",
                  "object_volume": "volume", "mount_path": str(mount), "checkpoint": str(checkpoint),
                  "source_commit": current["source"]}
    (parent / "job.json").write_text(json.dumps(parent_job))
    (parent / "config.json").write_text(json.dumps(cfg.to_dict()))
    child = mount / "eval-child"
    child.mkdir()
    child_job = {**parent_job, "stage": "evaluate-resume", "run_id": child.name, "job_name": "child-job",
                 "source_commit": "2" * 40, "resume_eval_run_id": parent.name, "resume_eval_job_id": "job-parent",
                 "resume_eval_job_name": parent_job["job_name"],
                 "resume_eval_provenance_sha256": sha256((parent / "evaluate/provenance.json").read_bytes()).hexdigest()}
    (child / "config.json").write_text(json.dumps(cfg.to_dict()))
    return SimpleNamespace(cfg=cfg, data=data, parent=parent, child=child, job=child_job,
                           full_rows=full_rows, full_summary=full_summary, current=current, calls=calls)


def prepare(env, count):
    raw = b"".join(env.full_rows[:count])
    (env.parent / "evaluate/attempts.jsonl").write_bytes(raw)
    env.job["resume_eval_attempts_sha256"] = sha256(raw).hexdigest()
    (env.child / "job.json").write_text(json.dumps(env.job))
    env.current["source"] = env.job["source_commit"]
    env.calls.clear()
    return raw


@pytest.mark.parametrize("count", [1, 36, 37, 39])
def test_continuation_matches_original_full_schedule(environment, count):
    env = environment
    retained = prepare(env, count)
    result = continuation.continue_evaluation(env.child)
    output = env.child / "evaluate"
    assert (output / "attempts.jsonl").read_bytes().startswith(retained)
    assert (output / "attempts.jsonl").read_bytes().splitlines() == [line.rstrip(b"\n") for line in env.full_rows]
    keys = [(t.task_id, attempt) for t in env.data.tasks for attempt in range(5)]
    expected_seeds = [42 + EVALUATION_IDS.index(task_id) * 1000 + attempt for task_id, attempt in keys[count:]]
    assert env.calls == [42, *expected_seeds]
    for key in ("successful_attempts", "total_attempts", "tasks_solved", "total_tasks", "per_task", "outer_updates"):
        assert result[key] == env.full_summary[key]
    assert result["total_attempts"] == 40 and result["successful_attempts"] == 24
    assert result["preserved_attempts_count"] == count
    assert result["computed_attempts_count"] == 40 - count
    lineage = json.loads((output / "resume_lineage.json").read_text())
    assert lineage["preserved_attempts"] == [list(key) for key in keys[:count]]
    assert lineage["computed_attempts"] == [list(key) for key in keys[count:]]
    assert lineage["parent_source_commit"] == "1" * 40
    assert lineage["source_commit"] == "2" * 40
    with pytest.raises(ValueError, match="already exists"):
        continuation.continue_evaluation(env.child)


@pytest.mark.parametrize("kind", ["hash", "duplicate", "seed", "test_input", "checkpoint", "config", "protocol", "source"])
def test_parent_mismatch_is_rejected_before_computation(environment, kind):
    env = environment
    prepare(env, 37)
    attempts_path = env.parent / "evaluate/attempts.jsonl"
    provenance_path = env.parent / "evaluate/provenance.json"
    if kind in ("duplicate", "seed", "test_input"):
        rows = [json.loads(line) for line in attempts_path.read_text().splitlines()]
        if kind == "duplicate":
            rows[-1] = rows[-2]
        elif kind == "seed":
            rows[-1]["seed"] += 1
        else:
            rows[-1]["predictions"] = []
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        attempts_path.write_bytes(raw)
        env.job["resume_eval_attempts_sha256"] = sha256(raw).hexdigest()
    elif kind == "hash":
        attempts_path.write_bytes(attempts_path.read_bytes() + b" ")
    elif kind == "config":
        config = env.cfg.to_dict()
        config["temperature"] = 1.0
        (env.parent / "config.json").write_text(json.dumps(config))
    else:
        provenance = json.loads(provenance_path.read_text())
        key = {"checkpoint": "checkpoint", "protocol": "protocol_fingerprint", "source": "source_commit"}[kind]
        provenance[key] = "wrong"
        provenance_path.write_text(json.dumps(provenance))
        env.job["resume_eval_provenance_sha256"] = sha256(provenance_path.read_bytes()).hexdigest()
    (env.child / "job.json").write_text(json.dumps(env.job))
    with pytest.raises(ValueError):
        continuation.continue_evaluation(env.child)
    assert env.calls == []
    assert not (env.child / "evaluate").exists()


def test_continuation_output_must_match_reviewed_run_directory(environment, tmp_path):
    env = environment
    prepare(env, 37)
    wrong = tmp_path / "wrong-output"
    wrong.mkdir()
    for name in ("job.json", "config.json"):
        (wrong / name).write_bytes((env.child / name).read_bytes())
    with pytest.raises(ValueError, match="run directory mismatch"):
        continuation.continue_evaluation(wrong)
    assert env.calls == []
    assert not (wrong / "evaluate").exists()
