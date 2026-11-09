"""Continue only missing ARC attempts from a verified interrupted evaluation.

The numerical model, checkpoint loader, adaptation and scoring functions are
imported unchanged from sleep_repro. This controller records its own identity
and the exact retained parent history alongside the normal evaluation evidence.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import time

import torch

from sleep_repro import runner
from sleep_repro.arc import EVALUATION_IDS, content_hash, expand_directive, parse_directive
from sleep_repro.engine import CausalEngine
from sleep_repro.metrics import Attempt, evaluate_attempts
from sleep_repro.runtime import ExperimentConfig


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def verified_parent(run_dir, cfg, data, job):
    """Validate an ordered parent prefix without consulting its answer scores."""
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job["run_id"]), "Unsafe continuation run id")
    expected_run_dir = Path(job["mount_path"]) / job["run_id"]
    require(Path(run_dir).resolve() == expected_run_dir.resolve(), "Continuation run directory mismatch")
    parent_id = job["resume_eval_run_id"]
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", parent_id), "Unsafe parent run id")
    require(parent_id != job["run_id"], "Continuation must use a fresh run id")
    parent = Path(job["mount_path"]) / parent_id
    parent_job = read_json(parent / "job.json")
    require(parent_job["stage"] in ("evaluate", "evaluate-resume"), "Parent is not an evaluation")
    for key in ("org", "team", "resource_spec", "image", "object_volume", "mount_path", "checkpoint"):
        require(parent_job.get(key) == job.get(key), f"Parent {key} mismatch")
    require(parent_job["run_id"] == parent_id and parent_job["job_name"] == job["resume_eval_job_name"],
            "Parent job identity mismatch")
    require(content_hash(read_json(parent / "config.json")) == content_hash(cfg.to_dict()),
            "Parent experiment configuration mismatch")
    provenance_bytes = (parent / "evaluate/provenance.json").read_bytes()
    attempts_bytes = (parent / "evaluate/attempts.jsonl").read_bytes()
    require(sha256(provenance_bytes).hexdigest() == job["resume_eval_provenance_sha256"],
            "Parent provenance hash mismatch")
    require(sha256(attempts_bytes).hexdigest() == job["resume_eval_attempts_sha256"],
            "Parent attempts hash mismatch")
    provenance = json.loads(provenance_bytes)
    require(provenance["source_commit"] == parent_job["source_commit"], "Parent source mismatch")
    require(provenance["protocol_fingerprint"] == runner.protocol_fingerprint(),
            "Parent numerical protocol mismatch")
    require(content_hash(provenance["config"]) == content_hash(cfg.to_dict()),
            "Parent provenance configuration mismatch")
    require(provenance["checkpoint"] == job["checkpoint"], "Parent checkpoint mismatch")
    require(provenance["data_hashes"] == dict(data.file_hashes), "Parent data hashes mismatch")
    require(tuple(task.task_id for task in data.tasks) == EVALUATION_IDS and cfg.attempts == 5,
            "Continuation requires the fixed eight-task five-attempt schedule")
    schedule = [(task.task_id, index) for task in data.tasks for index in range(cfg.attempts)]
    rows = [json.loads(line) for line in attempts_bytes.splitlines()]
    require(0 < len(rows) < len(schedule), "Parent must contain a nonempty incomplete evaluation prefix")
    attempts = []
    for offset, row in enumerate(rows):
        key = (row["task_id"], row["attempt_index"])
        require(key == schedule[offset], "Parent attempt order, duplicate or gap")
        task_index = offset // cfg.attempts
        require(type(row["seed"]) is int and row["seed"] == cfg.seed + task_index * 1000 + row["attempt_index"],
                "Parent attempt seed mismatch")
        attempt = Attempt(row["task_id"], row["attempt_index"], row["predictions"])
        require(len(attempt.predictions) == len(data.tasks[task_index].test_inputs),
                "Parent attempt omits test inputs")
        attempts.append(attempt)
    lineage = {
        "parent_run_id": parent_id,
        "parent_job_id": job["resume_eval_job_id"],
        "parent_job_name": job["resume_eval_job_name"],
        "parent_source_commit": provenance["source_commit"],
        "parent_attempts_sha256": sha256(attempts_bytes).hexdigest(),
        "parent_provenance_sha256": sha256(provenance_bytes).hexdigest(),
        "numerical_protocol_fingerprint": runner.protocol_fingerprint(),
        "controller_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": job["checkpoint"],
        "preserved_attempts": [list(key) for key in schedule[:len(rows)]],
        "computed_attempts": [],
    }
    return rows, attempts, attempts_bytes, lineage


def continue_evaluation(run_dir):
    run_dir = Path(run_dir)
    job = read_json(run_dir / "job.json")
    raw_config = read_json(run_dir / "config.json")
    cfg = ExperimentConfig.from_dict(raw_config)
    require(job["stage"] == "evaluate-resume" and cfg.mode == "sleep", "Unexpected continuation stage or mode")
    require(content_hash(cfg.to_dict()) == content_hash(raw_config),
            "Configuration must contain the resolved registered settings")
    checkpoint = Path(job["checkpoint"])
    require(checkpoint.is_absolute() and checkpoint.is_relative_to(Path(job["mount_path"]))
            and ".." not in checkpoint.parts, "Checkpoint must be inside the persistent volume")
    output = run_dir / "evaluate"
    require(not output.exists(), "Continuation output already exists")
    data = runner.dataset_for(cfg, "eval")
    rows, attempts, retained_bytes, lineage = verified_parent(run_dir, cfg, data, job)
    runner.seed_all(cfg.seed)
    policy, tokenizer = runner.build_model(cfg)
    progress = runner.load_checkpoint(checkpoint, policy, None, cfg, evaluation=True)
    metadata = runner.evidence(cfg, policy, data)
    require(metadata["source_commit"] == job["source_commit"], "Controller source differs from job pin")
    lineage["source_commit"] = metadata["source_commit"]
    output.mkdir()
    started = time.monotonic()
    runner.write_json(output / "provenance.json", {**metadata, "checkpoint": str(checkpoint), "resume": lineage})
    runner.write_json(output / "resume_lineage.json", lineage)
    (output / "attempts.jsonl").write_bytes(retained_bytes + (b"" if retained_bytes.endswith(b"\n") else b"\n"))
    preserved = {(row["task_id"], row["attempt_index"]) for row in rows}
    for task_index, task in enumerate(data.tasks):
        for attempt_index in range(cfg.attempts):
            if (task.task_id, attempt_index) in preserved:
                continue
            attempt_seed = cfg.seed + task_index * 1000 + attempt_index
            runner.seed_all(attempt_seed)
            episode, events = runner.wake_episode(policy, tokenizer, cfg, task)
            engine = CausalEngine(episode, tokenizer, cfg, torch.device(cfg.device))
            directive = None
            invalid = None
            with runner.dream_routing(episode, True):
                raw = engine.generate_directive(task)
            try:
                directive = parse_directive(raw)
            except (ValueError, TypeError) as exc:
                invalid = str(exc)
            inner_losses = []
            if directive:
                inner_losses = engine.sft(
                    runner.examples_for(task, directive, cfg=cfg), adapter="inner",
                    epochs=directive.epochs, lr=cfg.inner_lr * directive.lr_scale,
                )
            predictions = engine.predict(task) if invalid is None else tuple(None for _ in task.test_inputs)
            attempt = Attempt(task.task_id, attempt_index, predictions)
            runner.append_event(output / "attempts.jsonl", {
                **asdict(attempt), "raw_predictions": getattr(engine, "last_prediction_texts", []),
                "seed": attempt_seed, "directive": raw, "invalid_directive": invalid,
                "wake": events, "inner_optimizer_steps": len(inner_losses), "inner_losses": inner_losses,
                "sft_example_hash": content_hash(runner.examples_for(task, directive, cfg=cfg)) if directive else None,
                "expanded_example_hash": content_hash([asdict(pair) for pair in expand_directive(task, directive)]) if directive else None,
            })
            attempts.append(attempt)
            lineage["computed_attempts"].append([task.task_id, attempt_index])
            runner.write_json(output / "resume_lineage.json", lineage)
    summary = evaluate_attempts(data.tasks, data.solutions, attempts, expected_attempts=cfg.attempts)
    summary.update(elapsed_seconds=time.monotonic() - started, mode=cfg.mode,
                   policy_trained=progress["outer_updates"] > 0, outer_updates=progress["outer_updates"],
                   paper_result_reproduced=False, comparison_status="protocol reconstruction",
                   preserved_attempts_count=len(rows), computed_attempts_count=len(lineage["computed_attempts"]))
    runner.write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: continue_evaluation.py RUN_DIR")
    continue_evaluation(Path(sys.argv[1]))
