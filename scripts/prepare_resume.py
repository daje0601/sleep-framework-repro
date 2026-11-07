"""Copy the committed episode log prefix before a cloud training continuation."""

import json
from pathlib import Path
import sys


def prepare_resume(run_dir):
    job = json.loads((run_dir / "job.json").read_text())
    previous_id = job.get("resume_run_id")
    if previous_id is None:
        return None

    import torch
    from sleep_repro.arc import TRAIN_IDS
    from sleep_repro.runtime import ExperimentConfig

    previous = run_dir.parent / previous_id
    if (
        previous.resolve().parent != run_dir.parent.resolve()
        or previous.resolve() == run_dir.resolve()
    ):
        raise ValueError("Resume requires a different run on this volume")
    parent_job = json.loads((previous / "job.json").read_text())
    if parent_job.get("stage") != "train" or parent_job.get("run_id") != previous_id:
        raise ValueError("Resume parent is not the requested training run")
    checkpoint = previous / "train" / "checkpoint.pt"
    state = torch.load(checkpoint, map_location="meta", weights_only=True)
    cfg = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    if state.get("purpose") != "training" or state["config"] != cfg.to_dict():
        raise ValueError(
            "Resume requires a training checkpoint with the same experiment configuration"
        )
    provenance = json.loads((previous / "train" / "provenance.json").read_text())
    if (
        provenance["source_commit"] != parent_job["source_commit"]
        or provenance["protocol_fingerprint"] != state["protocol_fingerprint"]
    ):
        raise ValueError(
            "Parent training provenance does not match its job and checkpoint"
        )

    next_episode = state["next_episode"]
    if (
        type(next_episode) is not int
        or not 0 < next_episode < len(TRAIN_IDS) * cfg.train_epochs
    ):
        raise ValueError("Resume requires a partially completed training schedule")
    lines = []
    updates = 0
    with (previous / "train" / "training.jsonl").open() as history:
        for index in range(next_episode):
            line = history.readline()
            if not line:
                raise ValueError("Parent training log ends before checkpoint progress")
            event = json.loads(line)
            steps = len(event["outer_losses"])
            updates += steps
            if (
                event["episode"] != index
                or event["task_id"] != TRAIN_IDS[index % len(TRAIN_IDS)]
                or event["generated"] != cfg.train_dreams
                or event["accepted_count"] != len(event["accepted"])
                or event["episode_outer_optimizer_steps"] != steps
                or event["outer_updates"] != updates
            ):
                raise ValueError(
                    f"Parent training record disagrees with the schedule or counts at episode {index}"
                )
            lines.append(line.rstrip("\n") + "\n")
    if updates != state["outer_updates"]:
        raise ValueError(
            "Checkpoint and parent training log disagree on policy updates"
        )
    # A failed save can leave one more log entry than committed episodes.
    # Keep the parent untouched and replay only those episodes from the checkpoint.
    output = run_dir / "train"
    output.mkdir()
    with (output / "training.jsonl").open("x") as history:
        history.writelines(lines)
    return checkpoint


if __name__ == "__main__":
    checkpoint = prepare_resume(Path(sys.argv[1]))
    if checkpoint is not None:
        print(checkpoint)
