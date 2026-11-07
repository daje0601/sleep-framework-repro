"""One reviewed training prefix that predates the consolidation recovery fix."""

from hashlib import sha256
import json
from pathlib import Path

from .arc import content_hash


# Keep the target outside runner.py, which itself participates in this hash.
SOURCE_PROTOCOL = "2e4f6e2091a6f66316f5e983c7ae48ce4015c308126d519813aad077102b9302"
TARGET_PROTOCOL = "0101c9f153b7dc82e5c93fbf219c806474a7396e30837542d83c2fa627956260"
PARENT_RUN = "sleep-train-20261107-h100-v2-seed42-12"
SEED42_CONFIG = "970584c2ecc717d91dd22dcbd01c42ca8532dd6fa5ad6b5feb0de98e6a023514"
FIRST_EPISODE = "421588ae33d7c05889530d491a9a71151888d4559d100ec2dfb9bf0fe9210e86"
PARENT_PROVENANCE = "a1bf6477fbb1a5f974282e419174925c5d1858f3cd636e85ff7e57c799916723"
PARENT_CHECKPOINT_BYTES = 9096773009


def reviewed_continuation(path, state, target_protocol, output):
    """Validate the single known prefix and describe its provenance."""
    output = Path(output)
    checkpoint = Path(path).resolve()
    if (
        state.get("protocol_fingerprint") != SOURCE_PROTOCOL
        or target_protocol != TARGET_PROTOCOL
        or state.get("purpose") != "training"
        or state.get("next_episode") != 1
        or state.get("outer_updates") != 0
        or state.get("optimizer", {}).get("state") != {}
        or content_hash(state["config"]) != SEED42_CONFIG
    ):
        raise ValueError("Checkpoint protocol mismatch: incompatible training prefix")
    run = output.parent
    expected_checkpoint = (
        run.parent / PARENT_RUN / "train" / "checkpoint.pt"
    ).resolve()
    job = json.loads((run / "job.json").read_text())
    if (
        job.get("resume_run_id") != PARENT_RUN
        or job.get("stage") != "train"
        or checkpoint != expected_checkpoint
        or checkpoint.parent == output.resolve()
        or checkpoint.stat().st_size != PARENT_CHECKPOINT_BYTES
    ):
        raise ValueError(
            "Checkpoint protocol mismatch: this continuation needs its reviewed parent"
        )
    provenance = (checkpoint.parent / "provenance.json").read_bytes()
    with (checkpoint.parent / "training.jsonl").open("rb") as parent_log:
        prefix = parent_log.readline()
    if (
        sha256(provenance).hexdigest() != PARENT_PROVENANCE
        or sha256(prefix).hexdigest() != FIRST_EPISODE
        or prefix != (output / "training.jsonl").read_bytes()
    ):
        raise ValueError(
            "Checkpoint protocol mismatch: parent episode evidence has changed"
        )
    return {
        "parent_run": PARENT_RUN,
        "parent_source_commit": json.loads(provenance)["source_commit"],
        "source_commit": job["source_commit"],
        "parent_protocol_fingerprint": SOURCE_PROTOCOL,
        "protocol_fingerprint": target_protocol,
        "config_sha256": SEED42_CONFIG,
        "parent_provenance_sha256": PARENT_PROVENANCE,
        "parent_checkpoint_bytes": PARENT_CHECKPOINT_BYTES,
        "training_prefix_sha256": FIRST_EPISODE,
        "next_episode": 1,
        "completed_task_ids": ["44f52bb0"],
        "accepted_count": 0,
        "outer_updates": 0,
    }
