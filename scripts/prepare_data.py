#!/usr/bin/env python3
"""Fetch and verify the pinned SEAL ARC files using only the standard library.

The committed registry fixes upstream URLs and byte hashes. A complete dataset is
staged beside its destination and renamed into place only after all checks pass.
Existing data are verified offline, and modified/partial data are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Callable
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Support the documented command before an editable package installation.
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from sleep_repro.arc import (  # noqa: E402
    EVALUATION_IDS,
    SEAL_COMMIT,
    TRAIN_IDS,
    load_arc,
    reconstruction_manifest,
)

PROVENANCE_PATH = REPOSITORY_ROOT / "configs" / "arc_data_manifest.json"
SOURCE_NAMES = {
    "train_challenges.json": "arc-agi_training_challenges_filtered_1B_training_set.json",
    "train_solutions.json": "arc-agi_training_solutions_filtered_1B_training_set.json",
    "eval_challenges.json": "arc-agi_evaluation_challenges_filtered_1B_eval_set.json",
    "eval_solutions.json": "arc-agi_evaluation_solutions_filtered_1B_eval_set.json",
}
SOURCE_BASE = f"https://raw.githubusercontent.com/Continual-Intelligence/SEAL/{SEAL_COMMIT}/few-shot/data/"
SPLIT_RATIONALE = (
    "Public protocol reconstruction: 11 unique training tasks reached by the "
    "upstream self-edit generation command, and 8 unique evaluation tasks in the "
    "saved upstream final results. Retain every test subtask of those fixed source "
    "tasks; do not inherit CLI truncation, skips, shuffling, or result-dependent filtering. "
    "This is not a verified author-provided Sleep task manifest."
)


def download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "sleep-framework-repro-data/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _read_provenance(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read pinned provenance registry: {path}") from error
    if not isinstance(record, dict) or record.get("seal_commit") != SEAL_COMMIT:
        raise ValueError("provenance must use the pinned SEAL commit")
    files = record.get("files")
    if not isinstance(files, dict) or set(files) != set(SOURCE_NAMES):
        raise ValueError("provenance must name all four ARC source files")
    for local_name, source_name in SOURCE_NAMES.items():
        info = files[local_name]
        if not isinstance(info, dict) or info.get("url") != SOURCE_BASE + source_name:
            raise ValueError(f"unpinned source URL: {local_name}")
        digest = info.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"invalid expected SHA256 hash: {local_name}")
    return record


def _check_hash(path: Path, expected: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"missing or unreadable dataset file: {path}") from error
    if actual != expected:
        raise ValueError(f"dataset hash mismatch: {path.name}; refusing to overwrite")


def _build_manifest(directory: Path, provenance: dict) -> dict:
    datasets = {}
    for split, ids in (("train", TRAIN_IDS), ("eval", EVALUATION_IDS)):
        datasets[split] = load_arc(
            directory / f"{split}_challenges.json",
            directory / f"{split}_solutions.json",
            ids,
        )
    manifest = reconstruction_manifest()
    manifest["files"] = {
        name: {"url": info["url"], "sha256": info["sha256"]}
        for name, info in provenance["files"].items()
    }
    manifest["data_hashes"] = {
        name: info["sha256"] for name, info in manifest["files"].items()
    }
    manifest["split_rationale"] = SPLIT_RATIONALE
    manifest["train_subtask_ids"] = [
        f"{task.task_id}:{index}"
        for task in datasets["train"].tasks
        for index in range(len(task.test_inputs))
    ]
    manifest["evaluation_subtask_ids"] = [
        f"{task.task_id}:{index}"
        for task in datasets["eval"].tasks
        for index in range(len(task.test_inputs))
    ]
    manifest["upstream_source_task_counts"] = {
        split: len(
            json.loads(
                (directory / f"{split}_challenges.json").read_text(encoding="utf-8")
            )
        )
        for split in ("train", "eval")
    }
    return manifest


def prepare_data(
    output: str | Path,
    *,
    downloader: Callable[[str], bytes] = download_bytes,
    provenance_path: str | Path = PROVENANCE_PATH,
) -> dict:
    """Create a complete verified dataset or verify an existing one offline."""
    output = Path(output)
    provenance = _read_provenance(Path(provenance_path))
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"dataset destination is not a directory: {output}")
        for name, info in provenance["files"].items():
            _check_hash(output / name, info["sha256"])
        expected_manifest = _build_manifest(output, provenance)
        try:
            existing_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                "missing or invalid dataset manifest; refusing to overwrite"
            ) from error
        if existing_manifest != expected_manifest:
            raise ValueError("dataset manifest mismatch; refusing to overwrite")
        return expected_manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output.name}-stage-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        for name, info in provenance["files"].items():
            payload = downloader(info["url"])
            if not isinstance(payload, bytes):
                raise ValueError("downloader must return bytes")
            if hashlib.sha256(payload).hexdigest() != info["sha256"]:
                raise ValueError(f"download hash mismatch: {name}")
            (staging / name).write_bytes(payload)
        manifest = _build_manifest(staging, provenance)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise ValueError(
                "dataset destination appeared during download; refusing to overwrite"
            )
        staging.rename(output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "data" / "arc")
    args = parser.parse_args(argv)
    manifest = prepare_data(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "seal_commit": manifest["seal_commit"],
                "train_tasks": len(manifest["train_ids"]),
                "evaluation_tasks": len(manifest["evaluation_ids"]),
                "manifest": str((args.output / "manifest.json").resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
