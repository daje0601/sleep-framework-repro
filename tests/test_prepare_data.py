import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from sleep_repro.arc import EVALUATION_IDS, SEAL_COMMIT, TRAIN_IDS, load_arc


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)

SOURCE_NAMES = {
    "train_challenges.json": "arc-agi_training_challenges_filtered_1B_training_set.json",
    "train_solutions.json": "arc-agi_training_solutions_filtered_1B_training_set.json",
    "eval_challenges.json": "arc-agi_evaluation_challenges_filtered_1B_eval_set.json",
    "eval_solutions.json": "arc-agi_evaluation_solutions_filtered_1B_eval_set.json",
}


def fixture_downloads(tmp_path):
    payloads = {}
    for split, ids in (("train", TRAIN_IDS), ("eval", EVALUATION_IDS)):
        challenges = {}
        solutions = {}
        for index, task_id in enumerate((*ids, "unselected")):
            test_count = 2 if index == 0 else 1
            challenges[task_id] = {
                "train": [
                    {"input": [[1]], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]},
                ],
                "test": [{"input": [[5]]}] * test_count,
            }
            solutions[task_id] = [[[6]]] * test_count
        payloads[f"{split}_challenges.json"] = json.dumps(challenges).encode()
        payloads[f"{split}_solutions.json"] = json.dumps(solutions).encode()
    files = {}
    by_url = {}
    for local_name, source_name in SOURCE_NAMES.items():
        url = f"https://raw.githubusercontent.com/Continual-Intelligence/SEAL/{SEAL_COMMIT}/few-shot/data/{source_name}"
        files[local_name] = {
            "url": url,
            "sha256": hashlib.sha256(payloads[local_name]).hexdigest(),
        }
        by_url[url] = payloads[local_name]
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"seal_commit": SEAL_COMMIT, "files": files}))
    return provenance, payloads, by_url


def test_prepare_downloads_pinned_bytes_and_records_full_source_and_selected_subtasks(
    tmp_path,
):
    provenance, payloads, by_url = fixture_downloads(tmp_path)
    output = tmp_path / "data" / "arc"
    manifest = prepare_data.prepare_data(
        output, downloader=by_url.__getitem__, provenance_path=provenance
    )
    for name, expected_bytes in payloads.items():
        assert (output / name).read_bytes() == expected_bytes
        assert (
            manifest["files"][name]["sha256"]
            == hashlib.sha256(expected_bytes).hexdigest()
        )
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert manifest["train_ids"] == list(TRAIN_IDS)
    assert manifest["evaluation_ids"] == list(EVALUATION_IDS)
    assert len(manifest["train_subtask_ids"]) == 12
    assert manifest["train_subtask_ids"][:2] == ["44f52bb0:0", "44f52bb0:1"]
    assert len(manifest["evaluation_subtask_ids"]) == 9
    dataset = load_arc(
        output / "train_challenges.json",
        output / "train_solutions.json",
        manifest["train_ids"],
    )
    assert len(dataset.tasks) == 11


def test_existing_valid_data_is_verified_without_another_download(tmp_path):
    provenance, _, by_url = fixture_downloads(tmp_path)
    output = tmp_path / "arc"
    expected = prepare_data.prepare_data(
        output, downloader=by_url.__getitem__, provenance_path=provenance
    )

    def unavailable(_):
        raise AssertionError("verified data should work offline")

    assert (
        prepare_data.prepare_data(
            output, downloader=unavailable, provenance_path=provenance
        )
        == expected
    )


@pytest.mark.parametrize("mutation", ["data", "manifest", "missing_file"])
def test_existing_modified_or_partial_data_is_never_silently_overwritten(
    tmp_path, mutation
):
    provenance, _, by_url = fixture_downloads(tmp_path)
    output = tmp_path / "arc"
    prepare_data.prepare_data(
        output, downloader=by_url.__getitem__, provenance_path=provenance
    )
    if mutation == "data":
        (output / "train_challenges.json").write_bytes(b"modified")
    elif mutation == "manifest":
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["train_ids"] = []
        (output / "manifest.json").write_text(json.dumps(manifest))
    else:
        (output / "train_solutions.json").unlink()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError):
        prepare_data.prepare_data(
            output, downloader=by_url.__getitem__, provenance_path=provenance
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_untrusted_download_hash_is_rejected_before_data_directory_appears(tmp_path):
    provenance, _, by_url = fixture_downloads(tmp_path)
    output = tmp_path / "arc"

    def corrupted(url):
        return by_url[url] + b" "

    with pytest.raises(ValueError, match="hash"):
        prepare_data.prepare_data(
            output, downloader=corrupted, provenance_path=provenance
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".arc-stage-*"))


def test_interrupted_download_leaves_no_partial_dataset(tmp_path):
    provenance, _, by_url = fixture_downloads(tmp_path)
    output = tmp_path / "arc"
    first_url = next(iter(by_url))

    def interrupted(url):
        if url != first_url:
            raise OSError("download interrupted")
        return by_url[url]

    with pytest.raises(OSError, match="interrupted"):
        prepare_data.prepare_data(
            output, downloader=interrupted, provenance_path=provenance
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".arc-stage-*"))


def test_provenance_must_reference_pinned_upstream_and_all_four_files(tmp_path):
    provenance, _, by_url = fixture_downloads(tmp_path)
    record = json.loads(provenance.read_text())
    record["files"]["train_challenges.json"]["url"] = (
        "https://example.com/unpinned.json"
    )
    provenance.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="source"):
        prepare_data.prepare_data(
            tmp_path / "arc", downloader=by_url.__getitem__, provenance_path=provenance
        )
