import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from sleep_repro.arc import EVALUATION_IDS, TRAIN_IDS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_seal_baseline.py"
spec = importlib.util.spec_from_file_location("seal_baseline", SCRIPT)
seal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seal)


def fixtures(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    config = json.loads((repository / "configs" / "seal_upstream_ttt.json").read_text())
    sources = {
        path: b"# controlled upstream fixture\n" for path in config["source_files"]
    }
    sources["few-shot/ttt.py"] = b"""import os, json

def main():
    # Load tasks
    tasks = read_tasks_from_single_file(
        challenge_file="data/arc-agi_training_challenges.json",
        solution_file="data/arc-agi_training_solutions.json"
    )
    # Setup tokenizer and model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    n_tasks = 350
    output_dir=f"loras/lora-1B-train-350/{task.name}"
"""
    evaluation = b"""tasks = read_tasks_from_single_file(args.data_file, solution_file=args.solution_file, test=True)
if args.num_examples is not None:
    tasks = tasks[: args.num_examples]
all_predictions_file = os.path.join(args.experiment_folder, "all_predictions.json")
submission_file = os.path.join(args.experiment_folder, "submission.json")
"""
    sources["few-shot/eval-self-edits.py"] = evaluation
    sources["few-shot/eval-self-edits-baseline.py"] = evaluation
    urls = {}
    for path, info in config["source_files"].items():
        info["sha256"] = hashlib.sha256(sources[path]).hexdigest()
        urls[info["url"]] = sources[path]
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    directory = tmp_path / "data"
    directory.mkdir()
    registry = {"files": {}}
    for split, ids in [("train", TRAIN_IDS), ("eval", EVALUATION_IDS)]:
        challenges = {
            task_id: {
                "train": [
                    {"input": [[1]], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]},
                ],
                "test": [{"input": [[5]], "output": [[9]]}],
            }
            for task_id in ids
        }
        challenges[ids[0]]["test"].append({"input": [[6]], "output": [[9]]})
        solutions = {
            task_id: [[[8]]] * len(record["test"])
            for task_id, record in challenges.items()
        }
        for kind, payload in [("challenges", challenges), ("solutions", solutions)]:
            name = f"{split}_{kind}.json"
            content = json.dumps(payload).encode()
            (directory / name).write_bytes(content)
            registry["files"][name] = {"sha256": hashlib.sha256(content).hexdigest()}
    registry_file = tmp_path / "data_registry.json"
    registry_file.write_text(json.dumps(registry))
    return config_file, directory, registry_file, urls


def test_preparation_builds_exact_source_and_subtask_manifest_without_gpu_imports(
    tmp_path,
):
    config, data, registry, urls = fixtures(tmp_path)
    output = tmp_path / "package"
    manifest = seal.prepare_baseline(
        config, data, output, downloader=urls.__getitem__, data_registry_path=registry
    )
    selected = json.loads((output / "data" / "eval_challenges.json").read_text())
    assert set(selected) == set(EVALUATION_IDS)
    assert sum(len(task["test"]) for task in selected.values()) == 9
    assert all(
        set(pair) == {"input"} for task in selected.values() for pair in task["test"]
    )
    assert len(manifest["train_subtask_ids"]) == 12
    assert len(manifest["evaluation_subtask_ids"]) == 9
    assert (output / "upstream" / "original" / "LICENSE").read_bytes() == urls[
        json.loads(config.read_text())["source_files"]["LICENSE"]["url"]
    ]
    assert manifest["gpu_executed"] is False
    plan = json.loads((output / "commands.json").read_text())
    assert len(plan["adaptation_commands"]) == 5
    assert plan["evaluation_command"]["argv"][-2:] == ["--n_self_edits", "5"]
    assert "--new_format" in plan["evaluation_command"]["argv"]
    assert (output / "source.patch").stat().st_size > 0
    compile((output / "upstream/runtime/few-shot/ttt.py").read_text(), "ttt.py", "exec")


def test_config_and_source_hash_mismatch_never_create_partial_output(tmp_path):
    config, data, registry, urls = fixtures(tmp_path)
    first = next(iter(urls))
    urls[first] += b"changed"
    output = tmp_path / "package"
    with pytest.raises(ValueError, match="hash"):
        seal.prepare_baseline(
            config,
            data,
            output,
            downloader=urls.__getitem__,
            data_registry_path=registry,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".seal-stage-*"))


def test_patch_aborts_on_unreviewed_source_structure(tmp_path):
    config, data, registry, urls = fixtures(tmp_path)
    record = json.loads(config.read_text())
    path = "few-shot/ttt.py"
    url = record["source_files"][path]["url"]
    urls[url] = b"def main():\n    pass\n"
    record["source_files"][path]["sha256"] = hashlib.sha256(urls[url]).hexdigest()
    config.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="patch"):
        seal.prepare_baseline(
            config,
            data,
            tmp_path / "package",
            downloader=urls.__getitem__,
            data_registry_path=registry,
        )


def test_changed_data_or_prepared_artifact_is_not_overwritten(tmp_path):
    config, data, registry, urls = fixtures(tmp_path)
    output = tmp_path / "package"
    seal.prepare_baseline(
        config, data, output, downloader=urls.__getitem__, data_registry_path=registry
    )
    source = output / "upstream/runtime/few-shot/ttt.py"
    source.write_text("modified")
    with pytest.raises(ValueError, match="hash"):
        seal.prepare_baseline(
            config,
            data,
            output,
            downloader=urls.__getitem__,
            data_registry_path=registry,
        )
    assert source.read_text() == "modified"
    (data / "eval_challenges.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        seal.prepare_baseline(
            config,
            data,
            tmp_path / "other",
            downloader=urls.__getitem__,
            data_registry_path=registry,
        )


def test_legacy_package_without_per_attempt_preservation_is_rejected(tmp_path):
    config, data, registry, urls = fixtures(tmp_path)
    output = tmp_path / "package"
    seal.prepare_baseline(
        config, data, output, downloader=urls.__getitem__, data_registry_path=registry
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("integration_patch_version")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="patch"):
        seal.verify_package(output)


def evaluation_fixture_source():
    return """tasks = read_tasks_from_single_file(args.data_file, solution_file=args.solution_file, test=True)
if args.num_examples is not None:
    tasks = tasks[: args.num_examples]
all_predictions_file = os.path.join(args.experiment_folder, "all_predictions.json")
submission_file = os.path.join(args.experiment_folder, "submission.json")
"""


def test_patch_uses_public_inputs_and_refuses_truncation_when_executed():
    import os

    source = evaluation_fixture_source()
    patched = seal.patch_source("few-shot/eval-self-edits.py", source)
    from types import SimpleNamespace

    observed = []

    def read_tasks(path, *, solution_file, test):
        observed.append((path, solution_file, test))
        return [1, 2]

    context = {
        "args": SimpleNamespace(
            data_file="public.json",
            solution_file="secret.json",
            num_examples=2,
            experiment_folder="results",
        ),
        "read_tasks_from_single_file": read_tasks,
        "os": os,
        "lora_edit_idx": 0,
    }
    exec(compile(patched, "reviewed_patch", "exec"), context)
    assert observed == [("public.json", None, True)]
    assert context["tasks"] == [1, 2]
    context["args"].num_examples = 1
    with pytest.raises(ValueError, match="truncation"):
        exec(compile(patched, "reviewed_patch", "exec"), context)


@pytest.mark.parametrize(
    "source_path",
    ["few-shot/eval-self-edits.py", "few-shot/eval-self-edits-baseline.py"],
)
def test_indexed_raw_artifacts_survive_all_five_evaluation_attempts(
    tmp_path, source_path
):
    import os
    from types import SimpleNamespace

    patched = seal.patch_source(source_path, evaluation_fixture_source())
    context = {
        "args": SimpleNamespace(
            data_file="public.json",
            solution_file="private.json",
            num_examples=2,
            experiment_folder=str(tmp_path),
        ),
        "read_tasks_from_single_file": lambda *args, **kwargs: [1, 2],
        "os": os,
    }
    for attempt in range(5):
        context["lora_edit_idx"] = attempt
        exec(compile(patched, "reviewed_patch", "exec"), context)
        Path(context["all_predictions_file"]).write_text(
            json.dumps({"attempt": attempt, "raw": "prediction"})
        )
        Path(context["submission_file"]).write_text(
            json.dumps({"attempt": attempt, "guesses": [[[1]], [[2]]]})
        )
    for attempt in range(5):
        assert (
            json.loads((tmp_path / f"all_predictions_{attempt}.json").read_text())[
                "attempt"
            ]
            == attempt
        )
        assert (
            json.loads((tmp_path / f"submission_{attempt}.json").read_text())["attempt"]
            == attempt
        )
    assert len(list(tmp_path.iterdir())) == 10


def test_upstream_results_keep_two_guess_metric_separate_and_reject_missing_attempts(
    tmp_path,
):
    results = {
        str(i): {
            "task_id": {str(j): task_id for j, task_id in enumerate(EVALUATION_IDS)},
            "correct": {str(j): j < 4 for j in range(8)},
        }
        for i in range(5)
    }
    result = seal.summarize_results(results, task_ids=EVALUATION_IDS, attempts=5)
    assert result["metric"] == "upstream_two_guess_voting_task_attempt_success"
    assert result["successful_task_attempts"] == 20
    assert result["total_task_attempts"] == 40
    assert result["success_rate"] == 0.5
    assert result["paper_result_reproduced"] is False
    results.pop("4")
    with pytest.raises(ValueError, match="attempt"):
        seal.summarize_results(results, task_ids=EVALUATION_IDS, attempts=5)


def worker_config():
    return {'source_commit': 'a' * 40, 'mount_path': '/persistent', 'run_id': 'review-only',
            'image': 'pytorch/pytorch@sha256:3deb633094acb02ba490e4da21bb37337327860e884c5b2290042a6db7d6b572',
            'resource_spec': 'reviewed-gpu', 'timeout_seconds': 1800, 'stage': 'environment'}


def run_worker_validation(config):
    script = SCRIPT.with_name('cloud_seal_job.sh')
    return subprocess.run(['bash', str(script), '--validate-only'],
                          env={'PATH': os.environ['PATH'], 'SLEEP_SEAL_JOB_JSON': json.dumps(config)},
                          capture_output=True, text=True, timeout=5)


def test_cloud_worker_validation_prepares_no_jobs_or_gpu_imports():
    result = run_worker_validation(worker_config())
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'validated': True, 'stage': 'environment', 'submitted': False}


@pytest.mark.parametrize('field,value', [('timeout_seconds', 0), ('timeout_seconds', True),
                                         ('timeout_seconds', 86401), ('source_commit', 'main'),
                                         ('image', 'pytorch/pytorch:latest'), ('mount_path', '/tmp'),
                                         ('run_id', '../other'), ('stage', 'train'),
                                         ('HF_TOKEN', 'not-a-secret-reference')])
def test_cloud_worker_rejects_unbounded_unpinned_or_secret_bearing_specs(field, value):
    config = worker_config()
    config[field] = value
    result = run_worker_validation(config)
    assert result.returncode != 0
