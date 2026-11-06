#!/usr/bin/env python3
"""Prepare audited upstream SEAL baseline sources/commands; never run GPU code."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from typing import Callable
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleep_repro.arc import EVALUATION_IDS, TRAIN_IDS, SEAL_COMMIT, load_arc  # noqa: E402

DATA_REGISTRY = ROOT / "configs/arc_data_manifest.json"
SOURCE_BASE = (
    f"https://raw.githubusercontent.com/Continual-Intelligence/SEAL/{SEAL_COMMIT}/"
)
SOURCE_PATHS = (
    "LICENSE",
    "requirements.txt",
    "few-shot/ttt.py",
    "few-shot/eval-self-edits.py",
    "few-shot/eval-self-edits-baseline.py",
    "few-shot/arclib/__init__.py",
    "few-shot/arclib/arc.py",
    "few-shot/arclib/augmenters.py",
    "few-shot/arclib/eval.py",
    "few-shot/arclib/messagers.py",
    "few-shot/arclib/representers.py",
    "few-shot/arclib/update_model.py",
    "few-shot/arclib/voting.py",
    "few-shot/inference/__init__.py",
    "few-shot/inference/preprocess.py",
    "few-shot/inference/engine.py",
    "few-shot/inference/engine_no_lora.py",
    "few-shot/utils/__init__.py",
    "few-shot/utils/np_cache.py",
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def download_bytes(url: str) -> bytes:
    with urlopen(
        Request(url, headers={"User-Agent": "sleep-repro-seal-baseline/0.1"}),
        timeout=60,
    ) as response:
        return response.read()


def _read_config(path: Path) -> dict:
    cfg = _json(path)
    if cfg.get("schema_version") != 1 or cfg.get("seal_commit") != SEAL_COMMIT:
        raise ValueError("baseline config requires schema 1 and the pinned SEAL commit")
    if cfg.get("integration_patch_version") != 2:
        raise ValueError(
            "baseline config requires reviewed integration patch version 2"
        )
    if cfg.get("mode") not in ("ttt", "icl"):
        raise ValueError("supported upstream baselines are ttt and icl")
    if (
        type(cfg.get("seed")) is not int
        or type(cfg.get("attempts")) is not int
        or not 1 <= cfg["attempts"] <= 5
    ):
        raise ValueError("declare an integer seed and 1..5 adaptation attempts")
    if cfg["mode"] == "icl" and cfg["attempts"] != 1:
        raise ValueError("the upstream ICL CLI produces one two-guess evaluation")
    revision = cfg.get("model_revision", "")
    if (
        cfg.get("model") != "meta-llama/Llama-3.2-1B-Instruct"
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError(
            "baseline requires the named model and an immutable revision SHA"
        )
    if set(cfg.get("source_files", {})) != set(SOURCE_PATHS):
        raise ValueError(
            "source registry must cover exactly the reviewed upstream files"
        )
    for path, info in cfg["source_files"].items():
        digest = info.get("sha256", "")
        if (
            info.get("url") != SOURCE_BASE + path
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"invalid source provenance: {path}")
    return cfg


def _replace_once(text: str, old: str, new: str, path: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"reviewed patch precondition failed for {path}: {old[:65]!r}")
    return text.replace(old, new, 1)


def patch_source(path: str, text: str) -> str:
    """Exact, guarded integration patches; recorded verbatim as a unified diff."""
    if path == "few-shot/ttt.py":
        replacements = (
            (
                "def main():",
                'def main():\n    baseline_config = json.loads(os.environ["SEAL_BASELINE_CONFIG"])\n    np.random.seed(baseline_config["seed"])\n    torch.manual_seed(baseline_config["seed"])',
            ),
            (
                'challenge_file="data/arc-agi_training_challenges.json"',
                'challenge_file=baseline_config["challenge_file"]',
            ),
            ('solution_file="data/arc-agi_training_solutions.json"', "test=True"),
            (
                "    # Setup tokenizer and model",
                '    if {task.name.rsplit("-", 1)[0] for task in tasks} != set(baseline_config["task_ids"]):\n        raise ValueError("Unexpected locked task set")\n    if any(int(task.name.rsplit("-", 1)[1]) > 1 for task in tasks):\n        raise ValueError("Upstream skip rule only reviewed for at most two test subtasks")\n    # Setup tokenizer and model',
            ),
            (
                '    model_name = "meta-llama/Llama-3.2-1B-Instruct"',
                '    model_name = baseline_config["model_name"]',
            ),
            ("    n_tasks = 350", "    n_tasks = len(tasks)"),
            (
                'output_dir=f"loras/lora-1B-train-350/{task.name}"',
                "output_dir=f\"{baseline_config['adapter_dir']}/{task.name}/{baseline_config['attempt']}\"",
            ),
        )
    elif path in (
        "few-shot/eval-self-edits.py",
        "few-shot/eval-self-edits-baseline.py",
    ):
        replacements = (
            (
                "tasks = read_tasks_from_single_file(args.data_file, solution_file=args.solution_file, test=True)",
                "tasks = read_tasks_from_single_file(args.data_file, solution_file=None, test=True)",
            ),
            (
                "    tasks = tasks[: args.num_examples]",
                '    if args.num_examples < len(tasks):\n        raise ValueError("Locked manifest forbids task truncation")',
            ),
            (
                'all_predictions_file = os.path.join(args.experiment_folder, "all_predictions.json")',
                'all_predictions_file = os.path.join(args.experiment_folder, f"all_predictions_{lora_edit_idx}.json")',
            ),
            (
                'submission_file = os.path.join(args.experiment_folder, "submission.json")',
                'submission_file = os.path.join(args.experiment_folder, f"submission_{lora_edit_idx}.json")',
            ),
        )
    else:
        return text
    for old, new in replacements:
        text = _replace_once(text, old, new, path)
    compile(text, path, "exec")
    return text


def _datasets(data_dir: Path, registry_path: Path):
    registry = _json(registry_path)
    required = {
        f"{split}_{kind}.json"
        for split in ("train", "eval")
        for kind in ("challenges", "solutions")
    }
    if set(registry.get("files", {})) != required:
        raise ValueError("data provenance must cover all four ARC files")
    for name, info in registry["files"].items():
        if _digest((data_dir / name).read_bytes()) != info["sha256"]:
            raise ValueError(f"data hash mismatch: {name}")
    return {
        split: load_arc(
            data_dir / f"{split}_challenges.json",
            data_dir / f"{split}_solutions.json",
            ids,
        )
        for split, ids in (("train", TRAIN_IDS), ("eval", EVALUATION_IDS))
    }


def _command(argv, cwd, env=None):
    return {"argv": [str(arg) for arg in argv], "cwd": str(cwd), "env": env or {}}


def command_plan(cfg: dict, output: Path, test_subtasks: int) -> dict:
    python = output / ".venv/bin/python"
    working = output / "upstream/runtime/few-shot"
    model = output / "model_snapshot"
    adaptations = []
    for attempt in range(cfg["attempts"] if cfg["mode"] == "ttt" else 0):
        episode = {
            "challenge_file": str(output / "data/eval_challenges.json"),
            "model_name": str(model),
            "task_ids": list(EVALUATION_IDS),
            "adapter_dir": str(output / "adapters"),
            "attempt": attempt,
            "seed": cfg["seed"] + attempt,
        }
        adaptations.append(
            _command(
                [python, "ttt.py"],
                working,
                {"SEAL_BASELINE_CONFIG": json.dumps(episode, sort_keys=True)},
            )
        )
    script = (
        "eval-self-edits.py" if cfg["mode"] == "ttt" else "eval-self-edits-baseline.py"
    )
    arguments = [
        python,
        script,
        "--experiment_folder",
        output / "results",
        "--pretrained_checkpoint",
        model,
        "--temperature",
        "0",
        "--n_sample",
        "1",
        "--seed",
        str(cfg["seed"]),
        "--data_file",
        output / "data/eval_challenges.json",
        "--solution_file",
        output / "data/eval_solutions.json",
        "--max_lora_rank",
        "128",
        "--include_n",
        "1",
        "--new_format",
        "--num_examples",
        str(test_subtasks),
    ]
    if cfg["mode"] == "ttt":
        arguments += [
            "--lora_checkpoints_folder",
            output / "adapters",
            "--n_self_edits",
            str(cfg["attempts"]),
        ]
    return {
        "model_download": _command(
            [
                python,
                "-c",
                "from huggingface_hub import snapshot_download; snapshot_download(repo_id="
                + repr(cfg["model"])
                + ", revision="
                + repr(cfg["model_revision"])
                + ", local_dir="
                + repr(str(model))
                + ")",
            ],
            output,
        ),
        "adaptation_commands": adaptations,
        "evaluation_command": _command(arguments, working),
        "metric": "upstream_two_guess_voting_task_attempt_success",
        "gpu_executed": False,
    }


def _shell_command(command):
    argv = [
        *["env", "PYTHONHASHSEED=0"],
        *[f"{key}={value}" for key, value in command["env"].items()],
        *command["argv"],
    ]
    return "(cd " + shlex.quote(command["cwd"]) + " && " + shlex.join(argv) + ")"


def _scripts(cfg: dict, output: Path, plan: dict):
    python = output / ".venv/bin/python"
    setup = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(
            [
                shlex.join(["python3.11", "-m", "venv", str(output / ".venv")]),
                shlex.join(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(output / "upstream/original/requirements.txt"),
                    ]
                ),
                shlex.join([str(python), "-m", "pip", "check"]),
            ]
        )
        + "\n"
    )
    run = "#!/usr/bin/env bash\nset -euo pipefail\n"
    run += (
        shlex.join(
            [
                str(python),
                str(Path(__file__).resolve()),
                "verify",
                "--package",
                str(output),
            ]
        )
        + "\n"
    )
    run += "mkdir -p " + shlex.quote(str(output / "results")) + "\n"
    run += (
        shlex.join([str(python), "-m", "pip", "freeze"])
        + " > "
        + shlex.quote(str(output / "results/dependencies.freeze.txt"))
        + "\n"
    )
    for command in [
        plan["model_download"],
        *plan["adaptation_commands"],
        plan["evaluation_command"],
    ]:
        run += _shell_command(command) + "\n"
    run += (
        shlex.join(
            [
                "cp",
                str(output / "upstream/runtime/few-shot/final_results.json"),
                str(output / "results/final_results.json"),
            ]
        )
        + "\n"
    )
    run += (
        shlex.join(
            [
                str(python),
                str(Path(__file__).resolve()),
                "summarize",
                "--package",
                str(output),
            ]
        )
        + "\n"
    )
    return setup, run


def verify_package(output: Path) -> dict:
    manifest = _json(output / "manifest.json")
    if manifest.get("integration_patch_version") != 2:
        raise ValueError(
            "legacy integration patch lacks per-attempt preservation; prepare a fresh v2 package"
        )
    for path, expected in manifest["artifact_hashes"].items():
        candidate = output / path
        if not candidate.is_file() or _digest(candidate.read_bytes()) != expected:
            raise ValueError(
                f"prepared artifact hash mismatch: {path}; refusing overwrite"
            )
    return manifest


def prepare_baseline(
    config_path: str | Path,
    data_dir: str | Path,
    output: str | Path,
    *,
    downloader: Callable[[str], bytes] = download_bytes,
    data_registry_path: str | Path = DATA_REGISTRY,
) -> dict:
    config_path = Path(config_path)
    cfg = _read_config(config_path)
    datasets = _datasets(Path(data_dir), Path(data_registry_path))
    output = Path(output).resolve()
    config_hash = _digest(config_path.read_bytes())
    if output.exists():
        manifest = verify_package(output)
        if manifest["config_sha256"] != config_hash:
            raise ValueError("prepared package config hash mismatch")
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".seal-stage-", dir=output.parent) as temporary:
        stage = Path(temporary)
        diff = []
        for path, info in cfg["source_files"].items():
            payload = downloader(info["url"])
            if not isinstance(payload, bytes) or _digest(payload) != info["sha256"]:
                raise ValueError(f"upstream source hash mismatch: {path}")
            original = stage / "upstream/original" / path
            runtime = stage / "upstream/runtime" / path
            original.parent.mkdir(parents=True, exist_ok=True)
            runtime.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(payload)
            text = payload.decode("utf-8")
            patched = patch_source(path, text)
            runtime.write_bytes(patched.encode("utf-8"))
            diff.extend(
                difflib.unified_diff(
                    text.splitlines(keepends=True),
                    patched.splitlines(keepends=True),
                    fromfile="a/" + path,
                    tofile="b/" + path,
                )
            )
        for split, data in datasets.items():
            challenges = {
                task.task_id: {
                    "train": [asdict(pair) for pair in task.demonstrations],
                    "test": [{"input": grid} for grid in task.test_inputs],
                }
                for task in data.tasks
            }
            _write_json(stage / f"data/{split}_challenges.json", challenges)
            _write_json(stage / f"data/{split}_solutions.json", dict(data.solutions))
        train_subtasks = [
            f"{task.task_id}:{index}"
            for task in datasets["train"].tasks
            for index in range(len(task.test_inputs))
        ]
        eval_subtasks = [
            f"{task.task_id}:{index}"
            for task in datasets["eval"].tasks
            for index in range(len(task.test_inputs))
        ]
        plan = command_plan(cfg, output, len(eval_subtasks))
        _write_json(stage / "config.json", cfg)
        _write_json(stage / "commands.json", plan)
        (stage / "source.patch").write_text("".join(diff), encoding="utf-8")
        setup, run = _scripts(cfg, output, plan)
        (stage / "setup.sh").write_text(setup)
        (stage / "run.sh").write_text(run)
        hashes = {
            str(path.relative_to(stage)): _digest(path.read_bytes())
            for path in stage.rglob("*")
            if path.is_file()
        }
        manifest = {
            "seal_commit": SEAL_COMMIT,
            "config_sha256": config_hash,
            "mode": cfg["mode"],
            "integration_patch_version": cfg["integration_patch_version"],
            "integration_patch_sha256": _digest((stage / "source.patch").read_bytes()),
            "train_ids": list(TRAIN_IDS),
            "evaluation_ids": list(EVALUATION_IDS),
            "train_subtask_ids": train_subtasks,
            "evaluation_subtask_ids": eval_subtasks,
            "artifact_hashes": hashes,
            "source_files": cfg["source_files"],
            "input_data_hashes": {
                split: dict(data.file_hashes) for split, data in datasets.items()
            },
            "metric": plan["metric"],
            "gpu_executed": False,
            "dependency_resolution": "upstream direct pins recorded; Linux CUDA environment not validated",
            "deviations": [
                "explicit reconstructed 11/8 source task manifest",
                "immutable local model snapshot and declared attempt seeds",
                "TTT data/output paths configured; original trainer unchanged",
                "evaluation preprocessing sees input placeholders, never hidden outputs",
                "task truncation refused; all selected test subtasks retained",
                "indexed per-attempt raw prediction and submission filenames prevent overwrite",
            ],
        }
        _write_json(stage / "manifest.json", manifest)
        if output.exists():
            raise ValueError(
                "destination appeared during preparation; refusing overwrite"
            )
        stage.rename(output)
    return manifest


def summarize_results(results: dict, *, task_ids, attempts: int) -> dict:
    ids = tuple(task_ids)
    if set(results) != {str(index) for index in range(attempts)}:
        raise ValueError(
            "upstream result attempt indices must cover the declared attempts exactly"
        )
    outcomes = {task_id: [] for task_id in ids}
    for record in results.values():
        names, correct = record.get("task_id", {}), record.get("correct", {})
        if (
            set(names) != set(correct)
            or len(names) != len(ids)
            or set(names.values()) != set(ids)
        ):
            raise ValueError(
                "upstream results must cover every locked task exactly once"
            )
        for index, task_id in names.items():
            value = correct[index]
            if type(value) is not bool:
                raise ValueError("upstream task correctness must be boolean")
            outcomes[task_id].append(value)
    successes = sum(sum(values) for values in outcomes.values())
    return {
        "metric": "upstream_two_guess_voting_task_attempt_success",
        "successful_task_attempts": successes,
        "total_task_attempts": len(ids) * attempts,
        "success_rate": successes / (len(ids) * attempts),
        "per_task": [
            {"task_id": task_id, "attempt_outcomes": outcomes[task_id]}
            for task_id in ids
        ],
        "note": "Two voted guesses per test subtask; all subtasks required. Distinct from independent one-grid metric.",
        "paper_result_reproduced": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument(
        "--config", type=Path, default=ROOT / "configs/seal_upstream_ttt.json"
    )
    prepare.add_argument("--data-dir", type=Path, default=ROOT / "data/arc")
    prepare.add_argument("--output", type=Path, required=True)
    for command in ("verify", "summarize"):
        child = sub.add_parser(command)
        child.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_baseline(args.config, args.data_dir, args.output)
        print(
            json.dumps(
                {
                    "package": str(args.output.resolve()),
                    "gpu_executed": result["gpu_executed"],
                }
            )
        )
    else:
        manifest = verify_package(args.package)
        if args.command == "verify":
            print(
                json.dumps({"verified": True, "seal_commit": manifest["seal_commit"]})
            )
        else:
            cfg = _json(args.package / "config.json")
            summary = summarize_results(
                _json(args.package / "results/final_results.json"),
                task_ids=EVALUATION_IDS,
                attempts=cfg["attempts"],
            )
            _write_json(args.package / "results/upstream_summary.json", summary)
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
