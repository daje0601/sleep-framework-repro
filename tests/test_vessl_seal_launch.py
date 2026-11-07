import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/vessl_seal_launch.py"


def launcher():
    spec = importlib.util.spec_from_file_location("vessl_seal_launch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return {
        "org": "research",
        "team": "arc",
        "job_name": "seal-environment-01",
        "run_id": "seal-environment-01",
        "resource_spec": "reviewed-gpu",
        "object_volume": "reviewed-private-volume",
        "mount_path": "/persistent",
        "hf_secret": "reviewed-model-access",
        "repository": "https://github.com/daje0601/sleep-framework-repro.git",
        "source_commit": "ee39e0dbdfdb74edea0abe42283788fe6451b525",
        "image": "pytorch/pytorch@sha256:3deb633094acb02ba490e4da21bb37337327860e884c5b2290042a6db7d6b572",
        "python_version": "3.11",
        "timeout_seconds": 1800,
        "queue_timeout_seconds": 600,
        "stage": "environment",
    }


def test_builds_one_bounded_pinned_upstream_worker_command_and_env_whitelist():
    module = launcher()
    cfg = config()
    command = module.build_command(cfg)
    assert command[:7] == [
        "vesslctl",
        "--org",
        "research",
        "--team",
        "arc",
        "job",
        "create",
    ]
    assert command[command.index("--secret") + 1] == "HF_TOKEN=reviewed-model-access"
    assert (
        command[command.index("--object-volume") + 1]
        == "reviewed-private-volume:/persistent"
    )
    outer = shlex.split(command[command.index("--cmd") + 1])
    assert outer[:5] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "1800s",
        "bash",
    ]
    bootstrap = outer[-1]
    assert cfg["source_commit"] in bootstrap
    assert "scripts/cloud_seal_job.sh" in bootstrap
    assert "scripts/cloud_job.sh" not in bootstrap
    assert "python3.11" in bootstrap
    assert "SLEEP_DEADLINE_UNIX=" in bootstrap
    env = dict(
        command[i + 1].split("=", 1) for i, arg in enumerate(command) if arg == "--env"
    )
    assert set(env) == {
        "SLEEP_SEAL_JOB_JSON",
        "HF_HUB_DISABLE_TELEMETRY",
        "TOKENIZERS_PARALLELISM",
    }
    worker = json.loads(env["SLEEP_SEAL_JOB_JSON"])
    assert set(worker) == {
        "source_commit",
        "image",
        "resource_spec",
        "mount_path",
        "run_id",
        "timeout_seconds",
        "stage",
    }
    assert worker["stage"] == "environment"
    assert "HF_TOKEN" not in env
    assert "preflight_gate" not in worker
    subprocess.run(["bash", "-n"], input=bootstrap, text=True, check=True)


def test_icl_stage_reaches_worker_without_core_protocol_arguments():
    cfg = {**config(), "stage": "icl"}
    command = launcher().build_command(cfg)
    env = dict(
        command[i + 1].split("=", 1) for i, arg in enumerate(command) if arg == "--env"
    )
    assert json.loads(env["SLEEP_SEAL_JOB_JSON"])["stage"] == "icl"
    assert "SLEEP_EXPERIMENT_JSON" not in env


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "pytorch/pytorch:latest"),
        ("python_version", "3.12"),
        ("source_commit", "main"),
        ("source_commit", "a" * 40),
        ("timeout_seconds", 0),
        ("timeout_seconds", True),
        ("timeout_seconds", 86401),
        ("queue_timeout_seconds", 0),
        ("queue_timeout_seconds", False),
        ("queue_timeout_seconds", 86401),
        ("repository", "https://token@github.com/example/repo.git"),
        ("repository", "https://github.com/example/repo.git?token=secret"),
        ("repository", "https://github.com/../repo.git"),
        ("repository", "https://github.com/example/.."),
        ("hf_secret", "hf_not_a_secret_reference"),
        ("run_id", "../old"),
        ("mount_path", "/tmp"),
        ("mount_path", "/persistent/../etc"),
        ("stage", "ttt"),
        ("stage", "train"),
        ("env", {"HF_TOKEN": "forbidden"}),
        ("HF_TOKEN", "forbidden"),
        ("preflight_gate", "/tmp/fake.json"),
    ],
)
def test_unsafe_or_unsupported_job_specifications_fail_before_submission(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError):
        launcher().build_command(cfg)


def test_dry_run_never_submits_or_calls_provider_processes(
    tmp_path, monkeypatch, capsys
):
    module = launcher()
    path = tmp_path / "private-job.json"
    path.write_text(json.dumps(config()))
    monkeypatch.setattr(
        module,
        "submit_and_monitor",
        lambda *args, **kwargs: pytest.fail("dry run submitted"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry run contacted provider"),
    )
    assert module.main(["--config", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["stage"] == "environment"
    assert len(report["worker_config_sha256"]) == 64


def test_explicit_submit_reuses_one_monitored_job_with_separate_budgets(
    tmp_path, monkeypatch
):
    module = launcher()
    cfg = config()
    path = tmp_path / "private-job.json"
    path.write_text(json.dumps(cfg))
    calls = []

    def monitor(spec, command):
        calls.append((spec, command))
        return {"success": True}

    monkeypatch.setattr(module, "submit_and_monitor", monitor)
    assert module.main(["--config", str(path), "--submit"]) == 0
    assert len(calls) == 1
    assert calls[0][0]["queue_timeout_seconds"] == 600
    assert calls[0][0]["timeout_seconds"] == 1800
    assert calls[0][1][calls[0][1].index("--name") + 1] == cfg["job_name"]


def test_generated_worker_environment_is_accepted_by_published_worker_schema():
    command = launcher().build_command(config())
    forwarded = dict(
        command[i + 1].split("=", 1) for i, arg in enumerate(command) if arg == "--env"
    )
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("cloud_seal_job.sh")), "--validate-only"],
        env={"PATH": os.environ["PATH"], **forwarded},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "validated": True,
        "stage": "environment",
        "submitted": False,
    }
