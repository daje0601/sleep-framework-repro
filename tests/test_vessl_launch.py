import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shlex
import signal

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vessl_launch.py"


def launcher():
    spec = importlib.util.spec_from_file_location("vessl_launch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return {
        "org": "research",
        "team": "arc",
        "job_name": "sleep-preflight-01",
        "run_id": "preflight-01",
        "resource_spec": "resource-123",
        "image": "registry.example/pytorch@sha256:" + "a" * 64,
        "object_volume": "objectvol-123",
        "mount_path": "/persistent",
        "hf_secret": "hf-gated-model",
        "repository": "https://github.com/example/sleep.git",
        "source_commit": "b" * 40,
        "timeout_seconds": 1800,
        "stage": "preflight",
        "torch_version": "2.15.0",
        "linux_dependencies": ["numpy==2.5.4", "transformers==5.22.1"],
    }


EXPERIMENT = {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "revision": "c" * 40,
    "device": "cuda",
    "mode": "sleep",
}


def test_build_command_preserves_pins_secret_reference_and_finite_timeout():
    module = launcher()
    command = module.build_command(config(), EXPERIMENT, cli="vesslctl")
    assert command[:4] == ["vesslctl", "--org", "research", "--team"]
    assert command[command.index("--secret") + 1] == "HF_TOKEN=hf-gated-model"
    assert command[command.index("--object-volume") + 1] == "objectvol-123:/persistent"
    bootstrap = command[command.index("--cmd") + 1]
    assert "timeout --signal=TERM --kill-after=30s 1800s" in bootstrap
    assert "b" * 40 in bootstrap and "scripts/cloud_job.sh" in bootstrap
    assert "--image" in command and command[command.index("--image") + 1].endswith(
        "a" * 64
    )
    env = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
    assert "HF_HOME=/tmp/sleep-hf-cache/preflight-01" in env


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "pytorch:latest"),
        ("source_commit", "main"),
        ("timeout_seconds", 0),
        ("timeout_seconds", 10000000),
        ("resource_spec", "<RESOURCE>"),
        ("hf_secret", "hf_123456789012345678901234567890"),
        ("mount_path", "/tmp"),
        ("repository", "https://token@github.com/example/sleep.git"),
    ],
)
def test_invalid_or_unbounded_job_is_rejected(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError):
        launcher().build_command(cfg, EXPERIMENT)


def test_dry_run_does_not_execute_subprocess(tmp_path, monkeypatch, capsys):
    module = launcher()
    path = tmp_path / "job.json"
    cfg = config()
    cfg["experiment_config"] = str(tmp_path / "experiment.json")
    path.write_text(json.dumps(cfg))
    (tmp_path / "experiment.json").write_text(json.dumps(EXPERIMENT))
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: pytest.fail("dry-run submitted a job")
    )
    assert module.main(["--config", str(path)]) == 0
    assert "dry_run" in capsys.readouterr().out


def test_full_run_requires_matching_successful_preflight_gate(tmp_path):
    module = launcher()
    cfg = config()
    cfg["stage"] = "train"
    with pytest.raises(ValueError, match="preflight"):
        module.build_command(cfg, EXPERIMENT)
    gate = {
        "passed": True,
        "source_commit": cfg["source_commit"],
        "image": cfg["image"],
        "resource_spec": cfg["resource_spec"],
        "config_sha256": module.config_hash(EXPERIMENT),
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    cfg["preflight_gate"] = str(path)
    assert module.build_command(cfg, EXPERIMENT)
    gate["source_commit"] = "d" * 40
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="preflight"):
        module.build_command(cfg, EXPERIMENT)


def test_submission_rejects_unpinned_model_revision(tmp_path):
    module = launcher()
    experiment = dict(EXPERIMENT, revision="main")
    with pytest.raises(ValueError, match="revision"):
        module.build_command(config(), experiment)
    with pytest.raises(ValueError, match="model"):
        module.build_command(config(), dict(EXPERIMENT, model="tiny"))


def test_cloud_script_honors_expired_shared_deadline_before_work():
    env = dict(
        os.environ,
        SLEEP_TIMEOUT_SECONDS="300",
        SLEEP_DEADLINE_UNIX="1",
        SLEEP_JOB_JSON="{}",
        SLEEP_EXPERIMENT_JSON="{}",
        HF_TOKEN="unit-test-placeholder",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("cloud_job.sh"))],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 124
    assert "deadline" in result.stderr


@pytest.mark.parametrize("missing_venv", [True, False])
def test_bootstrap_installs_only_missing_system_prerequisites(tmp_path, missing_venv):
    bootstrap = shlex.split(
        launcher().build_command(config(), EXPERIMENT)[
            launcher().build_command(config(), EXPERIMENT).index("--cmd") + 1
        ]
    )[-1]
    tools = tmp_path / "bin"
    tools.mkdir()
    scripts = {
        "git": '#!/bin/bash\nif [[ "$1" == "clone" ]]; then target="${@: -1}"; mkdir -p "$target/scripts"; printf "exit 0\\n" > "$target/scripts/cloud_job.sh"; elif [[ "${@: -1}" == "HEAD" ]]; then printf "%s\\n" '
        + "b" * 40
        + "; fi\n",
        "python3": '#!/bin/bash\nexit "$MISSING_VENV"\n',
        "apt-get": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$APT_CALLS"\n',
    }
    for name, body in scripts.items():
        path = tools / name
        path.write_text(body)
        path.chmod(0o755)
    calls = tmp_path / "apt.txt"
    env = dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.defpath,
        MISSING_VENV=str(int(missing_venv)),
        APT_CALLS=str(calls),
    )
    result = subprocess.run(
        ["bash", "-c", bootstrap], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    if missing_venv:
        assert "python3-venv" in calls.read_text()
        assert " git" not in calls.read_text()
    else:
        assert not calls.exists()


class FakeCloud:
    def __init__(
        self, states, *, create_output='{"slug":"job-test123"}', show_error=False
    ):
        self.states = iter(states)
        self.state = "pending"
        self.created = False
        self.terminated = False
        self.create_output = create_output
        self.show_error = show_error
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        if "create" in args:
            self.created = True
            return subprocess.CompletedProcess(args, 0, self.create_output, "")
        if "list" in args:
            jobs = (
                [{"slug": "job-test123", "name": "sleep-preflight-01"}]
                if self.created
                else []
            )
            return subprocess.CompletedProcess(args, 0, json.dumps(jobs), "")
        if "terminate" in args:
            assert "job-test123" in args
            self.terminated = True
            self.state = "terminated"
            return subprocess.CompletedProcess(args, 0, "{}", "")
        if "billing" in args:
            return subprocess.CompletedProcess(args, 0, '{"burnRatePerHour":0}', "")
        if "show" in args:
            if self.show_error and not self.terminated:
                self.show_error = False
                raise RuntimeError("temporary status failure")
            if not self.terminated:
                self.state = next(self.states, self.state)
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "slug": "job-test123",
                        "name": "sleep-preflight-01",
                        "workloadState": self.state,
                    }
                ),
                "",
            )
        raise AssertionError(args)


class FakeTime:
    value = 0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def test_submit_monitors_until_terminal_and_reports_billing_without_termination():
    module = launcher()
    cloud = FakeCloud(["running", "succeeded"])
    clock = FakeTime()
    result = module.submit_and_monitor(
        config(),
        module.build_command(config(), EXPERIMENT),
        run=cloud,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result["job_id"] == "job-test123" and result["state"] == "succeeded"
    assert result["billing_burn_rate_per_hour"] == 0
    assert cloud.created and not cloud.terminated
    assert clock.value > 0


def test_queue_deadline_terminates_only_created_job_and_verifies_terminal():
    module = launcher()
    cloud = FakeCloud(["scheduling"])
    clock = FakeTime()
    cfg = config()
    cfg["timeout_seconds"] = 1
    with pytest.raises(TimeoutError):
        module.submit_and_monitor(
            cfg,
            module.build_command(cfg, EXPERIMENT),
            run=cloud,
            now=clock.now,
            sleep=clock.sleep,
        )
    assert cloud.terminated and cloud.state == "terminated"
    assert clock.value <= 76
    assert any("billing" in command for command in cloud.calls)


def test_monitor_exception_cleans_up_and_does_not_stop_unrelated_work():
    module = launcher()
    cloud = FakeCloud(["running"], show_error=True)
    clock = FakeTime()
    with pytest.raises(RuntimeError, match="status failure"):
        module.submit_and_monitor(
            config(),
            module.build_command(config(), EXPERIMENT),
            run=cloud,
            now=clock.now,
            sleep=clock.sleep,
        )
    assert cloud.terminated
    assert len([call for call in cloud.calls if "terminate" in call]) == 1


def test_keyboard_interrupt_cleans_up_created_job():
    module = launcher()
    cloud = FakeCloud(["running"])
    clock = FakeTime()

    def interrupt(seconds):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        module.submit_and_monitor(
            config(),
            module.build_command(config(), EXPERIMENT),
            run=cloud,
            now=clock.now,
            sleep=interrupt,
        )
    assert cloud.terminated


@pytest.mark.parametrize(
    "reply",
    ['✓ Job "sleep-preflight-01" created (slug: job-test123).', "unparseable response"],
)
def test_created_job_can_be_recovered_from_cli_message_or_unique_name(reply):
    module = launcher()
    cloud = FakeCloud(["succeeded"], create_output=reply)
    clock = FakeTime()
    result = module.submit_and_monitor(
        config(),
        module.build_command(config(), EXPERIMENT),
        run=cloud,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert result["job_id"] == "job-test123" and result["state"] == "succeeded"


def test_sigterm_during_main_submission_cleans_up_and_restores_handler(
    tmp_path, monkeypatch
):
    module = launcher()
    cloud = FakeCloud(["running"])
    path = tmp_path / "job.json"
    experiment = tmp_path / "experiment.json"
    cfg = config()
    cfg["experiment_config"] = str(experiment)
    path.write_text(json.dumps(cfg))
    experiment.write_text(json.dumps(EXPERIMENT))
    installed = []
    original = signal.getsignal(signal.SIGTERM)

    def install(sig, handler):
        assert sig == signal.SIGTERM
        installed.append(handler)
        return original

    def terminate(seconds):
        assert installed and callable(installed[0])
        installed[0](signal.SIGTERM, None)

    monkeypatch.setattr(module.signal, "signal", install)
    monkeypatch.setattr(module.subprocess, "run", cloud)
    monkeypatch.setattr(module.time, "sleep", terminate)
    with pytest.raises(KeyboardInterrupt):
        module.main(["--config", str(path), "--submit"])
    assert cloud.terminated
    assert installed[-1] is original


def test_interrupt_during_create_recovers_unique_new_job_for_cleanup():
    module = launcher()
    cloud = FakeCloud(["running"])
    clock = FakeTime()

    def create_interrupted(args, **kwargs):
        result = cloud(args, **kwargs)
        if "create" in args:
            raise KeyboardInterrupt
        return result

    with pytest.raises(KeyboardInterrupt):
        module.submit_and_monitor(
            config(),
            module.build_command(config(), EXPERIMENT),
            run=create_interrupted,
            now=clock.now,
            sleep=clock.sleep,
        )
    assert cloud.created and cloud.terminated
