import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_vessl_launch import EXPERIMENT, FakeCloud, FakeTime, SCRIPT, config, launcher


def resume_config():
    return dict(
        config(),
        stage="evaluate-resume",
        checkpoint="/persistent/training-parent/train/checkpoint.pt",
        resume_eval_run_id="evaluation-parent",
        resume_eval_job_id="job-parent123",
        resume_eval_job_name="sleep-evaluation-parent",
        resume_eval_attempts_sha256="d" * 64,
        resume_eval_provenance_sha256="e" * 64,
    )


def test_continuation_stage_requires_pinned_parent_and_checkpoint():
    module = launcher()
    cfg = resume_config()
    command = module.build_command(cfg, EXPERIMENT)
    environment = [command[i + 1] for i, value in enumerate(command) if value == "--env"]
    payload = next(value.split("=", 1)[1] for value in environment if value.startswith("SLEEP_JOB_JSON="))
    assert json.loads(payload) == cfg
    assert "scripts/cloud_job.sh" in command[command.index("--cmd") + 1]
    assert "preflight_gate" not in cfg


@pytest.mark.parametrize(
    "change",
    [
        {"checkpoint": None},
        {"checkpoint": "/persistent/preflight/fixture_checkpoint.pt"},
        {"resume_eval_run_id": "preflight-01"},
        {"resume_eval_job_name": "sleep-preflight-01"},
        {"resume_eval_job_id": "workspace-123"},
        {"resume_eval_attempts_sha256": "latest"},
        {"resume_eval_provenance_sha256": "e" * 63},
        {"resume_eval_provenance_sha256": None},
    ],
)
def test_continuation_rejects_incomplete_or_unsafe_parent_bindings(change):
    cfg = dict(resume_config(), **change)
    with pytest.raises(ValueError):
        launcher().build_command(cfg, EXPERIMENT)


def test_parent_fields_cannot_change_normal_stage_behavior():
    cfg = dict(resume_config(), stage="preflight")
    with pytest.raises(ValueError, match="require stage"):
        launcher().build_command(cfg, EXPERIMENT)


class ParentCloud(FakeCloud):
    def __init__(self, *, parent_changes=None):
        super().__init__(["running", "succeeded"])
        cfg = resume_config()
        self.parent = {
            "slug": cfg["resume_eval_job_id"],
            "name": cfg["resume_eval_job_name"],
            "workloadState": "failed",
            "resourceSpecSlug": cfg["resource_spec"],
            "imageUri": cfg["image"],
            **(parent_changes or {}),
        }

    def __call__(self, args, **kwargs):
        if "show" in args and args[args.index("show") + 1] == "job-parent123":
            self.calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(self.parent), "")
        return super().__call__(args, **kwargs)


def test_terminal_parent_is_verified_before_creating_one_child():
    module = launcher()
    cfg = resume_config()
    cloud = ParentCloud()
    clock = FakeTime()
    result = module.submit_and_monitor(
        cfg, module.build_command(cfg, EXPERIMENT), run=cloud, now=clock.now, sleep=clock.sleep
    )
    assert result["success"] and cloud.created and not cloud.terminated
    assert "show" in cloud.calls[0] and "job-parent123" in cloud.calls[0]
    assert sum("create" in call for call in cloud.calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"workloadState": "running"},
        {"workloadState": "scheduling"},
        {"name": "unrelated-job"},
        {"slug": "job-other123"},
        {"resourceSpecSlug": "another-resource"},
        {"imageUri": "registry.example/another@sha256:" + "a" * 64},
    ],
)
def test_parent_mismatch_or_active_parent_never_creates_or_terminates_a_job(changes):
    module = launcher()
    cfg = resume_config()
    cloud = ParentCloud(parent_changes=changes)
    clock = FakeTime()
    with pytest.raises(ValueError):
        module.submit_and_monitor(
            cfg, module.build_command(cfg, EXPERIMENT), run=cloud, now=clock.now, sleep=clock.sleep
        )
    assert not cloud.created and not cloud.terminated


@pytest.mark.parametrize("preflight_status", [0, 7])
def test_combined_worker_orders_genuine_preflight_gate_and_continuation(tmp_path, preflight_status):
    # Execute the actual worker's dispatch with lightweight command stand-ins.
    # Setup outside this section is covered by the existing bounded worker tests.
    worker = SCRIPT.with_name("cloud_job.sh").read_text()
    dispatch = worker[worker.index("run_preflight() {"):worker.rindex("\nBOUNDED_JOB")]
    commands = tmp_path / "bin"
    commands.mkdir()
    trace = tmp_path / "trace.txt"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sleep_repro = commands / "sleep-repro"
    sleep_repro.write_text('#!/bin/bash\nprintf "preflight\\n" >> "$TRACE"\nexit "$PREFLIGHT_STATUS"\n')
    python = commands / "python"
    python.write_text(
        '#!/bin/bash\nif [[ "$1" == "scripts/continue_evaluation.py" ]]; then\n'
        '  test -s "$SLEEP_RUN_DIR/preflight_gate.json" || exit 99\n'
        '  printf "continuation\\n" >> "$TRACE"\n'
        'else\n  exec "' + sys.executable + '" "$@"\nfi\n'
    )
    sleep_repro.chmod(0o755)
    python.chmod(0o755)
    cfg = resume_config()
    environment = dict(
        os.environ,
        PATH=str(commands) + os.pathsep + os.defpath,
        TRACE=str(trace),
        PREFLIGHT_STATUS=str(preflight_status),
        SLEEP_RUN_DIR=str(run_dir),
        SLEEP_JOB_JSON=json.dumps(cfg),
        SLEEP_EXPERIMENT_JSON=json.dumps(EXPERIMENT),
    )
    shell = 'set -euo pipefail\nsettings=(unused unused unused evaluate-resume)\nrun_dir="$SLEEP_RUN_DIR"\n' + dispatch
    result = subprocess.run(["bash", "-c", shell], env=environment, capture_output=True, text=True)
    assert result.returncode == preflight_status, result.stderr
    expected = ["preflight", "continuation"] if preflight_status == 0 else ["preflight"]
    assert trace.read_text().splitlines() == expected
    gate = run_dir / "preflight_gate.json"
    assert gate.exists() is (preflight_status == 0)
    if gate.exists():
        observed = json.loads(gate.read_text())
        assert observed == {
            "passed": True,
            "source_commit": cfg["source_commit"],
            "image": cfg["image"],
            "resource_spec": cfg["resource_spec"],
            "config_sha256": launcher().config_hash(EXPERIMENT),
        }
