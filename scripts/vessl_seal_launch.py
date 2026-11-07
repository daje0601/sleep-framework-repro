#!/usr/bin/env python3
"""Review or explicitly submit one bounded upstream SEAL environment/ICL job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import signal

# Reuse the tested monitor and its unique-name checks, deadlines and cleanup.
# Importing this standard-library module performs no CLI/API operation.
_monitor_spec = importlib.util.spec_from_file_location(
    "_sleep_vessl_existing_monitor", Path(__file__).with_name("vessl_launch.py")
)
_monitor = importlib.util.module_from_spec(_monitor_spec)
_monitor_spec.loader.exec_module(_monitor)
submit_and_monitor = _monitor.submit_and_monitor

WORKER_IMAGE = "pytorch/pytorch@sha256:3deb633094acb02ba490e4da21bb37337327860e884c5b2290042a6db7d6b572"
WORKER_SOURCE_COMMIT = "ee39e0dbdfdb74edea0abe42283788fe6451b525"
WORKER_PYTHON = "3.11"
REQUIRED_FIELDS = {
    "org",
    "team",
    "job_name",
    "run_id",
    "resource_spec",
    "object_volume",
    "mount_path",
    "hf_secret",
    "repository",
    "source_commit",
    "image",
    "python_version",
    "timeout_seconds",
    "queue_timeout_seconds",
    "stage",
}
ENV_WHITELIST = {
    "SLEEP_SEAL_JOB_JSON",
    "HF_HUB_DISABLE_TELEMETRY",
    "TOKENIZERS_PARALLELISM",
}


def _slug(value, field):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value
    ):
        raise ValueError(f"{field} must be a concrete account/run identifier")


def worker_config(cfg: dict) -> dict:
    """Validate the complete job and expose only the worker's exact schema."""
    if not isinstance(cfg, dict) or set(cfg) != REQUIRED_FIELDS:
        raise ValueError(
            "job fields must match the reviewed SEAL schema; arbitrary env/token fields are forbidden"
        )
    for field in (
        "org",
        "team",
        "job_name",
        "run_id",
        "resource_spec",
        "object_volume",
        "hf_secret",
    ):
        _slug(cfg[field], field)
    if cfg["hf_secret"].startswith("hf_"):
        raise ValueError(
            "hf_secret must name an existing Cloud secret, never contain a Hugging Face token"
        )
    if cfg["image"] != WORKER_IMAGE or cfg["python_version"] != WORKER_PYTHON:
        raise ValueError(
            "only the reviewed Torch 2.7 CUDA development image and Python 3.11 are supported"
        )
    if cfg["source_commit"] != WORKER_SOURCE_COMMIT:
        raise ValueError("source_commit must equal the published reviewed worker SHA")
    if not isinstance(cfg["repository"], str) or not re.fullmatch(
        r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?",
        cfg["repository"],
    ):
        raise ValueError(
            "repository must be a public HTTPS GitHub URL without credentials or query parameters"
        )
    owner, repository = cfg["repository"].removeprefix("https://github.com/").split("/")
    if owner in (".", "..") or repository.removesuffix(".git") in ("", ".", ".."):
        raise ValueError(
            "repository must identify an owner and repository, without path traversal"
        )
    for field in ("timeout_seconds", "queue_timeout_seconds"):
        if type(cfg[field]) is not int or not 1 <= cfg[field] <= 86400:
            raise ValueError(f"{field} must be an explicit integer in 1..86400")
    mount = cfg["mount_path"]
    if (
        not isinstance(mount, str)
        or not re.fullmatch(r"/[A-Za-z0-9_/-]+", mount)
        or ".." in PurePosixPath(mount).parts
        or mount.rstrip("/") in ("", "/tmp", "/var", "/etc", "/usr", "/root", "/home")
    ):
        raise ValueError("mount_path must identify a dedicated private output volume")
    if cfg["stage"] not in ("environment", "icl"):
        raise ValueError(
            "only upstream environment and ICL stages are enabled; TTT is not verified"
        )
    return {
        field: cfg[field]
        for field in (
            "source_commit",
            "mount_path",
            "run_id",
            "image",
            "resource_spec",
            "timeout_seconds",
            "stage",
        )
    }


def build_command(cfg: dict, *, cli: str = "vesslctl") -> list[str]:
    """Build one reviewable argv vector; no subprocess or provider I/O occurs."""
    worker = worker_config(cfg)
    if not isinstance(cli, str) or not cli or "\x00" in cli:
        raise ValueError("cli must name the local Cloud CLI executable")
    seconds = cfg["timeout_seconds"]
    commit = cfg["source_commit"]
    bootstrap = "\n".join(
        (
            "set -euo pipefail",
            f"export SLEEP_DEADLINE_UNIX=$(( $(date +%s) + {seconds} ))",
            "command -v python3.11 >/dev/null",
            "python3.11 -c 'import sys; assert sys.version_info[:2] == (3,11)'",
            "if ! command -v git >/dev/null; then",
            "  command -v apt-get >/dev/null",
            "  export DEBIAN_FRONTEND=noninteractive",
            "  apt-get -o Acquire::Retries=1 update",
            "  apt-get install -y --no-install-recommends git",
            "fi",
            "source_dir=$(mktemp -d /tmp/sleep-seal-source.XXXXXX)",
            f'git clone --depth 1 --no-checkout {shlex.quote(cfg["repository"])} "$source_dir"',
            f'git -C "$source_dir" fetch --depth 1 origin {commit}',
            f'git -C "$source_dir" checkout --detach {commit}',
            f'test "$(git -C "$source_dir" rev-parse HEAD)" = {commit}',
            'cd "$source_dir"',
            "test -f scripts/cloud_seal_job.sh",
            "exec bash scripts/cloud_seal_job.sh",
        )
    )
    bounded = shlex.join(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            f"{seconds}s",
            "bash",
            "-lc",
            bootstrap,
        ]
    )
    command = [
        cli,
        "--org",
        cfg["org"],
        "--team",
        cfg["team"],
        "job",
        "create",
        "--name",
        cfg["job_name"],
        "--resource-spec",
        cfg["resource_spec"],
        "--image",
        cfg["image"],
        "--image-pull-policy",
        "Always",
        "--object-volume",
        f"{cfg['object_volume']}:{cfg['mount_path']}",
        "--secret",
        f"HF_TOKEN={cfg['hf_secret']}",
        "--working-dir",
        "/tmp",
        "--cmd",
        bounded,
    ]
    environment = {
        "SLEEP_SEAL_JOB_JSON": json.dumps(
            worker, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if set(environment) != ENV_WHITELIST:
        raise RuntimeError("internal baseline environment whitelist mismatch")
    for key, value in environment.items():
        command += ["--env", f"{key}={value}"]
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="local private reviewed job JSON"
    )
    parser.add_argument("--cli", default="vesslctl")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="explicitly create exactly one billable Cloud job",
    )
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    command = build_command(cfg, cli=args.cli)
    fingerprint = hashlib.sha256(
        json.dumps(
            worker_config(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    print(
        json.dumps(
            {
                "dry_run": not args.submit,
                "stage": cfg["stage"],
                "worker_config_sha256": fingerprint,
                "queue_timeout_seconds": cfg["queue_timeout_seconds"],
                "runtime_timeout_seconds": cfg["timeout_seconds"],
                "command": shlex.join(command),
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.submit:
        return 0
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt_submission(signum, frame):
        raise KeyboardInterrupt(
            "Termination signal received; cleaning up this baseline submission"
        )

    signal.signal(signal.SIGTERM, interrupt_submission)
    try:
        result = submit_and_monitor(cfg, command)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
