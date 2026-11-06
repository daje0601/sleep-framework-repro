"""Review or explicitly submit one bounded VESSL Cloud job (never a sweep)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess


def config_hash(experiment: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            experiment, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _slug(value, name):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value
    ):
        raise ValueError(
            f"{name} must be a concrete account identifier, with no placeholders"
        )
    return value


def build_command(cfg: dict, experiment: dict, *, cli: str = "vesslctl") -> list[str]:
    """Validate a reviewable specification and build an argv vector without I/O."""
    allowed = {
        "org",
        "team",
        "job_name",
        "run_id",
        "resource_spec",
        "image",
        "object_volume",
        "mount_path",
        "hf_secret",
        "repository",
        "source_commit",
        "timeout_seconds",
        "stage",
        "torch_version",
        "linux_dependencies",
        "experiment_config",
        "preflight_gate",
        "checkpoint",
    }
    if set(cfg) - allowed:
        raise ValueError(
            f"unknown job fields: {sorted(set(cfg) - allowed)}; token values are forbidden"
        )
    for name in (
        "org",
        "team",
        "job_name",
        "run_id",
        "resource_spec",
        "object_volume",
        "hf_secret",
    ):
        _slug(cfg.get(name), name)
    # A Hugging Face access token is never an acceptable secret-reference name.
    if cfg["hf_secret"].startswith("hf_"):
        raise ValueError(
            "hf_secret must name a VESSL secret, never contain an HF token"
        )
    image = cfg.get("image", "")
    if not isinstance(image, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_./:-]*@sha256:[a-f0-9]{64}", image
    ):
        raise ValueError("image must be pinned by a concrete sha256 digest")
    commit = cfg.get("source_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("source_commit must be a full 40-character Git SHA")
    repository = cfg.get("repository", "")
    if not isinstance(repository, str) or not re.fullmatch(
        r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", repository
    ):
        raise ValueError(
            "repository must be a public HTTPS GitHub URL without credentials"
        )
    seconds = cfg.get("timeout_seconds")
    if type(seconds) is not int or not 1 <= seconds <= 86400:
        raise ValueError("timeout_seconds must be an integer in 1..86400")
    mount = cfg.get("mount_path", "")
    if (
        not isinstance(mount, str)
        or not re.fullmatch(r"/[A-Za-z0-9_/-]+", mount)
        or ".." in PurePosixPath(mount).parts
        or mount.rstrip("/") in ("", "/tmp", "/var", "/etc", "/usr", "/root", "/home")
    ):
        raise ValueError(
            "mount_path must identify a dedicated persistent output directory"
        )
    stage = cfg.get("stage", "preflight")
    if stage not in ("preflight", "train", "evaluate"):
        raise ValueError("stage must be preflight, train, or evaluate")
    if experiment.get("device") != "cuda":
        raise ValueError("cloud preflight/run must use device=cuda")
    if (
        not isinstance(experiment.get("model"), str)
        or not experiment["model"]
        or experiment["model"] == "tiny"
    ):
        raise ValueError(
            "cloud preflight requires an actual pretrained model, not the tiny unit-test model"
        )
    if not re.fullmatch(r"[a-f0-9]{40}", str(experiment.get("revision", ""))):
        raise ValueError(
            "model revision must be a pinned 40-character Hugging Face commit"
        )
    if not re.fullmatch(
        r"\d+\.\d+\.\d+(?:[a-z0-9.]*)?", str(cfg.get("torch_version", ""))
    ):
        raise ValueError(
            "torch_version must identify the version supplied by the pinned image"
        )
    dependencies = cfg.get("linux_dependencies")
    if (
        not isinstance(dependencies, list)
        or not dependencies
        or any(
            not isinstance(dep, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", dep)
            for dep in dependencies
        )
    ):
        raise ValueError("linux_dependencies must list exact package==version pins")
    if any(dep.lower().startswith(("torch==", "peft==")) for dep in dependencies):
        raise ValueError(
            "use image-supplied torch; PEFT is unused by this implementation"
        )
    checkpoint = cfg.get("checkpoint")
    if checkpoint is not None and (
        not isinstance(checkpoint, str)
        or not checkpoint.startswith(mount.rstrip("/") + "/")
        or ".." in PurePosixPath(checkpoint).parts
    ):
        raise ValueError(
            "checkpoint must be an absolute path inside the persistent volume"
        )
    if (
        stage == "evaluate"
        and experiment.get("mode") in ("sleep", "self_edit")
        and not checkpoint
    ):
        raise ValueError("learned-policy evaluation requires a persistent checkpoint")
    if stage != "preflight":
        gate_path = cfg.get("preflight_gate")
        if not gate_path:
            raise ValueError(
                "train/evaluate requires a downloaded successful preflight_gate artifact"
            )
        try:
            gate = json.loads(Path(gate_path).read_text())
        except (OSError, ValueError) as error:
            raise ValueError("preflight gate cannot be read") from error
        expected = {
            "passed": True,
            "source_commit": commit,
            "image": image,
            "resource_spec": cfg["resource_spec"],
            "config_sha256": config_hash(experiment),
        }
        if any(gate.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "preflight gate does not match config, image, source, and resource"
            )

    bootstrap = "\n".join(
        (
            "set -euo pipefail",
            f"export SLEEP_DEADLINE_UNIX=$(( $(date +%s) + {seconds} ))",
            "bootstrap_packages=()",
            "command -v git >/dev/null || bootstrap_packages+=(git)",
            "python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || bootstrap_packages+=(python3-venv)",
            "if (( ${#bootstrap_packages[@]} )); then",
            "  command -v apt-get >/dev/null",
            "  export DEBIAN_FRONTEND=noninteractive",
            "  apt-get -o Acquire::Retries=1 update",
            '  apt-get install -y --no-install-recommends "${bootstrap_packages[@]}"',
            "fi",
            "source_dir=$(mktemp -d /tmp/sleep-source.XXXXXX)",
            f'git clone --depth 1 --no-checkout {shlex.quote(repository)} "$source_dir"',
            f'git -C "$source_dir" fetch --depth 1 origin {commit}',
            f'git -C "$source_dir" checkout --detach {commit}',
            f'test "$(git -C "$source_dir" rev-parse HEAD)" = {commit}',
            'cd "$source_dir"',
            "exec bash scripts/cloud_job.sh",
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
        image,
        "--image-pull-policy",
        "Always",
        "--object-volume",
        f"{cfg['object_volume']}:{mount}",
        "--secret",
        f"HF_TOKEN={cfg['hf_secret']}",
        "--working-dir",
        "/tmp",
        "--cmd",
        bounded,
    ]
    environment = {
        "SLEEP_JOB_JSON": json.dumps(
            dict(cfg, stage=stage), separators=(",", ":"), allow_nan=False
        ),
        "SLEEP_EXPERIMENT_JSON": json.dumps(
            experiment, separators=(",", ":"), allow_nan=False
        ),
        "SLEEP_TIMEOUT_SECONDS": str(seconds),
        "HF_HOME": mount.rstrip("/") + "/hf-cache",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in environment.items():
        command.extend(["--env", f"{key}={value}"])
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cli", default="vesslctl")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="explicitly create exactly one billable job",
    )
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    config_path = Path(cfg["experiment_config"])
    if not config_path.is_absolute():
        config_path = args.config.parent / config_path
    experiment = json.loads(config_path.read_text())
    command = build_command(cfg, experiment, cli=args.cli)
    print(
        json.dumps(
            {
                "dry_run": not args.submit,
                "stage": cfg.get("stage", "preflight"),
                "config_sha256": config_hash(experiment),
                "command": shlex.join(command),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.submit:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
