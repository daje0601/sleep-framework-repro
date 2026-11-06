# VESSL Cloud jobs

This launcher targets **VESSL Cloud `vesslctl job create`**, not the older VESSL
Run CLI. Its flags were checked against `vesslctl` version `2026.11.05-03`.
VESSL documents [Cloud batch jobs and persistent object-volume mounts](https://vessl.ai/en/blog/vessl-cloud-batch-jobs)
and the [Cloud CLI](https://vessl.ai/en/blog/vesslctl-cli-launch).

The generic example contains deliberate placeholders. Resource availability,
image digest, storage, model access, and cost must come from the actual account.
See [experiment status](EXPERIMENT_STATUS.md) for executed work; the existence
of these scripts is not evidence of a successful GPU experiment.

## Prepare one reviewable preflight

1. Authenticate locally with `vesslctl auth login`. List the account's available
   resource specifications and volumes with `vesslctl resource-spec list` and
   `vesslctl volume list`. Create or select a private persistent object volume.
2. Select a Linux CUDA container with Python 3.12+, Bash, GNU `timeout`, Git,
   `venv`, pip, and a compatible PyTorch build. Pin its **actual image digest**
   as `registry/image@sha256:...` and record the image's PyTorch version. This
   project does not supply an invented image digest or claim a tested GPU image.
3. Store the gated-model access token as an account secret. Put only that
   secret's name in `hf_secret`; the launcher passes `--secret HF_TOKEN=NAME`.
   Never place a token in JSON, CLI `--env`, committed files, or job logs.
4. Publish the intended source commit to the public GitHub repository. Fill in
   its full 40-character SHA and HTTPS URL. Checkout is detached and verified.
5. Copy `configs/vessl.example.json` to a local job config and replace every
   placeholder. `experiment_config` is resolved relative to that job config.
   Pin the experiment's model/tokenizer `revision` to an actual Hugging Face
   commit. The launcher rejects mutable model revisions, image tags, and source
   branches. Choose an account resource based on a bounded memory preflight;
   no particular GPU count is represented as sufficient before measurement.
6. Give the attempt a unique `run_id`. The cloud script refuses to overwrite
   an existing output directory. Keep `stage` as `preflight`, and choose a
   concrete timeout of at most 86,400 seconds.

Review the exact one-job command without making an API call:

```bash
python scripts/vessl_launch.py --config configs/vessl.local.json
```

After reviewing that concrete job and its budget, explicit submission is:

```bash
python scripts/vessl_launch.py --config configs/vessl.local.json --submit
```

The launcher never authenticates, creates resources, submits by default, or
loops over seeds. The hard GNU `timeout` bounds checkout, dependency setup,
downloads, and work; it sends TERM at the deadline and KILL after 30 seconds.
The container must contain `timeout` before startup. A timeout is a process
guard, not a price guarantee or a provider scheduling deadline.

The bootstrap installs Git and `python3-venv` with APT only if they are missing,
inside that same deadline. It requires a Debian/Ubuntu image and a container
user allowed to install those missing packages. The official
[PyTorch 2.15 runtime Dockerfile](https://github.com/pytorch/pytorch/blob/v2.15.0/Dockerfile)
supplies Python 3.12 but omits those packages. One verified candidate is
`pytorch/pytorch@sha256:924707fa828f821c9a82d987b405675df6369eee5ce368aece6757b025ffcde5`:
[official Linux/amd64 CUDA 13.2 runtime manifest](https://hub.docker.com/layers/pytorch/pytorch/2.15.0-cuda13.2-cudnn9-runtime/images/sha256-924707fa828f821c9a82d987b405675df6369eee5ce368aece6757b025ffcde5).
The registry digest was verified with `docker manifest inspect --verbose`;
driver compatibility and successful GPU execution still require the preflight.

## Linux execution and artifacts

The checked-out `scripts/cloud_job.sh` installs exact direct dependency pins
from `linux_dependencies`, uses the image-supplied PyTorch version, and installs
this package with `--no-deps`. This avoids installing unused PEFT and avoids
treating the local macOS `requirements.lock` as a portable CUDA lock. These
direct version pins have CPU evidence locally; the chosen Linux/CUDA image is
unverified until the first successful job. The resolved Linux environment is
saved as `requirements-linux.txt` for the subsequent Linux lock/container.

The script downloads and verifies the pinned ARC files, then executes:

```bash
sleep-repro preflight --config RUN/config.json --output RUN/preflight
```

The CLI preflight is one bounded actual training-task adaptation check, not the
full benchmark. Its report records peak CUDA memory and real losses. A failed
command cannot produce the successful preflight gate. The output volume keeps
`RUN/console.log`, reviewed configs, environment/source metadata, the preflight
report, `exit.json`, and `preflight_gate.json`. `HF_HOME` is also on the private
volume; gated weights must remain private and must not be redistributed.

Inspect job status/logs with `vesslctl job show JOB_ID` and
`vesslctl job logs JOB_ID --follow`. Check the persistent artifacts after the
job finishes, including nonzero exit status or timeout. `exit.json` may be
missing after a hard kill; absence is not success. No local launcher output
constitutes a completed GPU experiment.

## Separately bounded train or evaluation

After a successful preflight, download its `preflight_gate.json`. Copy the
reviewed launcher config, change `stage` to `train` or `evaluate`, select a
fresh `run_id`/job name and an explicit timeout, and set `preflight_gate` to
the local downloaded artifact. The launcher requires a successful gate with
matching source commit, image, resource specification, and experiment hash.
An altered experiment or larger configuration requires another preflight.

For learned-policy evaluation, set `checkpoint` to its absolute path inside
the mounted volume, for example `/persistent/train-01/train/checkpoint.pt`.
ICL/TTT evaluation may omit that field. Training and evaluation are separate
one-job submissions; the launcher does not automatically start either after
preflight. Review and submit each with the same commands above. Keep baseline
configs and their provenance distinct. Run each declared seed explicitly only
after the configuration and measured budget are approved.
