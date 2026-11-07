# Pinned upstream SEAL baseline package

This entry point prepares the official SEAL ARC **fixed TTT and ICL baseline
code paths** at commit `6d9c9f9ee392c6cc618e771f399d436d190f6ca4`, on our declared
11-training / 8-evaluation source-task manifest. It preserves upstream chat
formatting, contextual augmentation, trainer updates, adapter targets, voting,
and two-guess scoring. Preparation has been exercised with the actual public
sources. No baseline GPU run or dependency-environment validation is claimed.

This is not a recovered Sleep author manifest. It does not implement SEAL's
learned self-edit/ReST-EM training command or replace our independent Sleep
runner. Keep this baseline's results separate from the independent one-grid
metric in [ARC_PROTOCOL.md](ARC_PROTOCOL.md).

## Prepare a reviewable artifact

After normal project installation and ARC preparation:

```sh
python scripts/prepare_data.py
python scripts/prepare_seal_baseline.py prepare \
  --config configs/seal_upstream_ttt.json --data-dir data/arc \
  --output runs/seal_upstream_ttt_package_v2
python scripts/prepare_seal_baseline.py verify \
  --package runs/seal_upstream_ttt_package_v2
```

For ICL, use `configs/seal_upstream_icl.json` and a separate output such as
`runs/seal_upstream_icl_package_v2`. The preparation script uses the standard
library and the project's standard-library ARC loader; it imports no GPU
training or inference library and submits no jobs.

The artifact contains:

- `upstream/original/`: 19 exact, SHA256-verified upstream source files, including
  the original MIT license and requirements file.
- `upstream/runtime/`: the same source files with the guarded integration changes
  described below.
- `source.patch`: the complete unified diff from original to runtime sources.
- `data/`: separate selected train/evaluation challenge and solution files.
  Challenge files contain visible demonstrations and test inputs only.
- `config.json`, `manifest.json`, and `commands.json`: fixed upstream/model
  revisions, input/source/artifact hashes, source and flattened subtask IDs,
  declared attempt seeds, command argument lists, and recorded deviations.
- `setup.sh` and `run.sh`: reviewed commands for a separate GPU environment.

The committed `configs/seal_upstream_ttt.json` and `configs/seal_upstream_icl.json`
contain a source URL and SHA256 for every fetched file. Source URLs must use the fixed commit. Modified
sources fail hash checks; unexpected patch locations fail closed. Preparation
uses an adjacent staging directory and atomic rename. Existing prepared files
are verified instead of silently replaced. Generated command paths are absolute;
regenerate the package in a new destination if its location changes.
Integration patch version 2 records its own SHA256 and preserves indexed raw
artifacts for every attempt. Legacy packages without this preservation are
rejected by verification; use a fresh `_v2` destination rather than replacing an
old artifact. The committed TTT/ICL source registries retain the unchanged
upstream hashes and declare integration patch version 2.

The original upstream files are downloaded into ignored run artifacts rather
than copied into this project's source tree. Their [MIT notice](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/LICENSE)
is retained in both source trees. ARC and model licenses remain separate.

## Separate dependency environment

The upstream [requirements file](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/requirements.txt)
pins a different stack from the independent runner:

| Dependency | Official direct pin |
| --- | --- |
| PyTorch | `2.7.0` |
| Transformers | `4.52.4` |
| PEFT | `0.15.2` |
| vLLM | `0.9.1` |
| Accelerate | `1.7.0` |
| Datasets | `3.6.0` |
| NumPy | `2.2.6` |
| FlashInfer Python | `0.2.2` |

Use a separate Linux NVIDIA/CUDA environment; do not install these into the
independent runner's environment. The generated setup command creates a Python
3.11 virtual environment, installs the unchanged upstream requirements, and runs
`pip check`. Python 3.11 is our environment selection, not a recovered author
Python version. The official file pins direct packages, but it is not a complete
transitive lock or a container image digest.

An official-PyPI metadata audit of all 17 direct pins is recorded in
[`configs/seal_upstream_dependencies.json`](../configs/seal_upstream_dependencies.json).
For Python 3.11/Linux x86_64, no contradiction was found between the declared
direct pins and their active requirements on other directly pinned packages.
This is a bounded metadata check, not a full transitive resolver run. In
particular:

- [vLLM 0.9.1 metadata](https://pypi.org/pypi/vllm/0.9.1/json) requires
  `torch==2.7.0`, `torchaudio==2.7.0`, `torchvision==0.22.0`, and Linux x86_64
  `xformers==0.0.30`. It requires Python `>=3.9,<3.13` and publishes only an
  x86_64 Linux wheel in that release. The current independent-runner PyTorch
  2.15 environment cannot satisfy these exact pins unchanged.
- On Python 3.12, the same vLLM metadata requires
  `setuptools>=77.0.3,<80`; the current independent Cloud setup's
  `setuptools==84.2.1` conflicts. A separate environment is necessary; do not
  inherit that setup's system packages.
- [FlashInfer Python 0.2.2 metadata](https://pypi.org/pypi/flashinfer-python/0.2.2/json)
  lists only a source distribution on PyPI, with `numpy`, `torch`, and `ninja`
  dependencies. A successful wheel-only resolution is therefore impossible
  against PyPI alone. CUDA development/build/JIT requirements still need to be
  tested; a runtime-only image is not established as sufficient.
- Plain [Transformers 4.52.4](https://pypi.org/pypi/transformers/4.52.4/json)
  requires `tokenizers>=0.21,<0.22`; this overlaps vLLM's `>=0.21.1` requirement.
  Its optional `torch`/`dev` extras declare `torch<2.7`, which would conflict with
  the official pin. The unchanged SEAL requirements request plain Transformers,
  so those extras must not be added implicitly.
- The generated setup invokes `python3.11`. A Cloud image containing only
  Python 3.12 will fail before installation even though Python 3.12 itself is
  within vLLM's supported metadata range. Select an image with the declared
  interpreter, or record an explicit interpreter/setup change.

**Resolver compatibility, CUDA/driver compatibility, binary availability,
FlashInfer behavior, and vLLM adapter loading have not been validated here.** A
successful package preparation proves none of these. Preserve any installation
failure and the resolver output; do not silently upgrade pinned packages to make
installation succeed. A dependency adjustment requires its own recorded variant.
The frozen environment and successful small GPU run are still required evidence.

On the intended GPU host, after checking the artifact and obtaining model access,
the concrete entry point is:

```sh
bash runs/seal_upstream_ttt_package_v2/setup.sh
bash runs/seal_upstream_ttt_package_v2/run.sh
```

These commands install packages, download the gated model snapshot, and then
perform GPU adaptation/inference. They have **not** been run by preparation.
Model access uses the authenticated Hugging Face environment; tokens must remain
in environment/account secrets. The model snapshot uses revision
`9213176726f574b556790deb65791e0c5aa438b6`. The run script saves `pip freeze`
output beside results; record the GPU, driver, CUDA/container identity, time, and
peak memory in the eventual execution report.

## Separate bounded VESSL Cloud worker

`scripts/cloud_seal_job.sh` is a separate execution entry point for **environment
validation and upstream ICL only**. It neither submits jobs nor calls the VESSL
API. It does not modify or dispatch through the independent runner's
`cloud_job.sh` / `vessl_launch.py`, whose stages and dependency rules differ.
Use only the already authorized Cloud organization/team, resource specifications,
private volume, and HF secret reference. No other account or provider is needed.

A metadata-reviewed official image candidate is:

```text
pytorch/pytorch@sha256:3deb633094acb02ba490e4da21bb37337327860e884c5b2290042a6db7d6b572
```

This is the Linux/amd64 manifest of
[`pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel`](https://hub.docker.com/layers/pytorch/pytorch/2.7.0-cuda12.6-cudnn9-devel/images/sha256-3deb633094acb02ba490e4da21bb37337327860e884c5b2290042a6db7d6b572).
`docker manifest inspect --verbose` verified that manifest digest. Its 20,687-byte
config blob has SHA256
`663476d9c40a1e5e7db63762c8795b92dd35555892cf33d18e95caaa74f7a087`;
hash-checked registry metadata records `PYTORCH_VERSION=2.7.0`,
`CUDA_VERSION=12.6.3`, and the Conda executable path. The official
[PyTorch v2.7.0 Dockerfile](https://github.com/pytorch/pytorch/blob/v2.7.0/Dockerfile)
defaults to Python 3.11. The image config does not expose the installed Python
version, so the worker checks `python3.11` and its actual version at startup.
Only manifests/config metadata were downloaded: no image layers were pulled and
this baseline image has not been executed here. Development CUDA is a candidate
for FlashInfer's source/JIT needs, not proof of successful kernel compilation.

The worker accepts a reviewed `SLEEP_SEAL_JOB_JSON` with exactly these fields:

| Field | Requirement |
| --- | --- |
| `source_commit` | Published full Git SHA containing this worker and the baseline preparer |
| `image` | The exact reviewed image digest above |
| `resource_spec` | Existing account resource identifier; runtime checks one A100/H100 with 80 GB |
| `mount_path` | Existing writable private object-volume mount, for ordinary output files |
| `run_id` | Fresh unique directory name; existing output is rejected |
| `timeout_seconds` | Explicit integer from 1 through 86,400, covering setup and workload |
| `stage` | `environment` or `icl`; no training or multi-seed sweep dispatch |

Validate a local private specification without invoking CUDA, downloading models,
or creating any job:

```sh
SLEEP_SEAL_JOB_JSON="$(cat /private/path/reviewed-seal-job.json)" \
  bash scripts/cloud_seal_job.sh --validate-only
```

The minimal separate Cloud job builder should reuse the existing account's
`vesslctl job create` flags and the existing single-job monitor/termination
behavior, but supply this worker as its workload. Its reviewed command must:

1. Start a finite shared deadline before bootstrap, clone the public repository,
   fetch/check out the exact `source_commit`, and verify it. Ensure Bash, Git,
   GNU `timeout`, `python3.11`/`venv`, and `nvcc` are available in the selected
   development image. Do not run the independent runner's dependency installer.
2. Set `SLEEP_DEADLINE_UNIX` and `SLEEP_SEAL_JOB_JSON`, mount the existing private
   volume, and inject `HF_TOKEN` through the existing Cloud secret reference for
   `stage=icl`. Never put token values in JSON or literal `--env` arguments.
3. Run `bash scripts/cloud_seal_job.sh` inside an outer GNU timeout that also
   bounds checkout. Keep the local monitor attached through queue/image-pull
   time and terminate only that newly created job if its deadline is exceeded.
4. Retrieve the private run artifacts and verified terminal state. An
   environment-only job does not automatically launch ICL; a subsequent ICL job
   is a separate explicitly reviewed submission.

Inside either stage the worker checks the source SHA, prepares pinned data and
the version-two target-safe ICL package, installs its separate upstream
environment, runs the original `pip check`, verifies every direct dependency
version, and imports Torch/Transformers/PEFT/vLLM/FlashInfer with CUDA. The
`environment` stage then exits without downloading the model or running ARC
inference. The `icl` stage performs the same checks before executing the
original patched ICL evaluation and its distinct two-guess summary.

All venvs, source trees, model snapshots, HF cache, extension caches, and atomic
staging directories live under job-local `/tmp` POSIX storage. The object-volume
mount receives streamed `console.log`, `environment.json`, the reviewed package
metadata/patch, partial or complete indexed results, and `exit.json`. Model
weights and venvs are not copied to that mount. A hard kill can prevent final
artifact copying; missing `exit.json` or summary is not success.

No separate Cloud job builder or launch configuration is supplied by this
worker, and no job has been submitted through it. Remaining blockers are a
published source commit, a reviewed same-account one-job specification, actual
image/interpreter availability, complete dependency resolution/source builds,
driver/kernel compatibility, and successful baseline execution. A successful
independent Sleep preflight does not validate this different baseline stack.

## Preserved upstream recipe

The fixed TTT path calls the original `ttt.py` and `arclib.update_model.TTT`:

- Original Llama chat templates and system/user/assistant example format.
- Leave-one- and leave-two-out contextual training tasks, original augmentation
  families enabled, color/example permutation, and at most 250 generated examples.
- Rank 128, alpha 16, dropout zero, with `q_proj`, `v_proj`, `gate_proj`,
  `down_proj`, and `up_proj` LoRA targets.
- Batch size two, gradient accumulation one, learning rate `1e-4`, two epochs,
  cosine schedule, 11 warmup steps, AdamW, and bfloat16 training.
- Output-token training through the original upstream masking implementation.
  Updates occur at trainer minibatch boundaries; epochs are not a fixed count
  of full-dataset optimizer updates.

The TTT package runs five fresh adaptation processes. Attempt `i` uses seed
`config.seed + i` and writes `adapters/task_id/i`; each process uses the upstream
LoRA reset behavior across tasks. These explicit attempt seeds are our declared
comparison choice. The ICL package performs one original baseline evaluation,
without task adapters. Its greedy query ensemble is not five independent
adaptation attempts.

Both evaluation paths preserve `--temperature 0 --n_sample 1 --include_n 1
--new_format`, original context permutations and spatial query augmentation,
inversion, and voting. Thus the upstream ICL path also uses a query ensemble; it
is not equivalent to our single-prompt ICL reconstruction. See the official
[TTT configuration](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/ttt.py#L353-L460),
[inner trainer](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/arclib/update_model.py#L227-L253),
and [evaluation CLI](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/README.md#L64-L99).

## Explicit integration changes

The generated patch makes these changes, without modifying the upstream trainer,
chat serializer, augmentation functions, voting implementation, or evaluator:

1. `ttt.py` has hardcoded paths and a 350-task loop rather than a CLI. Its runtime
   copy reads reviewed paths/seed/attempt metadata from `SEAL_BASELINE_CONFIG`,
   uses the immutable local model snapshot, and iterates the complete selected
   flattened list. The upstream `-1` duplicate-source skip is retained; a guard
   rejects an unreviewed task with more than two test subtasks. All selected
   source tasks receive one adapter per attempt.
2. Subset files contain exactly the fixed 11/8 source IDs and all 12/9 test
   subtasks. Evaluation refuses a `num_examples` value that would truncate the
   subset. The original shuffle is retained and cannot remove records.
3. Initial evaluation preprocessing reads challenges with `solution_file=None`.
   It uses input placeholders instead of final outputs when the upstream code
   computes example lengths. The original final evaluator still receives the
   separate solution file after predictions are generated. This intentionally
   prevents hidden-output lengths from influencing query filtering, and is a
   documented deviation from the upstream loader path.
4. The source paths, model revision, attempt seeds, and `PYTHONHASHSEED=0` are
   explicit. This stabilizes an upstream pipeline that deduplicates some tasks
   with Python sets, without claiming bitwise cross-device determinism.
5. Upstream evaluation overwrites `all_predictions.json` and `submission.json`
   inside its self-edit loop. Version 2 instead writes
   `all_predictions_{attempt}.json` and `submission_{attempt}.json`. This changes
   artifact names only; it leaves guesses, voting, scoring, and `final_results`
   unchanged. Both TTT and ICL evaluation scripts receive the explicit patch.

The 11 training tasks are included for manifest consistency and any later
training-only validation. Fixed TTT itself adapts only to visible demonstrations
of each evaluation task; it does not train an outer policy on those 11 tasks.
The entire final target mapping remains outside adaptation.

## Results and metric boundary

The original evaluator submits **two voted output guesses per test subtask**.
A source task succeeds in an adaptation attempt only if every test subtask has
a matching guess. The five fixed-TTT attempts therefore produce 40 task-attempt
outcomes, each with this two-guess allowance. ICL produces eight task outcomes
from one evaluation. Do not relabel either as our independent one-grid success
rate or as pass@5. See the upstream
[two-guess submission](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/eval-self-edits.py#L348-L372)
and [task success definition](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/arclib/eval.py#L54-L57).

The TTT run retains `results/all_predictions_0.json` through
`all_predictions_4.json` and `results/submission_0.json` through
`submission_4.json`; ICL retains the index-zero pair. The run script also copies
the upstream `final_results.json` into `results/` and creates
`results/upstream_summary.json`.
The summary rejects missing/duplicate task coverage, missing adaptation attempts,
and nonboolean correctness values. Its metric is explicitly named
`upstream_two_guess_voting_task_attempt_success`; it never overwrites independent
runner summaries. Summarization can also be rerun with:

```sh
python scripts/prepare_seal_baseline.py summarize \
  --package runs/seal_upstream_ttt_package_v2
```

The upstream console's competition denominator reads all tasks in the provided
file; our explicit subset makes that denominator eight. This is a documented
protocol reconstruction, not proof of the original Sleep authors' denominator.
A result must include the source patch, package manifest, raw predictions,
environment, and any runtime deviations. No paper reproduction claim follows
from preparing this package or passing its CPU tests.
