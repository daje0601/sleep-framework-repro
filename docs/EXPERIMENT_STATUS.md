# Experiment status

Updated 2026-11-07. This is an unofficial independent reconstruction. A [pretrained H100 preflight](../results/arc-h100-preflight-v2/README.md) passed. There are
**no quantitative paper reproduction results**.
Published paper percentages are not measurements from this repository.

The first L40S preflight loaded the pinned model, then failed during
consolidation audit snapshot allocation: 43.73 GiB allocated on a GPU exposing
44.39 GiB, with another 1,002 MiB requested. This used commit `9091350`.
The follow-up implementation moves audit and rollback copies to CPU and
releases already-accumulated wake gradients before creating snapshots.
The second preflight (`3f143bd`) failed before model loading because the shared
object-volume Hugging Face cache returned an invalid `config.json`. Therefore
it did not validate the memory correction. Both jobs reached terminal failure;
the account's compute burn rate was verified as zero afterward.
The third attempt (`923d9b3`) passed model loading and audit snapshot creation
but exhausted GPU memory in the KS student attention forward (42.65 GiB
allocated, another 532 MiB requested). Its terminal failure and zero account
burn rate were verified by the attached launcher. Follow-up changes offload
unused private gradient accumulators, select the explicit SDPA attention
backend, and preserve all checkpoint generation-stop tokens.
The fourth L40S attempt was explicitly terminated to move validation to an
A100/H100 with 80 GB. Both an A100 request and a subsequent H100 request
remained in scheduling without application execution; the account catalogue
reported both resource types unavailable. Both requests were terminated and
the account burn rate was verified as zero. Neither waiting request produced a pretrained preflight result.

The seventh request used H100 80 GB and source `b0d4a33`. It waited about
30 minutes before starting; dependencies installed and the pinned model weights
loaded at 23:22:21 UTC on 2026-11-06. The attached launcher's combined queue and
runtime deadline then terminated it before the preflight completed. This was a
launcher-budget problem, not a measured model OOM or accuracy failure. Termination
and a subsequently zero account burn rate were verified. An initial billing
read still showed the old rate, so a follow-up verification was required.
Subsequent launches can explicitly separate queue and runtime budgets while
retaining finite absolute and container deadlines.

The eighth H100 job was allocated but failed during dependency installation:
`click==8.3.3` conflicted with `huggingface-hub==2.4.0`. No model ran. The ninth
job used compatible `click==8.5.0`, passed the complete v2 preflight and exited
0. Peak CUDA allocation/reservation were 32.94/38.79 GiB, with four wake steps,
four consolidations and verified checkpoint/RNG restoration. The generated
directive was invalid; training-fixture updates were forced and held-out
training score was zero. No final evaluation occurred. The local monitor lost
connectivity; terminal success and zero burn were independently confirmed via
the provider after reconnection. See the [public report](../results/arc-h100-preflight-v2/report.json).

| Component / experiment | Implementation and available evidence | Execution / reproduction status |
| --- | --- | --- |
| ARC data provenance | Public SEAL files retrieved at the pinned commit; actual SHA256 values committed; explicit 11/8 source-task split and all retained test subtasks recorded | Data preparation and cached verification executed; this is not model evidence |
| Scheduled memory and conversion | Bounded expert banks, identity-initialized residual stages, fixed token-local routing, explicit periods | H100 four-token conversion probe passed with zero maximum logit difference |
| Transactional consolidation | Private teacher/student snapshots, receiver-only fitting, finite checks, atomic commit, rollback and optimizer reset | Four H100 consolidation events completed in one training-task preflight |
| KS/LTI | Masked forward KL, detached mixed-policy sampling, frozen teacher semantic judging, edit reward | CPU objective/engine checks available; semantic-judge quality and ARC effectiveness unmeasured |
| Dreaming and ReST-EM | Bounded JSON dreams, gradient ranking, isolated candidate adaptation, positive-reward outer updates | CPU correctness checks available; learned pretrained directive policy not trained |
| Synthetic integration | Tiny random Llama, synthetic grids, actual updates, checkpoint continuation | Exercised locally through CPU tests; deliberately not ARC benchmark evidence |
| Hosted CPU CI | Offline tests, synthetic CLI smoke, package build and uploaded evidence | [Initial public run passed](https://github.com/daje0601/sleep-framework-repro/actions/runs/42244614666) on commit `9091350` |
| ICL / TTT / consolidation | Reconstructed evaluation paths and configs provided | Pretrained ARC execution not run |
| `self_edit` | Reconstructed JSON self-edit policy without CMS | Pretrained training/evaluation not run; differs from official SEAL |
| Pinned upstream SEAL ICL/TTT baseline | SHA256-verified source package, explicit integration patch, separate setup/run commands and metric summary | Actual preparation exercised; dependency compatibility and GPU execution not yet validated |
| Reconstructed Sleep ARC | Training, evaluation, checkpoint/resume and seed aggregation entry points provided | H100 v2 preflight passed; full train/evaluation comparison pending |
| Reasoning, knowledge incorporation, classification, translation, long context | Future scope | Experiment pipelines not implemented; no execution or reproduction claim |

## Evidence that must accompany a result

The [upstream audit](PROTOCOL_V2.md) found additional scientific differences:
legacy full-dataset optimizer steps, raw prompts, isolated training pairs,
q/v-only adapters, and a one-grid evaluation metric. Opt-in v2 settings correct
the first four areas; they do not implement SEAL's complete augmentation,
scheduler, directive schema, or two-guess voting. A successful GPU preflight
alone would therefore still not establish numerical reproduction.

The CLI records resolved configuration, available source/model revisions,
dependency versions, parameter counts, data hashes, episode and attempt records,
and checkpoints. Preserve these files when reporting results. Add the frozen
dependency/container identity, hardware information, training-token/update
counts, measured peak memory, GPU-hours, job identity, and any protocol
deviations to the report. A CPU smoke report is not a substitute for these GPU
artifacts.

Checkpoint existence and a completed episode schedule are not evidence of a
successful policy update. Inspect accepted candidates and actual outer updates.
Invalid generated directives, an empty accepted set, adaptation failures, or
failure to recover the paper's improvement are reportable outcomes.

For the first numerical report, run the same fixed eight source tasks with five
attempts per task and three independent training seeds. Report each seed and
task, successful attempts over 40, and tasks solved over eight. Use tasks/seeds
as uncertainty units. The aggregate CLI provides descriptive seed variability;
it does not certify equivalence to the paper or produce independent-trial
confidence intervals from 40 correlated attempts.

## Remaining work before a paper comparison

1. Freeze all model/tokenizer revisions, dependencies, configuration, and
   training-only sensitivity choices; record the reconstruction deviations.
2. Verify pretrained conversion logits/greedy probes and receiver learning,
   then measure a small GPU preflight's memory, losses, parameter changes,
   expert eligibility, and checkpoint continuation.
3. Execute and verify the reconstructed baselines, and obtain the separately
   labeled official SEAL baseline through its upstream recipe.
4. Run the locked three-seed comparison, retain all raw artifacts, and publish
   measured outcomes and failures without changing final tasks after inspection.

The detailed protocol is in [ARC_PROTOCOL.md](ARC_PROTOCOL.md); architectural
and objective choices are in [RECONSTRUCTION.md](RECONSTRUCTION.md).
