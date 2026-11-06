# Experiment status

Updated 2026-11-07. This is an unofficial independent reconstruction. There are
**no successful pretrained GPU preflight and no quantitative reproduction results**.
Published paper percentages are not measurements from this repository.

The first L40S preflight loaded the pinned model, then failed during
consolidation audit snapshot allocation: 43.73 GiB allocated on a GPU exposing
44.39 GiB, with another 1,002 MiB requested. This used commit `9091350`.
The follow-up implementation moves audit and rollback copies to CPU and
releases already-accumulated wake gradients before creating snapshots.

| Component / experiment | Implementation and available evidence | Execution / reproduction status |
| --- | --- | --- |
| ARC data provenance | Public SEAL files retrieved at the pinned commit; actual SHA256 values committed; explicit 11/8 source-task split and all retained test subtasks recorded | Data preparation and cached verification executed; this is not model evidence |
| Scheduled memory and conversion | Bounded expert banks, identity-initialized residual stages, fixed token-local routing, explicit periods | CPU correctness checks available; pretrained conversion probes not run |
| Transactional consolidation | Private teacher/student snapshots, receiver-only fitting, finite checks, atomic commit, rollback and optimizer reset | CPU invariant checks available; pretrained GPU transfer not run |
| KS/LTI | Masked forward KL, detached mixed-policy sampling, frozen teacher semantic judging, edit reward | CPU objective/engine checks available; semantic-judge quality and ARC effectiveness unmeasured |
| Dreaming and ReST-EM | Bounded JSON dreams, gradient ranking, isolated candidate adaptation, positive-reward outer updates | CPU correctness checks available; learned pretrained directive policy not trained |
| Synthetic integration | Tiny random Llama, synthetic grids, actual updates, checkpoint continuation | Exercised locally through CPU tests; deliberately not ARC benchmark evidence |
| Hosted CPU CI | Offline tests, synthetic CLI smoke, package build and uploaded evidence | [Initial public run passed](https://github.com/daje0601/sleep-framework-repro/actions/runs/42244614666) on commit `9091350` |
| ICL / TTT / consolidation | Reconstructed evaluation paths and configs provided | Pretrained ARC execution not run |
| `self_edit` | Reconstructed JSON self-edit policy without CMS | Pretrained training/evaluation not run; differs from official SEAL |
| Official SEAL baseline | Upstream recipe identified and cited | External baseline not implemented or executed here |
| Reconstructed Sleep ARC | Training, evaluation, checkpoint/resume and seed aggregation entry points provided | First GPU preflight failed with OOM; corrected preflight and quantitative comparison pending |
| Reasoning, knowledge incorporation, classification, translation, long context | Future scope | Experiment pipelines not implemented; no execution or reproduction claim |

## Evidence that must accompany a result

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
