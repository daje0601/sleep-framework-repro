# Implementation validation — 2026-11-07

This records implementation evidence, not a numerical ARC reproduction.

- Initial Python 3.12 CPU suite: **144 passed**. After the GPU-memory correction:
  **146 passed, 1 CUDA-only test skipped** on the local CPU host.
- Current suite after shutdown monitoring, private accumulator offload, SDPA
  selection, and checkpoint-stop preservation: **158 passed, 2 CUDA-only
  tests skipped**. [Current hosted CI passed](https://github.com/daje0601/sleep-framework-repro/actions/runs/42249791037)
  for source commit `4894ea4`.
- Source distribution and wheel built successfully.
- The installed wheel ran the offline synthetic smoke command from outside the
  source checkout and restored its checkpoint/RNG successfully.
- The installed wheel verified the packaged data hashes and loaded the fixed
  11-training / 8-evaluation source-task split.
- A separate CPU preflight exercised all four configured wake steps on the
  first real training task using a tiny random model, with nonzero base and
  adapter updates. This remains diagnostic evidence, not a pretrained ARC score.
- [Hosted Linux CPU CI](https://github.com/daje0601/sleep-framework-repro/actions/runs/42244614666)
  passed tests, synthetic CLI execution, packaging, and artifact upload for
  source commit `90913508a492ce1954940bf3e50c5ef2917323ca`.

Independent review led to regression checks for causal routing, batch and
cached-generation invariance, committed consolidation behavior, per-expert
learning, rollback, isolated candidate lifetimes, token-weighted microbatch
updates, and exact checkpoint continuation through real optimizer steps.
Checkpoints with zero accepted outer updates remain explicitly untrained.

The fixed pretrained model revision is accessible after authentication and
loaded in CUDA attempts. Failures and corrections are recorded in
[experiment status](EXPERIMENT_STATUS.md). Subsequent A100/H100 requests did not
obtain allocation and were terminated; final account burn was zero. Successful
80 GB preflight and benchmark measurements remain pending. CPU success does not
establish GPU memory fit, semantic-judge quality, or the paper's reported accuracy.
