# Implementation validation — 2026-11-07

This records implementation evidence, not a numerical ARC reproduction.

- Initial Python 3.12 CPU suite: **144 passed**. After the GPU-memory correction:
  **146 passed, 1 CUDA-only test skipped** on the local CPU host.
- Earlier suite after shutdown monitoring, private accumulator offload, SDPA
  selection, and checkpoint-stop preservation: **158 passed, 2 CUDA-only
  tests skipped**. [Current hosted CI passed](https://github.com/daje0601/sleep-framework-repro/actions/runs/42249791037)
  for source commit `4894ea4`.
- Earlier suite after diagnostic-checkpoint isolation, configurable FFN LoRA,
  chat/contextual minibatch variants and pinned upstream preparation: **181
  passed, 2 CUDA-only tests skipped**. The [hosted CPU workflow also passed](https://github.com/daje0601/sleep-framework-repro/actions/runs/42310565556)
  for source commit `dfb9182`, including smoke, packaging, and artifact upload.
  This is CPU evidence, not a pretrained GPU run.
- After separate queue/runtime deadlines: **189 passed, 2 CUDA-only skipped**;
  [hosted CPU workflow passed](https://github.com/daje0601/sleep-framework-repro/actions/runs/42311751440)
  at `cead89d`.
- The actual pinned Llama tokenizer passed chat-prefix/BOS/masking checks.
  A token-length audit of all 19 fixed source tasks, using training holdouts
  and four rotations, found maximum SFT length 4,173 and maximum grid/directive
  prompt plus generation allowance 6,864, below the configured 16,384 limit.
  This does not measure generation quality or GPU memory.
- The upstream TTT/ICL source packages were prepared and verified from 19
  pinned public files. Per-attempt raw artifacts have indexed filenames.
  Direct-pin PyPI metadata was audited separately; no upstream installation
  or GPU execution is claimed.
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

The [H100 80 GB preflight](../results/arc-h100-preflight-v2/README.md) passed
using the pinned pretrained revision and v2 protocol on one training task.
Its forced updates and checkpoint roundtrip verify execution; the invalid
generated directive and zero held-out training score do not establish ARC
accuracy. Other failures and corrections are recorded in
[experiment status](EXPERIMENT_STATUS.md).
