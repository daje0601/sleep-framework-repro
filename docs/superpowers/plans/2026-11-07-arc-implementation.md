# ARC Sleep Reproduction Implementation Plan

> **For agentic workers:** Use subagent-driven development and meaningful red/green tests. User approved implementation; proceed without another planning approval.

**Goal:** Build executable independent ARC Sleep reproduction, CPU invariant tests, Hugging Face training/evaluation and VESSL launch instructions.
**Architecture:** PyTorch CMS wraps pretrained FFNs; transactional updates use teacher/student snapshots. ARC modules isolate solutions from adaptation; callback-based dreaming supports independent candidates and ReST-EM. CLI produces auditable run artifacts.
**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, pytest, JSON configs.
**Spec:** docs/superpowers/specs/2026-11-06-sleep-paper-reproduction-design.md

## Global Constraints
- Independent reconstruction, never label synthetic tests or unrun jobs paper reproduction.
- Preserve initial pretrained logits; new receiver experts alone train during consolidation.
- Final evaluation targets never reach training or candidate selection.
- Model/data/config revisions, seeds, raw predictions, and experimental deviations are persisted.
- CPU works offline with tiny randomly initialized Llama; quantitative ARC needs pretrained weights and GPUs.

## Review Focus
- Empty/invalid directives and malformed rectangular grids must fail predictably (ARC).
- Failed consolidation must leave live parameters unchanged (memory).
- No accepted dreams must produce a logged no-op (dreaming).
- Multiple ARC test pairs count as one task attempt only when all match (metrics).
- Resume restores optimizer/RNG/config; unsupported resume must fail explicitly (runner).

## File structure and interfaces
- `src/sleep_repro/memory.py`, `model.py`, `consolidation.py`: MemoryStage/MemoryChain, `attach_memory(model, levels, rank, capacity, periods)`, `iter_memory_chains(model)`, transaction API.
- `src/sleep_repro/arc.py`, `metrics.py`: ARC loading, public input projection, JSON directives, reversible augmentations, formatting/parsing, manifest, per-attempt metrics.
- `src/sleep_repro/objectives.py`, `dreaming.py`: masked causal KL, LTI policy gradients, JSON dream selection/adaptation and ReST-EM helper APIs.
- `src/sleep_repro/runner.py`, `cli.py`: pretrained/tiny model orchestration, train/evaluate/smoke commands, JSONL evidence.
- `configs/`, `scripts/`, `docs/`, `.github/workflows/`: published configs, VESSL launcher, reproducibility registry, CPU CI.

### Task 1: Transactional memory
- [x] Write tests for exact initial identity, finite bank capacity, immutable teacher/live rollback and receiver-only updates; run red.
- [x] Implement MemoryStage/MemoryChain with ordered periods and checkpointed routing state.
- [x] Implement Hugging Face Llama wrapping and copy-based consolidation; document exact interfaces for integrator.
- [x] Run tests green and report changes.

### Task 2: ARC protocol
- [x] Test rectangular grids, safe solution-free projections, roundtrip augmentations and whole-task success; run red.
- [x] Implement fixed reconstruction manifest (11 train / 8 eval), hash-pinned data loader, directives and metric summaries.
- [x] Test malformed outputs and dataset leakage boundaries; run green.

### Task 3: Objectives and dreaming
- [x] Test masked KL, policy gradient signs, isolated candidates, deterministic selection and no accepted dreams; run red.
- [x] Implement distillation + semantic/edit LTI, gradient-norm selection, candidate reward filtering and ReST-EM.
- [x] Provide model-agnostic callback contracts; run green.

### Task 4: Executable experiment
- [x] Integrate model, ARC, objectives and dreaming in offline CPU smoke plus HF GPU train/evaluate runner.
- [x] Implement strict JSON configuration validation, pinned data preparation, baseline modes, output records and checkpoints.
- [x] Run CPU integration using tiny Llama and synthetic grids; never count as paper evidence.
- [x] Add VESSL batch entrypoint, public docs/license/citation, CI and honest experiment status.

### Task 5: Review and release readiness
- [x] Independent scientific/code review; fix important findings and rerun affected tests.
- [x] Run complete CPU suite and CLI smoke, package build, clean artifact checks.
- [x] Inspect actual GitHub/VESSL access; publish/execute only through available authorized account and record blockers precisely.

## Execution ledger
- 2026-11-07: Approved design. Current folder has no Git repo. Use current folder and initialize a new development branch. Parallel modules have disjoint ownership; integration and mutation of git state stay with controller.

- 2026-11-07: Core ARC implementation complete. Independent review fixed causal routing, committed-function consistency, per-expert training eligibility, low-precision safety, dataset/checkpoint binding, policy-update accounting, and candidate memory lifetimes. CPU suite: 142 passed; built and installed wheel passed offline CLI smoke and packaged 11/8 data verification. GPU and hosted CI evidence remain pending.
- 2026-11-07: Final initial-release suite: 144 passed. Public source commit `9091350` published; hosted Linux CI passed. VESSL and model authentication verified. A single L40S preflight with a 1,800-second execution limit was submitted after explicit authorization to store the model-access token as a private VESSL secret. GPU outcome and quantitative reproduction remain pending.
