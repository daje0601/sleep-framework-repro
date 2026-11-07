# Sleep Framework Reproduction

An **unofficial, independent ARC reconstruction** of [Language Models Need Sleep](https://arxiv.org/html/2606.03979v2). This repository implements a runnable interpretation of the method and records its reconstruction choices. It is not affiliated with Google or the paper authors.

**No quantitative paper result has been reproduced.** The [seed 42 H100 training run](results/arc-h100-seed42-training-12/report.json) saved a checkpoint after one of 11 tasks with zero outer policy updates, then stopped during memory consolidation on task two. Final evaluation has not run. The earlier [H100 80 GB preflight](results/arc-h100-preflight-v2/README.md) verified the training path and checkpoint restoration using forced diagnostic updates. See the [equation and ARC protocol review (Korean)](docs/PAPER_AUDIT_KO.md) for the differences from the paper and the [experiment status](docs/EXPERIMENT_STATUS.md) for run evidence.

## What is implemented

The PyTorch implementation combines scheduled memory stages, transactional consolidation, Knowledge Seeding/Learning to Imitate, gradient-ranked dreaming, isolated candidate adaptation, and ReST-EM policy updates. Important builder choices include identity-initialized residual memory, fixed token-local expert routing, custom dual LoRA branches, and bounded JSON dreams. These are documented in the [reconstruction registry](docs/RECONSTRUCTION.md).

| Path | Implementation | Pretrained ARC results |
| --- | --- | --- |
| Reconstructed Sleep | Memory, consolidation, dreaming, inner adaptation, outer policy update | H100 preflight passed; seed 42 training stopped after 1/11 tasks; no final accuracy |
| `icl` | Visible examples and greedy grid decoding | Not run |
| `ttt` | Inner LoRA on visible pairs with an identity augmentation directive | Not run |
| `self_edit` | Learned JSON directives and inner LoRA, without CMS | Not run |
| `consolidation` | Wake and consolidation without dreaming | Not run |
| Pinned upstream SEAL ICL/TTT | Verified source package, explicit integration patch and commands; separate environment required | Not run |

`self_edit` differs from official SEAL and must retain its own label. The initial target is the paper's ARC experiment; reasoning, knowledge incorporation, continual learning, translation, and long-context benchmarks remain outside the implemented experiment scope.

The [upstream audit and v2 variants](docs/PROTOCOL_V2.md) document differences in
chat formatting, optimizer updates, contextual examples, adapter targets, and
evaluation metrics. The opt-in `arc_*_v2.json` configs correct several training
differences; they remain independent reconstructions. The [upstream baseline
package](docs/SEAL_BASELINE.md) preserves SEAL's two-guess voting metric separately
from this runner's one-grid metric.

## Install and run CPU checks

Python 3.12 or newer is required. From a checkout:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest
sleep-repro smoke --output runs/smoke
python -m build
```

The smoke command uses a tiny randomly initialized Llama and synthetic grids, runs real gradient updates, and checks checkpoint continuation. It needs neither model access nor an ARC download. Its report is `runs/smoke/smoke.json`.

`requirements.lock` records the locally tested Python 3.12 environment, including optional installed packages. CUDA jobs use the pinned container's PyTorch and save their own complete Linux dependency snapshot. The implementation uses its own dual LoRA layers and does not require PEFT.

## Prepare ARC and run an experiment

```sh
python scripts/prepare_data.py
sleep-repro check-config --config configs/arc_sleep.json
```

Preparation verifies four public files against committed SHA256 values. The fixed reconstruction selects 11 training and eight evaluation tasks, keeping every selected task's test pair. See [ARC protocol](docs/ARC_PROTOCOL.md) for provenance, the split discrepancy, JSON schema, and target isolation.

The commands below are **unexecuted GPU experiment instructions**. They require CUDA, access to `meta-llama/Llama-3.2-1B-Instruct`, and an authenticated Hugging Face environment. The supplied configs pin a model revision; freeze the dependency environment before a measured run. Full-model transactional snapshots and candidate copies can consume substantial memory. An 80 GB GPU is a candidate for preflight, not a measured requirement or fit guarantee. The preflight checks one actual training task, parameter changes, conversion logits, checkpoint continuation, and peak memory. Its forced optimizer fixtures are diagnostics, not policy-training evidence. See [VESSL instructions](docs/VESSL.md) for bounded cloud jobs.

```sh
sleep-repro preflight --config configs/arc_sleep.json --output runs/preflight
sleep-repro train --config configs/arc_sleep.json --output runs/example_train
sleep-repro evaluate --config configs/arc_sleep.json \
  --checkpoint runs/example_train/checkpoint.pt --output runs/example_eval
```

Resume training at an episode boundary with `train --resume runs/example_train/checkpoint.pt` and the same config/output. Use a fresh evaluation directory for every evaluation. `icl`, `ttt`, and `consolidation` can be evaluated directly with their corresponding `configs/arc_*.json` files; `sleep` and `self_edit` require their trained checkpoint for the primary comparison. `--allow-untrained` explicitly requests an untrained-policy ablation.

For the declared three training seeds, generate configs and run each complete train/evaluate pair:

```sh
python - <<'PY'
import json
from pathlib import Path
base = json.loads(Path('configs/arc_sleep.json').read_text())
directory = Path('runs/configs')
directory.mkdir(parents=True, exist_ok=True)
for seed in (42, 43, 44):
    (directory / f'arc_sleep_seed{seed}.json').write_text(
        json.dumps({**base, 'seed': seed}, indent=2) + '\n')
PY
for seed in 42 43 44; do
  sleep-repro train --config "runs/configs/arc_sleep_seed${seed}.json" \
    --output "runs/sleep_seed${seed}_train"
  sleep-repro evaluate --config "runs/configs/arc_sleep_seed${seed}.json" \
    --checkpoint "runs/sleep_seed${seed}_train/checkpoint.pt" \
    --output "runs/sleep_seed${seed}_eval"
done
sleep-repro aggregate runs/sleep_seed42_eval runs/sleep_seed43_eval \
  runs/sleep_seed44_eval --output runs/sleep_3seed_summary.json
```

The aggregator checks distinct seeds and matching configurations, task/attempt sets, and data hashes. It reports descriptive variability across seeds. Report successful **task attempts / 40** separately from **tasks solved / 8**; all test outputs of a task must match for an attempt to succeed. Correlated attempts are not independent statistical trials.

Training writes provenance, JSONL episode records, an episode-boundary checkpoint, and a training summary. Evaluation writes provenance, per-attempt records, and `summary.json`. Preserve these artifacts, the source commit, resolved configuration, environment/container identity, and execution metadata when reporting a result. Keep gated weights, credentials, downloaded data, and run directories out of Git.

## Contributing and attribution

Run the CPU suite before submitting changes. Add meaningful regression tests for behavior changes, update the reconstruction registry when a scientific choice changes, and include raw artifacts for experimental claims. Cite this software using [CITATION.cff](CITATION.cff) and cite the original work separately:

- [Sleep paper, v2](https://arxiv.org/html/2606.03979v2)
- [Nested Learning](https://arxiv.org/abs/2512.24695)
- [SEAL paper](https://arxiv.org/abs/2506.10943) and [official code](https://github.com/Continual-Intelligence/SEAL)
- [ARC-AGI dataset](https://github.com/fchollet/ARC-AGI)

Our original code is [MIT licensed](LICENSE). SEAL's code has its own [MIT notice](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/LICENSE). Downloaded ARC data retain the upstream [Apache 2.0 license](https://github.com/fchollet/ARC-AGI/blob/master/LICENSE). Meta's model weights are governed separately by the [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/LICENSE.txt) and associated acceptable-use terms. This repository does not redistribute model weights or relicense third-party material.
