# H100 execution preflight (2026-11-07 KST)

One pretrained Llama-3.2-1B-Instruct run on NVIDIA H100 80 GB completed at
source `cead89d728a2713d1dc93312b71a5cc6ccef3e56` using
`configs/arc_sleep_v2.json`, seed 42, and training task `44f52bb0`. The provider
confirmed success and the process exited 0. This is **not an ARC benchmark
result**: no final evaluation targets were scored.

| Measurement | Value |
| --- | ---: |
| Wake steps / level-0 consolidations | 4 / 4 |
| Max absolute initial conversion-logit difference (four-token probe) | 0 |
| Base / inner / outer parameter L1 changes | 102.9051 / 11742.5641 / 2470.9443 |
| Peak CUDA allocated / reserved | 32.94 / 38.79 GiB |
| Preflight elapsed (excludes container setup) | 667.74 s |
| Diagnostic checkpoint and RNG restoration | Passed |
| Generated directive valid | No |
| Held-out **training** demonstration score | 0 |

The model generated a grid before its directive JSON. Strict parsing classified
that output as invalid. The preflight then used its predetermined identity
fixture to test optimizer, adapter, and checkpoint behavior; it did not accept a
model-generated candidate or train the real policy. No expert-by-expert counts
are available in the serialized report.

[Report](report.json) includes the resolved config, model revision, data and
protocol hashes, original generated text, loss traces, fixture scope, and source
report SHA256. Only the job identifier and private checkpoint path were omitted.
[Environment](environment.json) records the hardware and CUDA/PyTorch versions.
The container digest is in the report. No gated weights or checkpoints are
redistributed. A separate fresh job is needed for actual train/evaluate results.
