# Explicit protocol variants after the upstream audit

The `arc_*_v2.json` files opt into corrections identified by comparing the
independent runner with pinned SEAL commit
`6d9c9f9ee392c6cc618e771f399d436d190f6ca4`. Legacy `arc_*.json` settings retain
their old behavior for interpreting the failed preflights. Neither variant is
an exact author recipe or a successful paper reproduction.

| Setting | Legacy configuration | Opt-in v2 configuration |
| --- | --- | --- |
| Prompt serialization | Raw text with BOS | Pinned tokenizer chat template; one user message and assistant generation prefix |
| Inner SFT updates | One update per entire expanded dataset per epoch | Shuffled minibatches of two examples; physical microbatch one |
| Outer SFT updates | One full-dataset update, one epoch | Minibatches of five accepted directives, eight epochs |
| Visible supervision | Isolated transformed input/output pairs | Leave-one-visible-pair-out contextual examples, separately within each transformation |
| LoRA targets | Attention q/v | Attention q/v plus FFN gate/down/up |
| Update budget | Epoch count | Actual optimizer-step count, maximum 375 per SFT call; oversized requests rejected |

The chat template uses the fixed date string `26 Jul 2024` to prevent runtime
date changes. Generation and SFT share exactly the same serialized prompt.
On-policy teacher prefixes remain inside the assistant response and are masked
from loss. Completion EOS remains the pinned tokenizer EOS. The actual pinned
Llama tokenizer was checked locally: one BOS, identical generation/SFT prefix,
and correct prefix masking. This is tokenizer evidence, not GPU execution.

Inner adaptation uses the configured directive's 1–5 epochs. Token-weighted
microbatch accumulation keeps effective-batch updates independent of physical
microbatch size. All lengths are validated before the first update; no example
is truncated. If a training reward holdout leaves only one visible pair,
supervision uses that pair alone, without restoring the held-out example.
All FFN policy/inner adapter weights remain frozen during CMS wake/base updates;
only the selected adapter branch changes during its SFT stage.

The runner records expanded-pair hashes, actual prompt/completion hashes,
per-step losses, and completed inner/outer optimizer-step counts. Preflight and
smoke checkpoints carry `purpose=diagnostic` and cannot be loaded by normal
train/evaluate commands. Older checkpoints without a purpose are rejected.

## Remaining differences

These corrections do not make v2 identical to SEAL. The independent runner
still uses a different bounded JSON directive schema, limited augmentation
families, a fixed raw generation count, a different contextual text format,
constant learning rates rather than the upstream warmup/cosine schedule, and
one greedy output per test subtask. Its one-epoch identity TTT configuration
is not the upstream two-epoch augmented TTT baseline. CMS routing, memory
schedule, KS/LTI weighting, and semantic-judge choices remain reconstructions.
The exact Sleep evaluation manifest has not been recovered.

The original SEAL evaluator uses augmentation, inversion, voting, and **two
output guesses per subtask**, separately from five adaptation attempts. Directly
comparing the independent one-grid metric with that metric is invalid.
Use the separately prepared [upstream baseline package](SEAL_BASELINE.md), which
preserves the upstream trainer and evaluator and explicitly records integration
deviations. Its environment and GPU execution still need validation.

Relevant primary code: [inner trainer](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/arclib/update_model.py#L227-L253),
[TTT recipe](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/ttt.py#L257-L368),
[chat preprocessing](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/inference/preprocess.py#L80-L108),
and [two-guess evaluation](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/eval-self-edits.py#L348-L372).

## Running the variant

Use a fresh run directory and checkpoint for v2. A preflight from another source
commit or configuration does not authorize the launcher's train/evaluate gate.

```sh
sleep-repro check-config --config configs/arc_sleep_v2.json
sleep-repro preflight --config configs/arc_sleep_v2.json --output runs/v2_preflight
sleep-repro train --config configs/arc_sleep_v2.json --output runs/v2_train
sleep-repro evaluate --config configs/arc_sleep_v2.json \
  --checkpoint runs/v2_train/checkpoint.pt --output runs/v2_eval
```

These GPU commands remain unexecuted. Fit on an 80 GB GPU and measured ARC
accuracy are not established by CPU tests or by configuration corrections.
