# Sleep Framework: independent paper reproduction

Status: approved on 2026-11-07; implementation and evidence status are tracked in
`docs/EXPERIMENT_STATUS.md`. This document preserves the original design rationale.
Date: 2026-11-06 (Asia/Seoul).

## Intent and agreed constraints

Implement the methods in *Language Models Need Sleep: Learning to Self-Modify
and Consolidate Memories*, arXiv:2606.03979v2, and publish reproducible code and
experimental evidence on GitHub. The user explicitly chose **paper
reproduction** and can obtain GPUs through VESSL.ai.

Success means an independently runnable, auditable reproduction attempt with
reported measurements. A working toy model alone is not a paper reproduction.
Matching the paper's numbers is a hypothesis to test, not a release promise.

The workspace is currently empty apart from browser investigation files. It has
no Git repository, source code, dependencies, or configured remote.

## Proposed first release

Use **ARC few-shot adaptation (paper Table 4 and Appendix B.3)** as the first
quantitative target. Implement the complete consolidation and dreaming paths
needed for this target, reproduce the available baselines, and publish the
resulting comparison. This is a first experiment in a broader paper
reproduction project; it does not establish reproduction of all paper results.

Why start here: the paper identifies a Llama-3.2-1B backbone and an 11-training /
8-evaluation task protocol, and points to SEAL, whose official repository
contains few-shot code and filtered ARC data. In contrast, reconstructing the
mathematical reasoning experiment also requires a training-data recipe that
the Sleep paper does not fully specify.

Proposed project name: `sleep-framework-repro`; Python import: `sleep_repro`.
The existing local folder name can stay unchanged. Use Python, PyTorch,
Transformers, and PEFT. Pin tested versions and upstream revisions when the
implementation environment is built. These are the assistant's proposed
implementation choices.

## Scope and evidence standards

Publish a matrix that separately tracks method implementation, baseline
execution, and quantitative reproduction for each experiment. Later targets
are Qwen3-1.7B reasoning (Table 2), knowledge incorporation (Table 3), then
continual classification, translation, and long-context evaluation. Do not
describe these as implemented or reproduced before their individual runs.

Initial release includes:

- CMS memory blocks with explicitly recorded update periods and bounded,
  preallocated low-rank expert banks.
- Transactional compute–consolidate–update, Knowledge Seeding, and Learning to
  Imitate, including both semantic and edit-distance reward components.
- Dream generation with additional random-expert routing, gradient-based
  candidate selection, isolated per-candidate LoRA adaptation, and ReST-EM.
- ARC train, reward/validation, and final evaluation paths with pinned task IDs
  and documented ancestry from SEAL.
- Baselines, ablations, checkpoint/resume, machine-readable result artifacts,
  VESSL execution instructions, and CPU correctness tests.
- English README, paper citation, limitations, third-party attribution,
  contribution instructions, and a proposed MIT license for our own code.

The release must identify itself as an **unofficial independent reproduction**.
Do not claim Google affiliation, universal absence of forgetting, 10M-token
capability, or reproduction of unrelated experiments from an ARC result.

## Method design

### Memory model and scheduling

Represent the memory as an ordered chain of MLP blocks with explicit integer
update periods, as described in Section 2.2. Track accumulated gradients,
active expert masks, update counters, routing state, and optimizer state.
Reject an invalid period hierarchy. Process simultaneous update boundaries
from fast to slow, using fresh transactional snapshots for each adjacent
transfer. This ordering is a reconstruction choice and must be tested and
documented.

The pretrained-model integration is a bounded **post-training** reconstruction;
full architectural pretraining is outside the first release. Preserve Llama's
attention, normalization, and outer residual paths. Inside each selected FFN
position, the first memory stage is the existing pretrained MLP. Append slower
residual MLP stages initialized to the identity (zero-output trainable branch),
then apply the unchanged outer residual. Each stage owns a separately scheduled
base branch and a preallocated bank of rank-64 expert branches. Start with two
levels; make the same graph configurable for four levels. Trace the underlying
memory-chain construction to the Sleep and Nested Learning papers, but label
the identity/residual embedding in pretrained Llama as **our reconstruction**.

Use routed additive expert output within each stage, with an always-on base
branch and explicitly logged mixing coefficients. The router selects from
trained active experts; during dreaming it additionally chooses an active
expert outside the normal selected set without changing stored masks. If no
such expert exists, log the absence and do not count that event as evidence
for random-expert dreaming. Ensure the full-method run has eligible experts
before its measured dreaming steps. Router scoring and its initialization
are builder choices to freeze in the reconstruction registry before training.
This routing is a reconstruction choice; neither the graph nor routing is
claimed to be the authors' unpublished implementation.

Before training, conversion must preserve logits within a declared dtype-aware
tolerance and preserve greedy outputs on a frozen probe set. Then verify that
the appended stages receive useful gradients and can reduce a held-out
teacher/student mismatch in an isolated transfer. If this path fails, fix the
conversion or report the failure before beginning numerical reproduction;
do not silently initiate a new pretraining project or replace CMS with plain
adapter copying. A simpler adapter-only variant can be a named ablation.

Expert capacity is finite. Exhausting a bank must stop with a useful error or
follow an explicitly configured policy; it must not overwrite an old expert
or silently grow the parameter budget. Reset optimizer moments when resetting
an expert. Report total, active, and trainable parameter counts separately.

### Consolidation transaction

For a sender block and its adjacent slower receiver:

1. Compute the sender's proposed base-weight update without changing the live
   model. Preserve an immutable pre-update teacher snapshot.
2. Create an isolated student with that proposed sender update, cleared sender
   experts, and one newly allocated receiver expert.
3. Optimize only that new receiver expert using teacher/student rollouts,
   distillation, and the LTI objective. No gradient passes through sampling;
   the semantic reward model is frozen.
4. Commit the proposed base update, sender reset, and receiver activation
   together. On failure, preserve the live model and its optimizer state.

Record divergence direction, temperature, mixture coefficients, generation
parameters, masking, reduction, and policy-gradient estimator in the config.
The semantic scorer checkpoint and prompt must be pinned. Exact string reward
alone is an explicitly named ablation, not a substitute for the full method.

### Dreaming

Generate candidates with the configured task context and an extra randomly
selected expert in each applicable router. Score each candidate with a
documented scalar statistic of its SFT gradient; the paper supplies a gradient
vector but leaves its scalar ranking unspecified. Use a declared norm-based
choice initially and record which parameter set it measures. Select top-k and
additional random candidates without duplication.

For each selected candidate, start from the same post-consolidation model,
adapter, optimizer, and RNG state. Adapt with LoRA, then compare performance
against the unadapted snapshot on the permitted reward examples. Candidate
changes must never leak into another candidate. Keep positive-reward samples
for the ReST-EM update. Specify which parameters that outer update changes and
save its optimizer state. An empty accepted set is a logged no-op.

Ground-truth final test outputs must not enter the context, training loss,
candidate selector, semantic reward, or hyperparameter selection.

For ARC, define a dream as a SEAL-compatible **structured augmentation and
adaptation directive**, serialized as JSON. Deterministically expand it into
supervised ARC examples using only the task's visible demonstration pairs.
Inner LoRA adaptation and gradient importance use these expanded examples;
outer ReST-EM learns the successful directive's token sequence. Log both the
directive and expanded-example hashes. Scoring the expanded examples rather
than treating the JSON directive itself as ARC training data is an explicit
task-specific reconstruction. Invalid directives are recorded and rejected.

### ARC episode state

A wake step is one supervised minibatch over serialized visible ARC
demonstrations, accumulating gradients according to the memory schedule.
Use configurable periods, initially `[1, 4]` for the two-level reconstruction;
the units are wake minibatches, not generated tokens or task counts. This
initial schedule is ours and must receive a training-only sensitivity check.
Complete pending valid consolidation transactions before dreaming.

Each training episode starts from the current outer policy checkpoint, with
fresh episode-local memory counters, accumulated gradients, active expert
masks, and optimizers. Keep only accepted outer ReST-EM updates across
episodes. Those updates train a separately identified policy LoRA adapter;
candidate-specific inner adapters and episodic memory never persist into
another episode. Evaluate candidate reward using the training task's permitted
held-out examples. The exact reward split is included in the frozen manifest.

During final evaluation, start every task and each of its five attempts from
the same frozen trained policy checkpoint and fresh episode state, varying
only the declared sampling seed. Use visible demonstrations for wake,
consolidation, directive generation, and inner adaptation. Do not perform a
ReST-EM policy update using evaluation-task outcomes. Score every attempt
after adaptation; do not use final outputs to select the best dream. If an
evaluation-time selector is enabled, its score may use leave-one-demonstration-
out validation only and is a separately reported variant.

This episodic ARC lifecycle is distinct from the persistent memory lifecycle
required by later continual-learning benchmarks. The ARC runner must not be
used to claim those benchmarks have been implemented.

## ARC protocol and comparisons

Before any measured run, create an immutable manifest containing the upstream
SEAL commit, data hashes, source task IDs, flattened subtask IDs, split rules,
checkpoint repository and revision, tokenizer revision, prompt format,
augmentation rules, and generation/LoRA configuration.

Use `meta-llama/Llama-3.2-1B-Instruct`, which is identified by the cited SEAL
paper and official commands. Sleep abbreviates the backbone name; record
this provenance instead of implying that Sleep supplies the complete model
identifier.

Upstream SEAL commit `6d9c9f9ee392c6cc618e771f399d436d190f6ca4` contains
12 training / 10 evaluation tasks. Its scripts flatten test subtasks,
truncate, skip some subtasks, and shuffle, so CLI counts are not unique task
counts. The README commands and saved final results do not imply the same
8-task evaluation set. This is a confirmed reproducibility gap.

For the initial reconstruction, use the 11 unique training tasks reached by
the upstream self-edit generation command, and the 8 unique evaluation tasks
in the saved upstream final results. Freeze their IDs before running our
models. This is our proposed choice, not proof of the authors' exact split.
Do not inherit later CLI truncation/shuffling implicitly.

Training IDs: `44f52bb0`, `4be741c5`, `5582e5ca`, `5614dbcf`, `5bd6f4ac`,
`6150a2bd`, `67e8384a`, `6d0aefbc`, `6e02f1e3`, `8be77c9e`, `8d5021e8`.

Evaluation IDs: `59341089`, `66e6c45b`, `68b67ca3`, `903d1b4a`, `b1fc8b8e`,
`bc4146bd`, `c7d4e6ad`, `ce8d95cc`.

Construct validation examples only from training tasks, with a recorded
split, before tuning. Final task outputs remain inaccessible to the
adaptation/reward pipeline. Use the public protocol reconstruction label on
all resulting comparisons until the authors' exact manifest can be
independently established. Do not filter tasks based on our final results.

Paper Appendix B.3 describes 60 training dreams, retaining 15, and five
independent adaptation attempts per evaluation task. Report the per-attempt
success metric as well as per-task solved counts, with explicit denominators.
For 8 tasks and 5 attempts, Table 4's 80% corresponds to 32/40 attempts; this
is not an 80% pass@5 result. Attempts within a task are correlated.

Run ICL, the precisely defined TTT variant, and SEAL on the same manifest as
Sleep. Also run consolidation-only and dreaming-related ablations. Preserve
the original baseline recipe where available. Report training tokens, update
steps, active capacity, GPU-hours, and any budget differences rather than
asserting compute matching without measurement.

Use three declared independent training seeds for the first report, increasing
replicates if variance makes the comparison inconclusive. Report each seed
and each task; summarize uncertainty using tasks/seeds as the sampling units.
Do not treat 40 attempts as 40 independent tasks. Tune only on training or
validation tasks. Lock configuration before final evaluation.

Table 4 reference values are ICL 0%, TTT 10%, SEAL 72.5%, and Sleep 80%.
The referenced SEAL work reports some differently named baseline values;
preserve both citations and explain any mismatch in definitions. Never
silently swap a stronger or weaker baseline under the same name.

## Unspecified details and reconstruction registry

Maintain a table with columns: setting, source location, published value,
chosen value, rationale, sensitivity run, and effect on comparability.

Known items include CMS integration and residual paths, routing/gating,
expert initialization and mixing, simultaneous-boundary ordering, memory
periods, divergence and coefficients, semantic reward checkpoint/prompt,
edit-distance threshold, LTI optimizer, dream-gradient scalarization,
dreaming update targets, model checkpoint identity, and task-manifest
reconstruction. Appendix B.4's learning rate / batch / LoRA values should not
be applied to every experiment without checking their scope against SEAL.

Choosing practical values is permitted. Omitting an essential mechanism and
calling the result the full paper method is not.

## Validation and artifacts

CPU tests must exercise nontrivial correctness properties: teacher immutability,
frozen/inactive parameter preservation, schedule counts, transaction rollback,
new-expert-only gradients, pruning with optimizer-state reset, deterministic
checkpoint continuation, candidate isolation, and strict split separation.
Small synthetic data are for these checks only.

GPU validation proceeds from a one-task smoke run to baseline verification to
the frozen multi-seed experiment. A completion claim requires actual finite
losses, parameter-delta checks, successful resume, and produced evaluation
artifacts. No fabricated metrics, placeholder success reports, or publishing
an unexecuted command as a completed experiment.

Each run saves the resolved config, source commit, dependency lock, container
identity, model/data revisions and hashes, seeds, device metadata, training
trace, checkpoint path, candidate rewards, per-attempt predictions, aggregate
metrics, runtime, and VESSL job identity. Serialize JSON/JSONL and generate
comparison tables from those records.

## VESSL and public release

Use a pinned CUDA container and a persistent volume for checkpoints and run
artifacts. Target VESSL Cloud batch jobs, which release resources when the
command exits; keep the Python entry points usable outside VESSL as well.
The older VESSL Run service uses a different CLI, so bind the launcher to the
actual product/account instead of mixing their command formats.

Start with the smallest resource spec that fits the measured model/snapshot
memory. The upstream SEAL README describes experiments on two A100/H100 GPUs;
this is a starting reference, not a measured requirement for our code. A
preflight run determines peak VRAM and expected duration before larger runs.
Account resource identifiers and an explicit per-job timeout are required
configuration. Paid jobs are submitted only after code and a concrete run
configuration are reviewable; selecting GPU access here does not constitute
an unlimited sweep budget.

Keep model and VESSL tokens in account secrets. Do not redistribute gated
model weights. Preserve upstream notices if reusing MIT-licensed SEAL code;
cite Sleep, Nested Learning, SEAL, and the datasets used. Publish measured
failures and protocol deviations along with successes.

The proposed initial publication contains code, configs, documentation, and
an honest experiment-status matrix. Release evidence is added as jobs finish.
The GitHub owner, available repository name, actual VESSL resource identifiers,
and existing model-access grants are operational inputs to resolve before
their dependent external actions.

## Builder decisions and delivery order

The builder may choose package layout, typed interfaces, logging library,
dependency tooling, and internal test fixtures. Proposed own-code license is
MIT; the repository name is a proposal, not an already created remote.

Delivery order:

1. Freeze the architecture/protocol reconstruction registry and ARC manifest.
2. Implement and verify CMS, the consolidation transaction, and KS/LTI.
3. Implement and verify dreaming and baseline-compatible ARC adapters.
4. Package, document, run CPU CI, and prepare the VESSL experiment commands.
5. Run the GPU smoke/baseline checks and the frozen multi-seed comparisons;
   publish their artifacts and limitations.
6. Expand the result matrix to the remaining paper experiments.

Method implementation, successful experiment execution, and numerical
reproduction are three separate completion criteria. If the paper's claimed
improvement is not recovered, report that outcome and investigate the
reconstruction choices without changing the locked test protocol.

## Primary sources

- Sleep paper v2, especially Sections 2.2–3.4, Table 4 and Appendix B:
  <https://arxiv.org/html/2606.03979v2>
- Google Research overview:
  <https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/>
- Official SEAL code and baseline documentation:
  <https://github.com/Continual-Intelligence/SEAL>
- SEAL paper:
  <https://arxiv.org/abs/2506.10943>
- Pinned SEAL few-shot recipe:
  <https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/README.md>
- Pinned upstream evaluation results used to choose the reconstructed split:
  <https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/final_results.json>
- Nested Learning / Hope:
  <https://arxiv.org/abs/2512.24695>
- Official VESSL Cloud job description and CLI examples:
  <https://vessl.ai/en/blog/vessl-cloud-batch-jobs>
- VESSL AI documentation index for the separate Run service:
  <https://docs.vessl.ai/llms.txt>
