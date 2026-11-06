# Reconstruction registry

This registry separates the published method from the choices made in this
independent implementation. It accompanies the [ARC protocol](ARC_PROTOCOL.md),
which specifies the fixed task IDs, data provenance, JSON directive schema, and
attempt metric. The implementation is a bounded post-training reconstruction;
it does not recreate the authors' architectural pretraining or establish their
exact unpublished recipe.

Sources: [Sleep v2](https://arxiv.org/html/2606.03979v2), particularly Sections
2.2–3.4, Table 4, and Appendix B.3;
[Nested Learning](https://arxiv.org/abs/2512.24695);
[SEAL](https://arxiv.org/abs/2506.10943), with the
[pinned upstream implementation](https://github.com/Continual-Intelligence/SEAL/tree/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot).

## Architecture and transactions

| Setting | Source / recovered published detail | Chosen implementation and rationale | Sensitivity / comparability |
| --- | --- | --- | --- |
| Pretrained integration | Sleep §2.2 and Nested Learning motivate a memory chain; the exact post-training Llama embedding was not recovered | Keep the original Llama FFN as stage zero; append zero-output nonlinear low-rank residual stages inside the unchanged outer residual. This preserves the initial function while allowing later stages to learn | CPU identity and gradient checks exist. Pretrained-logit/greedy-output equivalence and transfer quality still need GPU validation. This is our architecture reconstruction |
| Base and expert branches | Scheduled memory with expert consolidation; exact builder graph unspecified | An always-on stage base plus a finite preallocated bank of rank-limited nonlinear experts; slower stage bases use `up(silu(down(x)))` | Default expert rank 64 and bank capacity eight are builder choices; capacity exhaustion raises. Parameter counts must accompany results |
| Normal routing | Exact routing/gating recipe not recovered | Frozen, checkpointed random unit projections score each token independently; stable top-k selection among active trained experts, with additive coefficient 1 | Fixed projections are not learned routing. Token-local scoring prevents future tokens, padding, and co-batched examples from changing a token's route. Router/expert-scale sensitivity is pending |
| Dream routing | Sleep dreaming adds a random expert | Add one eligible expert outside each token's normal selected set. A sampled remaining-rank offset is shared across tokens. Masks are not persistently changed | No extra expert is added when no eligible alternative exists. A measured full-method run must establish and record eligibility |
| Update schedule | Explicit memory periods; exact ARC schedule not recovered | Periods count completed wake minibatches; initial two-level config uses `[1,4]` and four wake steps, covering the terminal boundary. Simultaneous boundaries process fast to slow with fresh snapshots | Training-only period and episode-length sensitivity is pending |
| Base optimizer | Proposed sender update must precede fitting receiver memory | Mean accumulated wake gradients define a prospective SGD base update. A private Adam optimizer fits only each newly allocated receiver expert | Memory base updates currently require float32: validated master-weight support for low precision is absent. Optimizer/lr/batch sensitivity is pending |
| Transaction boundary | Compute–consolidate–update | Immutable pre-update teacher and isolated prospective student. Commit sender update, sender-bank reset, and fitted receiver experts together after finite-state checks | CPU tests cover rollback, frozen-state preservation, teacher immutability, and optimizer reset. Full-model copies impose a substantial unmeasured GPU memory cost |
| Receiver fitting graph | Preserve transferred knowledge | The newly allocated receiver is made eligible under the same token-local routing graph used after commit; other parameters remain frozen | CPU tests check graph consistency and receiver-only changes. Successful finite fitting is not evidence that all teacher behavior was preserved |
| Terminal level | No adjacent receiver exists | Commit its pending base update while retaining its expert bank | Explicit terminal behavior is a builder choice; longer episodes and additional levels require separate runs |
| Adapter implementation | LoRA-style inner adaptation and outer learning | Custom dual low-rank branches on attention `q_proj` and `v_proj`: separate persistent `policy` and episode-local `inner` branches; base projections stay frozen during adapter SFT | This is original PyTorch code rather than a PEFT adapter recipe. Target modules, scaling, ranks, and interaction with CMS affect comparability |

The period hierarchy must strictly increase and each period must divide the
next. Expert banks do not grow during forward calls. Expert resets clear
applicable optimizer moments. Allocation, counters, gradient accumulation, and
routing buffers are checkpoint state. Four-level chains can be constructed,
but the initial experiment config uses two levels; four-level ARC results have
not been produced.

Audit and rollback state copies live in CPU memory. The teacher and prospective
student execution models stay on the configured device. Accumulated wake
gradients remain in checkpointed buffers; redundant parameter `.grad` tensors
are released before snapshots. This trades host RAM and transfer time for GPU
memory without changing the optimization or commit rules.
Private teacher/student accumulation buffers also reside on CPU after the
prospective update: forward calls never use these buffers, and commit copies
their exact values back to the live device. The declared `attention_backend`
defaults to PyTorch SDPA, with `eager` available for comparison. See the
[Transformers backend interface](https://huggingface.co/docs/transformers/attention_interface).

## Knowledge Seeding and Learning to Imitate

| Setting | Source / recovered published detail | Chosen implementation and rationale | Sensitivity / comparability |
| --- | --- | --- | --- |
| Distillation direction | Sleep KS/LTI distillation | Completion-masked causal `KL(teacher || student)`, token mean, temperature-scaled by `T²` | Direction, reduction, and temperature are explicit; alternative choices are untested |
| Trajectory mixture | Teacher/off-policy and student/on-policy rollouts with prefix conditioning | Weight teacher-generated trajectories by `1-on_policy_fraction` and student continuations by `on_policy_fraction`; initial fraction 0.5 | Exact sampling mixture was not recovered. Training-only sensitivity is pending |
| Student conditioning | Random prefix conditioning in the reconstructed LTI path | Random token prefix of the teacher response conditions a detached student rollout. Mask original prompt and supplied prefix from its loss | Tokenization and boundary behavior are tested on small inputs; quantitative effect is unknown |
| Semantic reward | Frozen semantic scoring is part of the objective | The frozen transactional teacher answers a fixed YES/NO equivalence prompt about a reference and prediction; positive verdict maps to 1, other verdicts to 0 | Reusing the teacher as judge is our reconstruction, not a recovered separate reward checkpoint. ARC judgment accuracy and prompt sensitivity are unmeasured |
| Edit component | Edit-distance reward accompanies semantics | Normalized character-level Levenshtein similarity, `1-distance/max(lengths,1)` | This uses continuous similarity rather than an asserted author threshold. Threshold/formulation sensitivity is pending |
| Composite reward | Both semantic and edit components are needed | Initial reward is 0.5 semantic score plus 0.5 edit similarity | Exact string reward alone is not the full implementation. Judge quality and coefficient sensitivity remain open |
| LTI estimator | Policy-gradient contribution alongside distillation | `alpha*KL - (1-alpha)*mean(reward*sequence_log_probability)`, initial alpha 0.5. Sum log probabilities over valid completion tokens; sampling and reward are detached | Sequence length affects the estimator; no baseline or length-normalized replacement is silently assumed. Sensitivity is pending |

Prompts and padding are masked from completion losses. Overlength examples and
prompt-plus-generation budgets fail explicitly rather than silently truncating
ARC information. The same configured temperature currently controls sampled
generation and distillation. That coupling is a declared implementation choice
that should be tested before a locked numerical comparison.
Generation preserves the pinned checkpoint's complete EOS/turn-end token
configuration; it does not replace it with the tokenizer's single EOS ID.

## Dreaming and episodic learning

| Setting | Source / recovered published detail | Chosen implementation and rationale | Sensitivity / comparability |
| --- | --- | --- | --- |
| Dream representation | Sleep Appendix B.3 refers to ARC/SEAL adaptation | Bounded JSON augmentation/adaptation directives; expanded visible demonstration pairs supply inner-training data | Schema and expansion differ from official SEAL. The `self_edit` mode is a separate reconstructed baseline, not an official SEAL implementation |
| Candidate generation | Appendix B.3 describes 60 training dreams, retaining 15 | Generate up to 60 responses; reject malformed directives; choose up to 12 highest-importance valid candidates plus three distinct random candidates | Fewer valid candidates produce fewer selected candidates. Count invalid, valid, selected, and accepted responses; do not replace invalid output with a successful fixture |
| Gradient scalar | Paper uses SFT gradient importance without a recovered scalar ranking | L2 norm of SFT gradients over the inner LoRA parameters on expanded visible pairs | The parameter set and norm are builder choices. Norm/rank sensitivity remains unrun |
| Candidate isolation | Compare adaptation benefits | Each candidate starts from identical episode model, optimizer, and RNG state. Score before and after adaptation on the permitted held-out demonstration | CPU isolation tests exist. The reward is exact whole-grid correctness on the held-out demonstration, not final test performance |
| Inner update | Candidate-specific LoRA adaptation | JSON selects 1–5 epochs and an lr multiplier 0.1–10; train only the `inner` branch on transformed visible examples, with microbatch size one | The supplied config sets the base inner lr and rank. Budget differs across directives and must be reported |
| Positive selection | ReST-EM learns successful self-modifications | Keep strictly positive post-minus-pre reward; supervised outer update targets accepted canonical directive strings | An empty accepted set is a logged no-op. A completed episode schedule does not prove the policy learned or any candidate succeeded |
| Outer update targets | Exact task-specific target parameter set not recovered | Update only the persistent `policy` LoRA branch; CMS and candidate-specific inner changes remain episode-local | This is an explicit reconstruction choice; no cross-episode persistent CMS benchmark is claimed |
| Training reward split | Final targets must remain isolated | Hold out the last visible demonstration of each training task before wake, generation, importance scoring, and adaptation; use its output only for candidate reward | Requires at least two demonstrations. Split sensitivity is training-only; final task outputs never tune it |
| Evaluation selection | Five independent adaptation attempts in Appendix B.3 | Start each task/attempt from the frozen policy and fresh episode state, sample one directive, adapt on visible pairs, then greedily predict all test inputs | There is no selection based on final outputs and no evaluation ReST-EM update. Leave-one-demonstration-out final selection would be a separately named variant |

The initial config uses policy rank 16, inner rank 128, LoRA alpha 16, one outer
SFT epoch, and one pass through the training task schedule. These are explicit
configuration values, not claims that Appendix B.4 supplies the same recipe for
ARC. Record actual successful outer updates, training tokens, adapter steps,
active capacity, and runtime; configured budgets alone are insufficient.

## Baselines and evidence gaps

`icl` uses visible demonstrations with greedy final decoding and no adaptation.
`ttt` applies one inner-adapter SFT pass using the identity directive. With
identical task state, greedy predictions can repeat across nominal attempts;
changing a seed does not make task outcomes statistically independent.
`consolidation` performs wake/consolidation without dreams. `self_edit` learns
and applies our JSON directives without memory stages. All use the declared
public task split; none has a completed pretrained ARC result yet.

The official SEAL baseline must be executed through a separately documented
upstream recipe, preserving its source revision and recording necessary
protocol changes. Until then, our baseline matrix must show it as external and
unimplemented. Published Table 4 values are references only.

Before a numerical reproduction claim, complete pretrained conversion probes,
GPU memory and finite-loss preflight, real-checkpoint resume checks, baseline
verification, training-only sensitivity decisions, and the locked three-seed
comparison. Publish failures and deviations with raw artifacts. See
[experiment status](EXPERIMENT_STATUS.md) for the current boundary of evidence.
