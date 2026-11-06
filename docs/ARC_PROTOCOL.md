# ARC public protocol reconstruction

This project is an unofficial independent reconstruction of the ARC experiment
in *Language Models Need Sleep*, Table 4 and Appendix B.3. Data preparation and
CPU correctness checks do not establish a quantitative reproduction. The
authors' exact task manifest and every baseline recipe have not been recovered.

## Prepare and verify the data

From the repository root, run:

```sh
python scripts/prepare_data.py
```

The script uses Python's standard library. `--output /path/to/arc` selects a
different destination. It downloads four files from the fixed SEAL commit
`6d9c9f9ee392c6cc618e771f399d436d190f6ca4`, checks their SHA256 hashes against
[`configs/arc_data_manifest.json`](../configs/arc_data_manifest.json), validates
the selected ARC records, and writes:

```text
data/arc/train_challenges.json
data/arc/train_solutions.json
data/arc/eval_challenges.json
data/arc/eval_solutions.json
data/arc/manifest.json
```

Downloads are assembled in a temporary directory beside the destination. The
complete directory is renamed into place after every hash and data check
passes. Interrupted or corrupt downloads leave no partial destination. When
the destination already exists, the script verifies its files and manifest
without network access. Modified files, a changed manifest, or an incomplete
dataset cause an error; the script does not silently replace them. Preserve a
modified dataset separately and choose a fresh destination when obtaining a
clean copy.

The files retain all upstream records; selection happens through explicit IDs,
not destructive filtering. The downloaded source contains 12 training and 10
evaluation tasks. The generated manifest records source URLs, byte hashes,
selected source IDs, every retained test-subtask ID, split rationale, and prompt
and directive versions. Model/tokenizer revisions, hyperparameters, generation
settings, and seeds must also be frozen in the experiment configuration before
measured runs.

## Fixed task split and the upstream discrepancy

The chosen training IDs are the 11 unique tasks reached by the pinned upstream
self-edit generation command:

```text
44f52bb0 4be741c5 5582e5ca 5614dbcf 5bd6f4ac 6150a2bd
67e8384a 6d0aefbc 6e02f1e3 8be77c9e 8d5021e8
```

The evaluation IDs are the eight unique tasks represented in the saved upstream
final results:

```text
59341089 66e6c45b 68b67ca3 903d1b4a b1fc8b8e bc4146bd
c7d4e6ad ce8d95cc
```

The upstream scripts flatten test subtasks and use truncation, skips, and
shuffling. Consequently a CLI count is not necessarily a count of unique source
tasks, and the README evaluation command does not establish the same eight-task
set as the saved final results. Our split is a declared reconstruction choice;
it is not evidence of the Sleep authors' exact split. See the pinned
[SEAL few-shot README](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/README.md)
and [saved final results](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/final_results.json).

We retain every test pair in each selected source task. The selected training
tasks have 12 test subtasks because `44f52bb0` has two. The selected evaluation
tasks have nine because `b1fc8b8e` has two. Subtask IDs use `task_id:zero_based_index`
and are recorded in the generated manifest. Source tasks remain the unit of
attempt success. Do not change the task list or discard difficult subtasks based
on final results.

## Model-facing data and reward separation

`Task` is an immutable public view containing `task_id`, visible demonstration
input/output pairs, and test inputs. Final targets live separately in
`ArcDataset.solutions`; even test outputs embedded in a challenge file are
discarded when constructing a `Task`. Only scoring code receives the separate
solution mapping. Adaptation, wake losses, dream generation, gradient selection,
and candidate reward must consume public task views rather than the dataset
object or solution files.

Training reward examples come from visible demonstrations of training tasks.
`holdout_demonstration` removes one complete demonstration before constructing
the adaptation view and exposes its input as the reward query. It requires at
least two demonstrations. Record the held-out index before tuning. The held-out
output is available only to the reward scorer. Final evaluation targets cannot
choose dreams, hyperparameters, or ReST-EM updates. A final-evaluation selector
using leave-one-demonstration-out reward is a separately reported variant.

## Bounded JSON dream schema

Our schema, `seal-like-bounded-json-v1`, is an original task-specific
reconstruction inspired by SEAL's augmentation/adaptation idea. It differs from
the official SEAL interface. It accepts JSON data and does not execute generated
Python, shell commands, arbitrary training code, or unknown fields. This
implementation alone must not be described as a faithful run of the official
SEAL baseline.

Example:

```json
{
  "rotations": [0, 1, 2, 3],
  "reflect": false,
  "color_permutation": [0, 2, 1, 3, 4, 5, 6, 7, 8, 9],
  "epochs": 2,
  "lr_scale": 1.0
}
```

| Field | Constraint | Default |
| --- | --- | --- |
| `rotations` | Required, nonempty list of distinct integer clockwise quarter turns in `0..3` | No parser default |
| `reflect` | Boolean; horizontal reflection before rotation | `false` |
| `color_permutation` | `null` or a bijection of integer colors `0..9`, mapping each old color to its listed value | `null` |
| `epochs` | Integer from 1 through 5 | `1` |
| `lr_scale` | Finite number from 0.1 through 10 | `1.0` |

Booleans are not integers in grid, rotation, epoch, or permutation validation.
Duplicate JSON keys, unknown fields, invalid bounds, and nonfinite constants
are rejected. Dream directives must be JSON objects without Markdown fencing.

Expansion transforms only the visible demonstration pairs. For each declared
rotation, in declared order, it applies horizontal reflection if requested,
clockwise rotation, and then color renaming to both input and output. This is
deterministic and reversible on rectangular grids. An `epochs` value controls
optimizer passes; expansion does not duplicate the dataset by epoch count.
Record the canonical directive and expanded-example hashes with each candidate.
Gradient importance is measured on expanded supervised examples, while the
outer ReST-EM target is the successful directive's token sequence. This choice
is part of our reconstruction.

ARC grid responses must be a rectangular JSON array, either alone or in one
JSON code fence, with 1 through 30 rows and columns and integer cells from 0
through 9. Extra explanation, ragged grids, invalid colors, and executable
syntax are invalid predictions. The prompt format version is `arc-json-v1`.

## Attempt metric and baseline interpretation

Final evaluation uses five declared independent adaptation attempts for each
of eight source tasks. Each task and attempt starts from the same frozen trained
policy checkpoint and fresh episodic state; only declared sampling seeds vary.
An attempt succeeds only if **all** of that task's test output grids match
exactly. For the two-test task `b1fc8b8e`, getting one grid right is insufficient.
Invalid predictions remain failed attempts in the denominator. Missing or
duplicate attempt indices, missing test predictions, unknown tasks, and an
incorrect declared attempt count are rejected by the metric aggregator.

Report successful attempts over 40 and, separately, source tasks solved by at
least one attempt over eight. The paper's 80% Table 4 value corresponds to
32/40 attempts, not an 80% pass@5 result. Attempts from one task are correlated;
uncertainty estimates should use tasks and independent training seeds as their
sampling units. Store per-task and per-attempt outcomes along with predictions.

The Table 4 reference values are ICL 0%, TTT 10%, SEAL 72.5%, and Sleep 80%.
They are published references, not measured results from this repository. A
comparison must identify the exact implemented ICL/TTT/SEAL recipes and any
changes in task grouping, compute, adaptation budget, or selection. The bounded
JSON directive path must not be silently labeled the official SEAL baseline.
See the [Sleep paper v2](https://arxiv.org/html/2606.03979v2) and
[SEAL paper](https://arxiv.org/abs/2506.10943) for the distinct published methods.
