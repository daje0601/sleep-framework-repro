"""Safe ARC data boundary and a reconstructed SEAL-like directive protocol.

This is an original implementation. The bounded JSON schema is our reconstruction,
not SEAL's unpublished or executable self-edit interface. Model-facing ``Task``
objects never carry final test outputs; those live in ``ArcDataset.solutions``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

Grid = tuple[tuple[int, ...], ...]
SEAL_COMMIT = "6d9c9f9ee392c6cc618e771f399d436d190f6ca4"
TRAIN_IDS = (
    "44f52bb0",
    "4be741c5",
    "5582e5ca",
    "5614dbcf",
    "5bd6f4ac",
    "6150a2bd",
    "67e8384a",
    "6d0aefbc",
    "6e02f1e3",
    "8be77c9e",
    "8d5021e8",
)
EVALUATION_IDS = (
    "59341089",
    "66e6c45b",
    "68b67ca3",
    "903d1b4a",
    "b1fc8b8e",
    "bc4146bd",
    "c7d4e6ad",
    "ce8d95cc",
)
PROMPT_VERSION = "arc-json-v1"
DIRECTIVE_SCHEMA_VERSION = "seal-like-bounded-json-v1"


def validate_grid(value: object) -> Grid:
    """Copy a rectangular 1..30 by 1..30 grid of integer colors into tuples."""
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 30:
        raise ValueError("grid must have 1..30 rows")
    rows = []
    width = None
    for row in value:
        if not isinstance(row, (tuple, list)) or not 1 <= len(row) <= 30:
            raise ValueError("grid rows must have 1..30 cells")
        if width is not None and len(row) != width:
            raise ValueError("grid must be rectangular")
        if any(type(cell) is not int or not 0 <= cell <= 9 for cell in row):
            raise ValueError("grid cells must be integers 0..9 (not booleans)")
        width = len(row)
        rows.append(tuple(row))
    return tuple(rows)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _json_loads(text: str):
    if not isinstance(text, str):
        raise ValueError("expected JSON text")
    try:
        return json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid JSON") from error


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid

    def __post_init__(self):
        object.__setattr__(self, "input", validate_grid(self.input))
        object.__setattr__(self, "output", validate_grid(self.output))


@dataclass(frozen=True)
class Task:
    """The complete model-facing view: demos and final test inputs only."""

    task_id: str
    demonstrations: tuple[Pair, ...]
    test_inputs: tuple[Grid, ...]

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty string")
        demonstrations = tuple(self.demonstrations)
        if not demonstrations or any(
            not isinstance(pair, Pair) for pair in demonstrations
        ):
            raise ValueError("task requires demonstration pairs")
        tests = tuple(validate_grid(grid) for grid in self.test_inputs)
        if not tests:
            raise ValueError("task requires test inputs")
        object.__setattr__(self, "demonstrations", demonstrations)
        object.__setattr__(self, "test_inputs", tests)


@dataclass(frozen=True)
class ArcDataset:
    tasks: tuple[Task, ...]
    solutions: Mapping[str, tuple[Grid, ...]]
    file_hashes: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(
            self,
            "solutions",
            MappingProxyType(
                {
                    key: tuple(validate_grid(grid) for grid in grids)
                    for key, grids in self.solutions.items()
                }
            ),
        )
        object.__setattr__(
            self, "file_hashes", MappingProxyType(dict(self.file_hashes))
        )


def load_arc(
    challenges_path: str | Path, solutions_path: str | Path, task_ids: Sequence[str]
) -> ArcDataset:
    """Load explicit ordered IDs from ARC challenge/solution JSON files.

    Embedded challenge test outputs, if present in a training-format file, are
    deliberately discarded. Only the separate solution file supplies scoring data.
    """
    ids = tuple(task_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("task IDs must be nonempty, with no duplicate IDs")
    challenge_bytes = Path(challenges_path).read_bytes()
    solution_bytes = Path(solutions_path).read_bytes()
    challenges = _json_loads(challenge_bytes.decode("utf-8"))
    solutions = _json_loads(solution_bytes.decode("utf-8"))
    if not isinstance(challenges, dict) or not isinstance(solutions, dict):
        raise ValueError("ARC files must contain task dictionaries")
    tasks = []
    selected_solutions = {}
    for task_id in ids:
        if task_id not in challenges or task_id not in solutions:
            raise ValueError(f"missing task or solutions: {task_id}")
        record = challenges[task_id]
        try:
            pairs = tuple(
                Pair(pair["input"], pair["output"]) for pair in record["train"]
            )
            tests = tuple(validate_grid(pair["input"]) for pair in record["test"])
            targets = tuple(validate_grid(grid) for grid in solutions[task_id])
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed ARC record: {task_id}") from error
        if len(targets) != len(tests):
            raise ValueError(f"solution/test count mismatch: {task_id}")
        tasks.append(Task(task_id, pairs, tests))
        selected_solutions[task_id] = targets
    return ArcDataset(
        tuple(tasks),
        selected_solutions,
        {
            "challenges_sha256": hashlib.sha256(challenge_bytes).hexdigest(),
            "solutions_sha256": hashlib.sha256(solution_bytes).hexdigest(),
        },
    )


def parse_grid_response(response: str) -> Grid:
    """Accept pure grid JSON or one fenced JSON block, never executable text."""
    if not isinstance(response, str):
        raise ValueError("expected a string response")
    text = response.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("expected a single JSON fence")
        text = match.group(1)
    return validate_grid(_json_loads(text))


@dataclass(frozen=True)
class DreamDirective:
    """Bounded augmentation/adaptation settings; no executable code fields."""

    rotations: tuple[int, ...] = (0,)
    reflect: bool = False
    color_permutation: tuple[int, ...] | None = None
    epochs: int = 1
    lr_scale: float = 1.0

    def __post_init__(self):
        if not isinstance(self.rotations, (list, tuple)) or not self.rotations:
            raise ValueError("rotations must be a nonempty list")
        if any(
            type(rotation) is not int or rotation not in range(4)
            for rotation in self.rotations
        ):
            raise ValueError("rotations must be quarter turns 0..3")
        if len(set(self.rotations)) != len(self.rotations):
            raise ValueError("duplicate rotations")
        if type(self.reflect) is not bool:
            raise ValueError("reflect must be boolean")
        if self.color_permutation is not None:
            permutation = self.color_permutation
            if (
                not isinstance(permutation, (list, tuple))
                or len(permutation) != 10
                or any(type(color) is not int for color in permutation)
                or set(permutation) != set(range(10))
            ):
                raise ValueError("color_permutation must be a bijection on 0..9")
            object.__setattr__(self, "color_permutation", tuple(permutation))
        if type(self.epochs) is not int or not 1 <= self.epochs <= 5:
            raise ValueError("epochs must be an integer 1..5")
        if (
            type(self.lr_scale) not in (int, float)
            or not math.isfinite(self.lr_scale)
            or not 0.1 <= self.lr_scale <= 10
        ):
            raise ValueError("lr_scale must be finite in 0.1..10")
        object.__setattr__(self, "rotations", tuple(self.rotations))
        object.__setattr__(self, "lr_scale", float(self.lr_scale))

    def to_json(self) -> str:
        return canonical_json(
            {
                "rotations": self.rotations,
                "reflect": self.reflect,
                "color_permutation": self.color_permutation,
                "epochs": self.epochs,
                "lr_scale": self.lr_scale,
            }
        )


def parse_directive(response: str) -> DreamDirective:
    """Parse JSON only; reject unknown fields and duplicate keys."""
    value = _json_loads(response)
    fields = {"rotations", "reflect", "color_permutation", "epochs", "lr_scale"}
    if not isinstance(value, dict) or "rotations" not in value or set(value) - fields:
        raise ValueError("directive requires rotations and only documented JSON fields")
    return DreamDirective(**value)


def _rotate(grid: Grid, turns: int) -> Grid:
    for _ in range(turns):
        grid = tuple(tuple(row) for row in zip(*grid[::-1]))
    return grid


def transform_grid(
    grid: Grid,
    rotation: int = 0,
    reflect: bool = False,
    color_permutation: tuple[int, ...] | None = None,
) -> Grid:
    """Reflect horizontally, rotate clockwise, then rename colors."""
    directive = DreamDirective((rotation,), reflect, color_permutation)
    result = validate_grid(grid)
    if reflect:
        result = tuple(row[::-1] for row in result)
    result = _rotate(result, rotation)
    if directive.color_permutation is not None:
        result = tuple(
            tuple(directive.color_permutation[cell] for cell in row) for row in result
        )
    return result


def inverse_transform_grid(
    grid: Grid,
    rotation: int = 0,
    reflect: bool = False,
    color_permutation: tuple[int, ...] | None = None,
) -> Grid:
    directive = DreamDirective((rotation,), reflect, color_permutation)
    result = validate_grid(grid)
    if directive.color_permutation is not None:
        inverse = {
            color: index for index, color in enumerate(directive.color_permutation)
        }
        result = tuple(tuple(inverse[cell] for cell in row) for row in result)
    result = _rotate(result, (-rotation) % 4)
    if reflect:
        result = tuple(row[::-1] for row in result)
    return result


def expand_directive(task: Task, directive: DreamDirective) -> tuple[Pair, ...]:
    """Transform visible pairs only; epochs affect optimizer passes, not copies."""
    if not isinstance(task, Task) or not isinstance(directive, DreamDirective):
        raise ValueError("expected a public Task and DreamDirective")
    return tuple(
        Pair(
            transform_grid(
                pair.input, rotation, directive.reflect, directive.color_permutation
            ),
            transform_grid(
                pair.output, rotation, directive.reflect, directive.color_permutation
            ),
        )
        for rotation in directive.rotations
        for pair in task.demonstrations
    )


def format_prompt(task: Task, test_index: int = 0) -> str:
    if type(test_index) is not int or not 0 <= test_index < len(task.test_inputs):
        raise ValueError("test index out of range")
    examples = "\n".join(
        f"Input: {canonical_json(pair.input)}\nOutput: {canonical_json(pair.output)}"
        for pair in task.demonstrations
    )
    return (
        "Infer the grid transformation from the examples. Return only the output grid as JSON.\n"
        f"{examples}\nInput: {canonical_json(task.test_inputs[test_index])}\nOutput:"
    )


def format_supervised_example(pair: Pair) -> tuple[str, str]:
    return (
        f"Transform this ARC input grid. Return only the output grid as JSON.\nInput: {canonical_json(pair.input)}\nOutput:",
        canonical_json(pair.output),
    )


def holdout_demonstration(task: Task, index: int = -1) -> tuple[Task, Pair]:
    """Remove a visible pair before adaptation and use only its input as a query."""
    if len(task.demonstrations) < 2:
        raise ValueError("holdout requires at least two demonstrations")
    if type(index) is not int or not -len(task.demonstrations) <= index < len(
        task.demonstrations
    ):
        raise ValueError("holdout index out of range")
    index %= len(task.demonstrations)
    heldout = task.demonstrations[index]
    visible = task.demonstrations[:index] + task.demonstrations[index + 1 :]
    return Task(task.task_id, visible, (heldout.input,)), heldout


def reconstruction_manifest(hashes: Mapping[str, str] | None = None) -> dict:
    """Protocol fragment; model/run revisions must be added before measured runs."""
    return {
        "protocol_label": "public protocol reconstruction",
        "seal_commit": SEAL_COMMIT,
        "train_ids": list(TRAIN_IDS),
        "evaluation_ids": list(EVALUATION_IDS),
        "data_hashes": dict(hashes or {}),
        "prompt_version": PROMPT_VERSION,
        "directive_schema": DIRECTIVE_SCHEMA_VERSION,
        "attempts_per_task": 5,
        "flatten_rule": "retain every test subtask of each explicit source task",
        "validation_rule": "held-out visible demonstration from training tasks only",
        "augmentation_rule": "horizontal reflection; clockwise quarter turns; color bijection",
        "directive_reconstruction": True,
    }
