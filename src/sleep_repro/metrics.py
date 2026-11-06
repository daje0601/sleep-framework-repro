"""Strict per-attempt ARC metrics, kept outside the adaptation data boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .arc import Grid, Task, validate_grid


@dataclass(frozen=True)
class Attempt:
    task_id: str
    attempt_index: int
    predictions: tuple[Grid | None, ...]

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("attempt requires a task ID")
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise ValueError("attempt index must be a nonnegative integer")
        object.__setattr__(
            self,
            "predictions",
            tuple(
                None if grid is None else validate_grid(grid)
                for grid in self.predictions
            ),
        )


def evaluate_attempts(
    tasks: Sequence[Task],
    solutions: Mapping[str, Sequence[Grid]],
    attempts: Iterable[Attempt],
    expected_attempts: int = 5,
) -> dict:
    """Require every declared attempt and all test outputs before counting success.

    None predictions represent invalid generations and remain in the denominator.
    Task solve rate is a separate descriptive any-attempt statistic. Attempts
    within a task are correlated; this function computes no binomial uncertainty.
    """
    if type(expected_attempts) is not int or expected_attempts < 1:
        raise ValueError("expected_attempts must be a positive integer")
    tasks = tuple(tasks)
    task_map = {task.task_id: task for task in tasks}
    if not tasks or len(task_map) != len(tasks):
        raise ValueError("tasks must be nonempty with no duplicate task IDs")
    if set(solutions) != set(task_map):
        raise ValueError("solutions must cover exactly the declared task IDs")
    targets = {}
    for task_id, task in task_map.items():
        targets[task_id] = tuple(validate_grid(grid) for grid in solutions[task_id])
        if len(targets[task_id]) != len(task.test_inputs):
            raise ValueError(f"solution/test count mismatch: {task_id}")
    by_task = {task_id: {} for task_id in task_map}
    for attempt in attempts:
        if not isinstance(attempt, Attempt) or attempt.task_id not in task_map:
            raise ValueError("attempt references an unknown task")
        if attempt.attempt_index >= expected_attempts:
            raise ValueError("attempt index exceeds declared attempt denominator")
        if attempt.attempt_index in by_task[attempt.task_id]:
            raise ValueError("duplicate task/attempt index")
        if len(attempt.predictions) != len(targets[attempt.task_id]):
            raise ValueError(
                "every attempt must include all test outputs (None if invalid)"
            )
        by_task[attempt.task_id][attempt.attempt_index] = (
            attempt.predictions == targets[attempt.task_id]
        )
    expected_indices = set(range(expected_attempts))
    per_task = []
    for task in tasks:
        outcomes = by_task[task.task_id]
        if set(outcomes) != expected_indices:
            raise ValueError(f"missing attempts for {task.task_id}")
        successes = sum(outcomes.values())
        per_task.append(
            {
                "task_id": task.task_id,
                "successful_attempts": successes,
                "total_attempts": expected_attempts,
                "solved": successes > 0,
                "attempt_outcomes": [outcomes[i] for i in range(expected_attempts)],
            }
        )
    successful_attempts = sum(item["successful_attempts"] for item in per_task)
    total_attempts = len(tasks) * expected_attempts
    tasks_solved = sum(item["solved"] for item in per_task)
    return {
        "successful_attempts": successful_attempts,
        "total_attempts": total_attempts,
        "attempt_success_rate": successful_attempts / total_attempts,
        "tasks_solved": tasks_solved,
        "total_tasks": len(tasks),
        "task_solve_rate": tasks_solved / len(tasks),
        "attempts_per_task": expected_attempts,
        "per_task": per_task,
        "uncertainty_note": "Attempts within a task are correlated; sample tasks/seeds for uncertainty.",
    }
