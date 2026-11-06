import json

import pytest

from sleep_repro.arc import Pair, Task
from sleep_repro.metrics import Attempt, evaluate_attempts


def test_attempt_metric_is_32_over_40_and_separate_from_task_solve_rate():
    tasks = [Task(str(i), (Pair(((0,),), ((1,),)),), (((2,),),)) for i in range(8)]
    solutions = {task.task_id: (((3,),),) for task in tasks}
    attempts = [
        Attempt(task.task_id, j, ((((3 if j < 4 else 0),),),))
        for task in tasks
        for j in range(5)
    ]
    stats = evaluate_attempts(tasks, solutions, attempts)
    assert stats["successful_attempts"] == 32
    assert stats["total_attempts"] == 40
    assert stats["attempt_success_rate"] == 0.8
    assert stats["tasks_solved"] == 8
    assert stats["total_tasks"] == 8
    assert stats["task_solve_rate"] == 1.0
    assert stats["attempts_per_task"] == 5
    assert all(item["successful_attempts"] == 4 for item in stats["per_task"])
    json.dumps(stats, allow_nan=False)


def test_multi_test_task_requires_all_outputs_per_attempt():
    task = Task("multi", (Pair(((0,),), ((1,),)),), (((2,),), ((3,),)))
    attempts = [
        Attempt("multi", i, (((4,),), ((5 if i == 0 else 0,),))) for i in range(5)
    ]
    stats = evaluate_attempts([task], {"multi": (((4,),), ((5,),))}, attempts)
    assert stats["successful_attempts"] == 1
    assert stats["total_attempts"] == 5
    assert stats["tasks_solved"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "unknown",
        "index",
        "output_count",
        "solution_missing",
        "task_duplicate",
    ],
)
def test_metrics_reject_invalid_or_incomplete_attempt_denominators(mutation):
    task = Task("a", (Pair(((0,),), ((1,),)),), (((2,),),))
    tasks = [task]
    solutions = {"a": (((3,),),)}
    attempts = [Attempt("a", i, (((3,),),)) for i in range(5)]
    if mutation == "duplicate":
        attempts[-1] = attempts[0]
    elif mutation == "missing":
        attempts.pop()
    elif mutation == "unknown":
        attempts[-1] = Attempt("other", 4, (((3,),),))
    elif mutation == "index":
        attempts[-1] = Attempt("a", 5, (((3,),),))
    elif mutation == "output_count":
        attempts[-1] = Attempt("a", 4, ())
    elif mutation == "solution_missing":
        solutions = {}
    elif mutation == "task_duplicate":
        tasks.append(task)
    with pytest.raises(ValueError):
        evaluate_attempts(tasks, solutions, attempts)


def test_invalid_generation_is_a_failed_attempt_and_retained_in_denominator():
    task = Task("a", (Pair(((0,),), ((1,),)),), (((2,),),))
    attempts = [Attempt("a", i, (None,)) for i in range(5)]
    result = evaluate_attempts([task], {"a": (((3,),),)}, attempts)
    assert result["successful_attempts"] == 0
    assert result["total_attempts"] == 5
