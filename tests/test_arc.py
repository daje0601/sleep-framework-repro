import dataclasses
import hashlib
import json

import pytest

from sleep_repro.arc import (
    Pair,
    Task,
    expand_directive,
    format_prompt,
    format_supervised_example,
    holdout_demonstration,
    inverse_transform_grid,
    load_arc,
    parse_directive,
    parse_grid_response,
    reconstruction_manifest,
    transform_grid,
)


def fixture_task():
    return Task(
        "example",
        (Pair(((1, 2, 3), (4, 5, 6)), ((2, 3),)), Pair(((7,),), ((8,),))),
        (((9,),),),
    )


@pytest.mark.parametrize(
    "value",
    [
        [],
        [[]],
        [[1], [2, 3]],
        [[True]],
        [[10]],
        [[-1]],
        [[1.0]],
        [[1] * 31],
        [[1]] * 31,
        "[[1]]",
        None,
    ],
)
def test_grid_parser_rejects_malformed_or_out_of_bounds_grids(value):
    with pytest.raises(ValueError):
        parse_grid_response(json.dumps(value))


def test_grid_response_accepts_only_json_or_single_json_fence():
    assert parse_grid_response("```json\n[[1,2],[3,4]]\n```") == ((1, 2), (3, 4))
    for response in (
        "answer: [[1]]",
        "[[1]] trailing",
        "```python\n[[1]]\n```",
        "[[NaN]]",
    ):
        with pytest.raises(ValueError):
            parse_grid_response(response)


def test_load_separates_targets_and_freezes_nested_data(tmp_path):
    challenges = tmp_path / "challenges.json"
    solutions = tmp_path / "solutions.json"
    challenges.write_text(
        json.dumps(
            {
                "abc": {
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]], "output": [[8]]}],
                }
            }
        )
    )
    solutions.write_text(json.dumps({"abc": [[[9]]]}))
    dataset = load_arc(challenges, solutions, ["abc"])
    task = dataset.tasks[0]
    assert dataclasses.asdict(task) == {
        "task_id": "abc",
        "demonstrations": ({"input": ((1,),), "output": ((2,),)},),
        "test_inputs": (((3,),),),
    }
    assert dataset.solutions["abc"] == (((9,),),)
    assert (
        dataset.file_hashes["challenges_sha256"]
        == hashlib.sha256(challenges.read_bytes()).hexdigest()
    )
    assert "[[9]]" not in format_prompt(task)
    assert "[[8]]" not in format_prompt(task)
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.task_id = "mutated"
    with pytest.raises(TypeError):
        dataset.solutions["abc"] = ()
    with pytest.raises(ValueError, match="duplicate"):
        load_arc(challenges, solutions, ["abc", "abc"])
    with pytest.raises(ValueError, match="missing"):
        load_arc(challenges, solutions, ["missing"])


def test_task_constructor_copies_mutable_grids_and_holdout_is_leak_free():
    source = [[1, 2]]
    task = Task("t", [Pair(source, [[3]])], [source])
    source[0][0] = 9
    assert task.test_inputs == (((1, 2),),)
    with pytest.raises(ValueError, match="two"):
        holdout_demonstration(task)
    visible, reward_pair = holdout_demonstration(fixture_task())
    assert visible.demonstrations == (fixture_task().demonstrations[0],)
    assert visible.test_inputs == (((7,),),)
    assert reward_pair.output == ((8,),)
    assert "[[8]]" not in format_prompt(visible)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"rotations": [4]},
        {"rotations": [True]},
        {"rotations": []},
        {"rotations": [0, 0]},
        {"rotations": [0], "reflect": "yes"},
        {"rotations": [0], "epochs": 0},
        {"rotations": [0], "epochs": 6},
        {"rotations": [0], "epochs": True},
        {"rotations": [0], "lr_scale": 0.01},
        {"rotations": [0], "lr_scale": 11},
        {"rotations": [0], "lr_scale": float("nan")},
        {"rotations": [0], "color_permutation": list(range(9))},
        {"rotations": [0], "color_permutation": [0] * 10},
        {"rotations": [0], "code": "raise RuntimeError()"},
    ],
)
def test_directive_schema_rejects_invalid_and_executable_fields(payload):
    with pytest.raises(ValueError):
        parse_directive(json.dumps(payload))


def test_directive_expansion_is_deterministic_uses_only_demonstrations():
    directive = parse_directive(
        '{"rotations":[0,1],"reflect":true,"epochs":2,"lr_scale":0.5}'
    )
    examples = expand_directive(fixture_task(), directive)
    assert len(examples) == 4
    assert examples[0] == Pair(((3, 2, 1), (6, 5, 4)), ((3, 2),))
    assert examples == expand_directive(fixture_task(), directive)
    assert not any(9 in row for pair in examples for row in pair.input)
    prompt, completion = format_supervised_example(examples[0])
    assert parse_grid_response(completion) == examples[0].output
    assert "[[3,2,1],[6,5,4]]" in prompt
    assert json.loads(directive.to_json())["epochs"] == 2


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("reflect", [False, True])
def test_spatial_and_color_transforms_invert_on_rectangles(rotation, reflect):
    grid = ((1, 2, 3), (4, 5, 6))
    permutation = tuple(reversed(range(10)))
    transformed = transform_grid(grid, rotation, reflect, permutation)
    assert inverse_transform_grid(transformed, rotation, reflect, permutation) == grid
    if rotation == 1 and not reflect:
        assert transformed == ((5, 8), (4, 7), (3, 6))


def test_manifest_freezes_public_reconstruction_splits_and_ancestry():
    manifest = reconstruction_manifest({"challenges_sha256": "a" * 64})
    assert len(manifest["train_ids"]) == 11
    assert len(manifest["evaluation_ids"]) == 8
    assert set(manifest["train_ids"]).isdisjoint(manifest["evaluation_ids"])
    assert manifest["seal_commit"] == "6d9c9f9ee392c6cc618e771f399d436d190f6ca4"
    assert manifest["protocol_label"] == "public protocol reconstruction"
    json.dumps(manifest, allow_nan=False)
