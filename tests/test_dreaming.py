import copy
import random

import numpy as np
import pytest
import torch

from sleep_repro.dreaming import evaluate_candidates, select_candidates


def test_selection_takes_best_then_seeded_remaining_without_duplicates():
    scores = [0.0, 4.0, 3.0, 2.0, 1.0]
    selected = select_candidates(scores, top_k=2, random_k=2, seed=12)
    assert selected[:2] == [1, 2]
    assert len(set(selected)) == 4
    assert selected == select_candidates(scores, top_k=2, random_k=2, seed=12)
    assert select_candidates(scores, top_k=0, random_k=0, seed=1) == []


@pytest.mark.parametrize(
    "scores,top_k,random_k",
    [
        ([float("nan")], 1, 0),
        ([float("inf")], 0, 1),
        ([1.0], 2, 0),
        ([1.0], 1, 1),
        ([1.0], -1, 0),
    ],
)
def test_selection_rejects_nonfinite_or_impossible_requests(scores, top_k, random_k):
    with pytest.raises(ValueError):
        select_candidates(scores, top_k=top_k, random_k=random_k, seed=1)


def make_state():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    return model, optimizer


def test_each_candidate_uses_same_model_optimizer_and_rng_and_only_positive_rest():
    model, optimizer = make_state()
    starts = []
    samples = []
    updates = []

    def adapt(inner, optim, candidate):
        starts.append(
            (
                inner.weight.item(),
                next(iter(optim.state.values()))["momentum_buffer"].item(),
            )
        )
        samples.append((random.random(), np.random.random(), torch.rand(1).item()))
        optim.zero_grad()
        (candidate * inner(torch.ones(1, 1)).sum()).backward()
        optim.step()
        return {"directive": candidate}

    def rest(live, optim, accepted):
        updates.extend(accepted)
        optim.zero_grad()
        live(torch.ones(1, 1)).sum().backward()
        optim.step()

    result = evaluate_candidates(
        model,
        optimizer,
        [-10.0, 10.0],
        adapt=adapt,
        score=lambda m: m.weight.item(),
        rest_update=rest,
    )
    assert starts[0] == starts[1] == pytest.approx((0.9, 1.0))
    assert samples[0] == samples[1]
    assert result.accepted == [-10.0]
    assert updates == [-10.0]
    assert result.records[0].reward > 0
    assert result.records[1].reward < 0
    assert result.records[0].output == {"directive": -10.0}
    assert model.weight.item() == pytest.approx(0.71)


def test_no_positive_candidate_is_noop_and_restores_all_rng_even_when_callback_raises():
    model, optimizer = make_state()
    before = copy.deepcopy(optimizer.state_dict())
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    expected = (random.random(), np.random.random(), torch.rand(1).item())
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    def adapt(inner, optim, candidate):
        random.random()
        np.random.random()
        torch.rand(1)
        inner.weight.data.fill_(-1)

    def forbidden(*args):
        raise AssertionError("empty accepted set must not trigger ReST")

    result = evaluate_candidates(
        model,
        optimizer,
        [0, 1],
        adapt=adapt,
        score=lambda m: m.weight.item(),
        rest_update=forbidden,
    )
    assert result.accepted == []
    assert model.weight.item() == pytest.approx(0.9)
    assert optimizer.state_dict()["state"][0]["momentum_buffer"].equal(
        before["state"][0]["momentum_buffer"]
    )
    assert (random.random(), np.random.random(), torch.rand(1).item()) == expected

    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    def exploding(*args):
        adapt(*args)
        raise RuntimeError("candidate failed")

    with pytest.raises(RuntimeError, match="candidate failed"):
        evaluate_candidates(
            model, optimizer, [0], adapt=exploding, score=lambda m: m.weight.item()
        )
    assert (random.random(), np.random.random(), torch.rand(1).item()) == expected
    assert model.weight.item() == pytest.approx(0.9)


def test_equal_score_is_rejected_and_nonfinite_score_is_error():
    model, optimizer = make_state()
    result = evaluate_candidates(
        model, optimizer, ["x"], adapt=lambda *args: None, score=lambda m: 1.0
    )
    assert result.records[0].accepted is False
    with pytest.raises(ValueError, match="finite"):
        evaluate_candidates(
            model,
            optimizer,
            ["x"],
            adapt=lambda *args: None,
            score=lambda m: float("nan"),
        )


def test_validation_uses_eval_even_when_adaptation_leaves_model_training():
    model, optimizer = make_state()
    model.eval()

    def score(candidate):
        # A mode-dependent score models the dropout discrepancy between clones.
        return candidate.weight.item() + float(candidate.training)

    result = evaluate_candidates(
        model,
        optimizer,
        ["x"],
        adapt=lambda inner, opt, candidate: inner.train(),
        score=score,
    )
    assert result.accepted == []
    assert result.records[0].reward == 0
    assert model.training is False


def test_scored_baseline_and_previous_candidate_clones_are_released_before_next_adaptation():
    import gc
    import weakref

    model, optimizer = make_state()
    baseline_refs = []
    previous_refs = []
    score_calls = 0

    def score(candidate):
        nonlocal score_calls
        if score_calls == 0:
            baseline_refs.extend(
                (weakref.ref(candidate), weakref.ref(candidate.weight))
            )
        score_calls += 1
        return candidate.weight.item()

    def adapt(inner, optim, candidate):
        gc.collect()
        assert all(reference() is None for reference in baseline_refs)
        assert all(reference() is None for reference in previous_refs)
        previous_refs[:] = [
            weakref.ref(inner),
            weakref.ref(optim),
            weakref.ref(inner.weight),
        ]
        return {"candidate": candidate}

    result = evaluate_candidates(model, optimizer, [0, 1], adapt=adapt, score=score)
    assert result.accepted == []
    gc.collect()
    assert all(reference() is None for reference in previous_refs)
