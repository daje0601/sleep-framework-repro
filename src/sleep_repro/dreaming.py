"""Candidate selection and isolated inner adaptation for dreaming/ReST-EM."""

from collections.abc import Callable, Sequence
import copy
from dataclasses import dataclass
import math
import random
from typing import Any, TypeVar

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

Candidate = TypeVar("Candidate")


@dataclass(frozen=True)
class DreamRecord:
    candidate: Any
    baseline_score: float
    adapted_score: float
    reward: float
    accepted: bool
    output: Any = None


@dataclass(frozen=True)
class DreamResult:
    records: list[DreamRecord]
    accepted: list[Any]


def select_candidates(
    scores: Sequence[float], *, top_k: int, random_k: int, seed: int
) -> list[int]:
    """Select highest importance indices, then seeded random remaining indices.

    Ties preserve input order. Random sampling uses a local generator and
    does not consume the application's Python RNG stream.
    """
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("candidate scores must be finite")
    if any(
        not isinstance(k, int) or isinstance(k, bool) or k < 0
        for k in (top_k, random_k)
    ):
        raise ValueError("selection counts must be nonnegative integers")
    if top_k + random_k > len(scores):
        raise ValueError("selection counts exceed candidate count")
    ranked = sorted(range(len(scores)), key=lambda index: -float(scores[index]))
    top = ranked[:top_k]
    remaining = [index for index in range(len(scores)) if index not in set(top)]
    return top + random.Random(seed).sample(remaining, random_k)


def _rng_snapshot() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _finite_score(model: nn.Module, score: Callable[[nn.Module], float]) -> float:
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            value = float(score(model))
    finally:
        for module, training in modes:
            module.training = training
    if not math.isfinite(value):
        raise ValueError("validation score must be finite")
    return value


def evaluate_candidates(
    model: nn.Module,
    optimizer: Optimizer,
    candidates: Sequence[Candidate],
    *,
    adapt: Callable[[nn.Module, Optimizer, Candidate], Any],
    score: Callable[[nn.Module], float],
    rest_update: Callable[[nn.Module, Optimizer, list[Candidate]], Any] | None = None,
) -> DreamResult:
    """Adapt each candidate from identical model/optimizer/RNG snapshots.

    ``adapt`` operates only on private clones and may return prediction or
    trace data for the record. ``score`` must use only permitted validation
    examples; higher is better. A strict improvement accepts the directive.
    After all candidates, ``rest_update`` receives only accepted directives
    and is the sole callback allowed to change the live outer policy. An
    empty accepted set never calls it. RNG streams are restored on all exits.

    Joint deepcopy preserves optimizer-to-parameter references and momentum
    while isolating buffers, training modes, and adapter state. This generic
    implementation prioritizes isolation over snapshot memory efficiency.
    """
    rng = _rng_snapshot()
    records: list[DreamRecord] = []
    accepted: list[Candidate] = []
    try:
        # Capture the baseline once before callbacks can inspect a clone.
        snapshot = copy.deepcopy((model, optimizer))
        baseline_model, baseline_optimizer = copy.deepcopy(snapshot)
        _restore_rng(rng)
        baseline = _finite_score(baseline_model, score) if candidates else None
        del baseline_model, baseline_optimizer
        for candidate in candidates:
            inner_model, inner_optimizer = copy.deepcopy(snapshot)
            _restore_rng(rng)
            output = adapt(inner_model, inner_optimizer, copy.deepcopy(candidate))
            # Validation randomness is identical for baseline and adapted policy.
            _restore_rng(rng)
            adapted = _finite_score(inner_model, score)
            reward = adapted - baseline
            if not math.isfinite(reward):
                raise ValueError("candidate improvement must be finite")
            positive = reward > 0
            directive = copy.deepcopy(candidate)
            records.append(
                DreamRecord(
                    directive,
                    baseline,
                    adapted,
                    reward,
                    positive,
                    copy.deepcopy(output),
                )
            )
            if positive:
                accepted.append(copy.deepcopy(candidate))
            # Release the previous private model/optimizer before allocating the
            # next candidate; optimizer parameter references retain weight memory.
            del inner_model, inner_optimizer, output
        del snapshot
        if accepted and rest_update is not None:
            _restore_rng(rng)
            rest_update(model, optimizer, copy.deepcopy(accepted))
        return DreamResult(records, accepted)
    finally:
        _restore_rng(rng)
