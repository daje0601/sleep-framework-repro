"""Compute–consolidate–commit transactions against an immutable live model.

Snapshots deliberately copy the full model, enabling objectives on actual HF
logits rather than disconnected FFN outputs. This correctness-first strategy
has substantial memory cost; GPU preflight must measure it before experiments.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np
from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .memory import CapacityError
from .model import iter_memory_chains


@dataclass(frozen=True)
class TransactionResult:
    committed: bool
    sender_level: int
    new_experts: tuple[int, ...]
    losses: tuple[float, ...]


def _validate_rate(rate: float, name: str) -> None:
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _rng_snapshot():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state().clone()
    return state


def _restore_rng(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _validate_finite(model: nn.Module) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite prospective state: {name}")


def _commit(
    live: nn.Module,
    student: nn.Module,
    reset_parameters: list[nn.Parameter],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    # Parameter objects and their requires_grad flags remain live, preserving
    # attached optimizer references and any separately configured policy adapter.
    old_state = copy.deepcopy(live.state_dict())
    old_optimizer = (
        copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    )
    try:
        live.load_state_dict(student.state_dict(), strict=True)
        if optimizer is not None:
            for parameter in reset_parameters:
                optimizer.state.pop(parameter, None)
    except BaseException:
        live.load_state_dict(old_state, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(old_optimizer)
        raise
    for parameter in reset_parameters:
        parameter.grad = None


def consolidate(
    model: nn.Module,
    sender_level: int,
    objective: Callable[[nn.Module, nn.Module], Tensor],
    *,
    base_lr: float,
    expert_lr: float,
    steps: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> TransactionResult:
    """Transfer every FFN chain at one pending adjacent-level boundary atomically.

    ``objective(teacher, student)`` returns a scalar differentiable loss. The
    teacher is an eval-mode pre-update snapshot with all parameters frozen;
    the eval-mode student contains proposed mean-gradient SGD base updates,
    cleared sender banks, and a new eligible receiver branch in every chain.
    Only those new branches require gradients. ``steps`` Adam iterations run
    on the isolated student under exactly the routing graph that will commit.
    One final no-gradient objective validation observes the post-fit parameters.
    All new branches activate together in the live model after finite
    loss, finite-gradient and finite-state checks. Exceptions leave live state,
    external optimizer state and Python/NumPy/torch RNG streams unchanged.

    Fresh snapshots per call implement fast-to-slow simultaneous boundaries.
    The final level uses :func:`apply_terminal_update`, since it has no receiver.
    """
    _validate_rate(base_lr, "base_lr")
    _validate_rate(expert_lr, "expert_lr")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(sender_level) is not int or sender_level < 0:
        raise ValueError("sender_level must be a nonnegative integer")
    live_chains = list(iter_memory_chains(model))
    if not live_chains:
        raise ValueError("model contains no memory chains")
    for chain in live_chains:
        if sender_level + 1 >= len(chain.stages):
            raise ValueError(
                "sender must have an adjacent receiver; use apply_terminal_update for the last level"
            )
        if not bool(chain.pending[sender_level]):
            raise ValueError(f"level {sender_level} has no pending update boundary")
        if bool(chain.stages[sender_level + 1].active.all()):
            raise CapacityError(
                f"receiver level {sender_level + 1} expert capacity exhausted"
            )
    rng = _rng_snapshot()
    try:
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        student = copy.deepcopy(model).eval().requires_grad_(False)
        # Snapshots can inherit stale .grad tensors from custom module deepcopy;
        # explicitly discard them so frozen tensors never carry wake gradients.
        teacher.zero_grad(set_to_none=True)
        student.zero_grad(set_to_none=True)
        teacher_before = {
            key: value.clone() for key, value in teacher.state_dict().items()
        }
        trainable = []
        expert_parameter_groups = []
        initial_expert_parameters = []
        new_experts = []
        for chain in iter_memory_chains(student):
            sender, receiver = chain.stages[sender_level : sender_level + 2]
            sender.propose_base_update(base_lr)
            sender.reset_experts()
            new_index = receiver.allocate_expert()
            # Make the private new branch normally eligible before fitting.
            # Adding it only after training would change top-k routing at commit.
            receiver.mark_trained(new_index)
            receiver.forced_expert.fill_(-1)
            receiver.experts[new_index].requires_grad_(True)
            expert_parameters = tuple(receiver.experts[new_index].parameters())
            trainable.extend(expert_parameters)
            expert_parameter_groups.append(expert_parameters)
            initial_expert_parameters.append(
                tuple(p.detach().clone() for p in expert_parameters)
            )
            new_experts.append(new_index)
            chain.pending[sender_level] = False
            chain.update_counts[sender_level].add_(1)
        prospective_before = {
            key: value.clone() for key, value in student.state_dict().items()
        }
        trainable_names = {
            name
            for name, parameter in student.named_parameters()
            if parameter.requires_grad
        }
        fit_optimizer = torch.optim.Adam(trainable, lr=expert_lr)
        losses = []
        for _ in range(steps):
            fit_optimizer.zero_grad(set_to_none=True)
            loss = objective(teacher, student)
            if not isinstance(loss, Tensor) or loss.ndim != 0:
                raise ValueError("consolidation objective must return a scalar tensor")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite consolidation objective")
            if not loss.requires_grad:
                raise ValueError(
                    "consolidation objective must depend on the new receiver expert"
                )
            loss.backward()
            if not any(parameter.grad is not None for parameter in trainable):
                raise ValueError(
                    "consolidation objective has no receiver expert gradient"
                )
            for parameter in trainable:
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError("nonfinite consolidation gradient")
            fit_optimizer.step()
            losses.append(float(loss.detach()))
        # Validate the exact post-optimizer function before making it live;
        # neither expert eligibility nor forced routing changes after this call.
        with torch.no_grad():
            final_loss = objective(teacher, student)
        if not isinstance(final_loss, Tensor) or final_loss.ndim != 0:
            raise ValueError("consolidation objective must return a scalar tensor")
        if not bool(torch.isfinite(final_loss)):
            raise FloatingPointError("nonfinite post-fit consolidation objective")
        for key, value in teacher.state_dict().items():
            if not torch.equal(value, teacher_before[key]):
                raise RuntimeError(f"objective modified immutable teacher state: {key}")
        # Frozen state may not be mutated inside a user-supplied objective.
        for key, value in student.state_dict().items():
            if key not in trainable_names and not torch.equal(
                value, prospective_before[key]
            ):
                raise RuntimeError(f"objective modified frozen student state: {key}")
        for chain_index, (parameters, initial) in enumerate(
            zip(expert_parameter_groups, initial_expert_parameters)
        ):
            if not any(
                not torch.equal(parameter.detach(), before)
                for parameter, before in zip(parameters, initial)
            ):
                raise ValueError(
                    f"receiver expert {chain_index}:{new_experts[chain_index]} received no parameter update; transaction aborted"
                )
        _validate_finite(student)
        reset_parameters = []
        for chain, index in zip(live_chains, new_experts):
            reset_parameters.extend(chain.stages[sender_level].base.parameters())
            reset_parameters.extend(
                p
                for expert in chain.stages[sender_level].experts
                for p in expert.parameters()
            )
            reset_parameters.extend(
                chain.stages[sender_level + 1].experts[index].parameters()
            )
        _commit(model, student, reset_parameters, optimizer)
        return TransactionResult(True, sender_level, tuple(new_experts), tuple(losses))
    except BaseException:
        _restore_rng(rng)
        raise


def apply_terminal_update(
    model: nn.Module, *, base_lr: float, optimizer: torch.optim.Optimizer | None = None
) -> TransactionResult:
    """Commit a pending slowest base SGD update, keeping its expert bank intact."""
    _validate_rate(base_lr, "base_lr")
    live_chains = list(iter_memory_chains(model))
    if not live_chains:
        raise ValueError("model contains no memory chains")
    for chain in live_chains:
        if not bool(chain.pending[-1]):
            raise ValueError("terminal level has no pending update boundary")
    student = copy.deepcopy(model)
    reset_parameters = []
    for chain, live_chain in zip(iter_memory_chains(student), live_chains):
        chain.stages[-1].propose_base_update(base_lr)
        chain.pending[-1] = False
        chain.update_counts[-1].add_(1)
        reset_parameters.extend(live_chain.stages[-1].base.parameters())
    _validate_finite(student)
    _commit(model, student, reset_parameters, optimizer)
    return TransactionResult(True, len(live_chains[0].stages) - 1, (), ())
