"""Compute–consolidate–commit transactions against an immutable live model.

Snapshots deliberately copy the full model, enabling objectives on actual HF
logits rather than disconnected FFN outputs. This correctness-first strategy
has substantial memory cost; GPU preflight must measure it before experiments.
Audit and rollback state copies own CPU storage to avoid extra GPU model copies.
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
    recovered_experts: tuple[tuple[int, int], ...] = ()
    recovery_losses: tuple[float, ...] = ()


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


def _cpu_snapshot(value):
    """Own independent CPU tensor storage throughout a nested checkpoint state.

    ``copy=True`` is essential even for tensors already on CPU: later live
    mutations must never change an audit reference or rollback backup.
    """
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        snapshot = type(value)(
            (key, _cpu_snapshot(item)) for key, item in value.items()
        )
        # Module state_dict version metadata is used by load_state_dict hooks.
        if hasattr(value, "_metadata"):
            snapshot._metadata = copy.deepcopy(value._metadata)
        return snapshot
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _offload_accumulated_gradients(model: nn.Module) -> None:
    """Move only private wake-gradient sums to CPU; forwards never read them.

    Call on a teacher before creating the student, and on a student only after
    applying its proposed sender updates. Parameters, routing buffers, counters,
    and the live model stay on their original execution devices. State-dict
    commit copies these values back into the live buffers on their own devices.
    """
    for chain in iter_memory_chains(model):
        for stage in chain.stages:
            for index, value in enumerate(stage.accumulated_gradients()):
                stage.register_buffer(
                    f"gradient_sum_{index}", value.detach().to(device="cpu", copy=True)
                )


def _commit(
    live: nn.Module,
    student: nn.Module,
    reset_parameters: list[nn.Parameter],
    optimizer: torch.optim.Optimizer | None,
) -> None:
    # Parameter objects and their requires_grad flags remain live, preserving
    # attached optimizer references and any separately configured policy adapter.
    old_state = _cpu_snapshot(live.state_dict())
    old_optimizer = (
        _cpu_snapshot(optimizer.state_dict()) if optimizer is not None else None
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


def _fit_receivers(teacher, student, objective, parameter_groups, expert_lr, steps):
    trainable = [p for group in parameter_groups for p in group]
    fit_optimizer = torch.optim.Adam(trainable, lr=expert_lr)
    losses = []
    # Distinguish a branch missed by the objective from a nonzero gradient
    # that produces no representable (or no net) weight change.
    observed_gradient = [False] * len(parameter_groups)
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
            raise ValueError("consolidation objective has no receiver expert gradient")
        for index, group in enumerate(parameter_groups):
            for parameter in group:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite consolidation gradient")
                if not observed_gradient[index] and bool(gradient.any()):
                    observed_gradient[index] = True
        fit_optimizer.step()
        losses.append(float(loss.detach()))
    return tuple(losses), observed_gradient


@torch.no_grad()
def _validate_fit(teacher, student, objective):
    final_loss = objective(teacher, student)
    if not isinstance(final_loss, Tensor) or final_loss.ndim != 0:
        raise ValueError("consolidation objective must return a scalar tensor")
    if not bool(torch.isfinite(final_loss)):
        raise FloatingPointError("nonfinite post-fit consolidation objective")


def _assert_frozen_state(module, before, mutable_names, label):
    for name, value in module.state_dict().items():
        if name not in mutable_names and not torch.equal(
            value.detach().cpu(), before[name]
        ):
            raise RuntimeError(f"objective modified {label} state: {name}")


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
    on the isolated student with ordinary routing. If hard routing leaves a
    new branch without a nonzero gradient, only those unchanged branches get
    one bounded fit with their slots included on all tokens. Ordinary routing
    is restored for the final objective validation and commit.
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
        # Teacher forwards never consume accumulation state. Release its GPU
        # sums before allocating the second full execution-model snapshot.
        teacher.zero_grad(set_to_none=True)
        _offload_accumulated_gradients(teacher)
        student = copy.deepcopy(model).eval().requires_grad_(False)
        # Snapshots can inherit stale .grad tensors from custom module deepcopy.
        student.zero_grad(set_to_none=True)
        teacher_before = _cpu_snapshot(teacher.state_dict())
        expert_parameter_groups = []
        initial_expert_parameters = []
        new_experts = []
        student_chains = list(iter_memory_chains(student))
        for chain in student_chains:
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
            expert_parameter_groups.append(expert_parameters)
            initial_expert_parameters.append(
                tuple(p.detach().clone() for p in expert_parameters)
            )
            new_experts.append(new_index)
            chain.pending[sender_level] = False
            chain.update_counts[sender_level].add_(1)
        # Sender SGD has now consumed its sums. Preserve all slower pending
        # sums on CPU while the isolated student performs forward/backward.
        _offload_accumulated_gradients(student)
        prospective_before = _cpu_snapshot(student.state_dict())
        trainable_names = {
            name
            for name, parameter in student.named_parameters()
            if parameter.requires_grad
        }
        losses, observed_gradient = _fit_receivers(
            teacher, student, objective, expert_parameter_groups, expert_lr, steps
        )
        _validate_fit(teacher, student, objective)
        _assert_frozen_state(teacher, teacher_before, (), "immutable teacher")
        _assert_frozen_state(
            student, prospective_before, trainable_names, "frozen student"
        )

        def changed(index):
            return any(
                not torch.equal(parameter.detach(), before)
                for parameter, before in zip(
                    expert_parameter_groups[index], initial_expert_parameters[index]
                )
            )

        recovered = []
        recovery_losses = ()
        for index, slot in enumerate(new_experts):
            if changed(index):
                continue
            if observed_gradient[index]:
                raise ValueError(
                    f"receiver expert {index}:{slot} had a nonzero gradient but no "
                    "net parameter update; transaction aborted"
                )
            recovered.append((index, slot))
        if recovered:
            # A top-k branch may see no tokens that contribute to the loss.
            # Give these branches a training signal without updating the peers
            # that have already fitted under the intended inference router.
            student.requires_grad_(False)
            recovery_groups = []
            for index, slot in recovered:
                receiver = student_chains[index].stages[sender_level + 1]
                receiver.experts[slot].requires_grad_(True)
                receiver.forced_expert.fill_(slot)
                recovery_groups.append(expert_parameter_groups[index])
            # Include already-fitted peers in the frozen-state audit.
            for name, parameter in student.named_parameters():
                if name in trainable_names and not parameter.requires_grad:
                    prospective_before[name] = _cpu_snapshot(parameter)
                    trainable_names.remove(name)
            recovery_losses, recovery_gradient = _fit_receivers(
                teacher, student, objective, recovery_groups, expert_lr, steps
            )
            for index, slot in recovered:
                receiver = student_chains[index].stages[sender_level + 1]
                if int(receiver.forced_expert) != slot:
                    raise RuntimeError("objective modified frozen student routing")
                receiver.forced_expert.fill_(-1)
            _validate_fit(teacher, student, objective)
            _assert_frozen_state(teacher, teacher_before, (), "immutable teacher")
            _assert_frozen_state(
                student, prospective_before, trainable_names, "frozen student"
            )
            for (index, slot), nonzero in zip(recovered, recovery_gradient):
                if not changed(index):
                    reason = (
                        "nonzero gradient but no net update"
                        if nonzero
                        else "no nonzero gradient"
                    )
                    raise ValueError(
                        f"receiver expert {index}:{slot} had {reason} during "
                        "forced fitting; transaction aborted"
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
        return TransactionResult(
            True,
            sender_level,
            tuple(new_experts),
            losses,
            tuple(recovered),
            recovery_losses,
        )
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
