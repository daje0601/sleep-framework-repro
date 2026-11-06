"""Bounded, explicitly scheduled memory chains for pretrained FFN positions.

The residual embedding and fixed-projection router are reconstruction choices.
Forward never allocates experts, changes masks, or advances the wake schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CapacityError(RuntimeError):
    """A preallocated memory bank has no unused expert slot."""


@dataclass(frozen=True)
class RoutingDecision:
    # Leading dimensions match the input token dimensions, never pooled.
    selected: Tensor
    extra: Tensor | None = None
    forced: int | None = None

    @property
    def indices(self) -> tuple[int, ...]:
        values = self.selected.flatten()
        if self.extra is not None:
            values = torch.cat((values, self.extra.flatten()))
        indices = tuple(int(index) for index in torch.unique(values))
        return tuple(
            dict.fromkeys(
                (*indices, *((self.forced,) if self.forced is not None else ()))
            )
        )

    def mask(self, index: int) -> Tensor:
        selected = (self.selected == index).any(dim=-1)
        if self.extra is not None:
            selected = selected | (self.extra == index)
        if self.forced == index:
            selected = torch.ones_like(selected)
        return selected


class LowRankBranch(nn.Module):
    """Trainable nonlinear rank-r branch, initially exactly zero-output."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden_size and rank must be positive")
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def reset_parameters(self) -> None:
        self.down.reset_parameters()
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(F.silu(self.down(x)))


class MemoryStage(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        hidden_size: int,
        period: int,
        capacity: int,
        rank: int,
        top_k: int,
        residual: bool,
        expert_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if capacity < 1 or top_k < 1 or top_k > capacity:
            raise ValueError("capacity must be positive and 1 <= top_k <= capacity")
        if not torch.isfinite(torch.tensor(expert_scale)) or expert_scale <= 0:
            raise ValueError("expert_scale must be finite and positive")
        self.base = base
        self.experts = nn.ModuleList(
            LowRankBranch(hidden_size, rank) for _ in range(capacity)
        )
        self.residual = residual
        self.top_k = top_k
        self.dreaming = False
        self.register_buffer("period", torch.tensor(period, dtype=torch.long))
        self.register_buffer("active", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("trained", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("forced_expert", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("expert_scale", torch.tensor(float(expert_scale)))
        # Fixed, checkpointed scores: each token independently dots random unit vectors.
        self.register_buffer(
            "routing_projection",
            F.normalize(torch.randn(capacity, hidden_size), dim=-1),
        )
        self.register_buffer("gradient_count", torch.tensor(0, dtype=torch.long))
        for index, parameter in enumerate(self.base.parameters()):
            self.register_buffer(f"gradient_sum_{index}", torch.zeros_like(parameter))

    def accumulated_gradients(self) -> tuple[Tensor, ...]:
        return tuple(
            getattr(self, f"gradient_sum_{i}")
            for i, _ in enumerate(self.base.parameters())
        )

    @torch.no_grad()
    def accumulate_gradients(self) -> None:
        any_gradient = False
        for parameter, total in zip(
            self.base.parameters(), self.accumulated_gradients()
        ):
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("nonfinite wake gradient")
                total.add_(parameter.grad)
                any_gradient = True
        if any_gradient:
            self.gradient_count.add_(1)

    @torch.no_grad()
    def clear_gradients(self) -> None:
        for total in self.accumulated_gradients():
            total.zero_()
        self.gradient_count.zero_()

    @torch.no_grad()
    def propose_base_update(self, learning_rate: float) -> None:
        """Apply mean accumulated SGD only to an isolated prospective copy."""
        denominator = max(int(self.gradient_count), 1)
        for parameter, total in zip(
            self.base.parameters(), self.accumulated_gradients()
        ):
            parameter.add_(total, alpha=-learning_rate / denominator)
        self.clear_gradients()

    @torch.no_grad()
    def allocate_expert(self) -> int:
        slots = (~self.active).nonzero(as_tuple=False).flatten()
        if not slots.numel():
            raise CapacityError(
                f"expert capacity exhausted ({len(self.experts)} slots); increase the frozen bank capacity"
            )
        index = int(slots[0])
        self.experts[index].reset_parameters()
        self.active[index] = True
        self.trained[index] = False
        return index

    @torch.no_grad()
    def mark_trained(self, index: int) -> None:
        if not 0 <= index < len(self.experts) or not bool(self.active[index]):
            raise ValueError("only an allocated active expert can be marked trained")
        self.trained[index] = True

    @torch.no_grad()
    def reset_experts(self) -> None:
        for expert in self.experts:
            expert.reset_parameters()
        self.active.zero_()
        self.trained.zero_()
        self.forced_expert.fill_(-1)

    def route(self, x: Tensor, *, dreaming: bool | None = None) -> RoutingDecision:
        """Route each token using only that token's hidden state.

        No reduction crosses token or batch dimensions: future completion tokens,
        padding, and co-batched examples cannot change another token's route.
        During dreaming one uniform remaining-rank offset is shared across
        tokens; each token still selects outside its own normal top-k set.
        """
        eligible = (self.active & self.trained).nonzero(as_tuple=False).flatten()
        k = min(self.top_k, eligible.numel())
        if eligible.numel():
            scores = x.detach().float() @ self.routing_projection[eligible].float().T
            # Stable sorting defines expert-index tie-breaking per token.
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            ranked = eligible[order]
            selected = ranked[..., :k]
        else:
            selected = torch.empty(
                (*x.shape[:-1], 0), dtype=torch.long, device=x.device
            )
        extra = None
        if (
            dreaming if dreaming is not None else self.dreaming
        ) and eligible.numel() > k:
            offset = int(torch.randint(eligible.numel() - k, ()).item())
            extra = ranked[..., k + offset]
        forced = int(self.forced_expert)
        if forced >= 0 and (
            forced >= len(self.experts) or not bool(self.active[forced])
        ):
            raise ValueError("forced expert must be an allocated active expert")
        return RoutingDecision(selected, extra, forced if forced >= 0 else None)

    def forward(self, x: Tensor) -> Tensor:
        output = self.base(x)
        routing = self.route(x)
        for index in routing.indices:
            mask = routing.mask(index).unsqueeze(-1).to(dtype=x.dtype)
            output = output + self.expert_scale * self.experts[index](x) * mask
        return x + output if self.residual else output


class MemoryChain(nn.Module):
    """Existing FFN followed by slower identity-initialized residual stages."""

    def __init__(
        self,
        base: nn.Module,
        *,
        hidden_size: int,
        periods: Iterable[int] = (1, 4),
        capacity: int = 8,
        rank: int = 64,
        top_k: int = 1,
        expert_scale: float = 1.0,
    ) -> None:
        super().__init__()
        periods = tuple(periods)
        if not periods or any(type(p) is not int or p <= 0 for p in periods):
            raise ValueError("periods must be positive integers")
        if any(b <= a or b % a for a, b in zip(periods, periods[1:])):
            raise ValueError(
                "period hierarchy must strictly increase with each period dividing the next"
            )
        self.stages = nn.ModuleList(
            MemoryStage(
                base if index == 0 else LowRankBranch(hidden_size, rank),
                hidden_size=hidden_size,
                period=period,
                capacity=capacity,
                rank=rank,
                top_k=top_k,
                residual=index > 0,
                expert_scale=expert_scale,
            )
            for index, period in enumerate(periods)
        )
        self.register_buffer("wake_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("pending", torch.zeros(len(periods), dtype=torch.bool))
        self.register_buffer(
            "update_counts", torch.zeros(len(periods), dtype=torch.long)
        )
        reference = next(base.parameters(), None)
        if reference is not None:
            self.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def record_wake(self) -> None:
        # Validate every current gradient before changing any schedule/accumulation buffer.
        for stage in self.stages:
            for parameter in stage.base.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError("nonfinite wake gradient")
        for stage in self.stages:
            stage.accumulate_gradients()
        self.wake_steps.add_(1)
        for index, stage in enumerate(self.stages):
            if int(self.wake_steps) % int(stage.period) == 0:
                self.pending[index] = True

    def pending_levels(self) -> tuple[int, ...]:
        return tuple(
            int(index) for index in self.pending.nonzero(as_tuple=False).flatten()
        )

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        return x
