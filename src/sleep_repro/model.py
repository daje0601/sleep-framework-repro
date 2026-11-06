"""Hugging Face Llama FFN conversion and explicit memory lifecycle helpers."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator, Iterable

import torch
from torch import nn

from .memory import MemoryChain


def iter_memory_chains(model: nn.Module) -> Iterator[MemoryChain]:
    yield from (module for module in model.modules() if isinstance(module, MemoryChain))


def wrap_llama_mlps(
    model: nn.Module,
    *,
    periods: Iterable[int] = (1, 4),
    capacity: int = 8,
    rank: int = 64,
    top_k: int = 1,
    layer_indices: Iterable[int] | None = None,
    expert_scale: float = 1.0,
) -> nn.Module:
    """Replace selected ``model.layers[*].mlp`` branches, keeping outer paths intact.

    Call before attaching LoRA; sharded/device-map conversion is not supported.
    The same model object is returned for convenience.
    """
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("expected a Hugging Face Llama model with model.layers")
    indices = (
        tuple(range(len(layers))) if layer_indices is None else tuple(layer_indices)
    )
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= len(layers) for i in indices)
    ):
        raise ValueError("layer_indices must identify distinct existing layers")
    periods = tuple(periods)
    # Prepare first, so validation/construction failures leave all original layers attached.
    replacements = []
    for index in indices:
        original = layers[index].mlp
        if isinstance(original, MemoryChain):
            raise ValueError("FFN already wrapped in a MemoryChain")
        hidden_size = getattr(getattr(original, "gate_proj", None), "in_features", None)
        if hidden_size is None:
            raise TypeError("expected a Llama gated MLP with gate_proj.in_features")
        chain = MemoryChain(
            original,
            hidden_size=hidden_size,
            periods=periods,
            capacity=capacity,
            rank=rank,
            top_k=top_k,
            expert_scale=expert_scale,
        )
        chain.train(original.training)
        replacements.append((index, chain))
    for index, chain in replacements:
        layers[index].mlp = chain
    return model


def configure_wake_training(model: nn.Module) -> None:
    """Accumulate base gradients while keeping nested policy/inner LoRA frozen."""
    from .runtime import LoRALinear

    model.requires_grad_(False)
    for chain in iter_memory_chains(model):
        for stage in chain.stages:
            stage.base.requires_grad_(True)
            # FFN-target adapters live inside the always-on base, but episode
            # wake SGD owns only its backbone weights and appended base branch.
            for module in stage.base.modules():
                if isinstance(module, LoRALinear):
                    for name in ("policy_a", "policy_b", "inner_a", "inner_b"):
                        getattr(module, name).requires_grad_(False)


def record_wake(model: nn.Module) -> tuple[int, ...]:
    """Record one already-backpropagated minibatch; return union of due levels."""
    chains = list(iter_memory_chains(model))
    if not chains:
        raise ValueError("model contains no memory chains")
    # Check globally first: a late nonfinite gradient must not advance earlier chains.
    for chain in chains:
        for stage in chain.stages:
            for parameter in stage.base.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError("nonfinite wake gradient")
    for chain in chains:
        chain.record_wake()
    return tuple(
        sorted({level for chain in chains for level in chain.pending_levels()})
    )


@contextmanager
def dream_routing(model: nn.Module, enabled: bool = True):
    """Temporarily enable an extra random trained expert outside normal top-k."""
    stages = [stage for chain in iter_memory_chains(model) for stage in chain.stages]
    before = [stage.dreaming for stage in stages]
    try:
        for stage in stages:
            stage.dreaming = enabled
        yield model
    finally:
        for stage, old in zip(stages, before):
            stage.dreaming = old


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    active = total
    for chain in iter_memory_chains(model):
        for stage in chain.stages:
            active -= sum(
                p.numel()
                for index, expert in enumerate(stage.experts)
                if not bool(stage.active[index])
                for p in expert.parameters()
            )
    return {
        "total": total,
        "active": active,
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
