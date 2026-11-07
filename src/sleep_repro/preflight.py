"""One-task execution preflight; forced fixtures are never learned-policy evidence."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .arc import (
    DreamDirective,
    content_hash,
    expand_directive,
    holdout_demonstration,
    parse_directive,
)
from .engine import CausalEngine
from .model import dream_routing, iter_memory_chains
from .runner import (
    build_model,
    dataset_for,
    evidence,
    examples_for,
    load_checkpoint,
    save_checkpoint,
    wake_episode,
    write_json,
)
from .runtime import seed_all, train_adapter


def _delta(before, after) -> float:
    """Compare in bounded chunks, without allocating another full model copy."""
    total = 0.0
    for left, right in zip(before, after, strict=True):
        left, right = left.detach().reshape(-1), right.detach().reshape(-1)
        if left.shape != right.shape:
            raise RuntimeError("parameter delta shape mismatch")
        for start in range(0, left.numel(), 1_000_000):
            a = left[start : start + 1_000_000].to(device="cpu", dtype=torch.float64)
            b = right[start : start + 1_000_000].to(device="cpu", dtype=torch.float64)
            total += (a - b).abs().sum().item()
    if not math.isfinite(total):
        raise FloatingPointError("nonfinite parameter delta")
    return total


def _snapshot(parameters):
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _check_update(delta: float, label: str) -> None:
    if delta <= 0:
        raise RuntimeError(
            f"{label} parameter delta is zero; refusing successful preflight"
        )


def _rng_probe(device):
    result = {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": torch.rand(3).tolist(),
    }
    if device.type == "cuda":
        result["cuda"] = torch.rand(3, device=device).cpu().tolist()
    return result


def preflight(cfg, output) -> dict:
    """Exercise full configured wake and one explicit inner/outer fixture update.

    Only training tasks are loaded. The first task contributes visible support
    demonstrations and a permitted held-out demonstration. A real directive is
    generated and recorded, while the identity directive is explicitly forced
    to test optimizer plumbing even when sampled JSON is invalid. Its outer
    update is a disposable diagnostic, never positive-reward policy training.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA preflight has no available CUDA device")
    seed_all(cfg.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.monotonic()
    data = dataset_for(cfg, "train")
    if not data.tasks:
        raise ValueError("training dataset is empty")
    task = data.tasks[0]
    support, heldout = holdout_demonstration(task)
    policy, tokenizer = build_model(cfg)
    wake_started = time.monotonic()
    episode, events = wake_episode(policy, tokenizer, cfg, support)
    wake_seconds = time.monotonic() - wake_started
    memory_delta = None
    if cfg.mode in ("sleep", "consolidation"):
        before_bases = [
            p
            for chain in iter_memory_chains(policy)
            for stage in chain.stages
            for p in stage.base.parameters()
        ]
        after_bases = [
            p
            for chain in iter_memory_chains(episode)
            for stage in chain.stages
            for p in stage.base.parameters()
        ]
        memory_delta = _delta(before_bases, after_bases)
        _check_update(memory_delta, "memory base")
        del before_bases, after_bases
    engine = CausalEngine(episode, tokenizer, cfg, device)
    directive_started = time.monotonic()
    with dream_routing(episode, cfg.mode == "sleep"):
        raw_directive = engine.generate_directive(support)
    directive_seconds = time.monotonic() - directive_started
    directive_error = None
    try:
        sampled = parse_directive(raw_directive)
    except (ValueError, TypeError) as error:
        sampled = None
        directive_error = str(error)

    fixture = DreamDirective()
    inner_parameters = train_adapter(episode, "inner")
    inner_before = _snapshot(inner_parameters)
    adaptation_started = time.monotonic()
    inner_losses = engine.sft(
        examples_for(support, fixture, cfg=cfg),
        adapter="inner",
        epochs=fixture.epochs,
        lr=cfg.inner_lr,
    )
    adaptation_seconds = time.monotonic() - adaptation_started
    inner_delta = _delta(inner_before, inner_parameters)
    _check_update(inner_delta, "inner")
    del inner_before
    # Scoring uses only this held-out training demonstration, never final targets.
    prediction_started = time.monotonic()
    validation_score = engine.score_demonstration(support, heldout)
    prediction_seconds = time.monotonic() - prediction_started
    del engine, episode, inner_parameters

    outer_parameters = train_adapter(policy, "policy")
    outer_before = _snapshot(outer_parameters)
    optimizer = torch.optim.AdamW(outer_parameters, lr=cfg.outer_lr)
    outer_engine = CausalEngine(policy, tokenizer, cfg, device)
    outer_losses = outer_engine.sft(
        [(outer_engine.directive_prompt(support), fixture.to_json())],
        adapter="policy",
        epochs=cfg.outer_epochs,
        optimizer=optimizer,
    )
    outer_delta = _delta(outer_before, outer_parameters)
    _check_update(outer_delta, "outer")
    del outer_before
    if not all(math.isfinite(loss) for loss in (*inner_losses, *outer_losses)):
        raise FloatingPointError("preflight losses must be finite")

    checkpoint = output / "fixture_checkpoint.pt"
    # Zero accepted outer updates prevents treating this forced fixture as a
    # genuinely trained Sleep/self-edit checkpoint in later evaluation.
    save_checkpoint(
        checkpoint, policy, optimizer, cfg, 1, outer_updates=0, purpose="diagnostic"
    )
    expected_rng = _rng_probe(device)
    one_parameter = outer_parameters[0]
    expected_parameter = one_parameter.detach().clone()
    with torch.no_grad():
        one_parameter.add_(1)
    restored = load_checkpoint(
        checkpoint, policy, optimizer, cfg, allow_diagnostic=True
    )
    if expected_rng != _rng_probe(device) or not torch.equal(
        one_parameter, expected_parameter
    ):
        raise RuntimeError("checkpoint model/RNG continuation failed")
    if restored != {"next_episode": 1, "outer_updates": 0}:
        raise RuntimeError("checkpoint fixture metadata did not resume")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        **evidence(cfg, policy, data),
        "kind": "one_training_task_execution_preflight",
        "passed": True,
        "gpu_validation": device.type == "cuda" and cfg.model != "tiny",
        "paper_result_reproduced": False,
        "task_id": task.task_id,
        "support_demonstration_count": len(support.demonstrations),
        "heldout_source": "visible demonstration of a training task",
        "wake_events": events,
        "memory_base_delta_l1": memory_delta,
        "generated_directive": raw_directive,
        "generated_directive_valid": sampled is not None,
        "generated_directive_error": directive_error,
        "generated_directive_hash": content_hash(raw_directive),
        "fixture_update": {
            "directive": fixture.to_json(),
            "forced": True,
            "accepted_policy_update": False,
            "candidate_count": 1,
            "purpose": "disposable optimizer/adapter correctness fixture",
            "expanded_example_hash": content_hash(
                [asdict(p) for p in expand_directive(support, fixture)]
            ),
        },
        "inner_losses": inner_losses,
        "outer_losses": outer_losses,
        "inner_parameter_delta_l1": inner_delta,
        "outer_parameter_delta_l1": outer_delta,
        "heldout_training_validation_score": validation_score,
        "checkpoint_resume_verified": True,
        "fixture_checkpoint": str(checkpoint),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "elapsed_seconds": time.monotonic() - started,
        "stage_wall_seconds": {
            "wake_and_consolidation": wake_seconds,
            "one_directive_generation": directive_seconds,
            "fixture_inner_sft": adaptation_seconds,
            "one_heldout_prediction": prediction_seconds,
        },
        "scope": "one training task; full configured wake; one generated directive; forced inner/outer fixture; no final evaluation",
    }
    write_json(output / "preflight.json", report)
    return report
