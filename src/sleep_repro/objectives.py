"""Explicit causal distillation, LTI rewards, and dream-gradient statistics.

Callers mix teacher-generated (off-policy) and student-generated (on-policy)
rollouts in the batch. Sampling and semantic model freezing belong to the
caller; neither sampling nor reward computation carries a policy gradient.
"""

from collections.abc import Callable, Iterable, Sequence
import math

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F


def _coefficient(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _causal_tokens(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError(
            "logits must be [batch, sequence, vocabulary] and labels [batch, sequence]"
        )
    if not logits.is_floating_point() or labels.dtype != torch.long:
        raise ValueError("logits must be floating point and labels must be int64")
    if logits.device != labels.device:
        raise ValueError("logits and labels must have the same device")
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    if not valid.any():
        raise ValueError("batch must contain at least one valid causal target token")
    targets = shifted_labels[valid]
    if torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
        raise ValueError("target token outside vocabulary")
    selected = logits[:, :-1][valid]
    if not torch.isfinite(selected).all():
        raise ValueError("valid-token logits must be finite")
    # Accumulate low-precision language-model objectives in float32.
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    return selected, targets, valid


def masked_causal_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Token-mean KL(teacher || student), shifted and masked, scaled by T².

    Labels of -100 exclude prompt/padding positions. An entirely masked batch
    raises instead of silently reporting a successful zero-loss update.
    Teacher logits are detached even when their producer retained gradients.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    student, _, _ = _causal_tokens(student_logits, labels)
    teacher, _, _ = _causal_tokens(teacher_logits.detach(), labels)
    if student.device != teacher.device:
        raise ValueError("teacher and student logits must have the same device")
    student_logp = F.log_softmax(student / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher / temperature, dim=-1)
    loss = F.kl_div(student_logp, teacher_logp, log_target=True, reduction="sum")
    loss = loss / student.shape[0] * temperature**2
    if not torch.isfinite(loss):
        raise ValueError("KL loss must be finite")
    return loss


def sequence_log_probs(
    logits: Tensor, labels: Tensor, *, normalize: bool = False
) -> Tensor:
    """Differentiable per-sequence causal log likelihood; empty rows yield zero.

    The default sums token log probabilities for the sequence REINFORCE
    estimator. ``normalize=True`` returns a token mean for diagnostics/SFT.
    """
    selected, targets, valid = _causal_tokens(logits, labels)
    token_logp = F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
    rows = valid.nonzero(as_tuple=True)[0]
    result = selected.new_zeros(logits.shape[0]).scatter_add(0, rows, token_logp)
    if normalize:
        result = result / valid.sum(dim=1).clamp_min(1)
    if not torch.isfinite(result).all():
        raise ValueError("sequence log probabilities must be finite")
    return result


def lti_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    rewards: Tensor,
    *,
    alpha: float = 0.5,
    temperature: float = 1.0,
) -> Tensor:
    """KS/LTI: alpha * forward KL - (1-alpha) * mean(reward * log p).

    Rewards must be one scalar per sequence and are always detached. Rows
    without supervised targets do not enter the policy-gradient mean.
    """
    _coefficient(alpha, "alpha")
    if rewards.shape != (student_logits.shape[0],):
        raise ValueError("rewards must have shape [batch]")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    kl = masked_causal_kl(
        student_logits, teacher_logits, labels, temperature=temperature
    )
    logp = sequence_log_probs(student_logits, labels)
    valid_rows = (labels[:, 1:] != -100).any(dim=1)
    reward = rewards.detach().to(device=logp.device, dtype=logp.dtype)
    loss = alpha * kl - (1 - alpha) * (reward[valid_rows] * logp[valid_rows]).mean()
    if not torch.isfinite(loss):
        raise ValueError("LTI loss must be finite")
    return loss


def _edit_similarity(prediction: str, reference: str) -> float:
    previous = list(range(len(reference) + 1))
    for i, left in enumerate(prediction, start=1):
        current = [i]
        for j, right in enumerate(reference, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return 1 - previous[-1] / max(len(prediction), len(reference), 1)


def composite_rewards(
    predictions: Sequence[str],
    references: Sequence[str],
    semantic_scorer: Callable[[Sequence[str], Sequence[str]], Sequence[float] | Tensor],
    *,
    semantic_weight: float = 0.5,
) -> Tensor:
    """Combine frozen semantic scores and normalized character edit similarity.

    The scorer callback must be pinned and frozen by the caller. ``no_grad``
    and detaching prevent reward gradients; this function does not change a
    scorer model's training mode or parameter ``requires_grad`` flags.
    """
    _coefficient(semantic_weight, "semantic_weight")
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not all(isinstance(item, str) for item in (*predictions, *references)):
        raise ValueError("predictions and references must be strings")
    with torch.no_grad():
        semantic = torch.as_tensor(
            semantic_scorer(predictions, references), dtype=torch.float32
        ).detach()
    if semantic.shape != (len(predictions),):
        raise ValueError("semantic scorer must return one scalar per prediction")
    if not torch.isfinite(semantic).all():
        raise ValueError("semantic scores must be finite")
    edits = semantic.new_tensor(
        [_edit_similarity(p, r) for p, r in zip(predictions, references, strict=True)]
    )
    return semantic_weight * semantic + (1 - semantic_weight) * edits


def gradient_importance(
    loss: Tensor, parameters: Iterable[Parameter], *, norm_type: float = 2.0
) -> float:
    """Norm of SFT gradients over exactly the declared trainable parameters.

    Uses autograd.grad, preserving existing ``.grad`` buffers and retaining the
    graph so callers can subsequently backpropagate the SFT loss normally.
    Unused declared parameters contribute zero.
    """
    params = tuple(parameters)
    if not params or any(not param.requires_grad for param in params):
        raise ValueError("provide a nonempty set of trainable parameters")
    if len({id(param) for param in params}) != len(params):
        raise ValueError("parameters must be unique")
    if not math.isfinite(norm_type) or norm_type <= 0:
        raise ValueError("norm_type must be positive and finite")
    if loss.numel() != 1 or not torch.isfinite(loss):
        raise ValueError("loss must be a finite scalar")
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            if not torch.isfinite(grad).all():
                raise ValueError("importance gradients must be finite")
            total += grad.detach().double().abs().pow(norm_type).sum().item()
    result = total ** (1 / norm_type)
    if not math.isfinite(result):
        raise ValueError("gradient importance must be finite")
    return result
