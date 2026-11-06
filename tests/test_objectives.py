import math

import pytest
import torch

from sleep_repro.objectives import (
    composite_rewards,
    gradient_importance,
    lti_loss,
    masked_causal_kl,
    sequence_log_probs,
)


def test_forward_kl_is_teacher_to_student_causal_masked_and_teacher_frozen():
    student = torch.zeros(1, 3, 2, requires_grad=True)
    teacher = torch.tensor(
        [[[math.log(3), 0.0], [90.0, -90.0], [0.0, 0.0]]], requires_grad=True
    )
    labels = torch.tensor([[-100, 0, -100]])
    loss = masked_causal_kl(student, teacher, labels)
    assert loss.item() == pytest.approx(0.13081204)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[0, 0].tolist() == pytest.approx([-0.25, 0.25])
    assert torch.count_nonzero(student.grad[0, 1:]) == 0


def test_kl_temperature_uses_squared_temperature_scaling():
    student = torch.zeros(1, 2, 2)
    teacher = torch.tensor([[[2 * math.log(3), 0.0], [0.0, 0.0]]])
    assert masked_causal_kl(
        student, teacher, torch.tensor([[-100, 0]]), temperature=2
    ).item() == pytest.approx(4 * 0.13081204)


def test_sequence_log_probs_sum_tokens_and_keep_policy_gradient():
    logits = torch.zeros(2, 3, 2, requires_grad=True)
    labels = torch.tensor([[-100, 1, 0], [-100, -100, 1]])
    actual = sequence_log_probs(logits, labels)
    assert actual.tolist() == pytest.approx([-2 * math.log(2), -math.log(2)])
    assert sequence_log_probs(logits, labels, normalize=True).tolist() == pytest.approx(
        [-math.log(2)] * 2
    )
    actual.sum().backward()
    assert logits.grad[0, 0].tolist() == pytest.approx([-0.5, 0.5])
    assert torch.count_nonzero(logits.grad[1, 0]) == 0


def test_lti_positive_reward_increases_sample_probability_without_reward_gradient():
    student = torch.zeros(1, 2, 2, requires_grad=True)
    reward = torch.tensor([2.0], requires_grad=True)
    loss = lti_loss(
        student, torch.zeros_like(student), torch.tensor([[-100, 1]]), reward, alpha=0
    )
    loss.backward()
    assert student.grad[0, 0].tolist() == pytest.approx([1.0, -1.0])
    assert reward.grad is None


def test_composite_reward_combines_semantics_with_normalized_levenshtein():
    reward = composite_rewards(
        ["cat", "", "kitten"],
        ["cut", "", "sitting"],
        lambda pred, ref: [0.2, 0.8, 0.0],
        semantic_weight=0.25,
    )
    assert reward.tolist() == pytest.approx([0.55, 0.95, 3 / 7])
    assert not reward.requires_grad


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -0.1},
        {"alpha": float("nan")},
        {"temperature": 0},
        {"temperature": float("inf")},
    ],
)
def test_invalid_lti_coefficients_are_rejected(kwargs):
    logits = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError):
        lti_loss(logits, logits, torch.tensor([[-100, 1]]), torch.ones(1), **kwargs)


def test_empty_and_nonfinite_training_signals_are_rejected():
    logits = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="valid"):
        masked_causal_kl(logits, logits, torch.full((1, 2), -100))
    with pytest.raises(ValueError, match="finite"):
        lti_loss(
            logits, logits, torch.tensor([[-100, 1]]), torch.tensor([float("nan")])
        )
    with pytest.raises(ValueError):
        lti_loss(logits, logits, torch.tensor([[-100, 1]]), torch.ones(2))


@pytest.mark.parametrize(
    "pred,ref,scorer,weight",
    [
        (["a"], [], lambda a, b: [1.0], 0.5),
        (["a"], ["b"], lambda a, b: [], 0.5),
        (["a"], ["b"], lambda a, b: [float("nan")], 0.5),
        (["a"], ["b"], lambda a, b: [1.0], 2),
    ],
)
def test_reward_shape_finiteness_and_weight_are_validated(pred, ref, scorer, weight):
    with pytest.raises(ValueError):
        composite_rewards(pred, ref, scorer, semantic_weight=weight)


def test_gradient_importance_uses_requested_parameters_without_mutating_grads():
    weight = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    unused = torch.nn.Parameter(torch.tensor(2.0))
    weight.grad = torch.tensor([8.0, 9.0])
    loss = (weight.square() / 2).sum()
    assert gradient_importance(loss, [weight, unused]) == pytest.approx(5.0)
    assert weight.grad.tolist() == [8.0, 9.0]
    assert unused.grad is None
    loss.backward()
    assert weight.grad.tolist() == [11.0, 13.0]
