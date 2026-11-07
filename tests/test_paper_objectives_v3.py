import math

import pytest
import torch

from sleep_repro.objectives import student_ks_loss, token_edit_reward


def test_equation_four_matches_pinned_llama_grid_tokens_at_inclusive_cutoff():
    # Pinned Llama 3.2 1B Instruct tokenizer, without special tokens.
    prediction = [15873, 16, 11, 17, 5163]  # [[1,2]]
    reference = [15873, 16, 11, 18, 5163]  # [[1,3]]
    assert token_edit_reward(prediction, reference, threshold=1) == 4 / 5
    assert token_edit_reward(prediction, reference, threshold=0) == 0


@pytest.mark.parametrize(
    "prediction,reference,threshold,expected",
    [
        ([], [], 0, 1),
        ([9], [], 1, 0),
        ([2, 4, 5], [2, 4, 5], 0, 1),
        ([3, 4], [2, 3, 4], 1, 2 / 3),
        ([3, 4, 5], [4, 5, 6], 2, 1 / 3),
        ([3, 4, 5], [4, 5, 6], 1, 0),
    ],
)
def test_edit_reward_is_symmetric_and_covers_insertions_and_deletions(
    prediction, reference, threshold, expected
):
    assert token_edit_reward(
        prediction, reference, threshold=threshold
    ) == pytest.approx(expected)
    assert token_edit_reward(
        reference, prediction, threshold=threshold
    ) == pytest.approx(expected)


def test_edit_reward_requires_explicit_token_ids_and_integer_threshold():
    with pytest.raises(ValueError, match="token IDs"):
        token_edit_reward("12", [12], threshold=2)
    with pytest.raises(ValueError, match="threshold"):
        token_edit_reward([1], [1], threshold=-1)


def test_student_ks_matches_manual_conditional_kl_score_function_and_gradient():
    # At T=2, p_teacher=(3/4,1/4), p_student=(1/3,2/3).
    # Rollout 1 has one sampled action (1); rollout 2 has two (0,1).
    student = (
        torch.tensor([0.0, 2 * math.log(2)], dtype=torch.float64)
        .repeat(2, 4, 1)
        .requires_grad_()
    )
    teacher = (
        torch.tensor([2 * math.log(3), 0.0], dtype=torch.float64)
        .repeat(2, 4, 1)
        .requires_grad_()
    )
    labels = torch.tensor([[-100, -100, 1, -100], [-100, 0, 1, -100]])
    rewards = torch.tensor([1.0, 0.25], dtype=torch.float64, requires_grad=True)
    alpha = 0.4
    loss = student_ks_loss(
        student, teacher, labels, rewards, alpha=alpha, temperature=2
    )
    kl = 0.75 * math.log(0.75 / (1 / 3)) + 0.25 * math.log(0.25 / (2 / 3))
    reward_logp = (math.log(2 / 3) + 0.25 * math.log((1 / 3) * (2 / 3))) / 2
    assert loss.item() == pytest.approx(alpha * kl - (1 - alpha) * reward_logp)
    loss.backward()
    assert teacher.grad is None and rewards.grad is None
    assert student.grad[0, 1, 0].item() == pytest.approx(
        (alpha * (1 / 3 - 0.75) + (1 - alpha) * (1 / 3)) / 4
    )
    assert student.grad[1, 0, 0].item() == pytest.approx(
        (alpha * (1 / 3 - 0.75) / 2 + (1 - alpha) * 0.25 * (1 / 3 - 1)) / 4
    )
    assert torch.count_nonzero(student.grad[0, [0, 2, 3]]) == 0
    assert torch.count_nonzero(student.grad[1, 2:]) == 0


def test_student_ks_rejects_rollout_with_no_sampled_actions():
    logits = torch.zeros(2, 3, 2)
    labels = torch.tensor([[-100, 1, 0], [-100, -100, -100]])
    with pytest.raises(ValueError, match="sampled action"):
        student_ks_loss(
            logits, logits, labels, torch.zeros(2), alpha=0.5, temperature=0.8
        )
