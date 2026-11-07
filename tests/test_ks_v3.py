import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sleep_repro.engine import CausalEngine
from sleep_repro.runtime import ByteTokenizer, ExperimentConfig


@pytest.mark.parametrize(
    "sample_stop", [False, True], ids=["length-limit", "alternate-eos"]
)
def test_student_ks_scores_the_actions_from_its_actual_sampling_policy(
    monkeypatch, sample_stop
):
    torch.manual_seed(12)
    cfg = ExperimentConfig(
        max_length=64,
        max_new_tokens=4,
        temperature=0.7,
        ks_protocol="student-tokens-v3",
        edit_distance_threshold=1,
        on_policy_fraction=1.0,
        ks_alpha=0.3,
        semantic_weight=0.25,
    )
    model_config = LlamaConfig(
        vocab_size=259,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    student = LlamaForCausalLM(model_config)
    teacher = LlamaForCausalLM(copy.deepcopy(model_config))
    for model in (teacher, student):
        model.generation_config.eos_token_id = [2, 4]
        # The pinned model's decoding defaults must not change the KS policy.
        model.generation_config.top_k = 1
        model.generation_config.top_p = 0.01
    tokenizer = ByteTokenizer()
    prompt = "p"
    prompt_ids = tokenizer.encode(prompt)
    engine = CausalEngine(student, tokenizer, cfg, "cpu")

    # Force one reproducible trajectory while inspecting the probabilities
    # passed to multinomial on every real cached model forward.
    teacher_actions = [94, 94, 66, 4]
    student_actions = [94, 66, 4] if sample_stop else [94, 67, 68]
    draws = iter(teacher_actions + student_actions)
    distributions = []

    def sample(distribution, num_samples):
        assert num_samples == 1 and not distribution.requires_grad
        assert torch.all(distribution > 0)
        distributions.append(distribution.detach().clone())
        return torch.tensor([[next(draws)]], device=distribution.device)

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(torch, "randint", lambda high, size: torch.tensor(1))
    # The original tokenization of the exact teacher prefix and sampled
    # continuation must survive even when a text round-trip is lossy.
    encode_calls = []

    def encode_once(text, add_special_tokens):
        encode_calls.append(text)
        assert text == prompt and add_special_tokens
        return prompt_ids

    monkeypatch.setattr(tokenizer, "encode", encode_once)
    judged = []

    def judge(frozen_model, predictions, references):
        assert frozen_model is teacher
        judged.append((predictions, references))
        return [1.0]

    monkeypatch.setattr(engine, "_semantic_scores", judge)
    loss = engine.knowledge_objective(
        teacher, student, [(prompt, "unused training target")]
    )
    assert encode_calls == [prompt]
    assert len(distributions) == 7
    student_content = student_actions[:-1] if sample_stop else student_actions
    prefix = teacher_actions[:1]
    assert judged == [
        (
            [tokenizer.decode(prefix + student_content)],
            [tokenizer.decode(teacher_actions[:-1])],
        )
    ]

    # Independent estimator from the probabilities that generated the
    # student's three actions (including a real stop only in the EOS case).
    with torch.no_grad():
        input_ids = torch.tensor([prompt_ids + prefix + student_actions])
        teacher.eval()
        teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits
        teacher_conditional = (teacher_logits[0, -4:-1] / cfg.temperature).softmax(-1)
        student_conditional = torch.cat(distributions[-3:])
        kl = (
            (
                teacher_conditional
                * (teacher_conditional.log() - student_conditional.log())
            )
            .sum(-1)
            .mean()
        )
        logp = student_conditional[range(3), student_actions].log().sum()
        # With a stop the full dreams match. Otherwise the two edits exceed z0.
        reward = 1.0 if sample_stop else cfg.semantic_weight
        expected = cfg.ks_alpha * kl - (1 - cfg.ks_alpha) * reward * logp
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert student.lm_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
