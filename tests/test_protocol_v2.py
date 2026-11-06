import copy

import pytest
import torch

from sleep_repro.arc import DreamDirective, Pair, Task
from sleep_repro.engine import CausalEngine
from sleep_repro.runner import build_model, examples_for
from sleep_repro.runtime import ByteTokenizer, ExperimentConfig, encode_batch, seed_all
import sleep_repro.runtime as runtime


class ChatTokenizer(ByteTokenizer):
    chat_template = "test-only template"

    def apply_chat_template(
        self, messages, *, tokenize, return_dict, add_generation_prompt, date_string
    ):
        assert tokenize and add_generation_prompt and return_dict is False
        assert date_string == "26 Jul 2024"
        assert messages[0]["role"] == "user"
        return (
            [1, 42]
            + self.encode(messages[0]["content"], add_special_tokens=False)
            + [43]
        )


def config(**overrides):
    return ExperimentConfig.from_dict(
        dict(
            model="tiny",
            mode="ttt",
            device="cpu",
            dtype="float32",
            policy_rank=2,
            inner_rank=2,
            max_length=2048,
            max_new_tokens=2,
            **overrides,
        )
    )


def test_chat_batch_masks_assistant_prefix_without_duplicate_bos():
    tok = ChatTokenizer()
    prefix = runtime.encode_prompt(
        tok, "query", "hf-chat-v1", assistant_prefix="already given:"
    )
    batch = encode_batch(
        tok,
        [("query", "new answer", "already given:")],
        "cpu",
        2048,
        prompt_format="hf-chat-v1",
    )
    assert batch["input_ids"][0, : len(prefix)].tolist() == prefix
    assert prefix.count(tok.bos_token_id) == 1
    assert (batch["labels"][0, : len(prefix)] == -100).all()
    assert batch["labels"][0, len(prefix) :].tolist() == tok.encode(
        "new answer", False
    ) + [tok.eos_token_id]


def test_generation_and_sft_share_chat_prefix():
    cfg = config(prompt_format="hf-chat-v1")
    model, _ = build_model(cfg)
    tok = ChatTokenizer()
    captured = []

    def generate(**kwargs):
        captured.append(kwargs["input_ids"][0].tolist())
        return torch.cat((kwargs["input_ids"], torch.tensor([[2]])), dim=1)

    model.generate = generate
    CausalEngine(model, tok, cfg, "cpu").generate("query", sample=False)
    assert captured == [runtime.encode_prompt(tok, "query", "hf-chat-v1")]


def test_minibatch_update_budget_counts_optimizer_steps_not_epochs():
    cfg = config(
        sft_update_mode="minibatch-v2", effective_batch_size=2, microbatch_size=1
    )
    model, tok = build_model(cfg)
    examples = [(f"q{i}", str(i)) for i in range(5)]
    assert len(CausalEngine(model, tok, cfg, "cpu").sft(examples, epochs=2)) == 6
    limited = config(
        sft_update_mode="minibatch-v2", effective_batch_size=2, sft_max_steps=5
    )
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="step"):
        CausalEngine(model, tok, limited, "cpu").sft(examples, epochs=2)
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before.items()
    )


def test_contextual_training_uses_only_other_visible_demonstrations():
    task = Task(
        "visible",
        (Pair(((1,),), ((6,),)), Pair(((2,),), ((7,),)), Pair(((3,),), ((8,),))),
        (((9,),),),
    )
    cfg = config(contextual_examples=True)
    examples = examples_for(task, DreamDirective(), cfg=cfg)
    assert len(examples) == 3
    for prompt, completion in examples:
        assert completion not in prompt
        assert sum(target in prompt for target in ("[[6]]", "[[7]]", "[[8]]")) == 2
        assert "[[9]]" not in prompt


def test_minibatch_physical_batch_size_does_not_change_updates():
    seed_all(16)
    cfg = config(
        sft_update_mode="minibatch-v2", effective_batch_size=2, microbatch_size=1
    )
    model, tok = build_model(cfg)
    other = copy.deepcopy(model)
    examples = [("short", "1"), ("a longer prompt", "234"), ("last", "56")]
    seed_all(22)
    CausalEngine(model, tok, cfg, "cpu").sft(examples)
    full = ExperimentConfig.from_dict({**cfg.to_dict(), "microbatch_size": 2})
    seed_all(22)
    CausalEngine(other, tok, full, "cpu").sft(examples)
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-5)


def test_invalid_later_example_cannot_partially_update_policy():
    cfg = config(sft_update_mode="minibatch-v2", outer_effective_batch_size=1)
    model, tok = build_model(cfg)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="length"):
        CausalEngine(model, tok, cfg, "cpu").sft(
            [("q", "1"), ("x" * 2048, "2")], adapter="policy"
        )
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before.items()
    )


def test_contextual_augmentations_keep_same_transform_in_each_group():
    task = Task(
        "visible", (Pair(((1, 2),), ((6, 7),)), Pair(((3, 4),), ((8, 9),))), (((0,),),)
    )
    examples = examples_for(
        task, DreamDirective(rotations=(0, 1)), cfg=config(contextual_examples=True)
    )
    assert len(examples) == 4
    assert "[[3,4]]" in examples[0][0]
    assert "[[3],[4]]" in examples[2][0]
    assert "[[3,4]]" not in examples[2][0]
    assert examples[0][1] == "[[6,7]]"
    assert examples[2][1] == "[[6],[7]]"


def test_contextual_single_visible_pair_falls_back_without_restoring_heldout_target():
    from sleep_repro.arc import holdout_demonstration

    task = Task(
        "two-demonstration-task",
        (Pair(((1,),), ((2,),)), Pair(((3,),), ((9, 8, 7),))),
        (((4,),),),
    )
    support, heldout = holdout_demonstration(task)
    examples = examples_for(
        support, DreamDirective(), cfg=config(contextual_examples=True)
    )
    assert len(examples) == 1
    prompt, completion = examples[0]
    assert "[[1]]" in prompt and completion == "[[2]]"
    assert "[[9,8,7]]" not in prompt + completion
    assert heldout not in support.demonstrations


def test_chat_on_policy_ks_masks_assistant_prefix_and_trains_only_ffn_receiver_expert():
    from sleep_repro.model import iter_memory_chains

    cfg = ExperimentConfig.from_dict(
        {
            **config(prompt_format="hf-chat-v1").to_dict(),
            "mode": "sleep",
            "rank": 2,
            "capacity": 2,
            "max_new_tokens": 4,
            "on_policy_fraction": 1.0,
            "lora_targets": ("q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"),
        }
    )
    seed_all(19)
    model, _ = build_model(cfg)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    student = copy.deepcopy(model).eval().requires_grad_(False)
    receiver = next(iter_memory_chains(student)).stages[1]
    index = receiver.allocate_expert()
    receiver.mark_trained(index)
    receiver.experts[index].requires_grad_(True)
    tok = ChatTokenizer()
    engine = CausalEngine(model, tok, cfg, "cpu")
    teacher_before = copy.deepcopy(teacher.state_dict())
    student_before = copy.deepcopy(student.state_dict())
    student_generation_inputs = []
    teacher_generation_count = 0

    # Control only the detached sampling boundary. Actual model forwards,
    # chat serialization, semantic/edit reward, KL/LTI, and backward stay real.
    def teacher_generate(**kwargs):
        nonlocal teacher_generation_count
        text = "AB" if teacher_generation_count == 0 else "YES"
        teacher_generation_count += 1
        suffix = torch.tensor([tok.encode(text, add_special_tokens=False)])
        assert suffix.shape[1] <= kwargs["max_new_tokens"]
        return torch.cat((kwargs["input_ids"], suffix), dim=1)

    def student_generate(**kwargs):
        student_generation_inputs.append(kwargs["input_ids"].clone())
        return torch.cat(
            (kwargs["input_ids"], torch.tensor([tok.encode("B", False)])), dim=1
        )

    teacher.generate = teacher_generate
    student.generate = student_generate
    logits_seen = []
    inputs_seen = []

    def capture_input(module, args, kwargs):
        inputs_seen.append(kwargs["input_ids"].clone())

    def capture_logits(module, args, output):
        output.logits.retain_grad()
        logits_seen.append(output.logits)

    input_hook = student.register_forward_pre_hook(capture_input, with_kwargs=True)
    logits_hook = student.register_forward_hook(capture_logits)
    # First randint(2) chooses prefix length one: supplied A, sampled B.
    torch.manual_seed(1)
    loss = engine.knowledge_objective(
        teacher, student, [("query", "unused visible target")]
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    input_hook.remove()
    logits_hook.remove()

    expected_prefix = (
        [1, 42] + tok.encode("query", False) + [43] + tok.encode("A", False)
    )
    assert len(student_generation_inputs) == 1
    assert student_generation_inputs[0].tolist() == [expected_prefix]
    assert inputs_seen[0].tolist() == [
        expected_prefix + tok.encode("B", False) + [tok.eos_token_id]
    ]
    # Prompt plus fixed assistant prefix are never supervised by KS/LTI.
    gradient = logits_seen[0].grad
    assert torch.count_nonzero(gradient[:, : len(expected_prefix) - 1]) == 0
    assert torch.count_nonzero(gradient[:, len(expected_prefix) - 1 : -1]) > 0
    nonzero = [
        name
        for name, parameter in student.named_parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad)
    ]
    assert nonzero and all("stages.1.experts.0." in name for name in nonzero)
    assert not any(parameter.grad is not None for parameter in teacher.parameters())
    for name, parameter in student.named_parameters():
        if name.endswith(("policy_a", "policy_b", "inner_a", "inner_b")):
            assert parameter.grad is None
    assert teacher_generation_count == 2  # reference and frozen semantic verdict
    assert all(
        torch.equal(value, teacher.state_dict()[name])
        for name, value in teacher_before.items()
    )
    assert all(
        torch.equal(value, student.state_dict()[name])
        for name, value in student_before.items()
    )
