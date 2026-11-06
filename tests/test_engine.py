import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sleep_repro.arc import Pair, Task
from sleep_repro.engine import CausalEngine
from sleep_repro.model import iter_memory_chains, wrap_llama_mlps
from sleep_repro.runtime import (
    ByteTokenizer,
    ExperimentConfig,
    attach_lora,
    encode_batch,
    train_adapter,
)


def build_engine(**overrides):
    torch.manual_seed(3)
    cfg = ExperimentConfig(
        max_length=512,
        max_new_tokens=2,
        policy_rank=2,
        inner_rank=2,
        inner_lr=0.03,
        outer_lr=0.03,
        **overrides,
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=259,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    attach_lora(model, cfg)
    return CausalEngine(model, ByteTokenizer(), cfg, "cpu")


def test_sft_uses_masked_causal_labels_and_only_selected_adapter_changes():
    engine = build_engine()
    batch = encode_batch(engine.tokenizer, [("input:", "answer")], "cpu", 512)
    with torch.no_grad():
        expected = engine.model(**batch).loss.item()
    before = {name: p.detach().clone() for name, p in engine.model.named_parameters()}
    labels_seen = []

    def observe(module, args, kwargs):
        labels_seen.append(kwargs["labels"].clone())

    hook = engine.model.register_forward_pre_hook(observe, with_kwargs=True)
    losses = engine.sft([("input:", "answer")], epochs=3)
    hook.remove()
    assert losses[0] == pytest.approx(expected)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert labels_seen[0][0, :7].tolist() == [-100] * 7
    changed = [
        name
        for name, p in engine.model.named_parameters()
        if not torch.equal(before[name], p)
    ]
    assert changed and all("inner_" in name for name in changed)
    assert losses[-1] < losses[0]


def test_generation_preserves_modes_and_state_and_enforces_token_budget():
    engine = build_engine()
    engine.model.train()
    engine.model.model.layers[0].self_attn.eval()
    before = copy.deepcopy(engine.model.state_dict())
    modes = [module.training for module in engine.model.modules()]
    seen = []

    generate = engine.model.generate

    def observe(**kwargs):
        seen.append(kwargs["attention_mask"].clone())
        return generate(**kwargs)

    engine.model.generate = observe
    output = engine.generate("hello", sample=False)
    engine.model.generate = generate
    assert isinstance(output, str)
    assert seen[0].tolist() == [[1] * 6]
    assert modes == [module.training for module in engine.model.modules()]
    assert all(
        torch.equal(value, engine.model.state_dict()[name])
        for name, value in before.items()
    )
    with pytest.raises(ValueError, match="length|context|budget"):
        engine.generate("x" * 510, max_new_tokens=3)
    with pytest.raises(ValueError):
        engine.generate("x", max_new_tokens=0)


def test_generation_preserves_all_checkpoint_stop_tokens():
    engine = build_engine()
    engine.model.generation_config.eos_token_id = [2, 3]
    observed = []

    def generate(**kwargs):
        observed.append(
            kwargs.get("eos_token_id", engine.model.generation_config.eos_token_id)
        )
        return torch.cat([kwargs["input_ids"], torch.tensor([[2]])], dim=1)

    engine.model.generate = generate
    engine.generate("p", sample=False)
    assert observed == [[2, 3]]


def test_sft_explicit_learning_rate_updates_supplied_optimizer_groups():
    engine = build_engine()
    parameters = train_adapter(engine.model, "inner")
    optimizer = torch.optim.SGD(parameters, lr=0.0)
    before = {name: p.detach().clone() for name, p in engine.model.named_parameters()}
    engine.sft([("p", "x")], lr=0.05, optimizer=optimizer)
    assert all(group["lr"] == 0.05 for group in optimizer.param_groups)
    assert any(
        not torch.equal(before[name], p) for name, p in engine.model.named_parameters()
    )


def test_directive_and_prediction_prompts_never_use_heldout_target():
    engine = build_engine()
    visible = Pair(((1,),), ((2,),))
    heldout = Pair(((3,),), ((9, 8, 7),))
    task = Task("t", (visible,), (((4,),),))
    prompt = engine.directive_prompt(task)
    assert "rotations" in prompt and "color_permutation" in prompt
    assert "[[9,8,7]]" not in prompt
    seen = []

    def generate(prompt, max_new_tokens=None, sample=True):
        seen.append((prompt, sample))
        return "[[9,8,7]]"

    engine.generate = generate
    assert engine.score_demonstration(task, heldout) == 1.0
    assert "[[3]]" in seen[0][0] and "[[9,8,7]]" not in seen[0][0]
    assert seen[0][1] is False
    assert engine.predict(task) == ((((9, 8, 7),)),)
    engine.generate = lambda *args, **kwargs: "invalid"
    assert engine.predict(task) == (None,)
    assert engine.last_prediction_texts == ["invalid"]


def test_knowledge_objective_is_finite_teacher_frozen_and_receiver_only_gradients():
    engine = build_engine()
    wrap_llama_mlps(engine.model, periods=(1, 4), capacity=2, rank=2)
    teacher = copy.deepcopy(engine.model).requires_grad_(False)
    student = copy.deepcopy(engine.model).requires_grad_(False)
    receiver = next(iter_memory_chains(student)).stages[1]
    index = receiver.allocate_expert()
    receiver.forced_expert.fill_(index)
    receiver.experts[index].requires_grad_(True)
    teacher_before = copy.deepcopy(teacher.state_dict())
    student_before = copy.deepcopy(student.state_dict())
    modes = [module.training for module in student.modules()]
    loss = engine.knowledge_objective(teacher, student, [("p:", "x")])
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    nonzero = [
        name
        for name, param in student.named_parameters()
        if param.grad is not None and torch.count_nonzero(param.grad)
    ]
    assert nonzero and all("stages.1.experts.0" in name for name in nonzero)
    assert all(param.grad is None for param in teacher.parameters())
    assert all(
        torch.equal(value, teacher.state_dict()[name])
        for name, value in teacher_before.items()
    )
    assert all(
        torch.equal(value, student.state_dict()[name])
        for name, value in student_before.items()
    )
    assert modes == [module.training for module in student.modules()]


def test_microbatch_accumulation_matches_full_batch_update():
    import copy
    from sleep_repro.runner import build_model
    from sleep_repro.runtime import ExperimentConfig, seed_all

    seed_all(12)
    cfg = ExperimentConfig.from_dict(
        dict(
            model="tiny",
            device="cpu",
            dtype="float32",
            mode="ttt",
            policy_rank=2,
            inner_rank=2,
            microbatch_size=1,
        )
    )
    model, tok = build_model(cfg)
    other = copy.deepcopy(model)
    examples = [("Short:", "1"), ("A longer prompt:", "2345")]
    from sleep_repro.engine import CausalEngine

    CausalEngine(model, tok, cfg, "cpu").sft(examples)
    full = ExperimentConfig.from_dict({**cfg.to_dict(), "microbatch_size": 2})
    CausalEngine(other, tok, full, "cpu").sft(examples)
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)


def test_microbatch_importance_matches_full_batch_gradient_without_mutation():
    from sleep_repro.objectives import gradient_importance

    engine = build_engine()
    examples = [("Short:", "1"), ("A longer prompt:", "2345")]
    parameters = train_adapter(engine.model, "inner")
    batch = encode_batch(engine.tokenizer, examples, "cpu", 512)
    expected = gradient_importance(engine.model(**batch).loss, parameters)
    observed = engine.importance(examples)
    assert observed == pytest.approx(expected, rel=1e-5, abs=1e-7)
    assert all(p.grad is None for p in parameters)
