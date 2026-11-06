"""Ownership boundaries for the explicit attention-and-FFN LoRA variant."""
import torch

from sleep_repro.consolidation import consolidate
from sleep_repro.engine import CausalEngine
from sleep_repro.model import configure_wake_training, iter_memory_chains, record_wake
from sleep_repro.runner import build_model
from sleep_repro.runtime import ExperimentConfig, LoRALinear, encode_batch, seed_all


TARGETS = ('q_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj')


def config(**overrides):
    values = dict(model='tiny', device='cpu', dtype='float32', mode='sleep', rank=2,
                  capacity=4, policy_rank=2, inner_rank=2, lora_targets=TARGETS,
                  max_length=512, max_new_tokens=2, train_dreams=2, top_k=1, random_k=1)
    values.update(overrides)
    return ExperimentConfig.from_dict(values)


def adapter_values(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if name.endswith(('policy_a', 'policy_b', 'inner_a', 'inner_b'))}


def test_attention_ffn_lora_and_cms_conversion_preserve_initial_logits_and_buffer_shapes():
    seed_all(31)
    control, _ = build_model(config(mode='icl', lora_targets=('q_proj', 'v_proj')))
    ids = torch.tensor([[1, 4, 5, 2]])
    with torch.no_grad():
        expected = control(ids, use_cache=False).logits
    seed_all(31)
    model, _ = build_model(config())
    stage = next(iter_memory_chains(model)).stages[0]
    for name in ('gate_proj', 'down_proj', 'up_proj'):
        layer = getattr(stage.base, name)
        assert isinstance(layer, LoRALinear)
        assert layer.in_features == layer.base.in_features
        assert layer.out_features == layer.base.out_features
    for parameter, accumulated in zip(stage.base.parameters(), stage.accumulated_gradients(), strict=True):
        assert parameter.shape == accumulated.shape
    with torch.no_grad():
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, rtol=0, atol=0)


def test_wake_and_receiver_only_consolidation_preserve_every_policy_and_inner_adapter():
    seed_all(11)
    model, tokenizer = build_model(config())
    # Existing nonzero adapter functions must remain constant through wake SGD.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(('policy_b', 'inner_b')):
                parameter.fill_(0.01)
    before = adapter_values(model)
    configure_wake_training(model)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name in before)
    stage = next(iter_memory_chains(model)).stages[0]
    assert stage.base.gate_proj.base.weight.requires_grad
    model.zero_grad(set_to_none=True)
    batch = encode_batch(tokenizer, [('Input:', '[[1]]')], 'cpu', 512)
    model(**batch, use_cache=False).loss.backward()
    assert record_wake(model) == (0,)
    for (name, parameter), accumulated in zip(stage.base.named_parameters(), stage.accumulated_gradients(), strict=True):
        if name.endswith(('policy_a', 'policy_b', 'inner_a', 'inner_b')):
            assert parameter.grad is None
            assert torch.count_nonzero(accumulated) == 0
    model.zero_grad(set_to_none=True)
    observations = []
    def objective(teacher, student):
        assert not any(parameter.requires_grad for parameter in teacher.parameters())
        trainable = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
        assert trainable and all('stages.1.experts.0.' in name for name in trainable)
        observations.append(True)
        return (student(batch['input_ids'], use_cache=False).logits -
                teacher(batch['input_ids'], use_cache=False).logits).square().mean()
    result = consolidate(model, 0, objective, base_lr=0.03, expert_lr=0.01, steps=3)
    assert result.committed and observations
    for name, parameter in model.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_inner_sft_updates_ffn_inner_branches_and_preserves_backbone_and_policy():
    seed_all(5)
    cfg = config()
    model, tokenizer = build_model(cfg)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    CausalEngine(model, tokenizer, cfg, 'cpu').sft([('Input:', '[[2]]')], adapter='inner', epochs=2, lr=0.02)
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])]
    assert changed and all('inner_' in name for name in changed)
    for target in ('gate_proj', 'down_proj', 'up_proj'):
        assert any(f'.{target}.inner_b' in name for name in changed)
