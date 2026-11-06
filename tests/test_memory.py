import pytest
import torch
from torch import nn

from sleep_repro.memory import CapacityError, MemoryChain
from sleep_repro.model import (
    configure_wake_training,
    dream_routing,
    iter_memory_chains,
    wrap_llama_mlps,
)


def test_zero_output_levels_preserve_ffn_and_receive_useful_gradient():
    torch.manual_seed(3)
    base = nn.Linear(6, 6, bias=False)
    x = torch.randn(2, 4, 6)
    expected = base(x).detach()
    chain = MemoryChain(base, hidden_size=6, periods=(1, 4), capacity=3, rank=2)
    torch.testing.assert_close(chain(x), expected, rtol=0, atol=0)
    chain(x).square().mean().backward()
    assert chain.stages[1].base.up.weight.grad.abs().sum() > 0
    assert not chain.stages[0].active.any()


@pytest.mark.parametrize("periods", [(0, 4), (4, 1), (2, 3), (1, 1)])
def test_invalid_period_hierarchy_is_rejected(periods):
    with pytest.raises(ValueError, match="period"):
        MemoryChain(nn.Linear(3, 3), hidden_size=3, periods=periods)


def test_dream_routes_only_trained_unselected_experts_without_mutating_masks():
    chain = MemoryChain(
        nn.Identity(), hidden_size=3, periods=(1, 4), capacity=3, rank=2
    )
    stage = chain.stages[0]
    x = torch.ones(1, 2, 3)
    assert stage.route(x, dreaming=True).extra is None
    first = stage.allocate_expert()
    stage.mark_trained(first)
    assert stage.route(x, dreaming=True).extra is None
    second = stage.allocate_expert()
    stage.mark_trained(second)
    before = stage.active.clone()
    routed = stage.route(x, dreaming=True)
    assert routed.selected.shape == (1, 2, 1)
    assert routed.extra.shape == (1, 2)
    assert torch.all(routed.extra != routed.selected.squeeze(-1))
    assert set(routed.extra.flatten().tolist()) <= {first, second}
    torch.testing.assert_close(before, stage.active)
    with dream_routing(chain):
        chain(x)
    torch.testing.assert_close(before, stage.active)


def test_preallocated_capacity_stops_and_inactive_expert_never_changes_output():
    chain = MemoryChain(
        nn.Identity(), hidden_size=3, periods=(1, 4), capacity=1, rank=2
    )
    stage = chain.stages[0]
    x = torch.randn(2, 3)
    stage.experts[0].up.weight.data.fill_(5)
    torch.testing.assert_close(chain(x), x)
    stage.allocate_expert()
    with pytest.raises(CapacityError):
        stage.allocate_expert()


def test_schedule_and_gradient_buffers_survive_checkpoint():
    chain = MemoryChain(nn.Linear(3, 3), hidden_size=3, periods=(1, 4), rank=2)
    for _ in range(4):
        chain.zero_grad(set_to_none=True)
        chain(torch.ones(1, 3)).sum().backward()
        chain.record_wake()
    assert chain.pending_levels() == (0, 1)
    assert chain.wake_steps.item() == 4
    restored = MemoryChain(nn.Linear(3, 3), hidden_size=3, periods=(1, 4), rank=2)
    restored.load_state_dict(chain.state_dict())
    assert restored.pending_levels() == (0, 1)
    for key, value in chain.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    assert sum(b.abs().sum() for b in restored.stages[0].accumulated_gradients()) > 0


def test_hf_llama_wrapping_preserves_logits_greedy_and_outer_structure():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(9)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=23,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).eval()
    ids = torch.tensor([[1, 5, 9, 2]])
    with torch.no_grad():
        expected = model(ids).logits
    attention = model.model.layers[0].self_attn
    wrap_llama_mlps(model, periods=(1, 4), capacity=2, rank=2)
    assert len(list(iter_memory_chains(model))) == 2
    assert model.model.layers[0].self_attn is attention
    with torch.no_grad():
        actual = model(ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    configure_wake_training(model)
    assert model.model.layers[0].mlp.stages[0].base.gate_proj.weight.requires_grad
    assert not model.model.layers[0].self_attn.q_proj.weight.requires_grad


def _distinct_routing_chain():
    chain = MemoryChain(
        nn.Identity(), hidden_size=2, periods=(1, 4), capacity=2, rank=2
    )
    stage = chain.stages[0]
    with torch.no_grad():
        stage.routing_projection.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        for multiplier in (1.0, 3.0):
            index = stage.allocate_expert()
            stage.mark_trained(index)
            stage.experts[index].down.weight.copy_(torch.eye(2))
            stage.experts[index].up.weight.copy_(torch.eye(2) * multiplier)
    return chain


def test_routing_prefix_is_invariant_to_future_completion_tokens():
    chain = _distinct_routing_chain()
    left = torch.tensor([[[1.0, 0.0], [10.0, 0.0]]])
    right = torch.tensor([[[1.0, 0.0], [-10.0, 0.0]]])
    torch.testing.assert_close(chain(left)[:, :1], chain(right)[:, :1], rtol=0, atol=0)


def test_routing_is_independent_of_cobatched_examples():
    chain = _distinct_routing_chain()
    single = torch.tensor([[[1.0, 0.0]]])
    together = torch.tensor([[[1.0, 0.0]], [[-10.0, 0.0]]])
    torch.testing.assert_close(chain(single)[0], chain(together)[0], rtol=0, atol=0)


def test_hf_trained_expert_routing_full_logits_match_cached_decoding():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=23,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).eval()
    wrap_llama_mlps(model, periods=(1, 4), capacity=2, rank=2)
    with torch.no_grad():
        for chain in iter_memory_chains(model):
            for stage in chain.stages:
                for _ in range(2):
                    index = stage.allocate_expert()
                    stage.mark_trained(index)
                    stage.experts[index].up.weight.normal_(std=0.3)
        ids = torch.tensor([[1, 5, 9, 2, 4]])
        full = model(ids, use_cache=False).logits
        cache = None
        incremental = []
        for position in range(ids.shape[1]):
            output = model(
                ids[:, position : position + 1], past_key_values=cache, use_cache=True
            )
            incremental.append(output.logits)
            cache = output.past_key_values
        cached = torch.cat(incremental, dim=1)
    torch.testing.assert_close(full, cached, rtol=1e-5, atol=1e-6)
