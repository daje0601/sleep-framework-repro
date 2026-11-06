import copy

import pytest
import torch
from torch import nn

from sleep_repro.memory import CapacityError, MemoryChain
from sleep_repro.consolidation import consolidate, apply_terminal_update


def setup_chain():
    torch.manual_seed(21)
    chain = MemoryChain(
        nn.Linear(4, 4, bias=False), hidden_size=4, periods=(1, 4), rank=3, capacity=2
    )
    x = torch.randn(3, 4)
    chain(x).square().mean().backward()
    chain.record_wake()
    return chain, x


def test_transaction_only_trains_new_receiver_and_keeps_teacher_immutable():
    chain, x = setup_chain()
    before = {k: v.clone() for k, v in chain.state_dict().items()}
    flags = [p.requires_grad for p in chain.parameters()]
    seen = []

    def objective(teacher, student):
        assert not any(p.requires_grad for p in teacher.parameters())
        trainable = [n for n, p in student.named_parameters() if p.requires_grad]
        assert trainable and all("stages.1.experts.0." in n for n in trainable)
        for key, value in before.items():
            torch.testing.assert_close(teacher.state_dict()[key], value)
        loss = (student(x) - teacher(x).detach()).square().mean()
        seen.append(float(loss.detach()))
        return loss

    result = consolidate(chain, 0, objective, base_lr=0.05, expert_lr=0.03, steps=30)
    assert result.committed and result.new_experts == (0,)
    assert seen[-1] < seen[0]
    assert chain.stages[1].active.tolist() == [True, False]
    assert chain.stages[1].trained.tolist() == [True, False]
    assert not torch.equal(chain.stages[0].base.weight, before["stages.0.base.weight"])
    assert [p.requires_grad for p in chain.parameters()] == flags
    assert chain.pending_levels() == ()


def test_nonfinite_loss_rolls_back_parameters_buffers_optimizer_and_rng():
    chain, x = setup_chain()
    optimizer = torch.optim.Adam(chain.parameters(), lr=0.1)
    optimizer.step()
    before = copy.deepcopy(chain.state_dict())
    opt_before = copy.deepcopy(optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    flags = [p.requires_grad for p in chain.parameters()]
    with pytest.raises(FloatingPointError):
        consolidate(
            chain,
            0,
            lambda t, s: s(x).sum() * float("nan"),
            base_lr=0.05,
            expert_lr=0.01,
            steps=1,
            optimizer=optimizer,
        )
    for key, value in before.items():
        torch.testing.assert_close(chain.state_dict()[key], value)
    assert optimizer.state_dict()["param_groups"] == opt_before["param_groups"]
    for key in opt_before["state"]:
        for moment, value in opt_before["state"][key].items():
            torch.testing.assert_close(
                optimizer.state_dict()["state"][key][moment], value
            )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert [p.requires_grad for p in chain.parameters()] == flags


def test_exception_in_objective_rolls_back_and_capacity_never_overwrites():
    chain, x = setup_chain()
    before = copy.deepcopy(chain.state_dict())

    def broken(t, s):
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        consolidate(chain, 0, broken, base_lr=0.01, expert_lr=0.01, steps=1)
    for key, value in before.items():
        torch.testing.assert_close(chain.state_dict()[key], value)
    for _ in range(2):
        chain.stages[1].allocate_expert()
    full = copy.deepcopy(chain.state_dict())
    with pytest.raises(CapacityError):
        consolidate(
            chain, 0, lambda t, s: s(x).sum(), base_lr=0.01, expert_lr=0.01, steps=1
        )
    for key, value in full.items():
        torch.testing.assert_close(chain.state_dict()[key], value)


def test_sender_reset_clears_active_bank_and_optimizer_moments():
    chain, x = setup_chain()
    index = chain.stages[0].allocate_expert()
    chain.stages[0].mark_trained(index)
    optimizer = torch.optim.Adam(chain.parameters(), lr=0.001)
    chain.zero_grad(set_to_none=True)
    chain(x).square().mean().backward()
    optimizer.step()
    assert any(
        p in optimizer.state for p in chain.stages[0].experts[index].parameters()
    )
    consolidate(
        chain,
        0,
        lambda t, s: (s(x) - t(x)).square().mean(),
        base_lr=0.01,
        expert_lr=0.01,
        steps=2,
        optimizer=optimizer,
    )
    assert not chain.stages[0].active.any()
    assert not chain.stages[0].trained.any()
    assert all(
        p not in optimizer.state
        for e in chain.stages[0].experts
        for p in e.parameters()
    )


def test_checkpoint_continuation_produces_same_transaction():
    chain, x = setup_chain()
    restored = copy.deepcopy(chain)
    objective = lambda t, s: (s(x) - t(x)).square().mean()
    rng = torch.get_rng_state()
    consolidate(chain, 0, objective, base_lr=0.03, expert_lr=0.01, steps=5)
    torch.set_rng_state(rng)
    consolidate(restored, 0, objective, base_lr=0.03, expert_lr=0.01, steps=5)
    for key, value in chain.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_terminal_base_update_respects_pending_schedule():
    chain, x = setup_chain()
    with pytest.raises(ValueError, match="pending"):
        apply_terminal_update(chain, base_lr=0.03)
    for _ in range(3):
        chain.zero_grad(set_to_none=True)
        chain(x).square().mean().backward()
        chain.record_wake()
    before = chain.stages[1].base.up.weight.detach().clone()
    apply_terminal_update(chain, base_lr=0.03)
    assert not torch.equal(chain.stages[1].base.up.weight, before)
    assert chain.pending_levels() == (0,)


def test_full_hf_logits_transfer_updates_every_wrapped_layer_atomically():
    from transformers import LlamaConfig, LlamaForCausalLM
    from sleep_repro.model import (
        configure_wake_training,
        iter_memory_chains,
        record_wake,
        wrap_llama_mlps,
    )

    torch.manual_seed(4)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=19,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
    ).eval()
    wrap_llama_mlps(model, periods=(1, 4), capacity=2, rank=2)
    configure_wake_training(model)
    ids = torch.tensor([[1, 3, 5, 2]])
    model(ids, labels=ids).loss.backward()
    assert record_wake(model) == (0,)
    before = model(ids).logits.detach().clone()
    teacher_predictions = []

    def logits_loss(teacher, student):
        with torch.no_grad():
            target = teacher(ids).logits
        teacher_predictions.append(target)
        return (student(ids).logits - target).square().mean()

    result = consolidate(model, 0, logits_loss, base_lr=0.01, expert_lr=0.002, steps=5)
    assert result.new_experts == (0, 0)
    for target in teacher_predictions:
        torch.testing.assert_close(target, before, rtol=0, atol=0)
    for chain in iter_memory_chains(model):
        assert chain.stages[1].trained.tolist() == [True, False]
        assert chain.update_counts.tolist() == [1, 0]


def test_frozen_teacher_or_student_mutation_aborts_the_entire_transaction():
    chain, x = setup_chain()
    before = copy.deepcopy(chain.state_dict())

    def mutates_teacher(teacher, student):
        teacher.stages[0].base.weight.data.add_(1)
        return (student(x) - teacher(x)).square().mean()

    with pytest.raises(RuntimeError, match="immutable teacher"):
        consolidate(chain, 0, mutates_teacher, base_lr=0.03, expert_lr=0.01, steps=1)
    for key, value in before.items():
        torch.testing.assert_close(chain.state_dict()[key], value)

    def mutates_frozen_student(teacher, student):
        student.stages[0].base.weight.data.add_(1)
        return (student(x) - teacher(x)).square().mean()

    with pytest.raises(RuntimeError, match="frozen student"):
        consolidate(
            chain, 0, mutates_frozen_student, base_lr=0.03, expert_lr=0.01, steps=1
        )
    for key, value in before.items():
        torch.testing.assert_close(chain.state_dict()[key], value)


def test_failed_objective_restores_python_numpy_and_mps_rng_streams():
    import random
    import numpy as np

    chain, x = setup_chain()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    mps_before = (
        torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    )

    def samples_then_fails(teacher, student):
        random.random()
        np.random.random()
        if mps_before is not None:
            torch.rand(2, device="mps")
        raise RuntimeError("sample failure")

    with pytest.raises(RuntimeError, match="sample failure"):
        consolidate(chain, 0, samples_then_fails, base_lr=0.03, expert_lr=0.01, steps=1)
    assert random.getstate() == python_before
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_before[0]
    assert np.array_equal(actual_numpy[1], numpy_before[1])
    assert actual_numpy[2:] == numpy_before[2:]
    if mps_before is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps_before)


def test_second_transfer_commits_the_same_function_as_final_objective_validation():
    chain, x = setup_chain()
    objective = lambda teacher, student: (student(x) - teacher(x)).square().mean()
    consolidate(chain, 0, objective, base_lr=0.05, expert_lr=0.03, steps=3)
    chain.zero_grad(set_to_none=True)
    chain(x).square().mean().backward()
    chain.record_wake()
    final_outputs = []

    def tracks_prospective_function(teacher, student):
        prediction = student(x)
        final_outputs.append(prediction.detach().clone())
        return (prediction - teacher(x)).square().mean()

    consolidate(
        chain, 0, tracks_prospective_function, base_lr=0.05, expert_lr=0.03, steps=3
    )
    assert chain.stages[1].trained.tolist() == [True, True]
    torch.testing.assert_close(chain(x), final_outputs[-1], rtol=0, atol=0)


def test_one_unselected_new_receiver_aborts_every_chain_and_preserves_live_state():
    chains = nn.ModuleList(
        [
            MemoryChain(
                nn.Linear(2, 2, bias=False),
                hidden_size=2,
                periods=(1, 4),
                capacity=2,
                rank=2,
            )
            for _ in range(2)
        ]
    )
    x = torch.ones(2, 2)
    with torch.no_grad():
        for chain in chains:
            chain.stages[0].base.weight.copy_(torch.eye(2))
        receiver = chains[1].stages[1]
        old = receiver.allocate_expert()
        receiver.mark_trained(old)
        # Every positive token selects the old expert, never the new second slot.
        receiver.routing_projection.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    sum(chain(x).square().mean() for chain in chains).backward()
    for chain in chains:
        chain.record_wake()
    before = copy.deepcopy(chains.state_dict())

    def objective(teacher, student):
        return sum(
            (new(x) - old(x)).square().mean() for old, new in zip(teacher, student)
        )

    with pytest.raises(ValueError, match="receiver expert.*update"):
        consolidate(chains, 0, objective, base_lr=0.03, expert_lr=0.01, steps=2)
    for key, value in before.items():
        torch.testing.assert_close(chains.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_cpu_snapshot_owns_independent_storage_for_model_and_nested_optimizer_state(
    device,
):
    from sleep_repro.consolidation import _cpu_snapshot

    model = nn.Linear(3, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.ones(1, 3, device=device)).sum().backward()
    optimizer.step()
    source = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    snapshot = _cpu_snapshot(source)
    expected_weight = model.weight.detach().cpu().clone()
    expected_moment = (
        next(iter(optimizer.state.values()))["exp_avg"].detach().cpu().clone()
    )
    assert snapshot["model"]["weight"].device.type == "cpu"
    assert not snapshot["model"]["weight"].requires_grad
    assert snapshot["model"]["weight"].data_ptr() != model.weight.data_ptr()
    moment_key = next(iter(snapshot["optimizer"]["state"]))
    for state in snapshot["optimizer"]["state"].values():
        assert all(
            value.device.type == "cpu"
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
    with torch.no_grad():
        model.weight.add_(10)
        next(iter(optimizer.state.values()))["exp_avg"].zero_()
    torch.testing.assert_close(
        snapshot["model"]["weight"], expected_weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        snapshot["optimizer"]["state"][moment_key]["exp_avg"],
        expected_moment,
        rtol=0,
        atol=0,
    )
    snapshot["model"]["weight"].zero_()
    assert model.weight.abs().sum() > 0
    assert snapshot["model"]._metadata == source["model"]._metadata


def test_cpu_rollback_backup_restores_live_weights_after_partial_commit_failure(
    monkeypatch,
):
    from sleep_repro.consolidation import _commit

    live = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(live.parameters(), lr=0.01)
    live(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    before = copy.deepcopy(live.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    student = copy.deepcopy(live)
    with torch.no_grad():
        student.weight.add_(5)
    actual_load = live.load_state_dict
    calls = 0

    def load_then_fail_once(state, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = actual_load(state, *args, **kwargs)
        if calls == 1:
            # Fail after applying prospective weights: rollback must read an
            # independent old snapshot rather than an alias of those weights.
            raise RuntimeError("commit failed after applying weights")
        return result

    monkeypatch.setattr(live, "load_state_dict", load_then_fail_once)
    with pytest.raises(RuntimeError, match="commit failed"):
        _commit(live, student, list(live.parameters()), optimizer)
    for key, value in before.items():
        torch.testing.assert_close(live.state_dict()[key], value, rtol=0, atol=0)
    for key, state in optimizer_before["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0
            )


def test_offloaded_accumulation_buffers_keep_independent_values_and_execution_parameters():
    from sleep_repro.consolidation import _offload_accumulated_gradients

    chain, x = setup_chain()
    source = chain.stages[0].accumulated_gradients()[0]
    expected = source.clone()
    parameters = tuple(chain.parameters())
    with torch.no_grad():
        output_before = chain(x).clone()
    _offload_accumulated_gradients(chain)
    stored = chain.stages[0].accumulated_gradients()[0]
    assert stored.device.type == "cpu"
    assert stored.data_ptr() != source.data_ptr()
    source.zero_()
    torch.testing.assert_close(stored, expected, rtol=0, atol=0)
    assert all(
        actual is before for actual, before in zip(chain.parameters(), parameters)
    )
    with torch.no_grad():
        torch.testing.assert_close(chain(x), output_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_private_accumulation_offload_preserves_slower_values_and_live_buffer_device(
    device,
):
    chain, x = setup_chain()
    chain = chain.to(device)
    x = x.to(device)
    before = [
        tuple(value.detach().cpu().clone() for value in stage.accumulated_gradients())
        for stage in chain.stages
    ]
    assert any(torch.count_nonzero(value) for value in before[1])
    seen = []

    def objective(teacher, student):
        for private in (teacher, student):
            assert all(
                parameter.device.type == device for parameter in private.parameters()
            )
            for stage in private.stages:
                assert all(
                    value.device.type == "cpu"
                    for value in stage.accumulated_gradients()
                )
                assert stage.active.device.type == device
        for level, stage in enumerate(teacher.stages):
            for actual, expected in zip(stage.accumulated_gradients(), before[level]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(
            student.stages[1].accumulated_gradients(), before[1]
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(
            value.device.type == device
            for stage in chain.stages
            for value in stage.accumulated_gradients()
        )
        seen.append(True)
        return (student(x) - teacher(x)).square().mean()

    result = consolidate(chain, 0, objective, base_lr=0.03, expert_lr=0.01, steps=2)
    assert result.committed and seen
    assert all(
        value.device.type == device
        for stage in chain.stages
        for value in stage.accumulated_gradients()
    )
    assert all(
        not torch.count_nonzero(value)
        for value in chain.stages[0].accumulated_gradients()
    )
    for actual, expected in zip(chain.stages[1].accumulated_gradients(), before[1]):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert chain.stages[1].gradient_count.item() == 1
