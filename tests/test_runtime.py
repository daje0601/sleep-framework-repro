import pytest
import torch
from sleep_repro.runtime import (
    ExperimentConfig,
    LoRALinear,
    encode_batch,
    ByteTokenizer,
)


def test_config_rejects_unknown_or_false_reproduction():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"imaginary_flag": True})
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"mode": "sleep", "attempts": 0})
    with pytest.raises(ValueError, match="dtype"):
        ExperimentConfig.from_dict({"dtype": "float16"})
    with pytest.raises(ValueError, match="revision"):
        ExperimentConfig.from_dict({"revision": "main"})


def test_student_ks_requires_its_declared_threshold_and_sampling_policy():
    cfg = {"ks_protocol": "student-tokens-v3", "on_policy_fraction": 1}
    for value in (None, True, -1, 1.5):
        with pytest.raises(ValueError, match="edit_distance_threshold"):
            ExperimentConfig.from_dict({**cfg, "edit_distance_threshold": value})
    with pytest.raises(ValueError, match="on_policy_fraction"):
        ExperimentConfig.from_dict({**cfg, "on_policy_fraction": 0.5, "edit_distance_threshold": 1})
    with pytest.raises(ValueError, match="legacy"):
        ExperimentConfig.from_dict({"edit_distance_threshold": 1})
    assert ExperimentConfig.from_dict({**cfg, "edit_distance_threshold": 0}).edit_distance_threshold == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("adapter", ["inner", "policy"])
def test_adam_first_zero_b_step_is_finite_and_updates(dtype, adapter):
    layer = LoRALinear(torch.nn.Linear(4, 4, dtype=dtype), 2, 2, 2, 2)
    layer.train_adapter(adapter)
    params = [p for p in layer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    layer(torch.ones(3, 4, dtype=dtype)).float().square().mean().backward()
    opt.step()
    assert all(torch.isfinite(p).all() for p in layer.parameters())
    assert getattr(layer, adapter + "_b").abs().sum() > 0


def test_lora_preserves_base_and_only_selected_adapter_learns():
    torch.manual_seed(12)
    base = torch.nn.Linear(4, 4)
    layer = LoRALinear(base, 2, 2, 2, 2)
    x = torch.randn(3, 4)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    layer.train_adapter("inner")
    before = layer.policy_b.detach().clone()
    opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    layer(x).square().mean().backward()
    opt.step()
    assert layer.inner_b.abs().sum() > 0
    torch.testing.assert_close(layer.policy_b, before)
    assert base.weight.grad is None


def test_supervised_batch_masks_prompt_and_rejects_truncation():
    tok = ByteTokenizer()
    batch = encode_batch(tok, [("input:", "[[1]]")], torch.device("cpu"), 100)
    assert (batch["labels"][0, : len(tok.encode("input:"))] == -100).all()
    assert (batch["labels"] != -100).sum() > 0
    with pytest.raises(ValueError, match="length"):
        encode_batch(tok, [("long input", "answer")], torch.device("cpu"), 3)
