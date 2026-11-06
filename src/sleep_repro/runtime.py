"""Explicit experiment configuration, dual LoRA adapters and text batching."""

from dataclasses import dataclass, asdict, fields
import math
import random
import re
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    revision: str = "9213176726f574b556790deb65791e0c5aa438b6"
    mode: str = "sleep"
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float32"
    attention_backend: str = "sdpa"
    levels: int = 2
    periods: tuple[int, ...] = (1, 4)
    rank: int = 64
    capacity: int = 8
    wake_steps: int = 2
    consolidation_steps: int = 2
    base_lr: float = 5e-6
    expert_lr: float = 5e-5
    policy_rank: int = 16
    inner_rank: int = 128
    lora_alpha: float = 16
    inner_lr: float = 1e-4
    outer_lr: float = 5e-5
    outer_epochs: int = 1
    train_epochs: int = 1
    microbatch_size: int = 1
    train_dreams: int = 60
    top_k: int = 12
    random_k: int = 3
    attempts: int = 5
    max_length: int = 4096
    max_new_tokens: int = 256
    temperature: float = 0.8
    ks_alpha: float = 0.5
    semantic_weight: float = 0.5
    on_policy_fraction: float = 0.5
    data_dir: str = "data/arc"

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        values = dict(values)
        if "periods" in values:
            values["periods"] = tuple(values["periods"])
        obj = cls(**values)
        if obj.mode not in ("sleep", "icl", "ttt", "self_edit", "consolidation"):
            raise ValueError("Unknown mode")
        integers = (
            "levels",
            "rank",
            "capacity",
            "wake_steps",
            "consolidation_steps",
            "policy_rank",
            "inner_rank",
            "outer_epochs",
            "train_epochs",
            "train_dreams",
            "attempts",
            "max_length",
            "max_new_tokens",
            "microbatch_size",
        )
        for name in integers:
            v = getattr(obj, name)
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(obj.periods) != obj.levels or any(
            type(p) is not int or p <= 0 for p in obj.periods
        ):
            raise ValueError("One positive integer period required per level")
        if any(b <= a or b % a for a, b in zip(obj.periods, obj.periods[1:])):
            raise ValueError("Periods must be increasing integer multiples")
        for name in (
            "base_lr",
            "expert_lr",
            "inner_lr",
            "outer_lr",
            "temperature",
            "lora_alpha",
        ):
            v = getattr(obj, name)
            if (
                not isinstance(v, (int, float))
                or isinstance(v, bool)
                or not math.isfinite(v)
                or v <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("ks_alpha", "semantic_weight", "on_policy_fraction"):
            v = getattr(obj, name)
            if (
                not isinstance(v, (int, float))
                or isinstance(v, bool)
                or not math.isfinite(v)
                or not 0 <= v <= 1
            ):
                raise ValueError(f"{name} must be between 0 and 1")
        if (
            any(type(k) is not int or k < 0 for k in (obj.top_k, obj.random_k))
            or not 0 < obj.top_k + obj.random_k <= obj.train_dreams
        ):
            raise ValueError("Invalid dream selection counts")
        if obj.dtype not in ("float32", "bfloat16"):
            raise ValueError(
                "Unsupported dtype; float16 Adam requires a mixed-precision implementation"
            )
        if obj.attention_backend not in ("eager", "sdpa"):
            raise ValueError("attention_backend must be eager or sdpa")
        if obj.mode in ("sleep", "consolidation") and obj.dtype != "float32":
            raise ValueError(
                "Memory base updates require float32 dtype until master-weight support is validated"
            )
        if obj.model != "tiny" and not re.fullmatch(r"[0-9a-f]{40}", obj.revision):
            raise ValueError("A full immutable model revision SHA is required")
        return obj

    def to_dict(self):
        return asdict(self)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LoRALinear(nn.Module):
    """Fixed base plus separate policy and per-attempt LoRA branches."""

    def __init__(self, base, policy_rank, inner_rank, policy_alpha, inner_alpha):
        super().__init__()
        self.base = base
        kw = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.policy_a = nn.Parameter(
            torch.randn(policy_rank, base.in_features, **kw)
            / math.sqrt(base.in_features)
        )
        self.policy_b = nn.Parameter(torch.zeros(base.out_features, policy_rank, **kw))
        self.inner_a = nn.Parameter(
            torch.randn(inner_rank, base.in_features, **kw)
            / math.sqrt(base.in_features)
        )
        self.inner_b = nn.Parameter(torch.zeros(base.out_features, inner_rank, **kw))
        self.policy_scale = policy_alpha / policy_rank
        self.inner_scale = inner_alpha / inner_rank
        self.requires_grad_(False)

    def forward(self, x):
        return (
            self.base(x)
            + (x @ self.policy_a.T @ self.policy_b.T) * self.policy_scale
            + (x @ self.inner_a.T @ self.inner_b.T) * self.inner_scale
        )

    def train_adapter(self, name):
        if name not in ("inner", "policy"):
            raise ValueError("Unknown adapter")
        self.requires_grad_(False)
        getattr(self, name + "_a").requires_grad_(True)
        getattr(self, name + "_b").requires_grad_(True)


def attach_lora(model, cfg):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in ("q_proj", "v_proj") and isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    LoRALinear(
                        child,
                        cfg.policy_rank,
                        cfg.inner_rank,
                        cfg.lora_alpha,
                        cfg.lora_alpha,
                    ),
                )


def train_adapter(model, name):
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.train_adapter(name)
    return [p for p in model.parameters() if p.requires_grad]


class ByteTokenizer:
    """Offline test tokenizer. Never used for pretrained quantitative runs."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [
            b + 3 for b in text.encode("utf-8")
        ]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i - 3 for i in ids if 3 <= i <= 258).decode(
            "utf-8", errors="replace"
        )


def encode_batch(tokenizer, examples, device, max_length):
    """Tokenize prompt and completion separately to make the loss boundary exact."""
    rows = []
    targets = []
    for prompt, completion in examples:
        prefix = tokenizer.encode(prompt, add_special_tokens=True)
        suffix = tokenizer.encode(completion, add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        ids = prefix + suffix
        if len(ids) > max_length:
            raise ValueError(
                f"Example length {len(ids)} exceeds max_length {max_length}; truncation is forbidden"
            )
        rows.append(ids)
        targets.append([-100] * len(prefix) + suffix)
    if not rows:
        raise ValueError("Empty batch")
    width = max(map(len, rows))
    return {
        "input_ids": torch.tensor(
            [r + [tokenizer.pad_token_id] * (width - len(r)) for r in rows],
            device=device,
        ),
        "attention_mask": torch.tensor(
            [[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device
        ),
        "labels": torch.tensor(
            [r + [-100] * (width - len(r)) for r in targets], device=device
        ),
    }
