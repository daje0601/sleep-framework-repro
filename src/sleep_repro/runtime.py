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
    prompt_format: str = "plain-v1"
    sft_update_mode: str = "full-dataset-v1"
    effective_batch_size: int = 2
    outer_effective_batch_size: int = 5
    sft_max_steps: int = 375
    contextual_examples: bool = False
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
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
    ks_protocol: str = "mixed-text-v2"
    edit_distance_threshold: int | None = None
    data_dir: str = "data/arc"

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        values = dict(values)
        if "periods" in values:
            values["periods"] = tuple(values["periods"])
        if "lora_targets" in values:
            values["lora_targets"] = tuple(values["lora_targets"])
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
            "effective_batch_size",
            "outer_effective_batch_size",
            "sft_max_steps",
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
        if obj.ks_protocol == "student-tokens-v3":
            if type(obj.edit_distance_threshold) is not int or obj.edit_distance_threshold < 0:
                raise ValueError("v3 requires a nonnegative integer edit_distance_threshold")
            if obj.on_policy_fraction != 1:
                raise ValueError("v3 KS requires on_policy_fraction=1")
        elif obj.ks_protocol != "mixed-text-v2" or obj.edit_distance_threshold is not None:
            raise ValueError("Unknown ks_protocol or unexpected legacy edit_distance_threshold")
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
        if obj.prompt_format not in ("plain-v1", "hf-chat-v1"):
            raise ValueError("Unknown prompt_format")
        if obj.sft_update_mode not in ("full-dataset-v1", "minibatch-v2"):
            raise ValueError("Unknown sft_update_mode")
        if type(obj.contextual_examples) is not bool:
            raise ValueError("contextual_examples must be boolean")
        allowed_targets = {"q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"}
        if (
            not obj.lora_targets
            or len(set(obj.lora_targets)) != len(obj.lora_targets)
            or not set(obj.lora_targets) <= allowed_targets
        ):
            raise ValueError("Invalid lora_targets")
        if obj.mode in ("sleep", "consolidation") and obj.dtype != "float32":
            raise ValueError(
                "Memory base updates require float32 dtype until master-weight support is validated"
            )
        if obj.model != "tiny" and not re.fullmatch(r"[0-9a-f]{40}", obj.revision):
            raise ValueError("A full immutable model revision SHA is required")
        return obj

    def to_dict(self):
        values = asdict(self)
        if self.ks_protocol == "mixed-text-v2" and self.edit_distance_threshold is None:
            # Preserve the serialized schema and hashes of historical checkpoints.
            del values["ks_protocol"], values["edit_distance_threshold"]
        return values


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
        self.in_features = base.in_features
        self.out_features = base.out_features
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
    """Attach exactly the configured projection targets before CMS if FFNs are selected."""
    targets = getattr(cfg, "lora_targets", ("q_proj", "v_proj"))
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
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


def encode_prompt(tokenizer, prompt, prompt_format="plain-v1", *, assistant_prefix=""):
    """Versioned prompt serialization shared by generation and supervised loss."""
    if prompt_format == "plain-v1":
        return tokenizer.encode(prompt + assistant_prefix, add_special_tokens=True)
    if prompt_format != "hf-chat-v1":
        raise ValueError("Unknown prompt_format")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("hf-chat-v1 requires a tokenizer chat template")
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        date_string="26 Jul 2024",
    )
    return list(ids) + tokenizer.encode(assistant_prefix, add_special_tokens=False)


def encode_batch(tokenizer, examples, device, max_length, *, prompt_format="plain-v1"):
    """Tokenize prompt and completion separately to make the loss boundary exact."""
    rows = []
    targets = []
    for example in examples:
        if len(example) not in (2, 3):
            raise ValueError("Expected (prompt,completion[,assistant_prefix])")
        prompt, completion = example[:2]
        assistant_prefix = example[2] if len(example) == 3 else ""
        prefix = encode_prompt(
            tokenizer, prompt, prompt_format, assistant_prefix=assistant_prefix
        )
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
