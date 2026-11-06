"""ARC train/evaluate orchestration with isolated episodes and evidence records."""

from __future__ import annotations
import copy
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time
import os
from hashlib import sha256
from importlib.resources import files
import numpy as np
import torch
from .runtime import ByteTokenizer, attach_lora, train_adapter, encode_batch, seed_all
from .arc import (
    DreamDirective,
    parse_directive,
    expand_directive,
    holdout_demonstration,
    format_supervised_example,
    content_hash,
    load_arc,
    TRAIN_IDS,
    EVALUATION_IDS,
)
from .metrics import Attempt, evaluate_attempts
from .model import (
    wrap_llama_mlps,
    configure_wake_training,
    record_wake,
    dream_routing,
    parameter_counts,
)
from .consolidation import consolidate, apply_terminal_update
from .dreaming import select_candidates, evaluate_candidates


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_model(cfg):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
    )

    if cfg.model == "tiny":
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=259,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=4096,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        tokenizer = ByteTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            revision=cfg.revision,
            dtype=getattr(torch, cfg.dtype),
            attn_implementation="eager",
        )
    model = model.to(device=cfg.device, dtype=getattr(torch, cfg.dtype))
    model.config.use_cache = False
    model.eval()
    probe_ids = torch.tensor(
        [tokenizer.encode("ARC conversion probe", add_special_tokens=True)],
        device=cfg.device,
    )
    with torch.no_grad():
        original_logits = model(input_ids=probe_ids, use_cache=False).logits
    if cfg.mode in ("sleep", "consolidation"):
        wrap_llama_mlps(
            model, periods=cfg.periods, rank=cfg.rank, capacity=cfg.capacity
        )
    attach_lora(model, cfg)
    model.eval()
    with torch.no_grad():
        converted_logits = model(input_ids=probe_ids, use_cache=False).logits
        torch.testing.assert_close(original_logits, converted_logits, rtol=0, atol=0)
        model._conversion_probe = {
            "passed": True,
            "max_absolute_logit_difference": 0.0,
            "token_count": probe_ids.shape[1],
            "scope": "short fixed input; initial conversion only",
        }
    return model, tokenizer


def protocol_fingerprint():
    """Bind checkpoints to the packaged data registry and executable protocol."""
    package = files("sleep_repro")
    names = (
        "arc_data_manifest.json",
        "arc.py",
        "metrics.py",
        "runtime.py",
        "engine.py",
        "memory.py",
        "model.py",
        "consolidation.py",
        "objectives.py",
        "dreaming.py",
        "runner.py",
    )
    return content_hash(
        {
            name: sha256(package.joinpath(name).read_bytes()).hexdigest()
            for name in names
        }
    )


def save_checkpoint(path, model, optimizer, cfg, next_episode, *, outer_updates=0):
    """Trusted local tensor/state checkpoint, resumable at episode boundaries."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "protocol_fingerprint": protocol_fingerprint(),
        "outer_updates": outer_updates,
        "next_episode": next_episode,
        "python_rng": random.getstate(),
        "numpy_rng": [
            np.random.get_state()[0],
            np.random.get_state()[1].tolist(),
            *np.random.get_state()[2:],
        ],
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, cfg, *, evaluation=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("protocol_fingerprint") != protocol_fingerprint():
        raise ValueError(
            "Checkpoint protocol mismatch: data registry or implementation changed"
        )
    if type(state.get("outer_updates")) is not int or state["outer_updates"] < 0:
        raise ValueError("Checkpoint lacks a valid outer-update count")
    expected = cfg.to_dict()
    observed = state["config"]
    if evaluation:
        ignored = {"device", "dtype", "attempts", "data_dir"}
        expected = {k: v for k, v in expected.items() if k not in ignored}
        observed = {k: v for k, v in observed.items() if k not in ignored}
    if expected != observed:
        raise ValueError("Checkpoint config mismatch; refuse silent protocol changes")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if not evaluation:
        random.setstate(state["python_rng"])
        n = state["numpy_rng"]
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return {
        "next_episode": state["next_episode"],
        "outer_updates": state["outer_updates"],
    }


def examples_for(task, directive=None):
    pairs = (
        task.demonstrations if directive is None else expand_directive(task, directive)
    )
    return [format_supervised_example(p) for p in pairs]


def wake_episode(policy, tokenizer, cfg, task):
    from .engine import CausalEngine

    episode = copy.deepcopy(policy)
    events = []
    if cfg.mode not in ("sleep", "consolidation"):
        return episode, events
    engine = CausalEngine(episode, tokenizer, cfg, torch.device(cfg.device))
    examples = examples_for(task)
    for step in range(cfg.wake_steps):
        configure_wake_training(episode)
        episode.zero_grad(set_to_none=True)
        sample = [examples[step % len(examples)]]
        batch = encode_batch(
            tokenizer, sample, torch.device(cfg.device), cfg.max_length
        )
        loss = episode(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite wake loss")
        loss.backward()
        due = record_wake(episode)
        # The schedule now owns these gradients in its accumulation buffers.
        # Release redundant .grad storage before allocating transfer snapshots.
        episode.zero_grad(set_to_none=True)
        events.append({"event": "wake", "step": step, "loss": float(loss.detach())})
        for level in due:
            if level == cfg.levels - 1:
                apply_terminal_update(episode, base_lr=cfg.base_lr)
            else:
                result = consolidate(
                    episode,
                    level,
                    lambda teacher, student: engine.knowledge_objective(
                        teacher, student, sample
                    ),
                    base_lr=cfg.base_lr,
                    expert_lr=cfg.expert_lr,
                    steps=cfg.consolidation_steps,
                )
                events.append(
                    {
                        "event": "consolidation",
                        "step": step,
                        "level": level,
                        "losses": list(result.losses),
                    }
                )
    episode.zero_grad(set_to_none=True)
    episode.requires_grad_(False)
    return episode, events


def dataset_for(cfg, split):
    directory = Path(cfg.data_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    registry = json.loads(
        files("sleep_repro").joinpath("arc_data_manifest.json").read_text()
    )
    expected_files = {
        name: {k: info[k] for k in ("url", "sha256")}
        for name, info in registry["files"].items()
    }
    if manifest.get("files") != expected_files:
        raise ValueError("Dataset provenance differs from the packaged pinned registry")
    for name, info in registry["files"].items():
        file = directory / name
        if (
            not file.is_file()
            or sha256(file.read_bytes()).hexdigest() != info["sha256"]
        ):
            raise ValueError(f"Data hash mismatch: {name}")
    return load_arc(
        directory / f"{split}_challenges.json",
        directory / f"{split}_solutions.json",
        TRAIN_IDS if split == "train" else EVALUATION_IDS,
    )


def evidence(cfg, model, dataset=None):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
    return {
        "config": cfg.to_dict(),
        "source_commit": commit,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "protocol_fingerprint": protocol_fingerprint(),
        "conversion_probe": getattr(model, "_conversion_probe", None),
        "dependencies": {
            p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")
        },
        "device": cfg.device,
        "parameter_counts": parameter_counts(model),
        "hardware": torch.cuda.get_device_name()
        if cfg.device.startswith("cuda") and torch.cuda.is_available()
        else cfg.device,
        "container_image": os.environ.get("SLEEP_CONTAINER_IMAGE"),
        "job_id": os.environ.get("VESSL_JOB_ID", os.environ.get("SLEEP_JOB_NAME")),
        "data_hashes": dict(dataset.file_hashes) if dataset else {},
        "protocol": "independent reconstructed ARC protocol; not exact author recipe",
        "paper_result_reproduced": False,
    }


def append_event(path, event):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")


def train(cfg, output, resume=None):
    from .engine import CausalEngine

    if cfg.mode not in ("sleep", "self_edit"):
        raise ValueError("Policy training supports sleep or self_edit modes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training.jsonl").exists() and resume is None:
        raise ValueError(
            "Output already contains training records; use resume or a new directory"
        )
    seed_all(cfg.seed)
    policy, tokenizer = build_model(cfg)
    data = dataset_for(cfg, "train")
    parameters = train_adapter(policy, "policy")
    optimizer = torch.optim.AdamW(parameters, lr=cfg.outer_lr)
    progress = (
        load_checkpoint(resume, policy, optimizer, cfg)
        if resume
        else {"next_episode": 0, "outer_updates": 0}
    )
    start = progress["next_episode"]
    outer_updates = progress["outer_updates"]
    write_json(output / "provenance.json", evidence(cfg, policy, data))
    schedule = list(data.tasks) * cfg.train_epochs
    if start > len(schedule):
        raise ValueError("Checkpoint episode exceeds training schedule")
    for index in range(start, len(schedule)):
        task = schedule[index]
        support, heldout = holdout_demonstration(task)
        episode, events = wake_episode(policy, tokenizer, cfg, support)
        engine = CausalEngine(episode, tokenizer, cfg, torch.device(cfg.device))
        valid = []
        scores = []
        rejected = []
        for dream_index in range(cfg.train_dreams):
            with dream_routing(episode, cfg.mode == "sleep"):
                text = engine.generate_directive(support)
            try:
                directive = parse_directive(text)
            except (ValueError, TypeError) as exc:
                rejected.append({"index": dream_index, "text": text, "error": str(exc)})
                continue
            score = engine.importance(examples_for(support, directive))
            valid.append(directive)
            scores.append(score)
        n = len(valid)
        k = min(cfg.top_k, n)
        b = min(cfg.random_k, n - k)
        selected = (
            select_candidates(scores, top_k=k, random_k=b, seed=cfg.seed + index)
            if n
            else []
        )
        candidates = [valid[i] for i in selected]
        parameters = train_adapter(episode, "inner")
        inner_optimizer = torch.optim.AdamW(parameters, lr=cfg.inner_lr)

        def adapt(model, opt, directive):
            return CausalEngine(model, tokenizer, cfg, torch.device(cfg.device)).sft(
                examples_for(support, directive),
                adapter="inner",
                epochs=directive.epochs,
                lr=cfg.inner_lr * directive.lr_scale,
                optimizer=opt,
            )

        def score(model):
            return CausalEngine(
                model, tokenizer, cfg, torch.device(cfg.device)
            ).score_demonstration(support, heldout)

        result = (
            evaluate_candidates(
                episode, inner_optimizer, candidates, adapt=adapt, score=score
            )
            if candidates
            else None
        )
        accepted = list(result.accepted) if result else []
        outer_losses = []
        if accepted:
            outer_engine = CausalEngine(
                policy, tokenizer, cfg, torch.device(cfg.device)
            )
            outer_losses = outer_engine.sft(
                [
                    (outer_engine.directive_prompt(support), d.to_json())
                    for d in accepted
                ],
                adapter="policy",
                epochs=cfg.outer_epochs,
                optimizer=optimizer,
            )
            outer_updates += len(outer_losses)
        event = {
            "episode": index,
            "task_id": task.task_id,
            "wake": events,
            "generated": cfg.train_dreams,
            "invalid": rejected,
            "valid_directives": [d.to_json() for d in valid],
            "expanded_example_hashes": [
                content_hash([asdict(p) for p in expand_directive(support, d)])
                for d in valid
            ],
            "gradient_scores": scores,
            "selected_indices": selected,
            "accepted": [d.to_json() for d in accepted],
            "candidate_records": [asdict(r) for r in result.records] if result else [],
            "outer_losses": outer_losses,
            "accepted_count": len(accepted),
            "outer_updates": outer_updates,
        }
        append_event(output / "training.jsonl", event)
        save_checkpoint(
            output / "checkpoint.pt",
            policy,
            optimizer,
            cfg,
            index + 1,
            outer_updates=outer_updates,
        )
    write_json(
        output / "training_summary.json",
        {
            "completed_episodes": len(schedule),
            "checkpoint": "checkpoint.pt",
            "outer_updates": outer_updates,
            "policy_trained": outer_updates > 0,
            "paper_result_reproduced": False,
        },
    )
    return output / "checkpoint.pt"


def evaluate(cfg, output, checkpoint=None, allow_untrained=False):
    from .engine import CausalEngine

    if (
        cfg.mode in ("sleep", "self_edit")
        and checkpoint is None
        and not allow_untrained
    ):
        raise ValueError(
            "Learned-policy evaluation requires --checkpoint; --allow-untrained is an explicit ablation"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "attempts.jsonl").exists():
        raise ValueError("Evaluation output already exists; choose a fresh directory")
    seed_all(cfg.seed)
    policy, tokenizer = build_model(cfg)
    progress = {"outer_updates": 0}
    if checkpoint:
        progress = load_checkpoint(checkpoint, policy, None, cfg, evaluation=True)
    data = dataset_for(cfg, "eval")
    attempts = []
    started = time.monotonic()
    write_json(
        output / "provenance.json",
        {
            **evidence(cfg, policy, data),
            "checkpoint": str(checkpoint) if checkpoint else None,
        },
    )
    for task_index, task in enumerate(data.tasks):
        for attempt_index in range(cfg.attempts):
            attempt_seed = cfg.seed + task_index * 1000 + attempt_index
            seed_all(attempt_seed)
            episode, events = wake_episode(policy, tokenizer, cfg, task)
            engine = CausalEngine(episode, tokenizer, cfg, torch.device(cfg.device))
            directive = None
            raw = None
            invalid = None
            if cfg.mode == "ttt":
                directive = DreamDirective()
            elif cfg.mode in ("sleep", "self_edit"):
                with dream_routing(episode, cfg.mode == "sleep"):
                    raw = engine.generate_directive(task)
                try:
                    directive = parse_directive(raw)
                except (ValueError, TypeError) as exc:
                    invalid = str(exc)
            if directive:
                engine.sft(
                    examples_for(task, directive),
                    adapter="inner",
                    epochs=directive.epochs,
                    lr=cfg.inner_lr * directive.lr_scale,
                )
            predictions = (
                engine.predict(task)
                if invalid is None
                else tuple(None for _ in task.test_inputs)
            )
            attempt = Attempt(task.task_id, attempt_index, predictions)
            attempts.append(attempt)
            append_event(
                output / "attempts.jsonl",
                {
                    **asdict(attempt),
                    "raw_predictions": getattr(engine, "last_prediction_texts", []),
                    "seed": attempt_seed,
                    "directive": raw,
                    "invalid_directive": invalid,
                    "wake": events,
                    "expanded_example_hash": content_hash(
                        [asdict(p) for p in expand_directive(task, directive)]
                    )
                    if directive
                    else None,
                },
            )
    summary = evaluate_attempts(
        data.tasks, data.solutions, attempts, expected_attempts=cfg.attempts
    )
    summary.update(
        {
            "elapsed_seconds": time.monotonic() - started,
            "mode": cfg.mode,
            "policy_trained": progress["outer_updates"] > 0,
            "outer_updates": progress["outer_updates"],
            "paper_result_reproduced": False,
            "comparison_status": "protocol reconstruction",
        }
    )
    write_json(output / "summary.json", summary)
    return summary
