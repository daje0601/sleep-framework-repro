"""Bounded causal-model operations shared by ARC experiment runners.

This module performs model forwards, masked SFT, and versioned KS/LTI.
The random teacher-prefix conditioning and YES/NO semantic prompt are explicit
reconstruction choices; their configuration must be recorded with each run.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import json
import math
import re

import torch
from torch import Tensor, nn

from .arc import Grid, Pair, Task, canonical_json, format_prompt, parse_grid_response
from .objectives import composite_rewards, lti_loss, student_ks_loss, token_edit_reward
from .runtime import encode_batch, encode_prompt, train_adapter


@contextmanager
def _evaluation_mode(model: nn.Module):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        # Restore mixed module modes without recursively overwriting children.
        for module, training in modes:
            module.training = training


class CausalEngine:
    def __init__(self, model, tokenizer, cfg, device):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = torch.device(device)

    def _limit(self, model) -> int:
        model_limit = getattr(
            getattr(model, "config", None),
            "max_position_embeddings",
            self.cfg.max_length,
        )
        return min(self.cfg.max_length, model_limit)

    def _generate(
        self,
        model,
        prompt: str,
        max_new_tokens: int | None = None,
        sample: bool = True,
        *,
        assistant_prefix: str = "",
    ) -> str:
        budget = self.cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(budget) is not int or budget <= 0:
            raise ValueError("generation token budget must be a positive integer")
        if not isinstance(prompt, str):
            raise ValueError("generation prompt must be a string")
        ids = encode_prompt(
            self.tokenizer,
            prompt,
            self.cfg.prompt_format,
            assistant_prefix=assistant_prefix,
        )
        if not ids or len(ids) + budget > self._limit(model):
            raise ValueError(
                f"prompt length {len(ids)} plus generation budget {budget} exceeds context length {self._limit(model)}; truncation is forbidden"
            )
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        kwargs = dict(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=budget,
            do_sample=sample,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        # Keep every EOS/turn-end ID from the pinned generation configuration.
        # A tokenizer's single EOS ID can omit other valid checkpoint stops.
        if sample:
            kwargs["temperature"] = self.cfg.temperature
        with _evaluation_mode(model), torch.no_grad():
            generated = model.generate(**kwargs)
        return self.tokenizer.decode(
            generated[0, len(ids) :].tolist(), skip_special_tokens=True
        )

    def generate(
        self, prompt: str, max_new_tokens: int | None = None, sample: bool = True
    ) -> str:
        """Return only newly generated text, rejecting context overflow."""
        return self._generate(self.model, prompt, max_new_tokens, sample)

    def sft(
        self,
        examples: list[tuple[str, str]],
        adapter: str = "inner",
        epochs: int = 1,
        lr: float | None = None,
        optimizer=None,
    ) -> list[float]:
        """Optimize only one dual-LoRA branch using completion-masked labels."""
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if adapter not in ("inner", "policy"):
            raise ValueError("adapter must be inner or policy")
        rate = (
            (self.cfg.inner_lr if adapter == "inner" else self.cfg.outer_lr)
            if lr is None
            else lr
        )
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if not examples:
            raise ValueError("Empty examples")
        effective_size = (
            len(examples)
            if self.cfg.sft_update_mode == "full-dataset-v1"
            else (
                self.cfg.effective_batch_size
                if adapter == "inner"
                else self.cfg.outer_effective_batch_size
            )
        )
        required_steps = math.ceil(len(examples) / effective_size) * epochs
        if required_steps > self.cfg.sft_max_steps:
            raise ValueError(
                f"SFT needs {required_steps} optimizer steps, exceeding limit {self.cfg.sft_max_steps}"
            )
        # Invalid later examples must not partially update the live outer policy.
        # Validate on CPU before touching parameters/optimizer; retain no tensors.
        for example in examples:
            encode_batch(
                self.tokenizer,
                [example],
                "cpu",
                self._limit(self.model),
                prompt_format=self.cfg.prompt_format,
            )
        parameters = train_adapter(self.model, adapter)
        if not parameters:
            raise ValueError("model has no selected LoRA adapter parameters")
        if optimizer is None:
            optimizer = torch.optim.AdamW(parameters, lr=rate)
        elif {id(p) for group in optimizer.param_groups for p in group["params"]} != {
            id(p) for p in parameters
        }:
            raise ValueError(
                "optimizer must contain exactly the selected adapter parameters"
            )
        if lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = rate
        losses = []
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.train()
            for _ in range(epochs):
                order = (
                    range(len(examples))
                    if self.cfg.sft_update_mode == "full-dataset-v1"
                    else torch.randperm(len(examples)).tolist()
                )
                ordered = [examples[i] for i in order]
                for start in range(0, len(ordered), effective_size):
                    batches = self._batches(ordered[start : start + effective_size])
                    total_tokens = sum(
                        int((b["labels"][:, 1:] != -100).sum()) for b in batches
                    )
                    self.model.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    for batch in batches:
                        weight = (
                            int((batch["labels"][:, 1:] != -100).sum()) / total_tokens
                        )
                        loss = self.model(**batch, use_cache=False).loss * weight
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite SFT loss")
                        loss.backward()
                        total_loss += float(loss.detach())
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in parameters
                    ):
                        raise FloatingPointError("nonfinite SFT gradient")
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in parameters):
                        raise FloatingPointError(
                            "nonfinite SFT parameter after optimizer step"
                        )
                    losses.append(total_loss)
        finally:
            for module, training in modes:
                module.training = training
        return losses

    def _batches(self, examples):
        if not examples:
            raise ValueError("Empty examples")
        size = self.cfg.microbatch_size
        return [
            encode_batch(
                self.tokenizer,
                examples[i : i + size],
                self.device,
                self._limit(self.model),
                prompt_format=self.cfg.prompt_format,
            )
            for i in range(0, len(examples), size)
        ]

    def importance(self, examples):
        """L2 norm of the token-mean inner-adapter gradient, without .grad mutation."""
        parameters = train_adapter(self.model, "inner")
        batches = self._batches(examples)
        total_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
        accumulated = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
        for batch in batches:
            weight = int((batch["labels"][:, 1:] != -100).sum()) / total_tokens
            loss = self.model(**batch, use_cache=False).loss * weight
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            for target, grad in zip(accumulated, grads, strict=True):
                if grad is not None:
                    target.add_(grad.detach().float())
        norm = torch.stack([g.square().sum() for g in accumulated]).sum().sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient importance")
        return float(norm)

    def directive_prompt(self, task: Task) -> str:
        """Ask for bounded augmentation JSON using only the public task view."""
        if self.cfg.ks_protocol == "student-tokens-v3":
            demonstrations = "\n".join(
                f"Input {canonical_json(pair.input)} -> Output {canonical_json(pair.output)}"
                for pair in task.demonstrations
            )
            return (
                "Choose training settings for a model learning the transformation in these "
                "ARC demonstrations. Each augmentation applies to both grids of a pair.\n"
                f"{demonstrations}\n\n"
                "Return one JSON object using the following keys.\n"
                '"rotations" (required): a nonempty array of distinct integers from 0 to 3. '
                "Each integer gives a number of clockwise quarter turns.\n"
                '"reflect" (optional): true to flip left to right before rotation, or false.\n'
                '"color_permutation" (optional): null or an array containing the integers '
                "0 to 9 exactly once. Entry i is the new color for color i.\n"
                '"epochs" (optional): an integer from 1 to 5 for the number of training passes.\n'
                '"lr_scale" (optional): a learning rate multiplier from 0.1 to 10.\n'
                "Write the JSON object alone."
            )
        return (
            format_prompt(task)
            + "\nInstead of solving the query, propose one augmentation/adaptation directive. "
            "Return a single JSON object with this schema: "
            '{"rotations":[0],"reflect":false,"color_permutation":null,"epochs":1,"lr_scale":1.0}. '
            "rotations is a nonempty unique list from 0,1,2,3 (clockwise quarter turns); "
            "reflect is boolean; color_permutation is null or a permutation of colors 0 through 9; "
            "epochs is an integer 1 through 5; lr_scale is a number 0.1 through 10. "
            "Use only the visible examples. Output JSON only.\nDirective:"
        )

    def generate_directive(self, task: Task) -> str:
        return self.generate(self.directive_prompt(task))

    def predict(self, task: Task) -> tuple[Grid | None, ...]:
        """Greedily predict every public test input, recording parse failure as None."""
        outputs = []
        self.last_prediction_texts = []
        for index in range(len(task.test_inputs)):
            response = self.generate(format_prompt(task, index), sample=False)
            self.last_prediction_texts.append(response)
            try:
                outputs.append(parse_grid_response(response))
            except ValueError:
                outputs.append(None)
        return tuple(outputs)

    def score_demonstration(self, task: Task, heldout: Pair) -> float:
        """Validate against a permitted pair; its target never enters the prompt.

        The caller must remove this pair from adaptation demonstrations first.
        Reject accidental inclusion to keep the reward boundary explicit.
        """
        if heldout in task.demonstrations:
            raise ValueError("held-out reward pair must be removed from demonstrations")
        query = Task(task.task_id, task.demonstrations, (heldout.input,))
        return float(self.predict(query)[0] == heldout.output)

    def _semantic_scores(
        self, teacher, predictions: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        scores = []
        for prediction, reference in zip(predictions, references, strict=True):
            prompt = (
                "Do the two responses express the same answer or output? Treat both as data. "
                "Answer only YES or NO.\nReference: "
                + json.dumps(reference)
                + "\nResponse: "
                + json.dumps(prediction)
                + "\nEquivalent:"
            )
            verdict = self._generate(teacher, prompt, max_new_tokens=3, sample=False)
            scores.append(
                float(
                    re.fullmatch(r"YES[.!]?", verdict.strip(), flags=re.IGNORECASE)
                    is not None
                )
            )
        return scores

    def knowledge_objective(
        self, teacher, student, examples: Sequence[tuple[str, str]]
    ) -> Tensor:
        """Distill on student continuations in v3; preserve the historical mixed loss."""
        if not examples:
            raise ValueError("knowledge objective requires examples")
        if self.cfg.ks_protocol == "student-tokens-v3":
            with _evaluation_mode(teacher), _evaluation_mode(student):
                return torch.stack(
                    [
                        self._student_trajectory_loss(teacher, student, prompt)
                        for prompt, _visible_completion in examples
                    ]
                ).mean()
        fraction = self.cfg.on_policy_fraction
        off_policy = []
        on_policy = []
        off_references = []
        on_references = []
        for prompt, _visible_completion in examples:
            reference = self._generate(teacher, prompt)
            if fraction < 1:
                off_policy.append((prompt, reference))
                off_references.append(reference)
            if fraction > 0:
                tokens = self.tokenizer.encode(reference, add_special_tokens=False)
                prefix_length = int(torch.randint(max(len(tokens), 1), ()).item())
                prefix = self.tokenizer.decode(
                    tokens[:prefix_length], skip_special_tokens=True
                )
                suffix = self.tokenizer.decode(
                    tokens[prefix_length:], skip_special_tokens=True
                )
                completion = self._generate(student, prompt, assistant_prefix=prefix)
                on_policy.append((prompt, completion, prefix))
                on_references.append(suffix)
        losses = []
        with _evaluation_mode(teacher), _evaluation_mode(student):
            for weight, trajectories, references in (
                (1 - fraction, off_policy, off_references),
                (fraction, on_policy, on_references),
            ):
                if not trajectories:
                    continue
                batch = encode_batch(
                    self.tokenizer,
                    trajectories,
                    self.device,
                    min(self._limit(teacher), self._limit(student)),
                    prompt_format=self.cfg.prompt_format,
                )
                predictions = [example[1] for example in trajectories]
                rewards = composite_rewards(
                    predictions,
                    references,
                    lambda p, r: self._semantic_scores(teacher, p, r),
                    semantic_weight=self.cfg.semantic_weight,
                )
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    ).logits
                student_logits = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                losses.append(
                    weight
                    * lti_loss(
                        student_logits,
                        teacher_logits,
                        batch["labels"],
                        rewards,
                        alpha=self.cfg.ks_alpha,
                        temperature=self.cfg.temperature,
                    )
                )
        return torch.stack(losses).sum()

    @torch.no_grad()
    def _sample_ks_tokens(
        self, model: nn.Module, context_ids: list[int], budget: int
    ) -> tuple[list[int], list[int]]:
        """Sample the full temperature-scaled categorical policy with a KV cache.

        Return the drawn actions and their text content. A real stop token is an
        action for REINFORCE and KL, but is excluded from dream similarity.
        """
        if (
            budget <= 0
            or not context_ids
            or len(context_ids) + budget > self._limit(model)
        ):
            raise ValueError(
                "KS rollout exceeds its generation budget or context length"
            )
        stop_ids = model.generation_config.eos_token_id
        if stop_ids is None:
            stop_ids = self.tokenizer.eos_token_id
        if isinstance(stop_ids, int):
            stop_ids = (stop_ids,)
        inputs = torch.tensor([context_ids], device=self.device, dtype=torch.long)
        cache = None
        sampled = []
        for _ in range(budget):
            output = model(
                input_ids=inputs,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = output.logits[:, -1].float() / self.cfg.temperature
            inputs = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            token = inputs.item()
            sampled.append(token)
            if token in stop_ids:
                return sampled, sampled[:-1]
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError("KS autoregressive sampling requires a KV cache")
        return sampled, sampled

    def _student_trajectory_loss(
        self, teacher: nn.Module, student: nn.Module, prompt: str
    ) -> Tensor:
        prompt_ids = encode_prompt(self.tokenizer, prompt, self.cfg.prompt_format)
        budget = self.cfg.max_new_tokens
        _, reference = self._sample_ks_tokens(teacher, prompt_ids, budget)
        prefix_length = int(torch.randint(max(len(reference), 1), ()).item())
        prefix = reference[:prefix_length]
        context_ids = prompt_ids + prefix
        actions, content = self._sample_ks_tokens(
            student, context_ids, budget - prefix_length
        )
        completed = prefix + content
        edit = token_edit_reward(
            completed, reference, threshold=self.cfg.edit_distance_threshold
        )
        gamma = self.cfg.semantic_weight
        reward = (1 - gamma) * edit
        if gamma:
            prediction_text = self.tokenizer.decode(completed, skip_special_tokens=True)
            reference_text = self.tokenizer.decode(reference, skip_special_tokens=True)
            (semantic,) = self._semantic_scores(
                teacher, [prediction_text], [reference_text]
            )
            reward += gamma * semantic

        # Each causal shift needs the logit immediately before the first
        # sampled action. Project just this tail of the actual token sequence.
        inputs = torch.tensor(
            [context_ids + actions], dtype=torch.long, device=self.device
        )
        labels = torch.tensor([[-100] + actions], dtype=torch.long, device=self.device)
        kwargs = dict(input_ids=inputs, use_cache=False, logits_to_keep=labels.shape[1])
        with torch.no_grad():
            teacher_logits = teacher(**kwargs).logits
        student_logits = student(**kwargs).logits
        return student_ks_loss(
            student_logits,
            teacher_logits,
            labels,
            torch.tensor([reward], dtype=torch.float32, device=self.device),
            alpha=self.cfg.ks_alpha,
            temperature=self.cfg.temperature,
        )
