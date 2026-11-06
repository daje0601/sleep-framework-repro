"""Command line entry points; no downloads or paid jobs occur implicitly."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import torch
from .runtime import ExperimentConfig, seed_all, train_adapter, encode_batch
from .runner import (
    build_model,
    wake_episode,
    write_json,
    save_checkpoint,
    load_checkpoint,
    examples_for,
    evidence,
)
from .arc import Task, Pair, DreamDirective
from .engine import CausalEngine
from .dreaming import evaluate_candidates


def smoke(output):
    """Exercise real gradients on tiny Llama with explicit synthetic fixtures."""
    torch.set_num_threads(1)
    cfg = ExperimentConfig.from_dict(
        dict(
            model="tiny",
            device="cpu",
            dtype="float32",
            rank=2,
            capacity=4,
            policy_rank=2,
            inner_rank=2,
            wake_steps=2,
            consolidation_steps=1,
            max_length=2048,
            max_new_tokens=2,
            train_dreams=2,
            top_k=1,
            random_k=1,
            inner_lr=0.01,
        )
    )
    seed_all(cfg.seed)
    policy, tok = build_model(cfg)
    task = Task(
        "synthetic_smoke", (Pair(((1,),), ((1,),)), Pair(((2,),), ((2,),))), (((3,),),)
    )
    episode, events = wake_episode(policy, tok, cfg, task)
    params = train_adapter(episode, "inner")
    opt = torch.optim.AdamW(params, lr=cfg.inner_lr)
    examples = examples_for(task)

    def adapt(model, optimizer, directive):
        return CausalEngine(model, tok, cfg, "cpu").sft(
            examples, optimizer=optimizer, lr=0.01
        )

    def score(model):
        batch = encode_batch(tok, examples, torch.device("cpu"), cfg.max_length)
        return -float(model(**batch).loss)

    result = evaluate_candidates(
        episode, opt, [DreamDirective()], adapt=adapt, score=score
    )
    outer = train_adapter(policy, "policy")
    outeropt = torch.optim.AdamW(outer, lr=cfg.outer_lr)
    outer_losses = CausalEngine(policy, tok, cfg, "cpu").sft(
        [("Return JSON:", DreamDirective().to_json())],
        adapter="policy",
        optimizer=outeropt,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output / "checkpoint.pt", policy, outeropt, cfg, 1)
    expected = torch.rand(4)
    load_checkpoint(output / "checkpoint.pt", policy, outeropt, cfg)
    torch.testing.assert_close(torch.rand(4), expected)
    report = {
        **evidence(cfg, policy),
        "kind": "synthetic_cpu_integration_check",
        "warning": "Random tiny model and handcrafted dream; no ARC benchmark or paper result.",
        "wake_events": events,
        "candidate_records": [asdict(r) for r in result.records],
        "outer_losses": outer_losses,
        "checkpoint_resume_verified": True,
    }
    write_json(output / "smoke.json", report)
    return {
        "output": str(output / "smoke.json"),
        "checkpoint_resume_verified": True,
        "paper_result_reproduced": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent Sleep ARC reproduction")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--output", default="runs/smoke")
    for name in ("train", "evaluate", "check-config", "preflight"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name != "check-config":
            p.add_argument("--output", required=True)
            if name in ("train", "evaluate"):
                p.add_argument("--checkpoint" if name == "evaluate" else "--resume")
        if name == "evaluate":
            p.add_argument("--allow-untrained", action="store_true")
    p = sub.add_parser("aggregate")
    p.add_argument("runs", nargs="+")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = smoke(args.output)
    elif args.command == "aggregate":
        summaries = [
            json.loads((Path(p) / "summary.json").read_text()) for p in args.runs
        ]
        provenances = [
            json.loads((Path(p) / "provenance.json").read_text()) for p in args.runs
        ]
        seeds = [p["config"]["seed"] for p in provenances]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds cannot count as independent runs")
        signatures = [
            {k: v for k, v in p["config"].items() if k != "seed"} for p in provenances
        ]
        if any(s != signatures[0] for s in signatures):
            raise ValueError("Config mismatch across seeds")
        tasksets = [
            [(t["task_id"], t["total_attempts"]) for t in s["per_task"]]
            for s in summaries
        ]
        if any(t != tasksets[0] for t in tasksets):
            raise ValueError("Task/attempt mismatch across runs")
        if any(p["data_hashes"] != provenances[0]["data_hashes"] for p in provenances):
            raise ValueError("Data mismatch")
        rates = [s["attempt_success_rate"] for s in summaries]
        result = {
            "seeds": seeds,
            "rates": rates,
            "mean": statistics.mean(rates),
            "std_across_seeds": statistics.stdev(rates) if len(rates) > 1 else None,
            "note": "Descriptive seed variability; correlated attempts are not independent trials.",
            "paper_result_reproduced": False,
        }
        write_json(args.output, result)
    else:
        cfg = ExperimentConfig.from_dict(json.loads(Path(args.config).read_text()))
        if args.command == "check-config":
            result = cfg.to_dict()
        elif args.command == "preflight":
            from .preflight import preflight

            result = preflight(cfg, args.output)
        elif args.command == "train":
            from .runner import train

            result = {"checkpoint": str(train(cfg, args.output, args.resume))}
        else:
            from .runner import evaluate

            result = evaluate(cfg, args.output, args.checkpoint, args.allow_untrained)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
