from types import SimpleNamespace

from sleep_repro.arc import (
    DreamDirective,
    Pair,
    Task,
    canonical_json,
    holdout_demonstration,
)
from sleep_repro.engine import CausalEngine


def test_v3_generation_prompt_depends_only_on_demonstrations(monkeypatch):
    cfg = SimpleNamespace(ks_protocol="student-tokens-v3")
    engine = CausalEngine(None, None, cfg, "cpu")
    # Capture the normal generation entry point before it calls the model.
    monkeypatch.setattr(engine, "generate", lambda prompt: prompt)
    demos = (Pair([[1, 2]], [[2, 1]]), Pair([[3, 4]], [[4, 3]]))
    task = Task("one", demos, ([[8, 8]], [[7, 7]]))
    other = Task("two", demos, ([[9, 9]],))
    prompt = engine.generate_directive(task)
    assert prompt == engine.generate_directive(other)
    for pair in demos:
        assert canonical_json(pair.input) in prompt
        assert canonical_json(pair.output) in prompt
    assert "{\"rotations\"" not in prompt
    assert DreamDirective().to_json() not in prompt
    assert "Return only the output grid" not in prompt

    support, heldout = holdout_demonstration(task)
    support_prompt = engine.generate_directive(support)
    assert canonical_json(heldout.input) not in support_prompt
    assert canonical_json(heldout.output) not in support_prompt


def test_legacy_directive_prompt_keeps_its_query_conditioning():
    engine = CausalEngine(None, None, SimpleNamespace(ks_protocol="legacy"), "cpu")
    task = Task("t", (Pair([[1]], [[2]]),), ([[8, 7]],))
    assert "[[8,7]]" in engine.directive_prompt(task)
