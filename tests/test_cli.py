import json
import torch
from sleep_repro.cli import main


def test_offline_smoke_runs_actual_model_and_marks_no_reproduction(tmp_path, capsys):
    main(["smoke", "--output", str(tmp_path)])
    report = json.loads((tmp_path / "smoke.json").read_text())
    assert report["checkpoint_resume_verified"]
    assert report["paper_result_reproduced"] is False
    assert len(report["candidate_records"]) == 1
    assert report["outer_losses"][0] > 0
    assert any(e["event"] == "consolidation" for e in report["wake_events"])

    assert torch.load(tmp_path / "checkpoint.pt", weights_only=True)["purpose"] == "diagnostic"
