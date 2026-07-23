# -*- coding: utf-8 -*-
"""Tests for the AFAC v2.0 supervisor package (synthetic, fast, fake clocks)."""
from __future__ import annotations

import json
from pathlib import Path

from afac_agent.supervisor import (
    HeartbeatState,
    HeartbeatWriter,
    ResumeManager,
    RunEventLog,
    RunSupervisor,
    StallDetector,
    UnbufferedTextLog,
)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


REQUIRED_HEARTBEAT_FIELDS = {
    "run_id", "task", "status", "stage", "substage", "current_experiment",
    "current_model", "rounds_used", "elapsed_seconds", "remaining_seconds",
    "cpu_active", "gpu_active", "latest_metric", "last_progress_time",
    "next_checkpoint", "estimated_completion", "safe_resume_point", "stall_status",
}


def test_heartbeat_force_tick_writes_all_fields(tmp_path: Path) -> None:
    clock = FakeClock()
    writer = HeartbeatWriter(tmp_path, interval_seconds=30.0, clock=clock, budget_seconds=3600.0)
    writer.update(run_id="run-1", task="A1", status="running", stage="profiling")
    assert writer.tick(force=True) is True
    payload = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert REQUIRED_HEARTBEAT_FIELDS <= set(payload)
    assert payload["run_id"] == "run-1"
    assert payload["stage"] == "profiling"
    assert payload["remaining_seconds"] == 3600.0


def test_heartbeat_not_rewritten_before_interval(tmp_path: Path) -> None:
    clock = FakeClock()
    writer = HeartbeatWriter(tmp_path, interval_seconds=30.0, clock=clock)
    assert writer.tick(force=True) is True
    clock.advance(5.0)
    assert writer.tick() is False  # interval (30s) not elapsed
    clock.advance(30.0)
    assert writer.tick() is True


def test_cpu_float_and_gpu_default_false(tmp_path: Path) -> None:
    clock = FakeClock()
    writer = HeartbeatWriter(tmp_path, clock=clock)
    writer.tick(force=True)
    assert isinstance(writer.state.cpu_active, float)
    assert writer.state.gpu_active is False
    probed = HeartbeatWriter(tmp_path / "gpu", clock=clock, gpu_probe=lambda: True)
    probed.tick(force=True)
    assert probed.state.gpu_active is True


def test_status_md_and_dashboard_contents(tmp_path: Path) -> None:
    clock = FakeClock()
    sup = RunSupervisor(
        tmp_path, task="A1", run_id="run-42", budget_seconds=1800.0, clock=clock,
    )
    sup.start("data_profiling")
    sup.complete_experiment("exp_baseline_lgbm")
    sup.heartbeat(current_model="ridge", latest_metric={"map@10": 0.0123})
    sup.complete_stage("data_profiling", outputs=["profile.json"])
    status_md = (tmp_path / "STATUS.md").read_text(encoding="utf-8")
    dashboard = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    for text in (status_md, dashboard):
        assert "run-42" in text
        assert "data_profiling" in text
        assert "ok" in text  # stall status
        assert "after:data_profiling" in text  # safe resume point
    assert "exp_baseline_lgbm" in dashboard
    assert 'http-equiv="refresh"' in dashboard
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["safe_resume_point"] == "after:data_profiling"


def test_event_log_appends_valid_jsonl(tmp_path: Path) -> None:
    clock = FakeClock()
    log = RunEventLog(tmp_path, clock=clock)
    log.append("stage_start", {"stage": "s1"})
    clock.advance(1.0)
    log.append("metric", {"value": 0.5})
    lines = (tmp_path / "run_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    assert events[0]["event_type"] == "stage_start"
    assert events[0]["payload"] == {"stage": "s1"}
    assert events[1]["event_type"] == "metric"
    assert all("timestamp" in e for e in events)
    text = UnbufferedTextLog(tmp_path)
    text.write("hello")
    text.write("world")
    assert (tmp_path / "unbuffered.log").read_text(encoding="utf-8").splitlines() == ["hello", "world"]


def test_stall_detector_transitions() -> None:
    clock = FakeClock()
    detector = StallDetector(suspect_after_seconds=180.0, stall_after_seconds=600.0, clock=clock)
    last_progress = clock()
    assert detector.status(last_progress) == "ok"
    clock.advance(179.0)
    assert detector.status(last_progress) == "ok"
    clock.advance(2.0)  # 181s idle
    assert detector.status(last_progress) == "suspected_stall"
    clock.advance(500.0)  # 681s idle
    assert detector.status(last_progress) == "stalled"


def test_resume_manager_skip_and_no_double_count(tmp_path: Path) -> None:
    manager = ResumeManager(tmp_path)
    manager.save_checkpoint("s1", rounds_used=3, remaining_budget=900.0)
    manager.mark_completed("s1", outputs=["a.json", "b.json"])
    manager.save_checkpoint("s2", rounds_used=2)

    reloaded = ResumeManager(tmp_path)
    assert reloaded.should_skip("s1") is True
    assert reloaded.should_skip("s2") is False
    assert reloaded.safe_resume_point() == "after:s1"
    assert reloaded.load()["completed_outputs"]["s1"] == ["a.json", "b.json"]
    # 3 (s1) + 2 (s2), summed from per-stage entries: no double counting
    assert reloaded.load()["scientific_rounds_used"] == 5
    # Re-checkpointing s1 replaces its rounds instead of accumulating
    reloaded.save_checkpoint("s1", rounds_used=3)
    assert ResumeManager(tmp_path).load()["scientific_rounds_used"] == 5
    # Safe resume point persists across reload
    assert ResumeManager(tmp_path).safe_resume_point() == "after:s1"


def test_run_supervisor_write_all(tmp_path: Path) -> None:
    clock = FakeClock()
    sup = RunSupervisor(tmp_path, task="B2", run_id="run-7", budget_seconds=600.0, clock=clock)
    sup.start("fold_build")
    sup.heartbeat(stage="fold_build", substage="hashing", rounds_used=1)
    sup.write_all()
    for name in ("heartbeat.json", "STATUS.md", "dashboard.html", "run_events.jsonl"):
        assert (tmp_path / name).exists(), name
    events = sup.events.load()
    assert [e["event_type"] for e in events] == ["stage_start"]
    state = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert state["task"] == "B2"
    assert state["rounds_used"] == 1
    assert isinstance(state["cpu_active"], float)
