"""Unit tests for `llm.gatekeeper` — pure-judge constraint reviewer."""

from __future__ import annotations

import json

import pytest

from llm.gatekeeper import gatekeeper_review, log_violations_jsonl
from models.schemas import EpisodeScript, ShotScript


def _shot(narration: str, scene: str = "close-up of hand", duration: float = 5.0) -> ShotScript:
    return ShotScript(
        scene_prompt=scene,
        narration_text=narration,
        duration_sec=duration,
        is_key_shot=False,
        characters=[],
    )


def _script(shots: list[ShotScript]) -> EpisodeScript:
    return EpisodeScript(
        episode_num=1,
        title="Test",
        chapter_range=(1, 2),
        shots=shots,
        characters_in_episode=[],
    )


def test_gatekeeper_passes_clean_script():
    shots = [
        _shot("Tôi mở nắp quan tài."),
        _shot("Tôi nghe tiếng cào nhỏ."),
        _shot("Tôi quay đầu lại nhìn."),
        _shot("Tôi thở gấp gáp."),
    ]
    result = gatekeeper_review(_script(shots))
    assert result.passed
    assert not result.blocking


def test_gatekeeper_blocks_english_hook():
    shots = [
        _shot("The coffin opened slowly in the dark."),
        _shot("Tôi nghe tiếng cào."),
    ]
    result = gatekeeper_review(_script(shots))
    assert not result.passed
    assert any(v.rule == "hook_language" for v in result.blocking)


def test_log_violations_jsonl_appends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shots = [_shot("The English hook"), _shot("Tôi đi.")]
    result = gatekeeper_review(_script(shots))
    log_violations_jsonl(1, result, attempt=0, final=False)
    log_violations_jsonl(1, result, attempt=1, final=True)
    log_path = tmp_path / "logs" / "retention_violations.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["episode"] == 1
    assert rec["blocking_count"] >= 1
    assert rec["attempt"] == 0
    assert rec["final"] is False


def test_summary_contains_violations():
    shots = [_shot("English here only"), _shot("Tôi đi.")]
    result = gatekeeper_review(_script(shots))
    summary = result.summary()
    assert "BLOCKING" in summary or "WARNING" in summary
