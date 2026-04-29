"""Unit tests for `llm.hook_judge` — competitive selection logic.

Generator + judge LLM calls are mocked. Tests verify:
- Pre-filter rejects English candidates
- Score weighting matches retention.hook_judge_weights
- Retry triggered when winner < threshold
- Persistence writes to hook_candidates/episode-NNN.json
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm.hook_judge import (
    _coerce_candidate,
    _score_candidate,
    select_hook,
)
from models.schemas import HookCandidate


# ---------------------------------------------------------------------------
# _coerce_candidate
# ---------------------------------------------------------------------------


def test_coerce_candidate_strips_and_builds():
    out = _coerce_candidate({"text": "  Hắn mở quan tài.  ", "visual_seed": " coffin opening "})
    assert out is not None
    assert out.text == "Hắn mở quan tài."
    assert out.visual_seed == "coffin opening"


def test_coerce_candidate_empty_text_returns_none():
    assert _coerce_candidate({"text": "", "visual_seed": "x"}) is None
    assert _coerce_candidate({"text": "   "}) is None


# ---------------------------------------------------------------------------
# _score_candidate
# ---------------------------------------------------------------------------


def test_score_candidate_applies_weights():
    c = HookCandidate(text="Hắn mở quan tài.", visual_seed="x")
    score = {
        "curiosity_gap": 1.0,
        "specificity": 0.5,
        "pattern_interrupt": 0.5,
        "rationale": "ok",
    }
    out = _score_candidate(c, score)
    # Default weights: 0.5, 0.25, 0.25 → 0.5*1.0 + 0.25*0.5 + 0.25*0.5 = 0.75
    assert out.total_score == pytest.approx(0.75)
    assert out.curiosity_score == 1.0
    assert out.rationale == "ok"


def test_score_candidate_clamps_out_of_range():
    c = HookCandidate(text="x", visual_seed="x")
    score = {"curiosity_gap": 1.5, "specificity": -0.3, "pattern_interrupt": "bad"}
    out = _score_candidate(c, score)
    assert out.curiosity_score == 1.0
    assert out.specificity_score == 0.0
    assert out.pattern_interrupt_score == 0.0


# ---------------------------------------------------------------------------
# select_hook end-to-end (mocked LLM)
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.settings.data_dir", str(tmp_path))
    return tmp_path


def test_select_hook_filters_english_then_judges(patch_data_dir):
    gen_response = [
        {"text": "Hắn mở nắp quan tài.", "visual_seed": "coffin lid opens"},
        {"text": "The coffin opened.", "visual_seed": "coffin lid"},  # English → filtered
        {"text": "Bàn tay nhỏ áp lên kính.", "visual_seed": "small hand on glass"},
    ]
    # Judge sees only 2 candidates (English filtered out).
    judge_response = [
        {"curiosity_gap": 0.9, "specificity": 0.8, "pattern_interrupt": 0.7, "rationale": "good"},
        {"curiosity_gap": 0.7, "specificity": 0.6, "pattern_interrupt": 0.6, "rationale": "ok"},
    ]

    with patch("llm.hook_judge._call_generator", return_value=gen_response), \
         patch("llm.hook_judge._call_judge", return_value=judge_response):
        result = select_hook("arc text", episode_num=5)

    assert result is not None
    winner, all_cands = result
    assert len(all_cands) == 2  # English candidate filtered before judging
    assert winner.text == "Hắn mở nắp quan tài."  # higher score
    # Persistence
    out_path = patch_data_dir / "hook_candidates" / "episode-005.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["winner_text"] == winner.text
    assert payload["winner_score"] == winner.total_score


def test_select_hook_retries_when_below_threshold(patch_data_dir, monkeypatch):
    monkeypatch.setattr("config.settings.settings.retention.hook_min_score", 0.9)

    weak_gen = [{"text": "Câu yếu.", "visual_seed": "x"}]
    weak_score = [
        {"curiosity_gap": 0.3, "specificity": 0.3, "pattern_interrupt": 0.3, "rationale": "weak"}
    ]
    strong_gen = [{"text": "Câu mạnh.", "visual_seed": "y"}]
    strong_score = [
        {"curiosity_gap": 1.0, "specificity": 1.0, "pattern_interrupt": 1.0, "rationale": "strong"}
    ]

    gen_calls = {"n": 0}

    def gen_side_effect(arc_text, episode_num, hint=""):
        gen_calls["n"] += 1
        return strong_gen if gen_calls["n"] == 2 else weak_gen

    judge_calls = {"n": 0}

    def judge_side_effect(cands):
        judge_calls["n"] += 1
        return strong_score if judge_calls["n"] == 2 else weak_score

    with patch("llm.hook_judge._call_generator", side_effect=gen_side_effect), \
         patch("llm.hook_judge._call_judge", side_effect=judge_side_effect):
        result = select_hook("arc text", episode_num=7)

    assert result is not None
    winner, all_cands = result
    assert winner.text == "Câu mạnh."  # retry winner replaces weak one
    assert len(all_cands) == 2  # both rounds persisted
    assert gen_calls["n"] == 2  # retry happened


def test_select_hook_returns_none_when_generator_fails(patch_data_dir):
    with patch("llm.hook_judge._call_generator", side_effect=RuntimeError("ollama down")):
        assert select_hook("arc", episode_num=1) is None


def test_select_hook_returns_none_when_all_filtered(patch_data_dir):
    # All candidates English → all rejected by pre-filter → no judge call → None.
    gen_response = [
        {"text": "First English line.", "visual_seed": "x"},
        {"text": "Second English line.", "visual_seed": "y"},
    ]
    with patch("llm.hook_judge._call_generator", return_value=gen_response), \
         patch("llm.hook_judge._call_judge") as mock_judge:
        result = select_hook("arc", episode_num=2)
    assert result is None
    mock_judge.assert_not_called()
