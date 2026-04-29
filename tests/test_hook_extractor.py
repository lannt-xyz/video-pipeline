"""Unit tests for `llm.hook_extractor` — coercion, caching, schema validation.

LLM call is mocked; we test the wiring + parsing logic, not LLM quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm.hook_extractor import (
    _HOOK_EXTRACTOR_VERSION,
    _coerce_moment,
    extract_viral_moments,
    load_viral_moments,
)
from models.schemas import ArcOverview, ViralMoment


@pytest.fixture
def fake_arc() -> ArcOverview:
    return ArcOverview(
        episode_num=42,
        arc_summary="Diệp Thiếu Dương khai quật ngôi mộ cổ.",
        key_events=[
            "Mở nắp quan tài, thấy xác đứa trẻ.",
            "Thanh Vân Tử xuất hiện chặn lại.",
        ],
        characters_in_episode=["Diệp Thiếu Dương", "Thanh Vân Tử"],
    )


def _setup_episode_files(tmp_path: Path, episode_num: int) -> None:
    """Create the chunks file the extractor reads."""
    summaries_dir = tmp_path / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    (summaries_dir / f"episode-{episode_num:03d}-chunks.json").write_text(
        json.dumps(
            [
                {
                    "chunk_index": 0,
                    "chapter_start": 1,
                    "chapter_end": 2,
                    "summary": "Diệp Thiếu Dương mở nắp quan tài cổ trong đêm tối.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# _coerce_moment
# ---------------------------------------------------------------------------


def test_coerce_moment_valid_dict():
    item = {
        "chapter_refs": [3, 4],
        "description": "Bàn tay nhỏ áp lên nắp quan tài từ bên trong.",
        "shock_factor": "child corpse moves",
        "mystery_seed": "ai bị chôn sống?",
    }
    m = _coerce_moment(item, fallback_chapter=1)
    assert isinstance(m, ViralMoment)
    assert m.chapter_refs == [3, 4]
    assert "Bàn tay nhỏ" in m.description


def test_coerce_moment_missing_description_returns_none():
    assert _coerce_moment({"chapter_refs": [1]}, fallback_chapter=1) is None


def test_coerce_moment_uses_fallback_chapter_when_refs_invalid():
    item = {
        "chapter_refs": ["not-a-number"],
        "description": "Mặt xác trắng bệch.",
    }
    m = _coerce_moment(item, fallback_chapter=7)
    assert m is not None
    assert m.chapter_refs == [7]


def test_coerce_moment_non_dict_returns_none():
    assert _coerce_moment("not a dict", fallback_chapter=1) is None
    assert _coerce_moment(None, fallback_chapter=1) is None


# ---------------------------------------------------------------------------
# extract_viral_moments wiring
# ---------------------------------------------------------------------------


def test_extract_viral_moments_writes_cache(tmp_path, monkeypatch, fake_arc):
    monkeypatch.setattr("config.settings.settings.data_dir", str(tmp_path))
    _setup_episode_files(tmp_path, fake_arc.episode_num)
    monkeypatch.setattr("llm.hook_extractor.load_arc_overview", lambda n: fake_arc)

    fake_response = {
        "moments": [
            {
                "chapter_refs": [1],
                "description": "Mặt xác trắng bệch trong quan tài.",
                "shock_factor": "corpse face reveal",
                "mystery_seed": "ai vừa chết?",
            },
            {
                "chapter_refs": [2],
                "description": "Tiếng thét xé lòng vọng từ rừng.",
                "shock_factor": "off-screen scream",
                "mystery_seed": "ai đang bị giết?",
            },
        ]
    }

    with patch("llm.hook_extractor._call_llm", return_value=fake_response):
        moments = extract_viral_moments(fake_arc.episode_num)

    assert len(moments) == 2
    out_path = tmp_path / "viral_moments" / f"episode-{fake_arc.episode_num:03d}.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["version"] == _HOOK_EXTRACTOR_VERSION
    assert len(payload["moments"]) == 2


def test_extract_viral_moments_uses_cache(tmp_path, monkeypatch, fake_arc):
    monkeypatch.setattr("config.settings.settings.data_dir", str(tmp_path))
    _setup_episode_files(tmp_path, fake_arc.episode_num)
    monkeypatch.setattr("llm.hook_extractor.load_arc_overview", lambda n: fake_arc)

    out_path = tmp_path / "viral_moments" / f"episode-{fake_arc.episode_num:03d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "version": _HOOK_EXTRACTOR_VERSION,
                "episode_num": fake_arc.episode_num,
                "moments": [
                    {
                        "chapter_refs": [1],
                        "description": "Cached moment.",
                        "shock_factor": "x",
                        "mystery_seed": "y",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with patch("llm.hook_extractor._call_llm") as mock_call:
        moments = extract_viral_moments(fake_arc.episode_num)
        mock_call.assert_not_called()
    assert len(moments) == 1
    assert moments[0].description == "Cached moment."


def test_extract_viral_moments_returns_empty_when_no_chunks(tmp_path, monkeypatch, fake_arc):
    monkeypatch.setattr("config.settings.settings.data_dir", str(tmp_path))
    monkeypatch.setattr("llm.hook_extractor.load_arc_overview", lambda n: fake_arc)
    # No chunks file created.
    moments = extract_viral_moments(fake_arc.episode_num)
    assert moments == []


def test_extract_viral_moments_empty_on_llm_failure(tmp_path, monkeypatch, fake_arc):
    monkeypatch.setattr("config.settings.settings.data_dir", str(tmp_path))
    _setup_episode_files(tmp_path, fake_arc.episode_num)
    monkeypatch.setattr("llm.hook_extractor.load_arc_overview", lambda n: fake_arc)

    with patch("llm.hook_extractor._call_llm", side_effect=RuntimeError("ollama down")):
        moments = extract_viral_moments(fake_arc.episode_num)
    assert moments == []


def test_extract_viral_moments_skips_invalid_items(tmp_path, monkeypatch, fake_arc):
    monkeypatch.setattr("config.settings.settings.data_dir", str(tmp_path))
    _setup_episode_files(tmp_path, fake_arc.episode_num)
    monkeypatch.setattr("llm.hook_extractor.load_arc_overview", lambda n: fake_arc)

    response = {
        "moments": [
            {"description": "good moment.", "chapter_refs": [1]},
            {"chapter_refs": [2]},  # missing description
            "not a dict at all",
        ]
    }
    with patch("llm.hook_extractor._call_llm", return_value=response):
        moments = extract_viral_moments(fake_arc.episode_num)
    assert len(moments) == 1
    assert moments[0].description == "good moment."


def test_load_viral_moments_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.settings.data_dir", str(tmp_path))
    assert load_viral_moments(99) == []
