import json
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import ArcOverview, EpisodeScript, ShotScript


# ── OllamaClient ──────────────────────────────────────────────────────────────

class TestOllamaClient:
    def test_generate_returns_text(self):
        from llm.client import OllamaClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello from Ollama"}
        mock_response.raise_for_status = MagicMock()

        with patch("llm.client.httpx.post", return_value=mock_response):
            client = OllamaClient()
            result = client.generate("test prompt")

        assert result == "Hello from Ollama"

    def test_generate_json_parses_response(self):
        from llm.client import OllamaClient

        payload = {"arc_summary": "Test arc", "key_events": [], "characters_in_episode": []}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": json.dumps(payload)}
        mock_response.raise_for_status = MagicMock()

        with patch("llm.client.httpx.post", return_value=mock_response):
            client = OllamaClient()
            result = client.generate_json("test prompt")

        assert result["arc_summary"] == "Test arc"

    def test_health_check_ok(self):
        from llm.client import OllamaClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("llm.client.httpx.get", return_value=mock_response):
            client = OllamaClient()
            assert client.health_check() is True

    def test_health_check_raises_on_failure(self):
        import httpx as _httpx
        from llm.client import OllamaClient

        with patch("llm.client.httpx.get", side_effect=_httpx.NetworkError("down")):
            client = OllamaClient()
            with pytest.raises(RuntimeError, match="not available"):
                client.health_check()


# ── Summarizer ─────────────────────────────────────────────────────────────────

class TestSummarizer:
    def test_load_arc_overview_not_found(self):
        from llm.summarizer import load_arc_overview

        with patch("llm.summarizer.settings") as mock_settings:
            mock_settings.data_dir = "/nonexistent"
            with pytest.raises(FileNotFoundError):
                load_arc_overview(999)

    def test_save_and_load_chunk_summary(self, tmp_path):
        from llm.summarizer import _save_chunk_summary
        from models.schemas import ChunkSummary

        chunk = ChunkSummary(
            episode_num=1,
            chunk_index=0,
            chapter_start=1,
            chapter_end=5,
            summary="Test summary content",
        )

        with patch("llm.summarizer.settings") as mock_settings:
            mock_settings.data_dir = str(tmp_path)
            _save_chunk_summary(chunk)
            _save_chunk_summary(chunk)  # idempotent — same index overwrites

        saved = json.loads(
            (tmp_path / "summaries" / "episode-001-chunks.json").read_text()
        )
        assert len(saved) == 1
        assert saved[0]["summary"] == "Test summary content"


# ── Scriptwriter ──────────────────────────────────────────────────────────────

class TestScriptwriter:
    def _make_shots(self, n=8, key_shots_at=None):
        key = set(key_shots_at or [0, 3])
        return [
            ShotScript(
                scene_prompt=f"Scene {i} anime style",
                narration_text=f"Narration {i}",
                duration_sec=6,
                is_key_shot=(i in key),
                characters=["Diep Thieu Duong"] if i == 0 else [],
            )
            for i in range(n)
        ]

    def test_normalize_promotes_key_shots_when_too_few(self):
        from llm.scriptwriter import _normalize_key_shots

        shots = self._make_shots(8, key_shots_at=[])  # 0 key shots
        normalized = _normalize_key_shots(shots, episode_num=1)
        key_count = sum(1 for s in normalized if s.is_key_shot)
        assert key_count == 2

    def test_normalize_caps_key_shots_when_too_many(self):
        from llm.scriptwriter import _normalize_key_shots

        shots = self._make_shots(8, key_shots_at=[0, 1, 2, 3, 4])  # 5 key shots
        normalized = _normalize_key_shots(shots, episode_num=1)
        key_count = sum(1 for s in normalized if s.is_key_shot)
        assert key_count == 3

    def test_write_episode_script_saves_json(self, tmp_path):
        from llm.scriptwriter import write_episode_script, load_episode_script

        arc = ArcOverview(
            episode_num=1,
            arc_summary="Arc summary text",
            key_events=["event1", "event2"],
            characters_in_episode=["Diep Thieu Duong"],
        )

        mock_raw = {
            "title": "Tập 1: Khởi Đầu",
            "shots": [
                {
                    "scene_prompt": f"Scene {i} anime style 9:16",
                    "narration_text": f"Lời dẫn {i}",
                    "duration_sec": 6,
                    "is_key_shot": i < 2,
                    "characters": [],
                }
                for i in range(8)
            ],
        }

        with (
            patch("llm.scriptwriter.load_arc_overview", return_value=arc),
            patch("llm.scriptwriter.ollama_client") as mock_llm,
            patch("llm.scriptwriter.settings") as mock_settings,
        ):
            mock_llm.generate_json.return_value = mock_raw
            mock_settings.data_dir = str(tmp_path)
            mock_settings.llm_max_retries = 3
            mock_settings.target_duration_sec = 60
            mock_settings.shot_transition_duration = 0.3

            script = write_episode_script(1)

        assert script.episode_num == 1
        assert len(script.shots) == 8
        assert script.title == "Tập 1: Khởi Đầu"

    def test_load_episode_script_not_found(self):
        from llm.scriptwriter import load_episode_script

        with patch("llm.scriptwriter.settings") as mock_settings:
            mock_settings.data_dir = "/nonexistent"
            with pytest.raises(FileNotFoundError):
                load_episode_script(999)


# ── Schemas validation ─────────────────────────────────────────────────────────

class TestSchemas:
    def test_shot_script_defaults(self):
        shot = ShotScript(
            scene_prompt="a scene",
            narration_text="lời dẫn",
        )
        assert shot.duration_sec == 6
        assert shot.is_key_shot is False
        assert shot.characters == []

    def test_episode_script_rejects_empty_shots(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EpisodeScript(episode_num=1, title="T", shots=[])

    def test_arc_overview_round_trip(self):
        arc = ArcOverview(
            episode_num=5,
            arc_summary="summary",
            key_events=["e1"],
            characters_in_episode=["char1"],
        )
        reloaded = ArcOverview(**json.loads(arc.model_dump_json()))
        assert reloaded.episode_num == 5
        assert reloaded.key_events == ["e1"]
