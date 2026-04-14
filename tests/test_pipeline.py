import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.state import StateDB


# ── StateDB schema & basics ───────────────────────────────────────────────────

class TestStateDB:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "test_pipeline.db")
        with patch("pipeline.state.settings") as mock_settings:
            mock_settings.db_path = db_path
            yield StateDB(db_path=db_path)

    def test_init_creates_tables(self, db):
        conn = sqlite3.connect(db.db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "chapters" in tables
        assert "episodes" in tables
        assert "episode_timings" in tables

    def test_upsert_chapter_and_get_status(self, db):
        from datetime import datetime, timezone

        db.upsert_chapter(
            chapter_num=1,
            title="Ch 1",
            url="http://example.com",
            file_path="/data/chuong-0001.txt",
            status="CRAWLED",
            crawled_at=datetime.now(timezone.utc),
        )
        assert db.get_chapter_status(1) == "CRAWLED"

    def test_get_chapter_status_returns_none_for_missing(self, db):
        assert db.get_chapter_status(9999) is None

    def test_upsert_chapter_is_idempotent(self, db):
        from datetime import datetime, timezone

        for _ in range(3):
            db.upsert_chapter(
                chapter_num=5,
                title="Ch 5",
                url="http://example.com",
                file_path="/data/chuong-0005.txt",
                status="CRAWLED",
                crawled_at=datetime.now(timezone.utc),
            )
        assert db.get_chapter_status(5) == "CRAWLED"

    def test_set_chapter_status_error(self, db):
        from datetime import datetime, timezone

        db.upsert_chapter(1, "Ch 1", "http://x.com", "/f", "CRAWLED", None)
        db.set_chapter_status(1, "ERROR", error_msg="timeout")
        assert db.get_chapter_status(1) == "ERROR"

    def test_get_crawled_chapters_range(self, db):
        from datetime import datetime, timezone

        for n in [1, 2, 3, 4, 5]:
            db.upsert_chapter(n, f"Ch{n}", "http://x.com", f"/f{n}", "CRAWLED", None)
        # Chapter 3 set to ERROR
        db.set_chapter_status(3, "ERROR")

        crawled = db.get_crawled_chapters(1, 5)
        assert set(crawled) == {1, 2, 4, 5}

    def test_upsert_episode_and_status(self, db):
        db.upsert_episode(1, chapter_start=1, chapter_end=35)
        assert db.get_episode_status(1) == "PENDING"

        db.set_episode_status(1, "CRAWLED")
        assert db.get_episode_status(1) == "CRAWLED"

    def test_upsert_episode_no_duplicate(self, db):
        db.upsert_episode(1, 1, 35)
        db.upsert_episode(1, 1, 35)  # should not raise
        assert db.get_episode_status(1) == "PENDING"

    def test_record_phase_timing(self, db):
        import time

        db.upsert_episode(1, 1, 35)
        db.record_phase_start(1, "crawl")
        time.sleep(0.05)
        db.record_phase_done(1, "crawl")

        avg = db.get_avg_phase_duration("crawl")
        assert avg is not None
        assert avg > 0.04

    def test_estimate_eta_none_without_data(self, db):
        assert db.estimate_eta(99) is None

    def test_reset_episode_to_phase(self, db):
        db.upsert_episode(1, 1, 35)
        db.set_episode_status(1, "VIDEO_DONE")
        db.reset_episode_to_phase(1, "video")
        assert db.get_episode_status(1) == "AUDIO_DONE"

    def test_reset_episode_to_crawl(self, db):
        db.upsert_episode(1, 1, 35)
        db.set_episode_status(1, "VALIDATED")
        db.reset_episode_to_phase(1, "crawl")
        assert db.get_episode_status(1) == "PENDING"


# ── VRAMManager ───────────────────────────────────────────────────────────────

class TestVRAMManager:
    def test_health_check_ollama_ok(self):
        from pipeline.vram_manager import VRAMManager

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("pipeline.vram_manager.httpx.get", return_value=mock_resp):
            vm = VRAMManager()
            assert vm.health_check_ollama(timeout=5) is True

    def test_health_check_ollama_raises_on_timeout(self):
        import httpx as _httpx
        from pipeline.vram_manager import VRAMManager

        with patch(
            "pipeline.vram_manager.httpx.get",
            side_effect=_httpx.NetworkError("fail"),
        ):
            vm = VRAMManager()
            with pytest.raises(RuntimeError, match="not responding"):
                vm.health_check_ollama(timeout=1)

    def test_health_check_comfyui_ok(self):
        from pipeline.vram_manager import VRAMManager

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("pipeline.vram_manager.httpx.get", return_value=mock_resp):
            vm = VRAMManager()
            assert vm.health_check_comfyui(timeout=5) is True


# ── ComfyUIClient ─────────────────────────────────────────────────────────────

class TestComfyUIClient:
    def test_health_check_ok(self):
        from image_gen.comfyui_client import ComfyUIClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch("image_gen.comfyui_client.httpx.get", return_value=mock_resp):
            client = ComfyUIClient()
            assert client.health_check() is True

    def test_health_check_raises_on_failure(self):
        import httpx as _httpx
        from image_gen.comfyui_client import ComfyUIClient

        with patch(
            "image_gen.comfyui_client.httpx.get",
            side_effect=_httpx.NetworkError("down"),
        ):
            client = ComfyUIClient()
            with pytest.raises(RuntimeError, match="not available"):
                client.health_check()

    def test_load_workflow_string_replacement(self, tmp_path):
        from image_gen.comfyui_client import ComfyUIClient

        workflow_file = tmp_path / "test.json"
        workflow_file.write_text(
            '{"node": {"inputs": {"text": "__SCENE_PROMPT__", "width": "__WIDTH__", "seed": "__SEED__"}}}',
            encoding="utf-8",
        )

        client = ComfyUIClient()
        result = client._load_workflow(
            str(workflow_file),
            {"SCENE_PROMPT": "anime girl", "WIDTH": 720, "SEED": 42},
        )

        assert result["node"]["inputs"]["text"] == "anime girl"
        assert result["node"]["inputs"]["width"] == 720
        assert result["node"]["inputs"]["seed"] == 42

    def test_submit_prompt_returns_id(self):
        from image_gen.comfyui_client import ComfyUIClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"prompt_id": "abc-123"}
        mock_resp.raise_for_status = MagicMock()

        with patch("image_gen.comfyui_client.httpx.post", return_value=mock_resp):
            client = ComfyUIClient()
            prompt_id = client.submit_prompt({"1": {"class_type": "test"}})

        assert prompt_id == "abc-123"

    def test_poll_result_detects_oom(self):
        from image_gen.comfyui_client import ComfyUIClient, ComfyUIOutOfMemoryError

        oom_entry = {"error": "CUDA out of memory. Tried to allocate 2GB"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"test-id": oom_entry}
        mock_resp.raise_for_status = MagicMock()

        with patch("image_gen.comfyui_client.httpx.get", return_value=mock_resp):
            client = ComfyUIClient()
            client.timeout = 1
            client.poll_interval = 0
            with pytest.raises(ComfyUIOutOfMemoryError):
                client.poll_result("test-id")
