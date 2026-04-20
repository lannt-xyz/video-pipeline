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
                status="CRAWLED",
                crawled_at=datetime.now(timezone.utc),
            )
        assert db.get_chapter_status(5) == "CRAWLED"

    def test_set_chapter_status_error(self, db):
        from datetime import datetime, timezone

        db.upsert_chapter(1, "Ch 1", "http://x.com", "CRAWLED", None)
        db.set_chapter_status(1, "ERROR", error_msg="timeout")
        assert db.get_chapter_status(1) == "ERROR"

    def test_get_crawled_chapters_range(self, db):
        from datetime import datetime, timezone

        for n in [1, 2, 3, 4, 5]:
            db.upsert_chapter(n, f"Ch{n}", "http://x.com", "CRAWLED", None)
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
        db.record_phase_start(1, "llm")
        time.sleep(0.05)
        db.record_phase_done(1, "llm")

        avg = db.get_avg_phase_duration("llm")
        assert avg is not None
        assert avg > 0.04

    def test_estimate_eta_none_without_data(self, db):
        assert db.estimate_eta(99) is None

    def test_reset_episode_to_phase(self, db):
        db.upsert_episode(1, 1, 35)
        db.set_episode_status(1, "VIDEO_DONE")
        db.reset_episode_to_phase(1, "video")
        assert db.get_episode_status(1) == "AUDIO_DONE"

    def test_reset_episode_to_llm(self, db):
        db.upsert_episode(1, 1, 35)
        db.set_episode_status(1, "VALIDATED")
        db.reset_episode_to_phase(1, "llm")
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


class TestImagePromptEnrichment:
    def test_detect_holding_context_true(self):
        from pipeline.orchestrator import _prompt_mentions_holding_context

        assert _prompt_mentions_holding_context(
            "daoist figure holding a glowing talisman in hand"
        ) is True

    def test_detect_holding_context_false(self):
        from pipeline.orchestrator import _prompt_mentions_holding_context

        assert _prompt_mentions_holding_context(
            "empty ruined shrine corridor, drifting fog"
        ) is False

    def test_artifact_tags_from_text_maps_weapon_and_material(self):
        from pipeline.orchestrator import _artifact_tags_from_text

        tags = _artifact_tags_from_text(
            "Kiếm trấn tà bằng đồng khắc phù văn, phát sáng"
        )
        assert "ornate daoist sword" in tags
        assert "aged bronze texture" in tags
        assert "glowing runic aura" in tags

    def test_build_artifact_prompt_tags_uses_character_hints(self):
        from pipeline.orchestrator import _build_artifact_prompt_tags
        from models.schemas import Character

        char = Character(name="Diệp Thiếu Dương", gender="male", description="1boy, solo")
        hints = {
            "diệp thiếu dương": ["ornate daoist sword", "intricate rune engravings"]
        }
        tags = _build_artifact_prompt_tags(
            "daoist figure holding weapon in hand",
            [(char, ["/tmp/anchor.png"])],
            hints,
        )
        assert "ornate daoist sword" in tags

    def test_compact_prompt_tags_removes_duplicates_and_limits_count(self):
        from pipeline.orchestrator import _compact_prompt_tags

        text = "a, b, c, a, d, e, f"
        compacted = _compact_prompt_tags(text, max_tags=4)
        assert compacted == "a, b, c, d"

    def test_single_character_prompt_is_identity_first(self):
        from pipeline.orchestrator import _build_shot_image_params
        from models.schemas import Character

        char = Character(
            name="Diệp Thiếu Dương",
            gender="male",
            description="1boy, solo, short black hair, sharp eyes, dark jacket",
        )
        _workflow, replacements = _build_shot_image_params(
            prompt_text="ruined shrine interior, daoist figure chanting",
            char_anchor_pairs=[(char, ["/tmp/a.png"])],
            seed=42,
            artifact_hints_by_name=None,
        )

        prompt = replacements["SCENE_PROMPT"]
        assert prompt.startswith("1boy, solo")


class TestThumbnailPrompt:
    def test_thumbnail_prompt_filters_dark_tags_and_adds_bright_tags(self):
        from pipeline.orchestrator import _build_thumbnail_scene_prompt

        src = "dark ruined shrine, moonlight fog, daoist figure, ritual altar"
        out = _build_thumbnail_scene_prompt(src)

        assert "dark ruined shrine" not in out
        assert "moonlight fog" not in out
        assert "daoist figure" in out
        assert "bright cinematic lighting" in out
        assert "high key lighting" in out


class TestLLMPhaseOrchestrator:
    def test_run_llm_skips_profile_build_when_arc_has_no_characters(self):
        from pipeline.orchestrator import run_llm
        from models.schemas import ArcOverview

        db = MagicMock()
        arc = ArcOverview(
            episode_num=1,
            arc_summary="Arc",
            key_events=["event1"],
            characters_in_episode=[],
        )

        with (
            patch("pipeline.orchestrator._episode_chapter_range", return_value=(1, 2)),
            patch("pipeline.orchestrator.vram_manager.acquire"),
            patch("pipeline.orchestrator.vram_manager.health_check_ollama"),
            patch("llm.summarizer.summarize_episode"),
            patch("llm.summarizer.load_arc_overview", return_value=arc),
            patch("llm.profile_builder.build_profiles_for_episode") as mock_build_profiles,
            patch("llm.scriptwriter.write_episode_script"),
        ):
            run_llm(episode_num=1, db=db, dry_run=False)

        mock_build_profiles.assert_not_called()
        db.set_episode_status.assert_any_call(1, "SUMMARIZED")
        db.set_episode_status.assert_any_call(1, "SCRIPTED")
