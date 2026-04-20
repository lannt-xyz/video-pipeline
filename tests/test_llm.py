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

    def test_generate_json_extracts_object_from_wrapped_text(self):
        from llm.client import OllamaClient

        payload = {
            "arc_summary": "Test arc",
            "key_events": ["e1"],
            "characters_in_episode": ["Diep Thieu Duong"],
        }
        wrapped = f"Result below:\n```json\n{json.dumps(payload)}\n```\nDone."

        client = OllamaClient()
        with patch.object(client, "generate", return_value=wrapped):
            result = client.generate_json("test prompt")

        assert result["key_events"] == ["e1"]

    def test_generate_json_repairs_malformed_once_then_parses(self):
        from llm.client import OllamaClient

        malformed = (
            '{"arc_summary":"Mo dau voi tu khoa bi loi "quote"",'
            '"key_events":[],"characters_in_episode":[]}'
        )
        repaired = json.dumps(
            {
                "arc_summary": "Mo dau voi tu khoa bi loi quote",
                "key_events": [],
                "characters_in_episode": [],
            }
        )

        client = OllamaClient()
        with patch.object(client, "generate", side_effect=[malformed, repaired]):
            result = client.generate_json("test prompt")

        assert result["arc_summary"] == "Mo dau voi tu khoa bi loi quote"

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
        from llm.summarizer import _save_chunk_summary, _SUMMARY_PROMPT_VERSION
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
        assert saved[0]["summary_version"] == _SUMMARY_PROMPT_VERSION

    def test_normalize_characters_filters_unresolved_id_tokens(self):
        from llm.summarizer import _normalize_characters_in_episode

        with patch(
            "llm.summarizer._load_character_lookup",
            return_value=(
                {"qing_yun_zi": "Thanh Vân Tử"},
                {"thanh vân tử": "Thanh Vân Tử", "thiếu dương": "Thiếu Dương"},
            ),
        ):
            out = _normalize_characters_in_episode(
                ["qing_yun_zi", "Thiếu Dương", "unknown_slug_name"]
            )

        assert out == ["Thanh Vân Tử", "Thiếu Dương"]

    def test_synthesize_arc_fallback_when_key_events_missing(self):
        from llm.summarizer import _synthesize_arc

        chunk_summaries = [
            "- Diệp Đại Bảo mở nắp quan tài trong khu mộ tổ tiên vào lúc nửa đêm.\n"
            "- Ông ta rút một lá bùa cháy dở và niệm chú giữa màn mưa nặng hạt.",
            "- Một bóng người xuất hiện ở cổng miếu và ném chuông đồng xuống sân đá.\n"
            "- Cả nhóm lao vào hành lang tối để truy đuổi kẻ lạ mặt.",
        ]

        with patch("llm.summarizer.ollama_client") as mock_llm:
            mock_llm.generate_json.return_value = {
                "arc_summary": "Tóm tắt vẫn có nhưng model quên key_events.",
                "characters_in_episode": ["Diệp Đại Bảo"],
            }
            arc = _synthesize_arc(chunk_summaries, episode_num=1)

        assert arc.arc_summary.startswith("Tóm tắt")
        assert len(arc.key_events) >= 1
        assert any("Diệp Đại Bảo" in ev or "quan tài" in ev for ev in arc.key_events)

    def test_synthesize_arc_accepts_string_characters_field(self):
        from llm.summarizer import _synthesize_arc

        with (
            patch("llm.summarizer.ollama_client") as mock_llm,
            patch(
                "llm.summarizer._load_character_lookup",
                return_value=({}, {"thiếu dương": "Thiếu Dương"}),
            ),
        ):
            mock_llm.generate_json.return_value = {
                "arc_summary": "Tóm tắt.",
                "key_events": ["Một cảnh hành động xảy ra ở sân miếu."],
                "characters_in_episode": "Thiếu Dương",
            }
            arc = _synthesize_arc(["Chunk 1"], episode_num=1)

        assert arc.characters_in_episode == ["Thiếu Dương"]

    def test_synthesize_arc_infers_characters_from_text_when_field_missing(self):
        from llm.summarizer import _synthesize_arc

        chunk_summaries = [
            "Diệp Đại Bảo mở nắp quan tài ở khu mộ tổ tiên.",
            "Thiếu Dương cầm kiếm trấn tà lao vào hành lang miếu cổ.",
        ]

        with (
            patch("llm.summarizer.ollama_client") as mock_llm,
            patch(
                "llm.summarizer._load_character_lookup",
                return_value=(
                    {},
                    {
                        "diệp đại bảo": "Diệp Đại Bảo",
                        "thiếu dương": "Thiếu Dương",
                    },
                ),
            ),
        ):
            mock_llm.generate_json.return_value = {
                "arc_summary": "Tóm tắt vẫn có nội dung.",
                "key_events": [
                    "Diệp Đại Bảo niệm chú trong mưa.",
                    "Thiếu Dương chém vỡ phong ấn ở cửa miếu.",
                ],
            }
            arc = _synthesize_arc(chunk_summaries, episode_num=1)

        assert "Diệp Đại Bảo" in arc.characters_in_episode
        assert "Thiếu Dương" in arc.characters_in_episode


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

    def test_write_episode_script_handles_shots_as_strings(self, tmp_path):
        from llm.scriptwriter import write_episode_script

        arc = ArcOverview(
            episode_num=1,
            arc_summary="Arc",
            key_events=["event1"],
            characters_in_episode=["Diep Thieu Duong"],
        )
        shot_dict = {
            "scene_prompt": "ruined shrine, fog",
            "narration_text": "Lời dẫn cảnh đầu.",
            "duration_sec": 6,
            "is_key_shot": False,
            "characters": [],
        }
        # Ollama returns shots as JSON strings — should be coerced to dicts
        mock_raw = {
            "title": "Tập 1: Lỗi Shots",
            "shots": [json.dumps(shot_dict) for _ in range(8)],
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

        assert len(script.shots) == 8
        assert "ruined shrine" in script.shots[0].scene_prompt

    def test_coerce_shot_item_returns_none_for_plain_string(self):
        from llm.scriptwriter import _coerce_shot_item

        result = _coerce_shot_item("just a plain sentence, not JSON", episode_num=1, index=0)
        assert result is None

    def test_coerce_shot_item_parses_json_string(self):
        from llm.scriptwriter import _coerce_shot_item

        shot = {"scene_prompt": "test scene", "narration_text": "lời dẫn", "duration_sec": 6}
        result = _coerce_shot_item(json.dumps(shot), episode_num=1, index=0)
        assert isinstance(result, dict)
        assert result["scene_prompt"] == "test scene"

    def test_load_episode_script_not_found(self):
        from llm.scriptwriter import load_episode_script

        with patch("llm.scriptwriter.settings") as mock_settings:
            mock_settings.data_dir = "/nonexistent"
            with pytest.raises(FileNotFoundError):
                load_episode_script(999)

    def test_align_scene_prompt_with_narration_adds_missing_visual_tags(self):
        from llm.scriptwriter import _align_scene_prompt_with_narration

        shots = [
            ShotScript(
                scene_prompt="muddy excavation pit, man digging with shovel, anime style, manhua art style, no text, no watermarks",
                narration_text="Diệp Đại Bảo dùng dù đỏ che mưa rồi cắm nhang trước mộ.",
                duration_sec=6,
            )
        ]

        out = _align_scene_prompt_with_narration(shots, episode_num=1)
        assert "red umbrella" in out[0].scene_prompt
        assert "burning incense sticks" in out[0].scene_prompt

    def test_align_injects_action_tag_early_and_removes_weak_pose(self):
        from llm.scriptwriter import _align_scene_prompt_with_narration

        shots = [
            ShotScript(
                scene_prompt="muddy cemetery at night, figure performing ritual, ancient stone gateway in background, cold moonlight",
                narration_text="Anh dùng xẻng khai quật phần mộ, phát hiện ra một quan tài màu đỏ tươi.",
                duration_sec=6,
            )
        ]

        out = _align_scene_prompt_with_narration(shots, episode_num=1)
        prompt = out[0].scene_prompt

        # Action tag must be present and come BEFORE the background tags
        assert "figure crouching and digging with long-handled shovel" in prompt
        # Weak generic pose tag must be replaced
        assert "figure performing ritual" not in prompt
        # Object tag (coffin) should also be appended
        assert "red lacquered coffin" in prompt

    def test_align_action_tag_inserted_before_background_not_at_end(self):
        from llm.scriptwriter import _align_scene_prompt_with_narration

        shots = [
            ShotScript(
                scene_prompt="wooden hall interior, figure standing, wooden pillars in background, warm lantern light",
                narration_text="Thanh Vân Tử chỉ thẳng vào Diệp Đại Bảo buộc tội hắn trước mặt mọi người.",
                duration_sec=6,
            )
        ]

        out = _align_scene_prompt_with_narration(shots, episode_num=1)
        prompt = out[0].scene_prompt

        assert "figure pointing accusingly" in prompt
        assert "figure standing" not in prompt  # weak pose replaced

    def test_build_characters_ref_skips_unresolved_id_like_tokens(self):
        from llm.scriptwriter import _build_characters_ref

        with patch("llm.character_extractor.load_all_characters", return_value=[]):
            ref = _build_characters_ref(["qing_yun_zi", "Thiếu Dương"])

        assert "qing_yun_zi" not in ref
        assert "Thiếu Dương" in ref


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
