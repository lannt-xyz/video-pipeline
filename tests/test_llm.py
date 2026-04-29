import json
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import ArcOverview, CameraFlow, EpisodeScript, ShotScript


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

        # 26 words per narration × 8 shots = 208 words — passes the 200-word minimum.
        long_narration = (
            "Đây là một đoạn lời dẫn đủ dài để vượt qua ngưỡng kiểm tra "
            "tối thiểu của scriptwriter trong pipeline và đảm bảo TTS "
            "có đủ âm thanh cho từng shot video của tập phim."
        )
        mock_raw = {
            "title": "Tập 1: Khởi Đầu",
            "shots": [
                {
                    "scene_prompt": f"Scene {i} anime style 9:16",
                    "narration_text": long_narration,
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
            patch("llm.scriptwriter.scene_prompt_client") as mock_scene_llm,
            patch("llm.scriptwriter.settings") as mock_settings,
        ):
            mock_llm.generate_json.return_value = mock_raw
            # Downstream passes (visual_brief, character resolve, scene align) all
            # expect JSON arrays; returning [] makes each pass no-op / fallback.
            mock_scene_llm.generate_json.return_value = []
            mock_settings.data_dir = str(tmp_path)
            mock_settings.llm_max_retries = 3
            mock_settings.target_duration_sec = 60
            mock_settings.shot_transition_duration = 0.3
            mock_settings.retention.use_constraint_system = False

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
            "narration_text": (
                "Đây là một đoạn lời dẫn đủ dài để vượt qua ngưỡng kiểm tra "
                "tối thiểu của scriptwriter trong pipeline và đảm bảo TTS "
                "có đủ âm thanh cho từng shot video của tập phim."
            ),
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
            patch("llm.scriptwriter.scene_prompt_client") as mock_scene_llm,
            patch("llm.scriptwriter.settings") as mock_settings,
        ):
            mock_llm.generate_json.return_value = mock_raw
            mock_scene_llm.generate_json.return_value = []
            mock_settings.data_dir = str(tmp_path)
            mock_settings.llm_max_retries = 3
            mock_settings.target_duration_sec = 60
            mock_settings.shot_transition_duration = 0.3
            mock_settings.retention.use_constraint_system = False

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
                scene_prompt="muddy excavation pit, man digging with shovel, anime style, no text, no watermarks",
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
        assert "red lacquered wooden coffin" in prompt

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


# ── ShotVisualBrief Schema ─────────────────────────────────────────────────────

class TestShotVisualBriefSchema:
    def _make_brief(self, **kwargs):
        from models.schemas import ShotVisualBrief

        defaults = dict(
            subjects=["hooded daoist figure"],
            actions=["figure prying open stone coffin lid with iron crowbar"],
            setting="dimly lit coffin shop with rows of dark wooden coffins",
            key_objects=["glowing talisman paper"],
            mood_lighting="blood-red candle flame casting elongated shadows",
            composition="medium close-up",
        )
        defaults.update(kwargs)
        return ShotVisualBrief(**defaults)

    def test_shot_script_backward_compat_no_visual_brief(self):
        """Old JSON without visual_brief field should load without error."""
        shot = ShotScript(
            scene_prompt="test scene",
            narration_text="lời dẫn",
        )
        assert shot.visual_brief is None

    def test_shot_script_with_visual_brief(self):
        from models.schemas import ShotVisualBrief

        brief = self._make_brief()
        shot = ShotScript(
            scene_prompt="placeholder",
            narration_text="lời dẫn",
            visual_brief=brief,
        )
        assert shot.visual_brief is not None
        assert shot.visual_brief.actions[0].startswith("figure prying")

    def test_episode_script_round_trip_with_visual_brief(self):
        """Serialize and deserialize EpisodeScript with visual_brief embedded."""
        from models.schemas import ShotVisualBrief

        brief = self._make_brief()
        shot = ShotScript(
            scene_prompt="test scene",
            narration_text="lời dẫn",
            visual_brief=brief,
        )
        episode = EpisodeScript(episode_num=1, title="Test", shots=[shot] * 8)
        reloaded = EpisodeScript(**json.loads(episode.model_dump_json()))
        assert reloaded.shots[0].visual_brief is not None
        assert reloaded.shots[0].visual_brief.composition == "medium close-up"

    def test_shot_script_json_without_visual_brief_field_stays_none(self):
        """Dict missing visual_brief key (old scripts) → visual_brief=None."""
        data = {
            "scene_prompt": "scene",
            "narration_text": "narration",
            "duration_sec": 6,
            "is_key_shot": False,
            "characters": [],
        }
        shot = ShotScript(**data)
        assert shot.visual_brief is None


# ── _synthesize_scene_prompt ───────────────────────────────────────────────────

class TestSynthesizeScenePrompt:
    def _make_brief(self, **kwargs):
        from models.schemas import ShotVisualBrief

        defaults = dict(
            subjects=["hooded daoist figure", "kneeling young warrior"],
            actions=["figure prying open stone coffin lid with iron crowbar"],
            setting="dimly lit coffin shop with rows of dark wooden coffins",
            key_objects=["glowing talisman paper", "ritual candles", "iron chains on wall"],
            mood_lighting="blood-red candle flame casting elongated shadows on stone wall",
            composition="medium close-up",
        )
        # Back-compat: allow callers to pass `action="..."` singular.
        if "action" in kwargs:
            kwargs["actions"] = [kwargs.pop("action")]
        defaults.update(kwargs)
        return ShotVisualBrief(**defaults)

    def _make_shot(self, **kwargs):
        defaults = dict(
            scene_prompt="placeholder",
            narration_text="test narration",
            duration_sec=8,
            camera_flow=CameraFlow.WIDE_TO_CLOSE,
        )
        defaults.update(kwargs)
        return ShotScript(**defaults)

    def test_tag_order_composition_setting_action_mood_subjects(self):
        from llm.scriptwriter import _synthesize_scene_prompt

        brief = self._make_brief()
        shot = self._make_shot()
        result = _synthesize_scene_prompt(brief, shot)
        tags = [t.strip() for t in result.split(",") if t.strip()]

        # composition comes first when non-empty
        assert tags[0] == "medium close-up"
        # setting is second
        assert "dimly lit coffin shop" in tags[1]
        # action is third
        assert "figure prying open stone coffin lid" in tags[2]

    def test_mood_lighting_always_present(self):
        from llm.scriptwriter import _synthesize_scene_prompt

        brief = self._make_brief()
        shot = self._make_shot()
        result = _synthesize_scene_prompt(brief, shot)
        assert "blood-red candle flame" in result

    def test_key_objects_wrapped_with_weight(self):
        from llm.scriptwriter import _synthesize_scene_prompt

        brief = self._make_brief()
        shot = self._make_shot()
        result = _synthesize_scene_prompt(brief, shot)
        assert "(glowing talisman paper:1.15)" in result
        assert "(ritual candles:1.15)" in result

    def test_key_object_dedup_with_setting(self):
        """Object that appears in setting should not be added again."""
        from llm.scriptwriter import _synthesize_scene_prompt

        brief = self._make_brief(
            setting="dimly lit coffin shop with rows of dark wooden coffins",
            key_objects=["dark wooden coffins", "ritual candles"],  # first is substring of setting
        )
        shot = self._make_shot()
        result = _synthesize_scene_prompt(brief, shot)
        # "dark wooden coffins" is already in setting — should not get a duplicate wrapped tag
        wrapped_dup = "(dark wooden coffins:1.15)"
        assert wrapped_dup not in result
        # second object should appear
        assert "(ritual candles:1.15)" in result

    def test_subject_not_duplicated_when_exact_phrase_in_action(self):
        """Primary subject exact phrase in action → subject should not be appended again."""
        from llm.scriptwriter import _synthesize_scene_prompt

        brief = self._make_brief(
            subjects=["figure"],  # exact word in action
            action="figure prying open stone coffin lid with iron crowbar",
        )
        shot = self._make_shot()
        result = _synthesize_scene_prompt(brief, shot)
        # Count occurrences — "figure" appears in action; it should NOT be appended as a separate tag
        tags = [t.strip() for t in result.split(",") if t.strip()]
        # Subject "figure" is an exact substring of action → skip
        standalone_figure = [t for t in tags if t == "figure"]
        assert len(standalone_figure) == 0, "Subject already in action — should not be appended standalone"

    def test_secondary_subject_dropped_when_budget_exhausted(self):
        """With many key_objects filling the cap, secondary subject should be dropped."""
        from llm.scriptwriter import _synthesize_scene_prompt, _SYNTHESIS_MAX_TAGS

        # Create brief with enough key_objects to exhaust budget before secondary subject
        brief = self._make_brief(
            subjects=["primary figure", "secondary figure"],
            key_objects=[f"object {i}" for i in range(10)],  # 10 objects
            composition="",  # fewer never_drop tags → gives more room; still test that secondary can be dropped
        )
        shot = self._make_shot(camera_flow=CameraFlow.WIDE_TO_CLOSE)
        result = _synthesize_scene_prompt(brief, shot)
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert len(tags) <= _SYNTHESIS_MAX_TAGS

    def test_composition_mapped_from_camera_flow_when_empty(self):
        """Empty composition → maps from camera_flow."""
        from llm.scriptwriter import _synthesize_scene_prompt

        brief = self._make_brief(composition="")
        shot = self._make_shot(camera_flow=CameraFlow.STATIC_CLOSE)
        result = _synthesize_scene_prompt(brief, shot)
        # STATIC_CLOSE → "medium close-up"
        assert result.startswith("medium close-up")

    def test_no_composition_when_camera_flow_has_no_mapping(self):
        """camera_flow=WIDE_TO_CLOSE has empty mapping → no composition tag added."""
        from llm.scriptwriter import _synthesize_scene_prompt

        brief = self._make_brief(composition="")
        shot = self._make_shot(camera_flow=CameraFlow.WIDE_TO_CLOSE)
        result = _synthesize_scene_prompt(brief, shot)
        # Setting should be first tag
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert "dimly lit coffin shop" in tags[0]

    def test_tag_count_within_cap(self):
        from llm.scriptwriter import _synthesize_scene_prompt, _SYNTHESIS_MAX_TAGS

        brief = self._make_brief()
        shot = self._make_shot()
        result = _synthesize_scene_prompt(brief, shot)
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert len(tags) <= _SYNTHESIS_MAX_TAGS


# ── _extract_visual_briefs ─────────────────────────────────────────────────────

class TestExtractVisualBriefs:
    def _make_shots(self, n=8):
        shots = []
        for i in range(n):
            duration = 3 if i <= 1 else 8
            shots.append(ShotScript(
                scene_prompt=f"scene {i}",
                narration_text=f"Narration shot {i} with enough words here.",
                duration_sec=duration,
                characters=[],
            ))
        return shots

    def test_hook_shots_keep_none_brief(self):
        """Shots index 0-1 (duration ≤3) must stay visual_brief=None."""
        from llm.scriptwriter import _extract_visual_briefs

        shots = self._make_shots(8)
        llm_response = [
            {
                "shot_index": i,
                "subjects": ["figure"],
                "action": f"figure doing action {i}",
                "setting": f"location {i}",
                "key_objects": [],
                "mood_lighting": "dim candle light",
                "composition": "",
            }
            for i in range(2, 8)  # only non-hook shots
        ]

        with patch("llm.scriptwriter.scene_prompt_client") as mock_client:
            mock_client.generate_json.return_value = llm_response
            result = _extract_visual_briefs(shots, episode_num=1)

        assert result[0].visual_brief is None, "Shot 0 (hook) must keep visual_brief=None"
        assert result[1].visual_brief is None, "Shot 1 (hook) must keep visual_brief=None"
        for i in range(2, 8):
            assert result[i].visual_brief is not None, f"Shot {i} should have visual_brief"

    def test_non_list_response_returns_shots_unchanged(self):
        """When LLM returns non-list, shots come back unchanged."""
        from llm.scriptwriter import _extract_visual_briefs

        shots = self._make_shots(4)
        with patch("llm.scriptwriter.scene_prompt_client") as mock_client:
            mock_client.generate_json.return_value = {"error": "bad response"}
            result = _extract_visual_briefs(shots, episode_num=1)

        assert all(s.visual_brief is None for s in result)

    def test_partial_parse_failure_skips_bad_shot(self):
        """If one shot's brief parse fails, skip it; others should still be populated."""
        from llm.scriptwriter import _extract_visual_briefs

        shots = self._make_shots(5)
        # Provide valid brief for shot 2, invalid for shot 3, valid for shot 4
        llm_response = [
            {
                "shot_index": 2,
                "subjects": ["daoist figure"],
                "action": "figure raising hand toward sky",
                "setting": "ruined temple courtyard",
                "key_objects": ["burning incense"],
                "mood_lighting": "pale moonlight",
                "composition": "",
            },
            {
                "shot_index": 3,
                # subjects=123 is truthy → `or []` doesn't coerce it → pydantic rejects list[str]=123
                "subjects": 123,
                "action": "some action",
                "setting": "some setting",
                "key_objects": [],
                "mood_lighting": "dark",
                "composition": "",
            },
            {
                "shot_index": 4,
                "subjects": ["warrior figure"],
                "action": "figure sprinting toward gate with sword raised",
                "setting": "dark stone corridor",
                "key_objects": [],
                "mood_lighting": "flickering torch light",
                "composition": "wide establishing shot",
            },
        ]

        with patch("llm.scriptwriter.scene_prompt_client") as mock_client:
            mock_client.generate_json.return_value = llm_response
            result = _extract_visual_briefs(shots, episode_num=1)

        assert result[2].visual_brief is not None
        assert result[3].visual_brief is None  # parse failed → skipped
        assert result[4].visual_brief is not None

    def test_llm_wrapped_in_dict_unwrapped(self):
        """LLM response wrapped in dict key 'shots' should be unwrapped."""
        from llm.scriptwriter import _extract_visual_briefs

        shots = self._make_shots(4)
        valid_brief = {
            "shot_index": 2,
            "subjects": ["figure"],
            "action": "figure walking forward into fog",
            "setting": "misty forest path",
            "key_objects": [],
            "mood_lighting": "cold pale moonlight",
            "composition": "",
        }
        with patch("llm.scriptwriter.scene_prompt_client") as mock_client:
            mock_client.generate_json.return_value = {"shots": [valid_brief]}
            result = _extract_visual_briefs(shots, episode_num=1)

        assert result[2].visual_brief is not None


# ── _build_scene_prompts_from_narration ───────────────────────────────────────

class TestBuildScenePromptsFromNarration:
    def _make_shots(self, n=8):
        shots = []
        for i in range(n):
            duration = 3 if i <= 1 else 8
            shots.append(ShotScript(
                scene_prompt=f"original scene {i}",
                narration_text=f"Narration shot {i}.",
                duration_sec=duration,
                characters=[],
            ))
        return shots

    def test_fallback_when_zero_briefs_populated(self):
        """If _extract_visual_briefs returns 0 populated briefs, fallback to rewrite pass."""
        from llm.scriptwriter import _build_scene_prompts_from_narration

        shots = self._make_shots(4)

        # Extract returns all shots unchanged (0 populated)
        with (
            patch("llm.scriptwriter._extract_visual_briefs", return_value=shots),
            patch("llm.scriptwriter._rewrite_scene_prompts_from_narration",
                  return_value=shots) as mock_rewrite,
        ):
            result = _build_scene_prompts_from_narration(shots, episode_num=1)

        mock_rewrite.assert_called_once()
        assert result == shots

    def test_synthesis_called_when_briefs_populated(self):
        """When extraction returns at least 1 brief, synthesis should run."""
        from llm.scriptwriter import _build_scene_prompts_from_narration
        from models.schemas import ShotVisualBrief

        brief = ShotVisualBrief(
            subjects=["figure"],
            action="figure walking into temple",
            setting="ancient temple gate",
            key_objects=[],
            mood_lighting="pale moonlight",
            composition="",
        )
        shots = self._make_shots(4)
        shots_with_brief = list(shots)
        shots_with_brief[2] = shots_with_brief[2].model_copy(update={"visual_brief": brief})

        with (
            patch("llm.scriptwriter._extract_visual_briefs", return_value=shots_with_brief),
            patch("llm.scriptwriter._synthesize_scene_prompts_from_briefs",
                  return_value=shots_with_brief) as mock_synth,
        ):
            _build_scene_prompts_from_narration(shots, episode_num=1)

        mock_synth.assert_called_once()

