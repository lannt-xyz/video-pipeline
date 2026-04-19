from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import ShotScript
from video.editor import sanitize_for_srt, generate_srt, _to_srt_time


# ── sanitize_for_srt ──────────────────────────────────────────────────────────

class TestSanitizeForSrt:
    def test_escapes_angle_brackets(self):
        text = "<b>bold</b>"
        result = sanitize_for_srt(text)
        assert "<b>" not in result
        assert "&lt;b&gt;" in result

    def test_escapes_backslash(self):
        result = sanitize_for_srt("path\\to\\file")
        assert "\\\\" in result

    def test_escapes_curly_braces(self):
        result = sanitize_for_srt("value {0}")
        assert "\\{" in result

    def test_strips_whitespace(self):
        result = sanitize_for_srt("  hello  ")
        assert result == "hello"

    def test_removes_carriage_return(self):
        result = sanitize_for_srt("line1\r\nline2")
        assert "\r" not in result

    def test_safe_plain_text_unchanged(self):
        text = "Diep Thieu Duong bước vào căn phòng."
        result = sanitize_for_srt(text)
        assert result == text


# ── generate_srt ──────────────────────────────────────────────────────────────

class TestGenerateSrt:
    def _make_shots(self, n=3):
        return [
            ShotScript(
                scene_prompt=f"Scene {i}",
                narration_text=f"Lời dẫn số {i}",
                duration_sec=6,
            )
            for i in range(n)
        ]

    def test_creates_srt_file(self, tmp_path):
        shots = self._make_shots(3)
        srt_path = tmp_path / "test.srt"
        generate_srt(shots, srt_path, intro_duration=0.0)

        assert srt_path.exists()
        content = srt_path.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "00:00:00,200 --> 00:00:06,000" in content

    def test_offset_by_intro_duration(self, tmp_path):
        shots = self._make_shots(1)
        srt_path = tmp_path / "test.srt"
        generate_srt(shots, srt_path, intro_duration=2.0)

        content = srt_path.read_text(encoding="utf-8")
        assert "00:00:02,200 --> 00:00:08,000" in content

    def test_sequential_timestamps(self, tmp_path):
        shots = self._make_shots(3)
        srt_path = tmp_path / "test.srt"
        generate_srt(shots, srt_path, intro_duration=0.0)

        content = srt_path.read_text(encoding="utf-8")
        # Shot 2 starts at 6s - 0.3s transition overlap + 0.2s lead-in = 5.9s
        assert "00:00:05,900 --> 00:00:11,699" in content

    def test_narration_appears_in_srt(self, tmp_path):
        shots = self._make_shots(1)
        srt_path = tmp_path / "test.srt"
        generate_srt(shots, srt_path, intro_duration=0.0)

        content = srt_path.read_text(encoding="utf-8")
        assert "Lời dẫn số 0" in content


# ── _to_srt_time ──────────────────────────────────────────────────────────────

class TestToSrtTime:
    def test_zero(self):
        assert _to_srt_time(0) == "00:00:00,000"

    def test_one_hour(self):
        assert _to_srt_time(3600) == "01:00:00,000"

    def test_fractional_seconds(self):
        assert _to_srt_time(1.5) == "00:00:01,500"

    def test_complex(self):
        assert _to_srt_time(3661.123) == "01:01:01,123"


# ── EpisodeValidator ───────────────────────────────────────────────────────────

class TestEpisodeValidator:
    def _make_probe(
        self,
        duration: float = 60.0,
        width: int = 720,
        height: int = 1280,
        has_audio: bool = True,
        audio_bitrate: int = 192000,
        file_size_mb: float = 10.0,
    ):
        streams = [
            {"codec_type": "video", "width": width, "height": height}
        ]
        if has_audio:
            streams.append(
                {"codec_type": "audio", "bit_rate": str(audio_bitrate)}
            )
        return {
            "format": {"duration": str(duration)},
            "streams": streams,
        }

    def test_passes_valid_episode(self, tmp_path):
        from pipeline.validator import EpisodeValidator

        ep_num = 1
        (tmp_path / "output").mkdir()
        video_path = tmp_path / "output" / "episode-001.mp4"
        video_path.write_bytes(b"0" * (6 * 1024 * 1024))  # 6MB fake

        (tmp_path / "thumbnails").mkdir()
        thumb_path = tmp_path / "thumbnails" / "episode-001.png"
        thumb_path.write_bytes(b"fake_image")

        probe_data = self._make_probe()
        val = EpisodeValidator()

        with (
            patch("pipeline.validator.settings") as mock_settings,
            patch.object(val, "_probe", return_value=probe_data),
        ):
            mock_settings.data_dir = str(tmp_path)
            mock_settings.image_width = 720
            mock_settings.image_height = 1280
            val.assert_episode(ep_num)  # should not raise

    def test_fails_wrong_duration(self, tmp_path):
        from pipeline.validator import EpisodeValidator, ValidationError

        ep_num = 1
        (tmp_path / "output").mkdir()
        video_path = tmp_path / "output" / "episode-001.mp4"
        video_path.write_bytes(b"0" * (6 * 1024 * 1024))

        (tmp_path / "thumbnails").mkdir()
        (tmp_path / "thumbnails" / "episode-001.png").write_bytes(b"img")

        probe_data = self._make_probe(duration=30.0)  # too short
        val = EpisodeValidator()

        with (
            patch("pipeline.validator.settings") as mock_settings,
            patch.object(val, "_probe", return_value=probe_data),
        ):
            mock_settings.data_dir = str(tmp_path)
            mock_settings.image_width = 720
            mock_settings.image_height = 1280
            with pytest.raises(ValidationError, match="Duration"):
                val.assert_episode(ep_num)

    def test_fails_missing_thumbnail(self, tmp_path):
        from pipeline.validator import EpisodeValidator, ValidationError

        ep_num = 1
        (tmp_path / "output").mkdir()
        video_path = tmp_path / "output" / "episode-001.mp4"
        video_path.write_bytes(b"0" * (6 * 1024 * 1024))
        # No thumbnail

        probe_data = self._make_probe()
        val = EpisodeValidator()

        with (
            patch("pipeline.validator.settings") as mock_settings,
            patch.object(val, "_probe", return_value=probe_data),
        ):
            mock_settings.data_dir = str(tmp_path)
            mock_settings.image_width = 720
            mock_settings.image_height = 1280
            with pytest.raises(ValidationError, match="Thumbnail"):
                val.assert_episode(ep_num)

    def test_fails_no_audio(self, tmp_path):
        from pipeline.validator import EpisodeValidator, ValidationError

        ep_num = 1
        (tmp_path / "output").mkdir()
        (tmp_path / "output" / "episode-001.mp4").write_bytes(b"0" * (6 * 1024 * 1024))
        (tmp_path / "thumbnails").mkdir()
        (tmp_path / "thumbnails" / "episode-001.png").write_bytes(b"img")

        probe_data = self._make_probe(has_audio=False)
        val = EpisodeValidator()

        with (
            patch("pipeline.validator.settings") as mock_settings,
            patch.object(val, "_probe", return_value=probe_data),
        ):
            mock_settings.data_dir = str(tmp_path)
            mock_settings.image_width = 720
            mock_settings.image_height = 1280
            with pytest.raises(ValidationError, match="audio"):
                val.assert_episode(ep_num)
