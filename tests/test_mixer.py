"""Tests for audio/mixer.py — narration + BGM mixing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings


# ---------------------------------------------------------------------------
# mix_narration_with_bgm
# ---------------------------------------------------------------------------


def test_mix_narration_with_bgm_calls_ffmpeg_amix(tmp_path: Path) -> None:
    """mix_narration_with_bgm uses amix with correct volume attenuation and AAC output."""
    narration = tmp_path / "narration.mp3"
    bgm = tmp_path / "bgm.mp3"
    output = tmp_path / "mixed.mp3"
    narration.write_bytes(b"narr")
    bgm.write_bytes(b"bgm")

    def fake_run(**kwargs):
        output.write_bytes(b"mixed")

    mock_stream = MagicMock()
    mock_stream.audio = mock_stream
    mock_stream.filter.return_value = mock_stream

    mock_ffmpeg = MagicMock()
    mock_ffmpeg.input.return_value = mock_stream
    mock_ffmpeg.filter.return_value = mock_stream
    mock_ffmpeg.output.return_value.overwrite_output.return_value.run = fake_run

    with (
        patch.dict("sys.modules", {"ffmpeg": mock_ffmpeg}),
        patch.object(settings, "bgm_volume_db", -15),
    ):
        import audio.mixer as mixer_module
        result = mixer_module.mix_narration_with_bgm(narration, bgm, output)

    assert result == output

    # Verify volume filter was applied with the configured dB value on BGM stream
    mock_stream.filter.assert_called_once_with("volume", "-15dB")

    # Verify amix was called with correct parameters: 2 inputs, first-stream duration
    mock_ffmpeg.filter.assert_called_once_with(
        [mock_stream, mock_stream],
        "amix",
        inputs=2,
        duration="first",
        dropout_transition=2,
    )

    # Verify output codec is AAC at 44.1 kHz (compatible with NVENC H.264 container)
    # ffmpeg.output(mixed, path, ...) is the top-level call pattern used in mixer.py
    mock_ffmpeg.output.assert_called_once_with(
        mock_stream,
        str(output),
        acodec="aac",
        audio_bitrate="192k",
        ar=44100,
    )


def test_mix_episode_audio_raises_without_bgm_tracks(tmp_path: Path) -> None:
    """mix_episode_audio raises FileNotFoundError when no BGM tracks exist."""
    music_dir = tmp_path / "music"
    music_dir.mkdir()

    with patch.object(settings, "assets_music_dir", str(music_dir)):
        import audio.mixer as mixer_module
        with pytest.raises(FileNotFoundError, match="No BGM tracks found"):
            mixer_module.mix_episode_audio(1, [])


def test_mix_episode_audio_selects_track_deterministically(tmp_path: Path) -> None:
    """BGM track selection is deterministic: episode_num % len(tracks)."""
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    track_a = music_dir / "track_a.mp3"
    track_b = music_dir / "track_b.mp3"
    track_a.write_bytes(b"a")
    track_b.write_bytes(b"b")

    selected_tracks: list[Path] = []

    def fake_mix(narration: Path, bgm: Path, out: Path) -> Path:
        selected_tracks.append(bgm)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mixed")
        return out

    narration_path = tmp_path / "shot-000.mp3"
    narration_path.write_bytes(b"n")

    with (
        patch("audio.mixer.mix_narration_with_bgm", side_effect=fake_mix),
        patch.object(settings, "assets_music_dir", str(music_dir)),
        patch.object(settings, "data_dir", str(tmp_path)),
    ):
        import audio.mixer as mixer_module

        mixer_module.mix_episode_audio(0, [narration_path])  # 0 % 2 = 0 → track_a
        mixer_module.mix_episode_audio(1, [narration_path])  # 1 % 2 = 1 → track_b
        mixer_module.mix_episode_audio(2, [narration_path])  # 2 % 2 = 0 → track_a (wrap)

    assert selected_tracks[0] == track_a
    assert selected_tracks[1] == track_b
    assert selected_tracks[2] == track_a
