"""Tests for audio/tts.py — TTS generation with edge and piper backends."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the already-loaded settings singleton so we can patch its attributes
from config.settings import settings
from models.schemas import ShotScript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shot(narration: str = "Xin chào thế giới") -> ShotScript:
    return ShotScript(
        scene_prompt="A misty mountain landscape",
        narration_text=narration,
        duration_sec=6,
    )


# ---------------------------------------------------------------------------
# generate_narration — edge backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_narration_edge_creates_file(tmp_path: Path) -> None:
    """edge-tts backend: output file is created when communicate.save succeeds."""
    output = tmp_path / "shot-000.mp3"

    mock_communicate = AsyncMock()
    mock_communicate.save = AsyncMock(side_effect=lambda p: Path(p).write_bytes(b"fake-mp3"))
    mock_edge_module = MagicMock()
    mock_edge_module.Communicate.return_value = mock_communicate

    with (
        patch.dict("sys.modules", {"edge_tts": mock_edge_module}),
        patch.object(settings, "tts_backend", "edge"),
        patch.object(settings, "edge_voice", "vi-VN-HoaiMyNeural"),
    ):
        import audio.tts as tts_module
        result = await tts_module.generate_narration("Xin chào", output)

    assert result == output
    assert output.exists()


@pytest.mark.asyncio
async def test_generate_narration_edge_skips_existing_file(tmp_path: Path) -> None:
    """edge-tts backend: if output file already exists, skip and return cached path."""
    output = tmp_path / "shot-000.mp3"
    output.write_bytes(b"cached")

    with patch("audio.tts._generate_edge", new_callable=AsyncMock) as mock_edge:
        import audio.tts as tts_module
        result = await tts_module.generate_narration("Xin chào", output)

    mock_edge.assert_not_called()
    assert result == output


# ---------------------------------------------------------------------------
# generate_narration — piper backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_narration_piper_calls_synthesize(tmp_path: Path) -> None:
    """piper backend: _piper_synthesize_blocking is called via to_thread."""
    output = tmp_path / "shot-001.mp3"

    def fake_synthesize_blocking(text: str, path: Path) -> Path:
        path.write_bytes(b"fake-piper-mp3")
        return path

    with (
        patch.object(settings, "tts_backend", "piper"),
        patch("audio.tts._piper_synthesize_blocking", side_effect=fake_synthesize_blocking),
    ):
        import audio.tts as tts_module
        result = await tts_module.generate_narration("Xin chào", output)

    assert result == output
    assert output.read_bytes() == b"fake-piper-mp3"


# ---------------------------------------------------------------------------
# generate_episode_audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_episode_audio_returns_all_paths(tmp_path: Path) -> None:
    """generate_episode_audio returns one path per shot."""
    shots = [_make_shot(f"Câu thoại {i}") for i in range(3)]

    async def fake_narration(text: str, path: Path) -> Path:
        path.write_bytes(b"audio")
        return path

    with (
        patch("audio.tts.generate_narration", side_effect=fake_narration),
        patch.object(settings, "data_dir", str(tmp_path)),
    ):
        import audio.tts as tts_module
        paths = await tts_module.generate_episode_audio(1, shots)

    assert len(paths) == 3


@pytest.mark.asyncio
async def test_generate_episode_audio_raises_on_failure(tmp_path: Path) -> None:
    """generate_episode_audio re-raises when any shot fails."""
    shots = [_make_shot()]

    async def failing_narration(text: str, path: Path) -> Path:
        raise RuntimeError("TTS network error")

    with (
        patch("audio.tts.generate_narration", side_effect=failing_narration),
        patch.object(settings, "data_dir", str(tmp_path)),
    ):
        import audio.tts as tts_module
        with pytest.raises(RuntimeError, match="TTS network error"):
            await tts_module.generate_episode_audio(1, shots)


# ---------------------------------------------------------------------------
# Piper blocking synthesis
# ---------------------------------------------------------------------------


def test_piper_synthesize_blocking_cleans_up_wav(tmp_path: Path) -> None:
    """_piper_synthesize_blocking removes intermediate WAV after MP3 conversion."""
    output = tmp_path / "shot-002.mp3"
    wav_path = output.with_suffix(".wav")

    mock_voice_instance = MagicMock()

    mock_piper_voice_cls = MagicMock()
    mock_piper_voice_cls.load.return_value = mock_voice_instance

    mock_piper_voice_module = MagicMock()
    mock_piper_voice_module.PiperVoice = mock_piper_voice_cls

    def fake_run(**kwargs):
        output.write_bytes(b"mp3")

    mock_ffmpeg_chain = MagicMock()
    mock_ffmpeg_chain.overwrite_output.return_value.run = fake_run
    mock_ffmpeg_input = MagicMock()
    mock_ffmpeg_input.output.return_value = mock_ffmpeg_chain
    mock_ffmpeg = MagicMock()
    mock_ffmpeg.input.return_value = mock_ffmpeg_input

    # Mock wave.open so the fake synthesize doesn't have to write real WAV headers
    mock_wave_file = MagicMock()
    mock_wave_ctx = MagicMock()
    mock_wave_ctx.__enter__ = MagicMock(return_value=mock_wave_file)
    mock_wave_ctx.__exit__ = MagicMock(return_value=False)
    mock_wave_open = MagicMock(return_value=mock_wave_ctx)

    with (
        patch.dict("sys.modules", {"piper.voice": mock_piper_voice_module}),
        patch.dict("sys.modules", {"ffmpeg": mock_ffmpeg}),
        patch("audio.tts.wave.open", mock_wave_open),
        patch.object(settings, "piper_model_path", "dummy.onnx"),
        patch.object(settings, "piper_config_path", "dummy.onnx.json"),
        patch.object(settings, "audio_bitrate", "128k"),
    ):
        import audio.tts as tts_module
        tts_module._piper_synthesize_blocking("text", output)

    assert not wav_path.exists(), "Intermediate WAV must be removed after MP3 conversion"
