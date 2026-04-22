from __future__ import annotations

import asyncio
import wave
from pathlib import Path

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_narration(text: str, output_path: Path) -> Path:
    """Generate TTS audio for one narration segment.

    Dispatches to the configured backend (edge or piper).
    output_path should have a .mp3 extension.
    Returns output_path on success.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already generated (resume-safe)
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.debug("TTS cache hit | path={}", output_path)
        return output_path

    logger.info("TTS generate | backend={} path={}", settings.tts_backend, output_path)

    if settings.tts_backend == "edge":
        return await _generate_edge(text, output_path)
    else:
        return await _generate_piper(text, output_path)


async def generate_episode_audio(episode_num: int, shots: list) -> list[Path]:
    """Generate TTS for all shots in an episode concurrently.

    Uses asyncio.gather for parallel generation — safe because:
    - edge-tts: network I/O only, no GPU
    - piper: CPU inference, wrapped in asyncio.to_thread, no GPU
    Both strategies are safe to run alongside ComfyUI (Phase 3).
    """
    output_dir = Path(settings.data_dir) / "audio" / f"episode-{episode_num:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        generate_narration(
            shot.narration_text,
            output_dir / f"shot-{i:03d}.mp3",
        )
        for i, shot in enumerate(shots)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    paths: list[Path] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("TTS failed | episode={} shot={} error={}", episode_num, i, result)
            raise result
        paths.append(result)  # type: ignore[arg-type]

    logger.info("TTS complete | episode={} shots={}", episode_num, len(paths))
    return paths


# ---------------------------------------------------------------------------
# Edge TTS backend (Azure cloud — requires internet)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _generate_edge(text: str, output_path: Path) -> Path:
    """Generate audio using edge-tts (Microsoft Azure TTS cloud).

    Requires internet. Voice is configured via settings.edge_voice.
    Retries up to 3 times with exponential backoff on any error.
    """
    import edge_tts  # lazy import — only required when backend = "edge"

    communicate = edge_tts.Communicate(text, settings.edge_voice)
    await communicate.save(str(output_path))
    logger.debug("edge-tts saved | path={}", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Piper TTS backend (local CPU — fully offline)
# ---------------------------------------------------------------------------


async def _generate_piper(text: str, output_path: Path) -> Path:
    """Generate audio using piper-tts (local ONNX model, CPU-only).

    Runs synthesize in a thread pool to keep the event loop free.
    No GPU usage — safe to run concurrently with ComfyUI image gen.
    """
    return await asyncio.to_thread(_piper_synthesize_blocking, text, output_path)


def _piper_synthesize_blocking(text: str, output_path: Path) -> Path:
    """Blocking piper synthesis — called via asyncio.to_thread.

    Synthesizes WAV → converts to MP3 via ffmpeg-python → removes WAV.
    Model files are validated at startup in config/settings.py.
    """
    import ffmpeg  # lazy import — only required when backend = "piper"
    from piper.voice import PiperVoice  # lazy import — only required when backend = "piper"

    wav_path = output_path.with_suffix(".wav")
    try:
        voice = PiperVoice.load(
            settings.piper_model_path,
            config_path=settings.piper_config_path,
            use_cuda=False,  # CPU only; GPU VRAM reserved for Ollama/ComfyUI
        )
        with wave.open(str(wav_path), "w") as wav_file:
            voice.synthesize(text, wav_file)

        # Convert WAV → MP3 using ffmpeg-python wrapper
        (
            ffmpeg.input(str(wav_path))
            .output(
                str(output_path),
                acodec="libmp3lame",
                audio_bitrate=settings.audio_bitrate,
            )
            .overwrite_output()
            .run(quiet=True)
        )
        logger.debug("piper TTS saved | path={}", output_path)
        return output_path
    finally:
        # Always clean up the intermediate WAV
        if wav_path.exists():
            wav_path.unlink()
