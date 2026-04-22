import asyncio
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import edge_tts
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from models.schemas import ShotScript

# Voices tried in order before giving up and synthesising silence.
_PRIMARY_VOICE = settings.tts_voice
_FALLBACK_VOICES = [
    v for v in ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"] if v != _PRIMARY_VOICE
]

# edge-tts chokes on text longer than ~900 chars in a single call.
_MAX_TTS_CHARS = 900


def _sanitize_tts_text(text: str) -> str:
    """Remove characters that cause edge-tts to return NoAudioReceived.

    Handles: control chars, zero-width chars, URLs, lone punctuation-only runs,
    and emoji that the Vietnamese TTS voice cannot render.
    """
    # Strip URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove zero-width / invisible Unicode chars
    text = re.sub(r"[\u200b-\u200f\u2028\u2029\ufeff\u00ad]", "", text)
    # Remove control characters (keep \n and \t for sentence breaks)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )
    # Remove emoji / pictographic symbols
    text = re.sub(r"[\U0001F300-\U0001FFFF]", "", text)
    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


async def _generate_silence(output_path: Path, duration_sec: float = 6.0) -> None:
    """Synthesise a silent MP3 via FFmpeg so the pipeline can continue."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(duration_sec),
        "-acodec", "libmp3lame", "-b:a", "192k",
        str(output_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


async def _tts_one_voice(text: str, voice: str, output_path: Path) -> None:
    """Single TTS call with tenacity retry. Raises on all failures."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=8, max=60),
        reraise=True,
    )
    async def _attempt() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))

    await _attempt()


async def generate_episode_tts(
    episode_num: int,
    shots: List[ShotScript],
    max_concurrent: int = 1,
) -> List[Path]:
    """Generate TTS for all shots in an episode.

    Returns list of audio paths in shot order. On per-shot failure (after all
    voice retries), writes a silent MP3 so the pipeline never crashes.
    """
    audio_dir = Path(settings.data_dir) / "audio" / f"episode-{episode_num:03d}"
    audio_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _generate_shot(
        shot_index: int,
        narration_text: str,
        output_path: Path,
        duration_sec: float,
    ) -> Tuple[int, Path]:
        raw_text = (narration_text or "").strip()
        text = _sanitize_tts_text(raw_text)

        if not text:
            logger.warning(
                "Empty narration after sanitize | shot={} raw={!r} — generating silence",
                shot_index, raw_text[:120],
            )
            await _generate_silence(output_path, duration_sec)
            return shot_index, output_path

        # Truncate if too long for a single edge-tts call
        if len(text) > _MAX_TTS_CHARS:
            logger.warning(
                "narration_text truncated {} → {} chars | shot={}",
                len(text), _MAX_TTS_CHARS, shot_index,
            )
            text = text[:_MAX_TTS_CHARS]

        logger.debug(
            "TTS request | shot={} voice={} chars={} text={!r}",
            shot_index, _PRIMARY_VOICE, len(text), text[:80],
        )

        async with semaphore:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            last_exc: Optional[Exception] = None
            for attempt, voice in enumerate([_PRIMARY_VOICE, *_FALLBACK_VOICES]):
                if attempt > 0:
                    # Brief cooldown between voice switches to avoid back-to-back 429s
                    await asyncio.sleep(3.0)
                try:
                    await _tts_one_voice(text, voice, output_path)
                    logger.debug(
                        "TTS ok | shot={} voice={} path={}", shot_index, voice, output_path
                    )
                    return shot_index, output_path
                except Exception as exc:
                    logger.warning(
                        "TTS failed | shot={} voice={} error={}", shot_index, voice, exc
                    )
                    last_exc = exc

            # All voices exhausted → write silence so the episode can finish
            logger.error(
                "All TTS voices failed | shot={} — writing silence. last_error={}",
                shot_index, last_exc,
            )
            await _generate_silence(output_path, duration_sec)
            return shot_index, output_path

    tasks = [
        _generate_shot(
            idx,
            shot.narration_text,
            audio_dir / f"shot-{idx:02d}.mp3",
            float(getattr(shot, "duration_sec", 6)),
        )
        for idx, shot in enumerate(shots)
    ]

    results = await asyncio.gather(*tasks)
    results_sorted = sorted(results, key=lambda x: x[0])
    paths = [r[1] for r in results_sorted]

    logger.info("TTS done | episode={} shots={}", episode_num, len(paths))
    return paths


def generate_episode_tts_sync(
    episode_num: int, shots: List[ShotScript]
) -> List[Path]:
    """Sync wrapper for generate_episode_tts."""
    return asyncio.run(generate_episode_tts(episode_num, shots))
