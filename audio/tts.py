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

# Sentence-ending punctuation used as preferred split points.
_SPLIT_PUNCT = re.compile(r'(?<=[.!?。！？…,，;；:\n])')


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
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode(errors="replace") if stderr else "(no stderr)"
        raise RuntimeError(f"FFmpeg silence generation failed for {output_path}: {err}")


def _split_text_into_chunks(text: str, max_chars: int = _MAX_TTS_CHARS) -> List[str]:
    """Split text into chunks <= max_chars, preferring sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    # Split at punctuation boundaries first
    parts = _SPLIT_PUNCT.split(text)
    chunks: List[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= max_chars:
            current += part
        else:
            if current:
                chunks.append(current.strip())
            # If a single part exceeds max_chars, hard-split by word
            while len(part) > max_chars:
                chunks.append(part[:max_chars].strip())
                part = part[max_chars:]
            current = part
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


async def _tts_one_voice(text: str, voice: str, output_path: Path) -> None:
    """Single TTS call. On first failure, retries with WordBoundary (pre-7.2.0 default).
    MS edge-tts server is intermittently rejecting SentenceBoundary requests (issue #473).
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=8, max=60),
        reraise=True,
    )
    async def _attempt(boundary: str) -> None:
        communicate = edge_tts.Communicate(text, voice, boundary=boundary)
        await communicate.save(str(output_path))

    # First try with SentenceBoundary (current default in ≥7.2.0)
    try:
        await _attempt("SentenceBoundary")
        return
    except Exception:
        pass

    # Fallback: WordBoundary (default before 7.2.0, more stable during MS outages)
    await _attempt("WordBoundary")


async def _tts_chunked(text: str, voice: str, output_path: Path) -> None:
    """Generate TTS for long text by splitting into chunks and concatenating with FFmpeg."""
    chunks = _split_text_into_chunks(text)
    if len(chunks) == 1:
        await _tts_one_voice(chunks[0], voice, output_path)
        return

    # Generate each chunk to a temp file
    tmp_dir = output_path.parent / f".tmp_{output_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: List[Path] = []
    try:
        for i, chunk in enumerate(chunks):
            chunk_path = tmp_dir / f"chunk_{i:03d}.mp3"
            await _tts_one_voice(chunk, voice, chunk_path)
            chunk_paths.append(chunk_path)

        # Build FFmpeg concat list
        concat_list = tmp_dir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in chunk_paths)
        )

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-acodec", "copy",
            str(output_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace") if stderr else "(no stderr)"
            raise RuntimeError(f"FFmpeg concat failed: {err}")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _generate_shot(
    episode_num: int,
    shot_index: int,
    narration_text: str,
    output_path: Path,
    duration_sec: float,
) -> Tuple[int, Path, bool]:
    """Generate TTS for a single shot. Returns (index, path, is_silence)."""
    raw_text = (narration_text or "").strip()
    text = _sanitize_tts_text(raw_text)

    if not text:
        logger.warning(
            "Empty narration after sanitize | shot={} raw={!r} — generating silence",
            shot_index, raw_text[:120],
        )
        await _generate_silence(output_path, duration_sec)
        return shot_index, output_path, True

    # Log if text will be chunked
    if len(text) > _MAX_TTS_CHARS:
        n_chunks = len(_split_text_into_chunks(text))
        logger.debug(
            "narration_text will be split into {} chunks ({} chars) | episode={} shot={}",
            n_chunks, len(text), episode_num, shot_index,
        )

    logger.debug(
        "TTS request | shot={} voice={} chars={} text={!r}",
        shot_index, _PRIMARY_VOICE, len(text), text[:80],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        if attempt > 0:
            # Longer cooldown — MS edge-tts server is intermittently flaky (issue #473)
            await asyncio.sleep(15.0)
        try:
            await _tts_chunked(text, _PRIMARY_VOICE, output_path)
            logger.debug(
                "TTS ok | shot={} voice={} path={}", shot_index, _PRIMARY_VOICE, output_path
            )
            return shot_index, output_path, False
        except Exception as exc:
            logger.warning(
                "TTS failed | shot={} attempt={} error={}", shot_index, attempt + 1, exc
            )
            last_exc = exc

    logger.error(
        "All TTS voices failed | shot={} — writing silence. last_error={}",
        shot_index, last_exc,
    )
    await _generate_silence(output_path, duration_sec)
    return shot_index, output_path, True


async def generate_episode_tts(
    episode_num: int,
    shots: List[ShotScript],
    max_concurrent: int = 1,
) -> List[Path]:
    """Generate TTS for all shots in an episode, serial with inter-shot delay.

    edge-tts is a free Microsoft service — concurrent requests cause throttling
    (NoAudioReceived). Serial execution with a short cooldown between shots is
    reliable and fast enough for 8-10 shots per episode.
    Returns list of audio paths in shot order.
    Raises RuntimeError if ALL shots fall back to silence (systemic failure).
    """
    audio_dir = Path(settings.data_dir) / "audio" / f"episode-{episode_num:03d}"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Inter-shot delay to avoid edge-tts throttling from back-to-back requests.
    _INTER_SHOT_DELAY_SEC = 1.5

    results: list[Tuple[int, Path, bool]] = []

    for idx, shot in enumerate(shots):
        if idx > 0:
            await asyncio.sleep(_INTER_SHOT_DELAY_SEC)
        result = await _generate_shot(
            episode_num,
            idx,
            shot.narration_text,
            audio_dir / f"shot-{idx:02d}.mp3",
            float(getattr(shot, "duration_sec", 6)),
        )
        results.append(result)

    results_sorted = sorted(results, key=lambda x: x[0])

    silence_shots = [idx for idx, _path, is_silence in results_sorted if is_silence]
    if silence_shots:
        logger.warning(
            "TTS silence fallback | episode={} shots_with_silence={}/{}  indices={}",
            episode_num, len(silence_shots), len(shots), silence_shots,
        )

    if len(silence_shots) == len(shots):
        raise RuntimeError(
            f"All {len(shots)} TTS shots failed for episode {episode_num}. "
            "Check network connectivity or edge-tts rate limits."
        )

    paths = [path for _idx, path, _is_silence in results_sorted]

    logger.info(
        "TTS done | episode={} shots={} silence={}",
        episode_num, len(paths), len(silence_shots),
    )
    return paths


def generate_episode_tts_sync(
    episode_num: int, shots: List[ShotScript]
) -> List[Path]:
    """Sync wrapper for generate_episode_tts."""
    return asyncio.run(generate_episode_tts(episode_num, shots))
