import asyncio
import re
import tempfile
import unicodedata
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import edge_tts
import httpx
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

# ---------------------------------------------------------------------------
# Piper TTS (local)
# ---------------------------------------------------------------------------

_piper_voice = None  # module-level singleton; loaded on first use


def _parse_piper_model_path(model_name: str) -> str:
    """Convert vi_VN-vivos-medium → vi/vi_VN/vivos/medium for HF URL."""
    parts = model_name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected piper model name format: {model_name!r}")
    locale = parts[0]       # vi_VN
    lang = locale.split("_")[0]  # vi
    dataset = parts[1]      # vivos
    quality = parts[2]      # medium
    return f"{lang}/{locale}/{dataset}/{quality}"


def _ensure_piper_model() -> Path:
    """Download model files from Hugging Face if not already cached. Returns .onnx path."""
    model_name = settings.piper_model
    model_dir = Path(settings.piper_models_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = model_dir / f"{model_name}.onnx"
    json_path = model_dir / f"{model_name}.onnx.json"

    if onnx_path.exists() and json_path.exists():
        return onnx_path

    hf_path = _parse_piper_model_path(model_name)
    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{hf_path}"

    for filename in [f"{model_name}.onnx", f"{model_name}.onnx.json"]:
        dest = model_dir / filename
        if dest.exists():
            continue
        url = f"{base_url}/{filename}"
        logger.info("Downloading piper model file: {} → {}", url, dest)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
        logger.info("Downloaded piper model file: {}", filename)

    return onnx_path


def _get_piper_voice():
    global _piper_voice
    if _piper_voice is None:
        from piper import PiperVoice  # lazy import — only loaded when provider=piper
        onnx_path = _ensure_piper_model()
        logger.info("Loading piper model: {}", onnx_path)
        _piper_voice = PiperVoice.load(str(onnx_path))
        logger.info("Piper model loaded")
    return _piper_voice


async def _piper_tts(text: str, output_path: Path) -> None:
    """Synthesise text via local Piper model and save as MP3."""
    tmp_wav = Path(tempfile.mktemp(suffix=".wav"))

    # synthesize is CPU/sync — run in executor to avoid blocking event loop
    def _synthesize() -> None:
        voice = _get_piper_voice()
        with wave.open(str(tmp_wav), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _synthesize)

        # Convert WAV → MP3 with post-processing:
        # - atempo: speed up slightly for more engaging delivery
        # - bass: +4dB warmth for horror narration
        # - treble: +2dB clarity for consonant articulation
        # - acompressor: normalize dynamics (Piper can be uneven)
        speed = settings.piper_speed
        af_chain = (
            f"atempo={speed},"
            "bass=g=4,"
            "treble=g=2,"
            "acompressor=threshold=-20dB:ratio=3:attack=5:release=50"
        )
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(tmp_wav),
            "-af", af_chain,
            "-acodec", "libmp3lame", "-b:a", "192k",
            str(output_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace") if stderr else "(no stderr)"
            raise RuntimeError(f"FFmpeg WAV→MP3 failed: {err}")
    finally:
        tmp_wav.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# F5-TTS Vietnamese (local GPU)
# ---------------------------------------------------------------------------

_f5tts_instance = None  # module-level singleton; loaded on first use

_F5TTS_HF_REPO = "hynt/F5-TTS-Vietnamese-ViVoice"
_F5TTS_CKPT_FILE = "model_last.pt"
_F5TTS_VOCAB_FILE = "config.json"  # repo uses config.json as vocab file

# Default ref audio text — a neutral Vietnamese sentence read at normal pace
_F5TTS_DEFAULT_REF_TEXT = (
    "Xin chào, tôi là người kể chuyện. "
    "Câu chuyện hôm nay sẽ đưa bạn vào một thế giới đầy bí ẩn và rùng rợn."
)


def _ensure_f5tts_model() -> tuple[Path, Path]:
    """Download VI checkpoint from HuggingFace if not cached. Returns (ckpt_path, vocab_path)."""
    model_dir = Path(settings.f5tts_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / _F5TTS_CKPT_FILE
    vocab_path = model_dir / "vocab.txt"  # saved locally as vocab.txt

    base_url = f"https://huggingface.co/{_F5TTS_HF_REPO}/resolve/main"
    for filename, dest in [(_F5TTS_CKPT_FILE, ckpt_path), ("config.json", vocab_path)]:
        if dest.exists():
            continue
        url = f"{base_url}/{filename}"
        logger.info("Downloading F5-TTS VI model: {} → {}", url, dest)
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
        logger.info("Downloaded: {}", filename)

    return ckpt_path, vocab_path


def _ensure_ref_audio() -> Path:
    """Return path to ref audio, generating from edge-tts if the configured file doesn't exist."""
    ref_path = Path(settings.f5tts_ref_audio)
    if ref_path.exists() and ref_path.stat().st_size > 0:
        return ref_path

    logger.info("ref audio not found at {} — generating from edge-tts", ref_path)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    # Generate ~10s reference clip via edge-tts synchronously (one-time setup)
    result = subprocess.run(
        ["python", "-c", f"""
import asyncio, edge_tts
async def gen():
    c = edge_tts.Communicate({_F5TTS_DEFAULT_REF_TEXT!r}, "vi-VN-HoaiMyNeural")
    await c.save("{ref_path}")
asyncio.run(gen())
"""],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0 or not ref_path.exists():
        raise RuntimeError(
            f"Failed to generate ref audio: {result.stderr.decode(errors='replace')}"
        )
    logger.info("Ref audio generated: {}", ref_path)
    return ref_path


def _get_f5tts():
    global _f5tts_instance
    if _f5tts_instance is None:
        from f5_tts.api import F5TTS
        ckpt_path, vocab_path = _ensure_f5tts_model()
        device = settings.f5tts_device
        logger.info("Loading F5-TTS VI model from {} on device={}", ckpt_path, device)
        try:
            # Use F5TTS_Base (NOT v1) — the VI fine-tune is on the original v0 architecture.
            _f5tts_instance = F5TTS(
                model="F5TTS_Base",
                ckpt_file=str(ckpt_path),
                vocab_file=str(vocab_path),
                device=device,
            )
        except Exception as exc:  # CUDA OOM, no GPU, etc.
            if device == "cuda":
                logger.warning(
                    "F5-TTS CUDA load failed ({}) — falling back to CPU", exc
                )
                _f5tts_instance = F5TTS(
                    model="F5TTS_Base",
                    ckpt_file=str(ckpt_path),
                    vocab_file=str(vocab_path),
                    device="cpu",
                )
            else:
                raise
        logger.info("F5-TTS model loaded")
    return _f5tts_instance


async def _f5tts_generate(text: str, output_path: Path) -> None:
    """Synthesise text via local F5-TTS CPU model and save as MP3.

    Acquires F5TTS consumer on the vram_manager so Ollama and ComfyUI
    are evicted from VRAM before the model loads.  This frees VRAM for
    the next image-gen / LLM phase when TTS is done.
    """
    from pipeline.vram_manager import VRAMConsumer, vram_manager

    # Evict Ollama + ComfyUI from VRAM before loading the 5 GB F5-TTS model.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, vram_manager.acquire, VRAMConsumer.F5TTS)

    ref_path = _ensure_ref_audio()
    ref_text = settings.f5tts_ref_text or _F5TTS_DEFAULT_REF_TEXT

    # The VI fine-tune was trained on lowercase text — must normalize input.
    gen_text_norm = text.lower()
    ref_text_norm = ref_text.lower()

    tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
    try:
        def _synthesize() -> None:
            tts = _get_f5tts()
            tts.infer(
                ref_file=str(ref_path),
                ref_text=ref_text_norm,
                gen_text=gen_text_norm,
                speed=settings.f5tts_speed,
                file_wave=str(tmp_wav),
                remove_silence=True,
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _synthesize)

        # Convert WAV → MP3
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(tmp_wav),
            "-acodec", "libmp3lame", "-b:a", "192k",
            str(output_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace") if stderr else "(no stderr)"
            raise RuntimeError(f"FFmpeg WAV→MP3 failed: {err}")
    finally:
        tmp_wav.unlink(missing_ok=True)


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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- F5-TTS Vietnamese (local GPU) ---
    if settings.tts_provider == "f5tts":
        logger.debug(
            "TTS request (f5tts) | shot={} chars={} text={!r}",
            shot_index, len(text), text[:80],
        )
        try:
            await _f5tts_generate(text, output_path)
            logger.debug("TTS ok (f5tts) | shot={} path={}", shot_index, output_path)
            return shot_index, output_path, False
        except Exception as exc:
            logger.error(
                "F5-TTS failed | shot={} error={} — falling back to edge-tts", shot_index, exc
            )
            # fall through to edge-tts below

    # --- Piper (local) provider ---
    if settings.tts_provider == "piper":
        logger.debug(
            "TTS request (piper) | shot={} model={} chars={} text={!r}",
            shot_index, settings.piper_model, len(text), text[:80],
        )
        try:
            await _piper_tts(text, output_path)
            logger.debug("TTS ok (piper) | shot={} path={}", shot_index, output_path)
            return shot_index, output_path, False
        except Exception as exc:
            logger.error(
                "Piper TTS failed | shot={} error={} — writing silence", shot_index, exc
            )
            await _generate_silence(output_path, duration_sec)
            return shot_index, output_path, True

    # --- edge-tts (cloud) provider ---
    # Log if text will be chunked
    if len(text) > _MAX_TTS_CHARS:
        n_chunks = len(_split_text_into_chunks(text))
        logger.debug(
            "narration_text will be split into {} chunks ({} chars) | episode={} shot={}",
            n_chunks, len(text), episode_num, shot_index,
        )

    logger.debug(
        "TTS request (edge) | shot={} voice={} chars={} text={!r}",
        shot_index, _PRIMARY_VOICE, len(text), text[:80],
    )

    voices_to_try = [_PRIMARY_VOICE] + _FALLBACK_VOICES
    last_exc: Optional[Exception] = None
    for attempt, voice in enumerate(voices_to_try):
        if attempt > 0:
            # Longer cooldown — MS edge-tts server is intermittently flaky (issue #473)
            await asyncio.sleep(15.0)
        try:
            await _tts_chunked(text, voice, output_path)
            logger.debug(
                "TTS ok | shot={} voice={} path={}", shot_index, voice, output_path
            )
            return shot_index, output_path, False
        except Exception as exc:
            logger.warning(
                "TTS failed | shot={} voice={} attempt={} error={}", shot_index, voice, attempt + 1, exc
            )
            last_exc = exc

    # All edge-tts voices failed — fall back to local Piper before giving up
    logger.warning(
        "edge-tts all voices failed | shot={} — falling back to Piper local TTS. last_error={}",
        shot_index, last_exc,
    )
    try:
        await _piper_tts(text, output_path)
        logger.info("Piper fallback ok | shot={} path={}", shot_index, output_path)
        return shot_index, output_path, False
    except Exception as exc:
        logger.error(
            "Piper fallback also failed | shot={} error={} — writing silence", shot_index, exc
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
    # Skipped when using local providers (piper, f5tts).
    _INTER_SHOT_DELAY_SEC = 0.0 if settings.tts_provider in ("piper", "f5tts") else 1.5

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
