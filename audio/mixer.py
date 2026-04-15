from pathlib import Path
from typing import Optional

import ffmpeg
from loguru import logger

from config.settings import settings


def mix_narration_with_bgm(
    narration_path: Path,
    output_path: Path,
    bgm_path: Optional[Path] = None,
) -> Path:
    """Mix narration (foreground) with optional BGM (background at bgm_volume_db).
    Output: AAC 44.1kHz stereo.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if bgm_path and bgm_path.exists():
        narration = ffmpeg.input(str(narration_path))
        bgm = ffmpeg.input(str(bgm_path), stream_loop=-1)  # loop BGM

        bgm_linear = _db_to_linear(settings.bgm_volume_db)

        # Pad narration: 200ms silence at start (prevents first syllable from being
        # hard-cut), 500ms silence at end (buffer before zoompan -t duration trim).
        narration_padded = (
            narration.audio
            .filter("adelay", "200:all=1")
            .filter("apad", pad_dur=0.5)
        )

        mixed = ffmpeg.filter(
            [narration_padded, bgm.audio],
            "amix",
            inputs=2,
            weights=f"1 {bgm_linear:.4f}",
            duration="first",
        )
        (
            ffmpeg.output(
                mixed,
                str(output_path),
                acodec="aac",
                audio_bitrate="192k",
                ar=44100,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    else:
        # No BGM — re-encode narration with 200ms lead-in silence only.
        # apad is intentionally omitted here: without a bounded second stream,
        # apad has no EOF signal and will loop forever.
        narration_padded = (
            ffmpeg.input(str(narration_path)).audio
            .filter("adelay", "200:all=1")
        )
        (
            ffmpeg.output(
                narration_padded,
                str(output_path),
                acodec="aac",
                audio_bitrate="192k",
                ar=44100,
            )
            .overwrite_output()
            .run(quiet=True)
        )

    logger.debug("Audio mixed | output={}", output_path)
    return output_path


def find_bgm(assets_dir: Optional[str] = None) -> Optional[Path]:
    """Return first BGM file found in assets/music/."""
    music_dir = Path(assets_dir or settings.assets_dir) / "music"
    if not music_dir.exists():
        return None
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        files = list(music_dir.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def _db_to_linear(db: float) -> float:
    """Convert dB to linear gain."""
    return 10 ** (db / 20)
