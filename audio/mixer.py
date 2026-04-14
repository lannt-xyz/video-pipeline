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

        mixed = ffmpeg.filter(
            [narration.audio, bgm.audio],
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
        # No BGM — just re-encode narration to AAC
        (
            ffmpeg.input(str(narration_path))
            .output(
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
