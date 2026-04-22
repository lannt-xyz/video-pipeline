from __future__ import annotations

from pathlib import Path

import ffmpeg
from loguru import logger

from config.settings import settings


def mix_narration_with_bgm(
    narration_path: Path,
    bgm_path: Path,
    output_path: Path,
) -> Path:
    """Mix foreground narration with background music.

    BGM is attenuated to settings.bgm_volume_db (default -15 dB) so the
    narration voice stays clearly audible.  Output is AAC 44.1 kHz stereo
    (compatible with the NVENC H.264 MP4 container used in video/editor.py).

    Returns output_path on success.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    narration = ffmpeg.input(str(narration_path))
    bgm = ffmpeg.input(str(bgm_path), stream_loop=-1)  # loop BGM to match narration length

    # Attenuate BGM volume
    bgm_attenuated = bgm.audio.filter("volume", f"{settings.bgm_volume_db}dB")

    # amix: duration=first keeps output length equal to narration
    mixed = ffmpeg.filter(
        [narration.audio, bgm_attenuated],
        "amix",
        inputs=2,
        duration="first",
        dropout_transition=2,
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

    logger.info("BGM mixed | narration={} bgm={} out={}", narration_path.name, bgm_path.name, output_path)
    return output_path


def mix_episode_audio(episode_num: int, shot_paths: list[Path]) -> list[Path]:
    """Mix narration + BGM for all shots in an episode.

    Selects a BGM track deterministically by episode_num % len(tracks),
    so re-runs of the same episode always use the same track.
    Each shot gets its own mixed audio file (narration + BGM, duration=narration).
    Returns list of mixed output paths.
    """
    music_dir = Path(settings.assets_music_dir)
    bgm_tracks = sorted(music_dir.glob("*.mp3"))
    if not bgm_tracks:
        raise FileNotFoundError(
            f"No BGM tracks found in '{music_dir}'. "
            "Add royalty-free .mp3 files to assets/music/."
        )

    # Deterministic BGM per episode (episode_num % track count)
    bgm_path = bgm_tracks[episode_num % len(bgm_tracks)]
    logger.info("BGM selected | episode={} track={}", episode_num, bgm_path.name)

    output_dir = Path(settings.data_dir) / "audio" / f"episode-{episode_num:03d}" / "mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed_paths: list[Path] = []
    for narration_path in shot_paths:
        out = output_dir / narration_path.name
        mixed_paths.append(mix_narration_with_bgm(narration_path, bgm_path, out))

    logger.info("BGM mix complete | episode={} shots={}", episode_num, len(mixed_paths))
    return mixed_paths
