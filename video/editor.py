import tempfile
from pathlib import Path
from typing import List, Optional

import ffmpeg
from loguru import logger

from config.settings import settings
from models.schemas import ShotScript


def sanitize_for_srt(text: str) -> str:
    """Sanitize narration text for SRT subtitles and FFmpeg drawtext filter.
    Escapes characters that could cause injection or parsing errors.
    """
    # Strip known problematic chars for FFmpeg subtitle filter
    text = text.replace("\\", "\\\\")
    text = text.replace("\r", "")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("{", "\\{").replace("}", "\\}")
    return text.strip()


def generate_srt(shots: List[ShotScript], output_path: Path, intro_duration: float = 2.0) -> None:
    """Write SRT subtitle file from shot narrations.
    Offset accounts for intro clip duration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    current_time = intro_duration  # offset by intro

    for idx, shot in enumerate(shots, start=1):
        start = current_time
        end = current_time + shot.duration_sec
        safe_text = sanitize_for_srt(shot.narration_text)

        lines.append(str(idx))
        lines.append(f"{_to_srt_time(start)} --> {_to_srt_time(end)}")
        lines.append(safe_text)
        lines.append("")

        current_time = end

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def assemble_episode(
    episode_num: int,
    shot_clips: List[Path],
    shots: List[ShotScript],
    intro_path: Optional[Path] = None,
    outro_path: Optional[Path] = None,
) -> Path:
    """Concatenate shot clips, burn SRT subtitles, encode with h264_nvenc.
    Returns path to final episode video.
    """
    output_dir = Path(settings.data_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode-{episode_num:03d}.mp4"

    # Determine intro duration for SRT offset
    intro_duration = 2.0 if (intro_path and intro_path.exists()) else 0.0

    # Generate SRT
    srt_path = (
        Path(settings.data_dir) / "scripts" / f"episode-{episode_num:03d}.srt"
    )
    generate_srt(shots, srt_path, intro_duration=intro_duration)

    # Build concat list
    clips_to_join: List[Path] = []
    if intro_path and intro_path.exists():
        clips_to_join.append(intro_path)
    clips_to_join.extend(shot_clips)
    if outro_path and outro_path.exists():
        clips_to_join.append(outro_path)

    concat_output = output_path.with_suffix(".concat.mp4")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        concat_file = Path(f.name)
        for clip in clips_to_join:
            f.write(f"file '{clip.resolve()}'\n")

    try:
        # Step 1: concat clips
        (
            ffmpeg.input(str(concat_file), format="concat", safe=0)
            .output(
                str(concat_output),
                vcodec="h264_nvenc",
                acodec="aac",
                audio_bitrate="192k",
                video_bitrate="4000k",
                r=settings.fps,
                **{"movflags": "+faststart"},
            )
            .overwrite_output()
            .run(quiet=True)
        )

        # Step 2: burn subtitles
        (
            ffmpeg.input(str(concat_output))
            .filter(
                "subtitles",
                str(srt_path),
                force_style="FontSize=18,PrimaryColour=&Hffffff,BorderStyle=3,Outline=1,Shadow=1",
            )
            .output(
                str(output_path),
                vcodec="h264_nvenc",
                acodec="copy",
                video_bitrate="4000k",
                r=settings.fps,
            )
            .overwrite_output()
            .run(quiet=True)
        )

    finally:
        concat_file.unlink(missing_ok=True)
        concat_output.unlink(missing_ok=True)

    logger.info("Episode assembled | episode={} path={}", episode_num, output_path)
    return output_path
