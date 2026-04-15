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


def _to_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_WORDS_PER_SEGMENT = 4


def generate_ass(shots: List[ShotScript], output_path: Path, intro_duration: float = 0.0) -> None:
    """Write ASS subtitle file with karaoke fill effect (\\kf per word).

    Each shot narration is split into segments of _WORDS_PER_SEGMENT words.
    Within each segment, \\kf tags distribute the segment duration across words
    so each word highlights as it is spoken (fill left→right).

    Style: white text, yellow unspoken, bold, black outline — bottom-center.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {settings.image_width}\n"
        f"PlayResY: {settings.image_height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # PrimaryColour=white, SecondaryColour=yellow (before kf timer), Alignment=2 bottom-center
        "Style: Karaoke,Arial,52,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,2.5,1.5,2,60,60,80,0\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )

    dialogue_lines: List[str] = []
    current_time = intro_duration

    for shot in shots:
        safe_narration = sanitize_for_srt(shot.narration_text)
        words = safe_narration.split()
        if not words:
            current_time += shot.duration_sec
            continue

        segments = [
            words[i: i + _WORDS_PER_SEGMENT]
            for i in range(0, len(words), _WORDS_PER_SEGMENT)
        ]
        seg_duration = shot.duration_sec / len(segments)

        for seg_words in segments:
            seg_start = current_time
            seg_end = current_time + seg_duration
            word_dur_cs = max(1, int(seg_duration * 100 / len(seg_words)))
            karaoke_text = " ".join(
                f"{{\\kf{word_dur_cs}}}{w}" for w in seg_words
            )
            dialogue_lines.append(
                f"Dialogue: 0,{_to_ass_time(seg_start)},{_to_ass_time(seg_end)},"
                f"Karaoke,,0,0,0,,{karaoke_text}"
            )
            current_time = seg_end

    output_path.write_text(
        header + "\n" + "\n".join(dialogue_lines),
        encoding="utf-8",
    )


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

    # Generate ASS karaoke subtitle file
    ass_path = (
        Path(settings.data_dir) / "scripts" / f"episode-{episode_num:03d}.ass"
    )
    generate_ass(shots, ass_path, intro_duration=intro_duration)

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
        # Step 1: concat clips — explicitly map both video and audio to avoid concat
        # demuxer silently dropping audio when no stream map is specified.
        inp_concat = ffmpeg.input(str(concat_file), format="concat", safe=0)
        (
            ffmpeg.output(
                inp_concat.video,
                inp_concat.audio,
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

        # Step 2: burn subtitles — apply subtitles filter to video only, then
        # re-join original audio stream, because ffmpeg-python's .filter() node
        # outputs only the filtered (video) stream and discards audio.
        inp_sub = ffmpeg.input(str(concat_output))
        video_with_subs = inp_sub.video.filter(
            "subtitles",
            str(ass_path),
        )
        (
            ffmpeg.output(
                video_with_subs,
                inp_sub.audio,
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
