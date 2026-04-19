import subprocess
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


def _resolve_subtitle_padding() -> tuple[float, float]:
    """Return (lead_in_sec, tail_pad_sec) used by subtitle timing.

    Tail padding is only applied when BGM exists, matching audio.mixer behavior.
    """
    lead = max(0.0, settings.tts_lead_in_sec)
    tail = 0.0
    try:
        from audio.mixer import find_bgm

        if find_bgm() is not None:
            tail = max(0.0, settings.tts_tail_padding_sec)
    except Exception as exc:
        logger.debug("Could not resolve BGM for subtitle padding | err={}", exc)
    return lead, tail


def _speech_window(shot_start: float, shot_duration: float, lead: float, tail: float) -> tuple[float, float]:
    """Compute spoken interval within one shot timeline.

    Subtitles should follow narration speech, not silence padding added to audio.
    """
    shot_end = shot_start + max(0.0, shot_duration)
    speech_start = min(shot_end, shot_start + max(0.0, lead))
    speech_end = min(shot_end, max(speech_start + 0.1, shot_end - max(0.0, tail)))
    return speech_start, speech_end


def generate_srt(shots: List[ShotScript], output_path: Path, intro_duration: float = 2.0) -> None:
    """Write SRT subtitle file from shot narrations.
    Offset accounts for intro clip duration and shot-to-shot transition overlap.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    current_time = intro_duration  # offset by intro
    transition_dur = settings.shot_transition_duration
    lead_in_sec, tail_pad_sec = _resolve_subtitle_padding()

    for idx, shot in enumerate(shots, start=1):
        shot_start = current_time
        shot_end = current_time + shot.duration_sec
        start, end = _speech_window(
            shot_start, shot.duration_sec, lead_in_sec, tail_pad_sec
        )
        safe_text = sanitize_for_srt(shot.narration_text)

        lines.append(str(idx))
        lines.append(f"{_to_srt_time(start)} --> {_to_srt_time(end)}")
        lines.append(safe_text)
        lines.append("")

        # Each shot overlaps with the next by transition_dur (except last)
        current_time = shot_end - transition_dur

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

# ASS BGR colour codes for keyword highlight
_DANGER_COLOUR = "&H0000FF&"  # red
_TWIST_COLOUR = "&H00FFFF&"   # yellow

# Words that trigger danger highlight (red) — genre-specific Vietnamese horror/action terms
_DANGER_KEYWORDS = frozenset([
    "chết", "xác", "mộ", "quỷ", "máu", "ám", "tà", "oan", "hồn",
    "thi", "hài", "quan", "tài", "phanh", "thây", "cắn", "xé",
    "gào", "thét", "biến", "mất", "sụp", "đổ", "giết", "má", "linh",
    "âm", "tử", "hồn", "ma", "thần", "kỳ", "dị", "nghiệp",
])

# Multi-word trigger phrases — checked in order before single-word match
_DANGER_PHRASES = ("thi hài", "quan tài", "phanh thây", "cắn xé", "gào thét", "biến mất", "sụp đổ")
_TWIST_PHRASES = ("thật ra", "không ngờ", "bất ngờ", "hóa ra", "thực chất", "đột ngột")


def _tag_word(word: str, word_dur_cs: int) -> str:
    """Return ASS-tagged word with karaoke timing and optional colour highlight.
    All override tags are in a single block per ASS spec.
    Falls back to plain kf tag on any error.
    Twist detection is phrase-based — call _tag_segment() for multi-word context.
    """
    try:
        lower = word.lower()
        if lower in _DANGER_KEYWORDS:
            return f"{{\\kf{word_dur_cs}\\c{_DANGER_COLOUR}}}{word}{{\\r}}"
    except Exception:
        logger.warning("Keyword tag failed for word={!r}, using plain tag", word)
    return f"{{\\kf{word_dur_cs}}}{word}"


def _tag_segment(seg_words: List[str], word_dur_cs: int) -> str:
    """Build karaoke text for a segment, applying danger or twist colours.
    Twist is detected at segment level (multi-word phrase); danger at word level.
    """
    try:
        seg_lower = " ".join(w.lower() for w in seg_words)
        # Check if segment contains a twist phrase — highlight entire segment yellow
        for phrase in _TWIST_PHRASES:
            if phrase in seg_lower:
                return " ".join(
                    f"{{\\kf{word_dur_cs}\\c{_TWIST_COLOUR}}}{w}{{\\r}}" for w in seg_words
                )
    except Exception:
        logger.warning("Segment phrase tag failed, falling back to word-level tagging")
    return " ".join(_tag_word(w, word_dur_cs) for w in seg_words)


def generate_ass(shots: List[ShotScript], output_path: Path, intro_duration: float = 0.0) -> None:
    """Write ASS subtitle file with karaoke fill effect (\\kf per word).

    Each shot narration is split into segments of _WORDS_PER_SEGMENT words.
    Within each segment, \\kf tags distribute the segment duration across words
    so each word highlights as it is spoken (fill left→right).
    Timing accounts for shot-to-shot transition overlap.

    Style: white text, yellow unspoken, bold, black outline — bottom-center.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transition_dur = settings.shot_transition_duration

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
    lead_in_sec, tail_pad_sec = _resolve_subtitle_padding()

    for shot_idx, shot in enumerate(shots):
        safe_narration = sanitize_for_srt(shot.narration_text)
        words = safe_narration.split()
        shot_start = current_time
        shot_end = current_time + shot.duration_sec
        speech_start, speech_end = _speech_window(
            shot_start, shot.duration_sec, lead_in_sec, tail_pad_sec
        )
        speech_duration = max(0.1, speech_end - speech_start)
        if not words:
            current_time = shot_end
            if shot_idx < len(shots) - 1:
                current_time -= transition_dur
            continue

        segments = [
            words[i: i + _WORDS_PER_SEGMENT]
            for i in range(0, len(words), _WORDS_PER_SEGMENT)
        ]
        seg_duration = speech_duration / len(segments)

        current_time = speech_start

        for seg_i, seg_words in enumerate(segments):
            seg_start = current_time
            if seg_i == len(segments) - 1:
                seg_end = speech_end
            else:
                seg_end = current_time + seg_duration
            word_dur_cs = max(1, int(seg_duration * 100 / len(seg_words)))
            karaoke_text = _tag_segment(seg_words, word_dur_cs)
            dialogue_lines.append(
                f"Dialogue: 0,{_to_ass_time(seg_start)},{_to_ass_time(seg_end)},"
                f"Karaoke,,0,0,0,,{karaoke_text}"
            )
            current_time = seg_end

        # Account for transition overlap with next shot (except last)
        current_time = shot_end
        if shot_idx < len(shots) - 1:
            current_time -= transition_dur

    output_path.write_text(
        header + "\n" + "\n".join(dialogue_lines),
        encoding="utf-8",
    )


def _probe_duration(path: Path) -> float:
    """Return file duration in seconds via ffprobe. Returns 0.0 on error."""
    try:
        info = ffmpeg.probe(str(path))
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


def _build_xfade_command(
    clips: List[Path],
    output_path: Path,
    transition: str = "dissolve",
    transition_dur: float = 0.3,
) -> List[str]:
    """Build raw ffmpeg command for xfade transition chain between clips.

    Strategy: chain pairwise xfade on video streams, amerge/amix audio streams.
    For N clips: N-1 xfade filters chained sequentially.
    Audio: concat filter (simpler, transition overlap handled by mixing).
    """
    n = len(clips)
    if n < 2:
        raise ValueError("Need at least 2 clips for xfade")

    # Probe actual durations
    durations = []
    for c in clips:
        d = _probe_duration(c)
        if d <= 0:
            d = 6.0  # fallback
        durations.append(d)

    # Build input args
    cmd = ["ffmpeg", "-y"]
    for c in clips:
        cmd.extend(["-i", str(c)])

    # Build video xfade chain
    # [0:v][1:v]xfade=transition=dissolve:duration=0.3:offset=D0-0.3[v01]
    # [v01][2:v]xfade=...offset=D0+D1-2*0.3[v012]
    # etc.
    filters = []
    cumulative_offset = 0.0

    for i in range(n - 1):
        if i == 0:
            src1 = "[0:v]"
            src2 = "[1:v]"
        else:
            src1 = f"[vx{i-1}]"
            src2 = f"[{i+1}:v]"

        offset = cumulative_offset + durations[i] - transition_dur
        if offset < 0.1:
            offset = 0.1

        out_label = f"[vx{i}]" if i < n - 2 else "[vout]"
        filters.append(
            f"{src1}{src2}xfade=transition={transition}:duration={transition_dur:.2f}"
            f":offset={offset:.4f}{out_label}"
        )
        cumulative_offset = offset

    # Audio: trim each clip's audio to the duration until the next xfade starts,
    # so audio switches in sync with when each video stream becomes dominant.
    # Without this, concat audio is (n-1)*transition_dur longer than the xfade
    # video, causing ~0.3s of accumulated A/V drift per shot transition.
    # Last clip keeps full audio since there is no following xfade.
    for i, d in enumerate(durations):
        if i < n - 1:
            trim_end = min(d, max(0.1, d - transition_dur))
            filters.append(
                f"[{i}:a]atrim=0:{trim_end:.4f},asetpts=PTS-STARTPTS[a{i}]"
            )
        else:
            filters.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")

    audio_concat_inputs = "".join(f"[a{i}]" for i in range(n))
    filters.append(f"{audio_concat_inputs}concat=n={n}:v=0:a=1[aout]")

    filter_str = ";".join(filters)

    cmd.extend([
        "-filter_complex", filter_str,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "h264_nvenc",
        "-b:v", "4000k",
        "-c:a", "aac",
        "-b:a", "192k",
        "-r", str(settings.fps),
        "-movflags", "+faststart",
        str(output_path),
    ])

    return cmd


def assemble_episode(
    episode_num: int,
    shot_clips: List[Path],
    shots: List[ShotScript],
    intro_path: Optional[Path] = None,
    outro_path: Optional[Path] = None,
) -> Path:
    """Join shot clips with xfade transitions, burn ASS subtitles, encode with h264_nvenc.
    Returns path to final episode video.
    """
    output_dir = Path(settings.data_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode-{episode_num:03d}.mp4"

    # Determine intro duration for subtitle offset
    intro_duration = 2.0 if (intro_path and intro_path.exists()) else 0.0

    # Generate ASS karaoke subtitle file
    ass_path = (
        Path(settings.data_dir) / "scripts" / f"episode-{episode_num:03d}.ass"
    )
    generate_ass(shots, ass_path, intro_duration=intro_duration)

    # Build clip list
    clips_to_join: List[Path] = []
    if intro_path and intro_path.exists():
        clips_to_join.append(intro_path)
    clips_to_join.extend(shot_clips)
    if outro_path and outro_path.exists():
        clips_to_join.append(outro_path)

    transition_type = settings.shot_transition_type
    transition_dur = settings.shot_transition_duration

    if len(clips_to_join) < 2:
        # Single clip — no transition needed, just copy
        transition_output = output_path.with_suffix(".trans.mp4")
        (
            ffmpeg.input(str(clips_to_join[0]))
            .output(str(transition_output), vcodec="h264_nvenc", acodec="aac",
                    video_bitrate="4000k", audio_bitrate="192k", r=settings.fps)
            .overwrite_output()
            .run(quiet=True)
        )
    else:
        transition_output = output_path.with_suffix(".trans.mp4")
        cmd = _build_xfade_command(
            clips_to_join, transition_output,
            transition=transition_type, transition_dur=transition_dur,
        )
        logger.debug("xfade command | {}", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "xfade failed, falling back to concat | stderr={}",
                (exc.stderr or "")[:500],
            )
            # Fallback: simple concat demuxer (no transitions)
            transition_output = _fallback_concat(clips_to_join, transition_output)

    try:
        # Burn subtitles onto the transition-joined video
        inp_sub = ffmpeg.input(str(transition_output))
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
        transition_output.unlink(missing_ok=True)

    logger.info("Episode assembled | episode={} path={}", episode_num, output_path)
    return output_path


def _fallback_concat(clips: List[Path], output_path: Path) -> Path:
    """Simple concat demuxer fallback when xfade fails."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        concat_file = Path(f.name)
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")

    try:
        inp_concat = ffmpeg.input(str(concat_file), format="concat", safe=0)
        (
            ffmpeg.output(
                inp_concat.video,
                inp_concat.audio,
                str(output_path),
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
    finally:
        concat_file.unlink(missing_ok=True)

    return output_path
