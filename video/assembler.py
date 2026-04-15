from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import ffmpeg
from loguru import logger

from config.settings import settings
from image_gen.comfyui_client import ComfyUIOutOfMemoryError, comfyui_client
from models.schemas import FrameScript, MotionDirection, ShotScript


# ── Ken Burns direction presets ───────────────────────────────────────────────
# Each direction returns (z_expr, x_expr, y_expr) for the zoompan filter.
# `d` is total_frames placeholder — substituted at call site.

def _zoompan_params(motion: MotionDirection, total_frames: int) -> dict:
    """Return zoompan filter kwargs for the given motion direction."""
    w, h = settings.image_width, settings.image_height
    d = total_frames
    base = {"s": f"{w}x{h}", "fps": settings.fps, "d": d}

    if motion == MotionDirection.ZOOM_IN:
        return {**base, "z": "zoom+0.0015", "x": "iw/2-(iw/zoom/2)", "y": "ih/2-(ih/zoom/2)"}
    elif motion == MotionDirection.ZOOM_OUT:
        return {**base, "z": f"if(eq(on,1),1.15,zoom-0.0015)", "x": "iw/2-(iw/zoom/2)", "y": "ih/2-(ih/zoom/2)"}
    elif motion == MotionDirection.PAN_LEFT:
        return {**base, "z": "1.12", "x": f"(iw-iw/zoom)*(1-on/{d})", "y": "(ih-ih/zoom)/2"}
    elif motion == MotionDirection.PAN_RIGHT:
        return {**base, "z": "1.12", "x": f"(iw-iw/zoom)*on/{d}", "y": "(ih-ih/zoom)/2"}
    elif motion == MotionDirection.PAN_UP:
        return {**base, "z": "1.12", "x": "(iw-iw/zoom)/2", "y": f"(ih-ih/zoom)*(1-on/{d})"}
    elif motion == MotionDirection.PAN_DOWN:
        return {**base, "z": "1.12", "x": "(iw-iw/zoom)/2", "y": f"(ih-ih/zoom)*on/{d}"}
    # Fallback: zoom in
    return {**base, "z": "zoom+0.0015", "x": "iw/2-(iw/zoom/2)", "y": "ih/2-(ih/zoom/2)"}


def _create_zoompan_clip(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float = 6.0,
    motion: MotionDirection = MotionDirection.ZOOM_IN,
) -> Path:
    """Ken Burns (zoompan) clip from static image + audio. Encoded with h264_nvenc."""
    if not image_path.exists():
        raise FileNotFoundError(
            f"Shot image missing: {image_path} — re-run from --from-phase images"
        )
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Shot audio missing: {audio_path} — re-run from --from-phase audio"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = settings.fps
    video_duration = duration + 1.0
    total_frames = int(video_duration * fps)

    zp = _zoompan_params(motion, total_frames)

    video = (
        ffmpeg.input(str(image_path), loop=1, t=video_duration, framerate=fps)
        .filter("zoompan", **zp)
        .filter("setpts", "PTS-STARTPTS")
    )
    audio = ffmpeg.input(str(audio_path))

    try:
        ffmpeg.output(
            video,
            audio.audio,
            str(output_path),
            vcodec="h264_nvenc",
            acodec="aac",
            audio_bitrate="192k",
            video_bitrate="4000k",
            r=fps,
            shortest=None,
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        logger.error("ffmpeg zoompan failed | path={} stderr={}", output_path, stderr)
        raise

    logger.debug("Zoompan clip created | path={} motion={}", output_path, motion.value)
    return output_path


def _create_multiframe_clip(
    frame_images: List[Path],
    frames: List[FrameScript],
    audio_path: Path,
    output_path: Path,
    duration: float = 6.0,
) -> Path:
    """Render a shot clip from 2 frames with crossfade + varied Ken Burns.

    Each frame gets roughly half the duration with a crossfade overlap.
    Uses FFmpeg filter_complex: zoompan(frame1) → zoompan(frame2) → xfade → audio merge.
    """
    if len(frame_images) < 2 or len(frames) < 2:
        # Single frame — delegate to simple zoompan
        motion = frames[0].motion if frames else MotionDirection.ZOOM_IN
        return _create_zoompan_clip(frame_images[0], audio_path, output_path, duration, motion)

    for img in frame_images:
        if not img.exists():
            raise FileNotFoundError(
                f"Frame image missing: {img} — re-run from --from-phase images"
            )
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Shot audio missing: {audio_path} — re-run from --from-phase audio"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = settings.fps
    xfade_dur = settings.crossfade_duration
    # Each frame displays for (duration + xfade_dur) / 2, with xfade overlap
    frame_dur = (duration + xfade_dur) / 2.0
    frame_video_dur = frame_dur + 1.0  # extra buffer for zoompan
    total_frames_per = int(frame_video_dur * fps)

    zp0 = _zoompan_params(frames[0].motion, total_frames_per)
    zp1 = _zoompan_params(frames[1].motion, total_frames_per)

    # Build filter_complex manually for proper xfade between two zoompan streams
    xfade_offset = max(0.1, frame_dur - xfade_dur)

    in0 = ffmpeg.input(str(frame_images[0]), loop=1, t=frame_video_dur, framerate=fps)
    in1 = ffmpeg.input(str(frame_images[1]), loop=1, t=frame_video_dur, framerate=fps)
    audio_in = ffmpeg.input(str(audio_path))

    v0 = (
        in0.filter("zoompan", **zp0)
        .filter("setpts", "PTS-STARTPTS")
        .filter("trim", duration=frame_dur)
        .filter("setpts", "PTS-STARTPTS")
    )
    v1 = (
        in1.filter("zoompan", **zp1)
        .filter("setpts", "PTS-STARTPTS")
        .filter("trim", duration=frame_dur)
        .filter("setpts", "PTS-STARTPTS")
    )

    merged = ffmpeg.filter([v0, v1], "xfade", transition="fade", duration=xfade_dur, offset=xfade_offset)

    try:
        ffmpeg.output(
            merged,
            audio_in.audio,
            str(output_path),
            vcodec="h264_nvenc",
            acodec="aac",
            audio_bitrate="192k",
            video_bitrate="4000k",
            r=fps,
            shortest=None,
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        logger.error("ffmpeg multiframe failed | path={} stderr={}", output_path, stderr)
        # Fallback: use first frame as simple zoompan
        logger.warning("Falling back to single-frame zoompan | path={}", output_path)
        return _create_zoompan_clip(
            frame_images[0], audio_path, output_path, duration, frames[0].motion
        )

    logger.debug(
        "Multiframe clip created | path={} frames={} motions={}",
        output_path, len(frame_images),
        [f.motion.value for f in frames],
    )
    return output_path


def _create_svd_clip(
    shot: ShotScript,
    shot_index: int,
    episode_num: int,
    image_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Generate SVD motion clip via ComfyUI. Falls back to zoompan on OOM."""
    try:
        svd_raw = output_path.with_suffix(".svd_raw.mp4")
        comfyui_client.generate_image(
            workflow_path="image_gen/workflows/svd_keyshot.json",
            replacements={
                "IMAGE_PATH": str(image_path),
                "WIDTH": settings.image_width,
                "HEIGHT": settings.image_height,
                "SEED": episode_num * 1000 + shot_index,
            },
            output_path=svd_raw,
        )

        pts_factor = shot.duration_sec / 4.2
        audio = ffmpeg.input(str(audio_path))

        (
            ffmpeg.input(str(svd_raw))
            .filter("setpts", f"{pts_factor:.4f}*PTS")
            .output(
                audio.audio,
                str(output_path),
                vcodec="h264_nvenc",
                acodec="aac",
                audio_bitrate="192k",
                video_bitrate="4000k",
                t=shot.duration_sec,
            )
            .overwrite_output()
            .run(quiet=True)
        )

        svd_raw.unlink(missing_ok=True)
        logger.info(
            "SVD clip created | episode={} shot={}", episode_num, shot_index
        )
        return output_path

    except ComfyUIOutOfMemoryError:
        logger.warning(
            "SVD OOM, falling back to zoompan | episode={} shot={}",
            episode_num,
            shot_index,
        )
        motion = shot.frames[0].motion if shot.frames else MotionDirection.ZOOM_IN
        return _create_zoompan_clip(image_path, audio_path, output_path, shot.duration_sec, motion)


def _resolve_frame_images(images_dir: Path, shot_idx: int, num_frames: int) -> List[Path]:
    """Resolve frame image paths for a shot. Supports both multi-frame and legacy naming."""
    if num_frames > 1:
        return [images_dir / f"shot-{shot_idx:02d}-frame-{f:02d}.png" for f in range(num_frames)]
    # Legacy single-frame: try frame-aware name first, then legacy name
    frame_path = images_dir / f"shot-{shot_idx:02d}-frame-00.png"
    if frame_path.exists():
        return [frame_path]
    return [images_dir / f"shot-{shot_idx:02d}.png"]


def assemble_shot_clips(
    episode_num: int,
    shots: List[ShotScript],
    audio_paths: List[Path],
) -> List[Path]:
    """Render per-shot clips.
    - Multi-frame shots: multiframe clip with crossfade (parallel, ThreadPoolExecutor(4))
    - Single-frame shots: zoompan with motion direction (parallel)
    - Key shots with SVD enabled: SVD (serial), uses first frame image
    Returns list of clip paths in shot order.
    """
    images_dir = (
        Path(settings.data_dir) / "images" / f"episode-{episode_num:03d}"
    )
    clips_dir = (
        Path(settings.data_dir) / "clips" / f"episode-{episode_num:03d}"
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    standard_jobs: List[tuple] = []
    svd_jobs: List[tuple] = []

    for idx, (shot, audio_path) in enumerate(zip(shots, audio_paths)):
        output_path = clips_dir / f"shot-{idx:02d}.mp4"
        num_frames = len(shot.frames) if shot.frames else 1
        frame_images = _resolve_frame_images(images_dir, idx, num_frames)

        if shot.is_key_shot and settings.enable_svd:
            svd_jobs.append((idx, shot, frame_images[0], audio_path, output_path))
        else:
            standard_jobs.append(
                (idx, shot, frame_images, audio_path, output_path)
            )

    clip_paths: Dict[int, Path] = {}

    # Standard clips (zoompan / multiframe) — parallel on CPU
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {}
        for idx, shot, frame_imgs, aud, out in standard_jobs:
            if len(frame_imgs) >= 2 and len(shot.frames) >= 2:
                future = executor.submit(
                    _create_multiframe_clip,
                    frame_imgs[:2], shot.frames[:2], aud, out, shot.duration_sec,
                )
            else:
                motion = shot.frames[0].motion if shot.frames else MotionDirection.ZOOM_IN
                future = executor.submit(
                    _create_zoompan_clip,
                    frame_imgs[0], aud, out, shot.duration_sec, motion,
                )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            clip_paths[idx] = future.result()

    # SVD clips — serial to avoid VRAM contention
    for idx, shot, image_path, audio_path, output_path in svd_jobs:
        clip_paths[idx] = _create_svd_clip(
            shot, idx, episode_num, image_path, audio_path, output_path
        )

    return [clip_paths[i] for i in sorted(clip_paths.keys())]
