from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import ffmpeg
from loguru import logger

from config.settings import settings
from image_gen.comfyui_client import ComfyUIOutOfMemoryError, comfyui_client
from models.schemas import ShotScript


def _create_zoompan_clip(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: int = 6,
) -> Path:
    """Ken Burns (zoompan) clip from static image + audio. Encoded with h264_nvenc."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = settings.fps
    total_frames = duration * fps

    video = (
        ffmpeg.input(str(image_path), loop=1, t=duration, framerate=fps)
        .filter(
            "zoompan",
            z="if(lte(zoom,1.0),1.05,zoom-0.0015)",
            d=total_frames,
            x="iw/2-(iw/zoom/2)",
            y="ih/2-(ih/zoom/2)",
            s=f"{settings.image_width}x{settings.image_height}",
            fps=fps,
        )
        .filter("setpts", "PTS-STARTPTS")
    )
    audio = ffmpeg.input(str(audio_path))

    (
        ffmpeg.output(
            video,
            audio.audio,
            str(output_path),
            vcodec="h264_nvenc",
            acodec="aac",
            audio_bitrate="192k",
            video_bitrate="4000k",
            r=fps,
            t=duration,
        )
        .overwrite_output()
        .run(quiet=True)
    )

    logger.debug("Zoompan clip created | path={}", output_path)
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

        # SVD outputs ~25 frames @6fps ≈ 4.2s; stretch to target duration via setpts
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
        return _create_zoompan_clip(image_path, audio_path, output_path, shot.duration_sec)


def assemble_shot_clips(
    episode_num: int,
    shots: List[ShotScript],
    audio_paths: List[Path],
) -> List[Path]:
    """Render per-shot clips.
    - Standard shots: zoompan (parallel, ThreadPoolExecutor(4))
    - Key shots with SVD enabled: SVD (serial)
    Returns list of clip paths in shot order.
    """
    images_dir = (
        Path(settings.data_dir) / "images" / f"episode-{episode_num:03d}"
    )
    clips_dir = (
        Path(settings.data_dir) / "clips" / f"episode-{episode_num:03d}"
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    zoompan_jobs: List[tuple] = []
    svd_jobs: List[tuple] = []

    for idx, (shot, audio_path) in enumerate(zip(shots, audio_paths)):
        image_path = images_dir / f"shot-{idx:02d}.png"
        output_path = clips_dir / f"shot-{idx:02d}.mp4"

        if shot.is_key_shot and settings.enable_svd:
            svd_jobs.append((idx, shot, image_path, audio_path, output_path))
        else:
            zoompan_jobs.append(
                (idx, image_path, audio_path, output_path, shot.duration_sec)
            )

    clip_paths: Dict[int, Path] = {}

    # Zoompan clips — parallel on CPU
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {
            executor.submit(
                _create_zoompan_clip, img, aud, out, dur
            ): idx
            for idx, img, aud, out, dur in zoompan_jobs
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            clip_paths[idx] = future.result()

    # SVD clips — serial to avoid VRAM contention
    for idx, shot, image_path, audio_path, output_path in svd_jobs:
        clip_paths[idx] = _create_svd_clip(
            shot, idx, episode_num, image_path, audio_path, output_path
        )

    return [clip_paths[i] for i in sorted(clip_paths.keys())]
