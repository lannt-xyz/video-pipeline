from pathlib import Path
from typing import Optional

import ffmpeg
from loguru import logger

from config.settings import settings


class ValidationError(Exception):
    """Raised when an episode fails quality validation."""


class EpisodeValidator:
    """Asserts quality criteria for a completed episode."""

    def assert_episode(self, episode_num: int) -> None:
        """Run all checks. Raises ValidationError if any check fails."""
        video_path = (
            Path(settings.data_dir) / "output" / f"episode-{episode_num:03d}.mp4"
        )
        thumbnail_path = (
            Path(settings.data_dir) / "thumbnails" / f"episode-{episode_num:03d}.png"
        )

        errors = []

        if not video_path.exists():
            errors.append(f"Video not found: {video_path}")
        else:
            probe = self._probe(video_path)
            if probe:
                errors.extend(self._check_duration(probe))
                errors.extend(self._check_resolution(probe))
                errors.extend(self._check_audio(probe))
                errors.extend(self._check_file_size(video_path))

        if not thumbnail_path.exists():
            errors.append(f"Thumbnail not found: {thumbnail_path}")

        if errors:
            msg = " | ".join(errors)
            logger.error("Validation failed | episode={} errors={}", episode_num, msg)
            raise ValidationError(
                f"Episode {episode_num} failed validation: {msg}"
            )

        logger.info("Validation passed | episode={}", episode_num)

    def _probe(self, path: Path) -> Optional[dict]:
        try:
            return ffmpeg.probe(str(path))
        except ffmpeg.Error as exc:
            logger.error("ffprobe failed | path={} error={}", path, exc)
            return None

    def _check_duration(self, probe: dict) -> list:
        duration = float(probe.get("format", {}).get("duration", 0))
        if not (58 <= duration <= 62):
            return [f"Duration {duration:.1f}s not in [58s, 62s]"]
        return []

    def _check_resolution(self, probe: dict) -> list:
        video_streams = [
            s for s in probe.get("streams", []) if s.get("codec_type") == "video"
        ]
        if not video_streams:
            return ["No video stream found"]
        w = video_streams[0].get("width", 0)
        h = video_streams[0].get("height", 0)
        if w != settings.image_width or h != settings.image_height:
            return [
                f"Resolution {w}x{h} != {settings.image_width}x{settings.image_height}"
            ]
        return []

    def _check_audio(self, probe: dict) -> list:
        audio_streams = [
            s for s in probe.get("streams", []) if s.get("codec_type") == "audio"
        ]
        if not audio_streams:
            return ["No audio stream"]
        bitrate = int(audio_streams[0].get("bit_rate", 0))
        if 0 < bitrate < 128_000:
            return [f"Audio bitrate {bitrate // 1000}kbps < 128kbps"]
        return []

    def _check_file_size(self, path: Path) -> list:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < 5:
            return [f"File size {size_mb:.1f}MB < 5MB"]
        return []


validator = EpisodeValidator()
