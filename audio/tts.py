import asyncio
from pathlib import Path
from typing import List, Tuple

import edge_tts
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from models.schemas import ShotScript


async def _generate_single(
    shot_index: int,
    narration_text: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, Path]:
    """Generate TTS MP3 for a single shot. Respects semaphore concurrency limit."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _tts() -> None:
        communicate = edge_tts.Communicate(narration_text, settings.tts_voice)
        await communicate.save(str(output_path))

    async with semaphore:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await _tts()
        logger.debug("TTS generated | shot={} path={}", shot_index, output_path)
        return shot_index, output_path


async def generate_episode_tts(
    episode_num: int,
    shots: List[ShotScript],
    max_concurrent: int = 5,
) -> List[Path]:
    """Generate TTS for all shots in an episode concurrently.
    Returns list of audio paths in shot order.
    """
    audio_dir = (
        Path(settings.data_dir) / "audio" / f"episode-{episode_num:03d}"
    )
    audio_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _generate_single(
            idx,
            shot.narration_text,
            audio_dir / f"shot-{idx:02d}.mp3",
            semaphore,
        )
        for idx, shot in enumerate(shots)
    ]

    results = await asyncio.gather(*tasks)
    results_sorted = sorted(results, key=lambda x: x[0])
    paths = [r[1] for r in results_sorted]

    logger.info("TTS done | episode={} shots={}", episode_num, len(paths))
    return paths


def generate_episode_tts_sync(
    episode_num: int, shots: List[ShotScript]
) -> List[Path]:
    """Sync wrapper for generate_episode_tts."""
    return asyncio.run(generate_episode_tts(episode_num, shots))
