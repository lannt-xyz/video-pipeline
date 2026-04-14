import json
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from crawler.storage import load_chapter_content
from llm.client import ollama_client
from models.schemas import ArcOverview, ChunkSummary

_CHUNK_SYSTEM = (
    "You are a story analyst. Summarize the provided Vietnamese story chapters "
    "in Vietnamese. Focus on: key events, character actions, emotional beats, "
    "and plot developments. Write a concise summary of ~200 words. "
    "Return plain text only, no JSON."
)

_ARC_SYSTEM = """You are a story analyst. Given multiple chunk summaries from consecutive story chapters, create an arc overview for this episode segment.
Return a JSON object with EXACTLY this schema:
{
  "arc_summary": "string — comprehensive narrative overview in Vietnamese, ~300 words",
  "key_events": ["string — list of 5-7 key events that happened"],
  "characters_in_episode": ["string — list of character names who appear"]
}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _summarize_chunk(chapters_text: str, chunk_index: int, episode_num: int) -> str:
    prompt = f"Tóm tắt các chương sau:\n\n{chapters_text}"
    logger.debug("Summarizing chunk {} for episode {}", chunk_index, episode_num)
    return ollama_client.generate(
        prompt=prompt, system=_CHUNK_SYSTEM, temperature=0.5
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _synthesize_arc(chunk_summaries: List[str], episode_num: int) -> ArcOverview:
    combined = "\n\n---\n\n".join(
        f"Chunk {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )
    prompt = f"Episode {episode_num} chunk summaries:\n\n{combined}"
    raw = ollama_client.generate_json(
        prompt=prompt, system=_ARC_SYSTEM, temperature=0.3
    )
    return ArcOverview(
        episode_num=episode_num,
        arc_summary=raw["arc_summary"],
        key_events=raw["key_events"],
        characters_in_episode=raw["characters_in_episode"],
    )


def summarize_episode(
    episode_num: int, chapter_start: int, chapter_end: int
) -> ArcOverview:
    """Two-pass summarization for one episode.
    Pass 1: groups of 5 chapters → chunk summaries.
    Pass 2: chunk summaries → ArcOverview.
    """
    chapters_per_chunk = 5
    chapter_nums = list(range(chapter_start, chapter_end + 1))
    chunk_summaries: List[str] = []

    # Pass 1 — one summary per 5-chapter chunk
    for chunk_idx, start in enumerate(range(0, len(chapter_nums), chapters_per_chunk)):
        batch = chapter_nums[start : start + chapters_per_chunk]
        texts = []
        for ch_num in batch:
            content = load_chapter_content(ch_num)
            if content:
                # Truncate long chapters to keep total prompt manageable
                texts.append(f"=== Chương {ch_num} ===\n{content[:3000]}")

        if not texts:
            logger.warning(
                "No content for chunk {} episode {}", chunk_idx, episode_num
            )
            continue

        summary = _summarize_chunk("\n\n".join(texts), chunk_idx, episode_num)
        chunk_summaries.append(summary)
        _save_chunk_summary(
            ChunkSummary(
                episode_num=episode_num,
                chunk_index=chunk_idx,
                chapter_start=batch[0],
                chapter_end=batch[-1],
                summary=summary,
            )
        )

    logger.info(
        "Pass 1 done | episode={} chunks={}", episode_num, len(chunk_summaries)
    )

    # Pass 2 — synthesize arc
    arc = _synthesize_arc(chunk_summaries, episode_num)

    arc_path = (
        Path(settings.data_dir)
        / "summaries"
        / f"episode-{episode_num:03d}-arc.json"
    )
    arc_path.parent.mkdir(parents=True, exist_ok=True)
    arc_path.write_text(arc.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "Pass 2 done | episode={} arc_len={}", episode_num, len(arc.arc_summary)
    )
    return arc


def load_arc_overview(episode_num: int) -> ArcOverview:
    path = (
        Path(settings.data_dir)
        / "summaries"
        / f"episode-{episode_num:03d}-arc.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Arc overview not found for episode {episode_num}")
    return ArcOverview(**json.loads(path.read_text(encoding="utf-8")))


def _save_chunk_summary(chunk: ChunkSummary) -> None:
    path = (
        Path(settings.data_dir)
        / "summaries"
        / f"episode-{chunk.episode_num:03d}-chunks.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[dict] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

    existing = [c for c in existing if c.get("chunk_index") != chunk.chunk_index]
    existing.append(chunk.model_dump())
    existing.sort(key=lambda x: x["chunk_index"])

    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
