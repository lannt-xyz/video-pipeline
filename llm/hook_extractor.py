"""Phase 2 — Viral Moment Extraction.

Stage tách riêng khỏi summarizer. Summarizer = faithful narrative; hook_extractor
= deliberately biased "what would shock TikTok viewers". Output drives chỉ hook +
1-2 key shot; narrative spine vẫn từ ArcOverview.

Behind feature flag `retention.use_constraint_system`. Persists to
`data/{slug}/viral_moments/episode-NNN.json` for calibration trace.

NOT a refactor of summarizer — additive, idempotent (cache check), reuses
summary_client to avoid acquiring extra VRAM context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import RetryError, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import summary_client as ollama_client
from llm.summarizer import load_arc_overview
from models.schemas import ArcOverview, ViralMoment

_HOOK_EXTRACTOR_VERSION = "viral-v1"

_HOOK_EXTRACTOR_SYSTEM = """You are a TikTok/YouTube Shorts hook scout for a Vietnamese supernatural-horror story.

YOUR JOB: scan the chapter summaries below and pick 3-5 SHOCKING moments that would make a viewer freeze and ask "what just happened?".

CRITICAL — you are NOT writing a summary. You are CHERRY-PICKING moments. Faithful narrative is summarizer's job.

WHAT MAKES A GOOD VIRAL MOMENT (rank by all of these):
- Visual shock: corpse face, wound, blood, supernatural reveal, ritual gone wrong
- Question seed: leaves a "why?" or "what is that?" hanging in air
- Specificity: a CONCRETE image, not abstract dread ("a child's pale hand pressed against the coffin lid", NOT "darkness fell")
- Drop context allowed: you may withhold the explanation — that's the whole point

WHAT TO AVOID:
- Generic horror clichés ("máu chảy", "bóng tối ập tới", "cái chết đến gần")
- Already-resolved moments (the explanation is part of the description)
- Long backstory or character introduction
- Anything that needs >1 sentence of setup to make sense

OUTPUT — return EXACTLY this JSON shape, nothing else:
{
  "moments": [
    {
      "chapter_refs": [<integer chapter numbers this moment spans>],
      "description": "<Vietnamese, 1-2 sentences, the concrete visual moment>",
      "shock_factor": "<English or Vietnamese, why this is shocking — 1 sentence>",
      "mystery_seed": "<the unanswered question this plants in the viewer's head, 1 short sentence>"
    },
    ...
  ]
}

Generate 3-5 moments. Order them by descending shock potential (best first)."""


def _viral_moments_path(episode_num: int) -> Path:
    return (
        Path(settings.data_dir)
        / "viral_moments"
        / f"episode-{episode_num:03d}.json"
    )


def _load_chunk_summaries(episode_num: int) -> List[dict]:
    """Reuse the same chunks file the scriptwriter uses."""
    path = (
        Path(settings.data_dir)
        / "summaries"
        / f"episode-{episode_num:03d}-chunks.json"
    )
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return raw


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("version") == _HOOK_EXTRACTOR_VERSION


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=15),
    retry=retry_if_not_exception_type(RetryError),
)
def _call_llm(arc: ArcOverview, scenes_text: str) -> dict:
    arc_event_text = "\n".join(
        f"Event {i + 1}: {e}" for i, e in enumerate(arc.key_events)
    )
    prompt = (
        f"Episode {arc.episode_num} — arc summary:\n{arc.arc_summary}\n\n"
        f"Key events (in order):\n{arc_event_text}\n\n"
        f"Detailed chapter summaries:\n\n{scenes_text}\n\n"
        "Pick 3-5 viral moments per the rules above. Return JSON only."
    )
    raw = ollama_client.generate_json(
        prompt=prompt, system=_HOOK_EXTRACTOR_SYSTEM, temperature=0.6
    )
    if not isinstance(raw, dict):
        raise ValueError(f"hook_extractor JSON must be object, got {type(raw).__name__}")
    return raw


def _coerce_moment(item: object, fallback_chapter: int) -> ViralMoment | None:
    if not isinstance(item, dict):
        return None
    desc = str(item.get("description", "") or "").strip()
    if not desc:
        return None
    refs_raw = item.get("chapter_refs")
    refs: List[int] = []
    if isinstance(refs_raw, list):
        for x in refs_raw:
            try:
                refs.append(int(x))
            except (TypeError, ValueError):
                continue
    if not refs:
        refs = [fallback_chapter]
    return ViralMoment(
        chapter_refs=refs,
        description=desc,
        shock_factor=str(item.get("shock_factor", "") or "").strip(),
        mystery_seed=str(item.get("mystery_seed", "") or "").strip(),
    )


def extract_viral_moments(episode_num: int) -> List[ViralMoment]:
    """Generate viral-moment candidates for an episode. Idempotent (file cache).

    Returns empty list on hard failure rather than raising — caller falls back
    to legacy hook generation.
    """
    out_path = _viral_moments_path(episode_num)
    if _is_cache_fresh(out_path):
        logger.info("Viral moments cached, skipping | episode={}", episode_num)
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            return [ViralMoment(**m) for m in data.get("moments", [])]
        except Exception as e:  # noqa: BLE001
            logger.warning("Cached viral moments unreadable, regenerating | err={}", e)

    arc = load_arc_overview(episode_num)
    chunks = _load_chunk_summaries(episode_num)
    if not chunks:
        logger.warning(
            "No chunk summaries for hook extraction | episode={}", episode_num
        )
        return []

    # Cap each chunk text to keep prompt within context window.
    _MAX_CHUNK_CHARS = 1500
    scenes_text = "\n\n---\n\n".join(
        f"Chunk {i + 1} (chương {c.get('chapter_start')}-{c.get('chapter_end')}):\n"
        f"{str(c.get('summary', ''))[:_MAX_CHUNK_CHARS]}"
        for i, c in enumerate(chunks)
    )

    try:
        raw = _call_llm(arc, scenes_text)
    except Exception as e:  # noqa: BLE001  (RetryError is a subclass of Exception)
        logger.error(
            "Hook extraction failed after retries | episode={} err={}",
            episode_num, e,
        )
        return []

    moments_raw = raw.get("moments") if isinstance(raw, dict) else None
    if not isinstance(moments_raw, list):
        logger.warning(
            "hook_extractor returned no 'moments' list | episode={} keys={}",
            episode_num, list(raw.keys()) if isinstance(raw, dict) else None,
        )
        return []

    fallback_ch = chunks[0].get("chapter_start", 1) if chunks else 1
    moments: List[ViralMoment] = []
    for item in moments_raw:
        m = _coerce_moment(item, fallback_chapter=int(fallback_ch))
        if m is not None:
            moments.append(m)

    if not moments:
        logger.warning("hook_extractor produced 0 valid moments | episode={}", episode_num)
        return []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _HOOK_EXTRACTOR_VERSION,
        "episode_num": episode_num,
        "moments": [m.model_dump() for m in moments],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Viral moments extracted | episode={} count={} path={}",
        episode_num, len(moments), out_path,
    )
    return moments


def load_viral_moments(episode_num: int) -> List[ViralMoment]:
    """Load cached viral moments without regenerating. Empty list if missing."""
    path = _viral_moments_path(episode_num)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [ViralMoment(**m) for m in data.get("moments", [])]
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to load viral moments | episode={} err={}", episode_num, e)
        return []
