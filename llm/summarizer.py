import json
import re
import sqlite3
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import RetryError, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import summary_client as ollama_client
from models.schemas import ArcOverview, ChunkSummary
from pipeline.state import StateDB

_SUMMARY_PROMPT_VERSION = "scene-detail-v2"

_CHUNK_SYSTEM = (
        "You are a story analyst for AI video generation. Summarize the provided "
        "Vietnamese story chapters in Vietnamese with SCENE-LEVEL detail that can be "
        "directly used to build visual prompts. Focus on: concrete actions, who does "
        "what, where the action happens, and notable props/artifacts/weapon usage. "
        "Do NOT write abstract commentary.\n"
        "\n"
        "Output format (plain text only, no JSON):\n"
        "- Bối cảnh chính: 2-3 câu mô tả địa điểm/thời điểm\n"
        "- Cảnh 1..N (6-10 cảnh, theo đúng thứ tự thời gian), mỗi cảnh 2-3 câu gồm:\n"
        "  + Nhân vật chính trong cảnh\n"
        "  + Hành động cụ thể (động từ rõ ràng)\n"
        "  + Đạo cụ/pháp khí/vũ khí được cầm hoặc sử dụng (nếu có)\n"
        "  + Kết quả trực tiếp dẫn sang cảnh kế\n"
        "- Mồi cliffhanger: 1-2 câu chưa giải quyết hoàn toàn\n"
        "\n"
        "Write concise but specific text (~260-380 words). Return plain text only."
)

_ARC_SYSTEM = """You are a story analyst. Given multiple chunk summaries from consecutive story chapters, create an arc overview for this episode segment.

CRITICAL: key_events MUST be written as CHRONOLOGICAL SCENE BEATS — in the exact same time order that events happen in the story.
Each item in key_events = one specific action or scene that happens, described in 1-2 sentences.
Do NOT merge scenes or write thematic statements. Do NOT reorder events.
The FIRST key_event = what happens first in the story. The LAST key_event = what happens last.

QUALITY RULES for each key_event (mandatory):
- Mention at least 1 concrete action verb (mở, rút, lao tới, niệm chú, đập vỡ, v.v.).
- Mention location/background context if available (miếu, mộ, nghĩa địa, nhà kho, hành lang, rừng...).
- If a character uses/carries artifact/weapon/talisman, explicitly mention it.
- Avoid vague phrases like "mọi chuyện căng thẳng hơn", "tình hình leo thang".

Return a JSON object with EXACTLY this schema:
{
    "arc_summary": "string — narrative overview in Vietnamese, ~180-240 words, concrete and scene-oriented",
    "key_events": ["string — 8-12 chronological scene beats, each describing a specific action/scene in story order with location and object detail when possible"],
  "characters_in_episode": ["string — list of character names who appear"]
}"""


# Vietnamese text averages ~2 chars/token; reserve tokens for system prompt + output headroom.
# Cap at 6000 tokens of content (~12000 chars) so each chunk processes ≤2 chapters at a time,
# keeping individual LLM calls within the model's reliable generation window.
_CHARS_PER_TOKEN = 2
_CONTEXT_OVERHEAD_TOKENS = 26768  # 32768 - 6000 = 6000 available content tokens
_ID_LIKE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$")


def _max_content_chars() -> int:
    """Max characters of chapter content that safely fit in one LLM call."""
    available = max(settings.llm_context_size - _CONTEXT_OVERHEAD_TOKENS, 2000)
    return available * _CHARS_PER_TOKEN


def _resolve_wiki_db_path() -> Path | None:
    """Resolve DB that contains wiki_characters table."""
    candidates = [
        Path(settings.db_path),
        Path("data") / f"{settings.story_slug}.db",
    ]
    for db_path in candidates:
        if not db_path.exists():
            continue
        try:
            con = sqlite3.connect(str(db_path))
            exists = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='wiki_characters'"
            ).fetchone()
            con.close()
        except Exception:
            continue
        if exists:
            return db_path
    return None


def _load_character_lookup() -> tuple[dict[str, str], dict[str, str]]:
    """Return (id_to_name, alias_or_name_to_name) lookup from wiki_characters."""
    db_path = _resolve_wiki_db_path()
    if db_path is None:
        return {}, {}

    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        cols = {
            row["name"] for row in con.execute("PRAGMA table_info(wiki_characters)").fetchall()
        }
        active_filter = " WHERE is_delete = 0" if "is_delete" in cols else ""
        rows = con.execute(
            f"SELECT character_id, name, aliases_json FROM wiki_characters{active_filter}"
        ).fetchall()
        con.close()
    except Exception as exc:
        logger.debug("Failed to load character lookup from DB | err={}", exc)
        return {}, {}

    id_to_name: dict[str, str] = {}
    alias_to_name: dict[str, str] = {}
    for row in rows:
        char_id = str(row["character_id"] or "").strip()
        name = str(row["name"] or "").strip()
        if not char_id or not name:
            continue

        id_to_name[char_id.lower()] = name
        alias_to_name[name.lower()] = name

        try:
            aliases = json.loads(row["aliases_json"] or "[]")
        except Exception:
            aliases = []
        for a in aliases:
            alias = str(a).strip()
            if alias:
                alias_to_name[alias.lower()] = name

    return id_to_name, alias_to_name


def _normalize_characters_in_episode(raw_names: List[str]) -> List[str]:
    """Normalize character list to canonical Vietnamese names.

    - Strips Vietnamese honorific prefixes ("Lão đạo sĩ X" → "X") before lookup.
    - Drops generic placeholders ("Người đàn ông bí ẩn", "Ai đó", etc.) that
      cannot anchor IPAdapter and cause per-shot face drift.
    - Maps character_id/alias to canonical name via wiki_characters.
    - Drops unresolved id-like tokens (e.g. qing_yun_zi).
    - Preserves order and deduplicates.
    """
    id_to_name, alias_to_name = _load_character_lookup()

    normalized: List[str] = []
    seen: set[str] = set()

    for raw in raw_names:
        token = str(raw or "").strip()
        if not token:
            continue

        lower = token.lower()

        # Filter generic placeholders BEFORE any lookup — these are narrative
        # stand-ins that never resolve to a named character with an anchor.
        if _is_placeholder_character(lower):
            logger.debug("Dropping placeholder character name | name={}", token)
            continue

        # Try exact alias/id lookup first (wiki canonical form).
        canonical = alias_to_name.get(lower) or id_to_name.get(lower)

        # If no exact match, try stripping common Vietnamese honorifics and
        # re-looking up — this handles "Lão đạo sĩ Thanh Vân Tử" → "Thanh Vân Tử".
        # Only accept the stripped form when it maps to a KNOWN canonical name;
        # otherwise keep the original (stripping "Hiệu trưởng Chu" → "Chu" would
        # break anchor lookup since the folder is `hieu_truong_chu/`).
        if canonical is None:
            stripped = _strip_vn_honorifics(token)
            if stripped and stripped.lower() != lower:
                stripped_lower = stripped.lower()
                canonical = (
                    alias_to_name.get(stripped_lower)
                    or id_to_name.get(stripped_lower)
                )

        if canonical is None and _ID_LIKE_RE.match(lower):
            logger.debug("Dropping unresolved character id token | token={}", token)
            continue

        final_name = canonical or token
        key = final_name.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(final_name)

    return normalized


# Vietnamese honorific prefixes that should be stripped when resolving to an
# anchored character name. Ordered longest-first so multi-word prefixes match
# before single-word ones (e.g. "Lão đạo sĩ " before "Lão ").
_VN_HONORIFICS = (
    "lão đạo sĩ ", "lão đạo sỹ ", "đại đạo sĩ ", "đạo sĩ ", "đạo sỹ ",
    "đại sư ", "tiểu sư phụ ", "sư phụ ", "sư thúc ", "sư huynh ", "sư đệ ",
    "hiệu trưởng ", "thầy ", "cô ",
    "lão ", "tiểu ", "đại ",
)

# Pure placeholder patterns — narrative "mysterious man" style labels that
# cannot be anchored. Matched as prefix (with optional adjective suffix).
_PLACEHOLDER_PREFIXES = (
    "người đàn ông", "người phụ nữ", "người con gái", "người con trai",
    "người lạ", "người vô danh",
    "ai đó", "một người", "có người", "có kẻ", "kẻ lạ", "kẻ nào đó",
    "cô gái lạ", "chàng trai lạ",
)


def _strip_vn_honorifics(name: str) -> str:
    """Strip Vietnamese honorific prefixes from a character name.

    Returns the original string if no honorific matches.
    """
    lower = name.lower()
    for prefix in _VN_HONORIFICS:
        if lower.startswith(prefix):
            return name[len(prefix):].strip()
    return name


def _is_placeholder_character(name_lower: str) -> bool:
    """Return True when name is a generic narrative placeholder (no real identity)."""
    for prefix in _PLACEHOLDER_PREFIXES:
        if name_lower.startswith(prefix):
            return True
    return False


def _infer_characters_from_texts(texts: List[str], max_count: int = 12) -> List[str]:
    """Infer canonical character names from free text using wiki alias lookup."""
    _id_to_name, alias_to_name = _load_character_lookup()
    if not alias_to_name:
        return []

    corpus = "\n".join(t for t in texts if t).lower()
    if not corpus:
        return []

    found: List[str] = []
    seen: set[str] = set()

    # Prefer longer aliases first to reduce accidental short-token matches.
    for alias in sorted(alias_to_name.keys(), key=len, reverse=True):
        alias_norm = alias.strip().lower()
        if len(alias_norm) < 4:
            continue
        if not re.search(rf"(?<!\\w){re.escape(alias_norm)}(?!\\w)", corpus):
            continue

        canonical = alias_to_name[alias]
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(canonical)
        if len(found) >= max_count:
            break

    return found


def _is_chunk_cache_fresh(chunk_obj: dict) -> bool:
    """Return True when cached chunk summary matches current prompt version."""
    return chunk_obj.get("summary_version") == _SUMMARY_PROMPT_VERSION


def _is_arc_cache_fresh(arc_path: Path) -> bool:
    """Return True when cached arc summary was generated by current prompt version."""
    if not arc_path.exists():
        return False
    try:
        data = json.loads(arc_path.read_text(encoding="utf-8"))
        return data.get("summary_version") == _SUMMARY_PROMPT_VERSION
    except Exception:
        return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=15),
    retry=retry_if_not_exception_type(RetryError),
)
def _summarize_chunk(chapters_text: str, chunk_index: int, episode_num: int) -> str:
    prompt = f"Tóm tắt các chương sau:\n\n{chapters_text}"
    logger.debug("Summarizing chunk {} for episode {}", chunk_index, episode_num)
    return ollama_client.generate(
        prompt=prompt, system=_CHUNK_SYSTEM, temperature=0.5
    )


def _coerce_str_list(value: object) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Accept simple single-string fallback from model output.
        return [text]

    return []


def _fallback_key_events_from_chunks(chunk_summaries: List[str]) -> List[str]:
    events: List[str] = []
    for summary in chunk_summaries:
        for raw_line in summary.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            clean = re.sub(r"^(?:[-*\d\.)\s]+)", "", line).strip()
            if len(clean) < 20:
                continue
            events.append(clean)
            if len(events) >= 10:
                return events

    if events:
        return events

    merged = " ".join(s.strip() for s in chunk_summaries if s.strip())
    if not merged:
        return ["Tình tiết diễn biến theo thứ tự thời gian trong chương."]
    return [merged[:240]]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=15),
    retry=retry_if_not_exception_type(RetryError),
)
def _synthesize_arc(chunk_summaries: List[str], episode_num: int) -> ArcOverview:
    # Cap each chunk summary to avoid exceeding context window
    _MAX_CHUNK_CHARS = 1500
    truncated = [s[:_MAX_CHUNK_CHARS] for s in chunk_summaries]
    combined = "\n\n---\n\n".join(
        f"Chunk {i + 1}:\n{s}" for i, s in enumerate(truncated)
    )
    prompt = f"Episode {episode_num} chunk summaries:\n\n{combined}"
    raw = ollama_client.generate_json(
        prompt=prompt, system=_ARC_SYSTEM, temperature=0.3
    )

    if not isinstance(raw, dict):
        raise ValueError(f"Arc JSON must be an object, got {type(raw).__name__}")

    arc_summary = str(raw.get("arc_summary", "") or "").strip()
    if not arc_summary:
        arc_summary = " ".join(s.strip() for s in chunk_summaries if s.strip())[:600]

    key_events = _coerce_str_list(raw.get("key_events"))
    if not key_events:
        logger.warning(
            "Arc JSON missing key_events; deriving fallback from chunk summaries | episode={}",
            episode_num,
        )
        key_events = _fallback_key_events_from_chunks(chunk_summaries)

    raw_characters = _coerce_str_list(raw.get("characters_in_episode"))
    normalized_characters = _normalize_characters_in_episode(raw_characters)
    if not normalized_characters:
        normalized_characters = _infer_characters_from_texts(key_events + chunk_summaries)
        if normalized_characters:
            logger.warning(
                "Arc JSON missing characters_in_episode; inferred {} character(s) from text | episode={}",
                len(normalized_characters),
                episode_num,
            )

    return ArcOverview(
        episode_num=episode_num,
        arc_summary=arc_summary,
        key_events=key_events,
        characters_in_episode=normalized_characters,
    )


def summarize_episode(
    episode_num: int, chapter_start: int, chapter_end: int
) -> ArcOverview:
    """Two-pass summarization for one episode.
    Pass 1: merge chapters greedily into context-window-sized batches → chunk summaries.
           Each batch sends as many full chapters as fit before hitting the LLM context limit.
           Skips already-cached chunks.
    Pass 2: chunk summaries → ArcOverview (skips if arc file already exists).
    """
    # Short-circuit: if arc already exists, skip both passes entirely
    arc_path = (
        Path(settings.data_dir)
        / "summaries"
        / f"episode-{episode_num:03d}-arc.json"
    )
    if _is_arc_cache_fresh(arc_path):
        logger.info("Arc overview cached, skipping summarization | episode={}", episode_num)
        return load_arc_overview(episode_num)
    if arc_path.exists():
        logger.info(
            "Arc cache is stale (old summary version), regenerating | episode={}",
            episode_num,
        )

    chapter_nums = list(range(chapter_start, chapter_end + 1))
    chunk_summaries: List[str] = []
    db = StateDB()

    # Load already-saved chunk summaries to avoid re-calling LLM
    cached_chunks: dict = {}
    chunks_path = (
        Path(settings.data_dir)
        / "summaries"
        / f"episode-{episode_num:03d}-chunks.json"
    )
    if chunks_path.exists():
        for c in json.loads(chunks_path.read_text(encoding="utf-8")):
            if _is_chunk_cache_fresh(c):
                cached_chunks[c["chunk_index"]] = c["summary"]

    # Pass 1 — dynamic packing: fill each LLM call as many full chapters as context allows
    max_chars = _max_content_chars()
    chunk_idx = 0
    current_nums: List[int] = []
    current_texts: List[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal chunk_idx, current_nums, current_texts, current_len
        if not current_texts:
            return
        if chunk_idx in cached_chunks:
            chunk_summaries.append(cached_chunks[chunk_idx])
            logger.debug("Chunk {} cached, skipping LLM | episode={}", chunk_idx, episode_num)
        else:
            logger.debug(
                "Summarizing chunk {} | episode={} chapters={}-{} chars={}",
                chunk_idx, episode_num, current_nums[0], current_nums[-1], current_len,
            )
            summary = _summarize_chunk("\n\n".join(current_texts), chunk_idx, episode_num)
            chunk_summaries.append(summary)
            _save_chunk_summary(
                ChunkSummary(
                    episode_num=episode_num,
                    chunk_index=chunk_idx,
                    chapter_start=current_nums[0],
                    chapter_end=current_nums[-1],
                    summary=summary,
                )
            )
        chunk_idx += 1
        current_nums = []
        current_texts = []
        current_len = 0

    for ch_num in chapter_nums:
        content = db.get_chapter_content(ch_num)
        if not content:
            continue
        ch_text = f"=== Chương {ch_num} ===\n{content}"
        # Flush current batch before adding if it would overflow context
        if current_texts and current_len + len(ch_text) > max_chars:
            _flush()
        current_nums.append(ch_num)
        current_texts.append(ch_text)
        current_len += len(ch_text)

    _flush()  # flush remaining chapters

    logger.info(
        "Pass 1 done | episode={} chunks={} max_chars_per_chunk={}",
        episode_num, len(chunk_summaries), max_chars,
    )

    # Pass 2 — synthesize arc
    arc = _synthesize_arc(chunk_summaries, episode_num)

    arc_path.parent.mkdir(parents=True, exist_ok=True)
    arc_payload = arc.model_dump()
    arc_payload["summary_version"] = _SUMMARY_PROMPT_VERSION
    arc_path.write_text(
        json.dumps(arc_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
    data = json.loads(path.read_text(encoding="utf-8"))
    data["characters_in_episode"] = _normalize_characters_in_episode(
        data.get("characters_in_episode", [])
    )
    return ArcOverview(**data)


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
    payload = chunk.model_dump()
    payload["summary_version"] = _SUMMARY_PROMPT_VERSION
    existing.append(payload)
    existing.sort(key=lambda x: x["chunk_index"])

    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


_THUMBNAIL_TAGS_SYSTEM = (
    "You are a visual prompt engineer for Stable Diffusion / Flux image generation.\n"
    "Given Vietnamese story scene beats, extract 8-12 English comma-separated visual tags "
    "for a thumbnail image.\n"
    "Rules:\n"
    "- English ONLY. No Vietnamese. No sentences. No markdown.\n"
    "- Identify the MOST visually dramatic and shocking moment from the scene beats.\n"
    "- Tags must be concrete visual elements: character action (verb+noun), location, "
    "key props, character expression, visual drama.\n"
    "- Prefer visual specificity over vague atmosphere.\n"
    "- Output format: one line, comma-separated tags only."
)


def distill_thumbnail_tags(episode_num: int) -> str:
    """Condense arc key_events into English visual tags for thumbnail prompt.

    Reads the cached ArcOverview and uses the summary LLM (already warm in LLM
    phase) to extract 8-12 English visual tags from the most dramatic key_events.
    Result is cached to summaries/episode-NNN-thumb-tags.txt so the images phase
    can read it without reloading the model.

    Returns empty string on any failure so callers can fall back gracefully.
    """
    cache_path = (
        Path(settings.data_dir)
        / "summaries"
        / f"episode-{episode_num:03d}-thumb-tags.txt"
    )
    if cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8").strip()
        if cached:
            logger.debug("Thumbnail tags cached | episode={}", episode_num)
            return cached

    try:
        arc = load_arc_overview(episode_num)
    except Exception as exc:
        logger.warning("Cannot load arc for thumbnail distillation | episode={} err={}", episode_num, exc)
        return ""

    if not arc.key_events:
        return ""

    # Use the second half of key_events (typically the dramatic climax).
    half = max(1, len(arc.key_events) // 2)
    dramatic_events = arc.key_events[half:]
    prompt = "Scene beats:\n" + "\n".join(f"- {e}" for e in dramatic_events)

    try:
        raw = ollama_client.generate(
            prompt=prompt,
            system=_THUMBNAIL_TAGS_SYSTEM,
            temperature=0.3,
        )
        tags = raw.strip().strip('"').strip("'")
        if not tags:
            return ""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(tags, encoding="utf-8")
        logger.info("Thumbnail tags distilled | episode={} tags={}", episode_num, tags[:80])
        return tags
    except Exception as exc:
        logger.warning("Thumbnail tag distillation failed | episode={} err={}", episode_num, exc)
        return ""
