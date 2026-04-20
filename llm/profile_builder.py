"""Build character visual-anchor profiles from wiki SQLite tables.

Flow per character:
  wiki_characters + wiki_relations + wiki_artifacts
      → build_markdown()  → saves <id>.md   (confirm step 1)
      → _derive_tags()    → saves <id>.raw.json (confirm step 2)
      → validate + sanitize → saves <id>.json    (final)

Run standalone:
  python -m llm.profile_builder
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.character_extractor import _sanitize_description, _slugify
from llm.client import ollama_client
from models.schemas import Character

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_PROFILE_SYSTEM = """You are a character appearance designer for AI image generation (Stable Diffusion / PonyXL Danbooru model).
Your job: read a Vietnamese character profile in Markdown and produce a clean Danbooru tag list for that character.

The Markdown profile contains:
- **Visual Anchor** — the character's fixed, defining appearance. This is your PRIMARY source.
- Personality traits — use these to infer visual equivalents when Visual Anchor is absent.
- Artifacts / weapons — if an artifact has "Sự kiện chủ chốt: có", add its visual to the description.
- Relationships — context only, do NOT extract appearance from here.

OUTPUT FORMAT — return a single JSON object (not an array):
{
  "name": "full Vietnamese name",
  "alias": ["alternative names"],
  "gender": "male | female | unknown",
  "description": "12+ comma-separated Danbooru tags — NO prose",
  "relationships": {"other_character_name": "relationship"}
}

GENDER RULES:
- If profile says "Giới tính: male"   → gender = "male",    description starts with: 1boy, solo
- If profile says "Giới tính: female" → gender = "female",  description starts with: 1girl, solo
- If profile says "Giới tính: unknown" → gender = "unknown", description starts with: 1other, solo, androgynous

MANDATORY DESCRIPTION RULES:
1. Minimum 12 tags. Categories required:
   - Hair: color + length + style
   - Eyes: color + expression
   - Face: notable features
   - Outfit: clothes matching modern urban ghost-hunter setting (unless story says otherwise)
   - Body type
   - Expression

2. Setting = modern urban ghost-hunter story (Mao Son Troc Quy Nhan).
   Default clothing:
     - Ghost hunters / students → modern casual or tactical wear. NO hanfu.
     - Daoist masters / elders  → daoist robes acceptable.
     - Ancient spirits / ghosts → traditional clothing acceptable.

3. FORBIDDEN abstract tags: mysterious aura, mysterious, scholar, ethereal, spiritual energy,
   exudes, symbolizing, enchanting, magical presence, otherworldly, ancient wisdom, cunning aura,
   dangerous aura, noble aura, cold aura.
   REPLACE with visual equivalents: serious expression, sharp eyes, cold expression, etc.

4. WEIGHTS — sparingly. Max 2 weighted tags. DO NOT weight every tag.

5. For male (1boy): forbidden hair = side ponytail, twin tails, pigtails (unless story-explicit).
6. For female (1girl): required at least 2 of: fully clothed, high collar, long sleeves,
   traditional attire, modern casual wear, formal wear. NO bare shoulders / revealing tags.

EXAMPLES:
Modern male ghost hunter: "1boy, solo, short black hair, side part, dark brown eyes, sharp eyes, dark jacket, dark pants, talisman in hand, athletic build, serious expression"
Daoist elder male: "1boy, solo, long white hair, low bun, white daoist robes, thin beard, wrinkled face, wise gaze, wooden staff, prayer beads"
Modern female: "1girl, solo, long black hair, straight hair, dark eyes, gentle expression, pale skin, slender, white blouse, dark skirt, modern casual wear, fully clothed, looking at viewer"
Unknown/abstract entity: "1other, solo, androgynous, pale skin, white hair, long hair, blank expression, minimal white robes, glowing eyes, slim build, ethereal silhouette"
"""

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


@contextmanager
def _open_db(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
    finally:
        con.close()


def _wiki_characters_has_is_delete(con: sqlite3.Connection) -> bool:
    """Return True when wiki_characters has soft-delete column is_deleted."""
    rows = con.execute("PRAGMA table_info(wiki_characters)").fetchall()
    return any(row["name"] == "is_deleted" for row in rows)


def _load_wiki_character(character_id: str, con: sqlite3.Connection) -> Optional[dict]:
    active_filter = " AND is_deleted = 0" if _wiki_characters_has_is_delete(con) else ""
    cur = con.execute(
        f"SELECT * FROM wiki_characters WHERE character_id = ?{active_filter}",
        (character_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    d["aliases_json"] = json.loads(d.get("aliases_json") or "[]")
    d["traits_json"] = json.loads(d.get("traits_json") or "[]")
    return d


def _load_relations(character_id: str, con: sqlite3.Connection) -> list[dict]:
    cur = con.execute(
        "SELECT related_name, description, chapter_start "
        "FROM wiki_relations WHERE character_id = ? "
        "ORDER BY chapter_start",
        (character_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def _load_artifacts(character_id: str, con: sqlite3.Connection) -> list[dict]:
    # Check tables exist and have rows first to avoid expensive JOIN on empty tables
    cur = con.execute("SELECT COUNT(*) FROM wiki_artifact_snapshots WHERE owner_id = ?", (character_id,))
    if cur.fetchone()[0] == 0:
        return []

    cur = con.execute(
        """
        SELECT a.artifact_id, a.name, a.rarity, a.material, a.visual_anchor, a.description,
               s.chapter_start, s.normal_state, s.active_state, s.condition, s.vfx_color,
               s.is_key_event
        FROM wiki_artifacts a
        JOIN (
            SELECT artifact_id, MIN(chapter_start) AS first_ch
            FROM wiki_artifact_snapshots WHERE owner_id = ?
            GROUP BY artifact_id
        ) first ON a.artifact_id = first.artifact_id
        JOIN (
            SELECT artifact_id, chapter_start, MAX(extraction_version) AS best_ver
            FROM wiki_artifact_snapshots WHERE owner_id = ?
            GROUP BY artifact_id, chapter_start
        ) ver ON a.artifact_id = ver.artifact_id AND ver.chapter_start = first.first_ch
        JOIN wiki_artifact_snapshots s ON s.artifact_id = ver.artifact_id
            AND s.chapter_start = ver.chapter_start
            AND s.extraction_version = ver.best_ver
        """,
        (character_id, character_id),
    )
    return [dict(r) for r in cur.fetchall()]


def _load_best_snapshot(character_id: str, con: sqlite3.Connection) -> Optional[dict]:
    """Load the highest visual_importance active snapshot for fallback."""
    cur = con.execute(
        """
        SELECT physical_description, outfit, weapon, vfx_vibes, level, chapter_start
        FROM wiki_snapshots
        WHERE character_id = ? AND is_active = 1
          AND (physical_description IS NOT NULL OR outfit IS NOT NULL)
        ORDER BY visual_importance DESC, chapter_start ASC
        LIMIT 1
        """,
        (character_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _load_extraction_coverage(con: sqlite3.Connection) -> Optional[str]:
    cur = con.execute(
        """
        SELECT MIN(chapter_start) AS ch_min, MAX(chapter_end) AS ch_max,
               COUNT(*) AS total,
               SUM(CASE WHEN status IN ('DONE','MERGED') THEN 1 ELSE 0 END) AS done,
               MAX(extraction_version) AS max_ver
        FROM wiki_batches
        """
    )
    row = cur.fetchone()
    if row is None or row["total"] == 0:
        return None
    return (
        f"ch.{row['ch_min']}–{row['ch_max']} | "
        f"{row['done']}/{row['total']} batches | "
        f"v{row['max_ver']}"
    )


def _load_supplemental_snapshots(character_id: str, con: sqlite3.Connection, limit: int = 5) -> list[dict]:
    """Load top snapshots by visual_importance to supplement sparse visual_anchor."""
    cur = con.execute(
        """
        SELECT physical_description, outfit, weapon, vfx_vibes, level, visual_importance, chapter_start
        FROM wiki_snapshots
        WHERE character_id = ? AND is_active = 1
          AND (outfit IS NOT NULL OR weapon IS NOT NULL)
        ORDER BY visual_importance DESC, chapter_start ASC
        LIMIT ?
        """,
        (character_id, limit),
    )
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Chapter-context fallback helpers
# ---------------------------------------------------------------------------

_CONTEXT_INFER_SYSTEM = """You are a character appearance designer for AI image generation (Stable Diffusion / PonyXL).
You receive raw Vietnamese story excerpts mentioning a character. Infer their appearance from context clues.

OUTPUT — return a single JSON object:
{
  "name": "full Vietnamese name as written in the text",
  "alias": [],
  "gender": "male | female | unknown",
  "description": "12+ comma-separated Danbooru tags — NO prose",
  "relationships": {}
}

RULES:
- Infer gender from: pronouns (ông/anh/cậu → male; bà/cô/chị → female), role titles, family titles.
- Infer age/build from: role (con trai = young man; vợ = adult woman; lão = elderly).
- MANDATORY description categories: hair color+style, eye expression, outfit (context-appropriate), body type, expression.
- Setting = modern Vietnamese rural/urban ghost-hunter story. Default: modern casual clothing.
- FORBIDDEN tags: mysterious aura, spiritual energy, ethereal, enchanting — use visual equivalents only.
- Minimum 12 tags. Start with gender count tag: 1boy/1girl/1other, solo.
- If no physical description clues at all, generate plausible appearance for role/age inferred from context."""

_CONTEXT_SNIPPET_WINDOW = 400  # chars around each mention
_CONTEXT_MAX_SNIPPETS = 5
_CONTEXT_MAX_CHARS = 2000  # total chars sent to LLM


def _load_chapter_context(
    character_id: str,
    char_name: str,
    aliases: list[str],
    con: sqlite3.Connection,
) -> Optional[str]:
    """Search chapters.content for the character name/aliases and return
    concatenated context snippets for LLM inference.

    Returns None when the chapters table doesn't exist or has no content.
    """
    # Check table exists
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "chapters" not in tables:
        return None

    search_names = [char_name] + aliases

    # Collect chapter_nums from wiki_mention_index first (fast path).
    mention_chapters: list[int] = []
    if "wiki_mention_index" in tables:
        rows = con.execute(
            "SELECT DISTINCT chapter_num FROM wiki_mention_index WHERE character_id = ? ORDER BY chapter_num",
            (character_id,),
        ).fetchall()
        mention_chapters = [r[0] for r in rows]

    # Fallback: LIKE search across all chapters if mention_index is empty.
    if not mention_chapters:
        like_clauses = " OR ".join(["content LIKE ?" for _ in search_names])
        params = [f"%{n}%" for n in search_names]
        rows = con.execute(
            f"SELECT chapter_num FROM chapters WHERE ({like_clauses}) AND content IS NOT NULL ORDER BY chapter_num LIMIT 10",
            params,
        ).fetchall()
        mention_chapters = [r[0] for r in rows]

    if not mention_chapters:
        return None

    snippets: list[str] = []
    total_chars = 0

    for ch_num in mention_chapters:
        if len(snippets) >= _CONTEXT_MAX_SNIPPETS or total_chars >= _CONTEXT_MAX_CHARS:
            break
        row = con.execute(
            "SELECT content FROM chapters WHERE chapter_num = ?", (ch_num,)
        ).fetchone()
        if not row or not row[0]:
            continue
        content: str = row[0]

        # Find first mention of any search name in this chapter.
        best_idx = -1
        for name in search_names:
            idx = content.find(name)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx

        if best_idx == -1:
            continue

        start = max(0, best_idx - _CONTEXT_SNIPPET_WINDOW // 2)
        end = min(len(content), best_idx + _CONTEXT_SNIPPET_WINDOW // 2)
        snippet = content[start:end].strip()
        snippets.append(f"[Chương {ch_num}] ...{snippet}...")
        total_chars += len(snippet)

    if not snippets:
        return None

    return "\n\n".join(snippets)


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------


def build_markdown(character_id: str, con: sqlite3.Connection) -> Optional[str]:
    """Build Markdown anchor profile for one character.

    Returns None if the character lacks enough data to build a meaningful anchor
    (visual_anchor, personality, and traits_json all null/empty).
    """
    char = _load_wiki_character(character_id, con)
    if char is None:
        logger.warning("Character not found in DB | id={}", character_id)
        return None

    visual_anchor: Optional[str] = char.get("visual_anchor")
    personality: Optional[str] = char.get("personality")
    traits: list[str] = char.get("traits_json") or []

    # Load best snapshot: use as fallback when visual_anchor is absent,
    # or as supplemental visual detail when visual_anchor is too short.
    _VA_WEAK_THRESHOLD = 80  # chars — below this the anchor is treated as insufficient
    snapshot: Optional[dict] = None
    va_is_weak = not visual_anchor or len((visual_anchor or "").strip()) < _VA_WEAK_THRESHOLD

    if not visual_anchor and not personality and not traits:
        # No data at all — need snapshot or skip
        snapshot = _load_best_snapshot(character_id, con)
        if snapshot is None:
            logger.debug("Skipping {} — no visual data", character_id)
            return None
        logger.debug("Using snapshot fallback for {} (ch.{})", character_id, snapshot["chapter_start"])
    elif va_is_weak:
        # visual_anchor exists but is too vague — load best snapshot for supplemental detail
        snapshot = _load_best_snapshot(character_id, con)
        if snapshot:
            logger.debug(
                "visual_anchor weak ({} chars); loading snapshot supplement for {} (ch.{})",
                len((visual_anchor or "").strip()), character_id, snapshot["chapter_start"],
            )

    lines: list[str] = []

    # Helper: treat DB string "null" / "none" / empty as missing value.
    def _val(v) -> Optional[str]:
        s = (v or "").strip()
        return s if s and s.lower() not in ("null", "none", "n/a") else None

    # Infer gender from visual_anchor text when DB gender is NULL.
    # Simple keyword match on the first 60 chars covers "Nam giới" / "Nữ giới" reliably.
    def _infer_gender_from_anchor(anchor_text: Optional[str]) -> Optional[str]:
        if not anchor_text:
            return None
        sample = anchor_text[:60].lower()
        if any(k in sample for k in ("nam giới", "nam nhân", "đàn ông", "con trai", "chàng trai", "lão ông", "cụ ông")):
            return "male"
        if any(k in sample for k in ("nữ giới", "nữ nhân", "đàn bà", "con gái", "thiếu nữ", "cô gái", "cô nương", "nương tử")):
            return "female"
        return None

    # --- Header ---
    lines.append(f"# {char['name']}")
    lines.append("")

    # Gender on its own line — matches system-prompt GENDER RULES pattern exactly.
    gender = _val(char.get("gender"))
    if gender is None:
        # Visual anchor often starts with "Nam giới" or "Nữ giới" — infer from there.
        inferred = _infer_gender_from_anchor(_val(char.get("visual_anchor")))
        if inferred:
            gender = inferred
            logger.debug(
                "Gender inferred from visual_anchor | id={} name={} → {}",
                char.get("id"), char.get("name"), gender,
            )
    gender = gender or "unknown"
    lines.append(f"**Giới tính**: {gender}")

    meta_parts: list[str] = []
    faction = _val(char.get("faction"))
    if faction:
        meta_parts.append(f"**Phe**: {faction}")

    ver = char.get("remaster_version")
    if ver:
        meta_parts.append(f"**Dữ liệu v{ver}**")

    if meta_parts:
        lines.append(" | ".join(meta_parts))

    if char["aliases_json"]:
        lines.append(f"**Tên khác**: {', '.join(char['aliases_json'])}")

    if traits:
        lines.append(f"**Tính cách**: {', '.join(traits)}")

    if personality:
        lines.append(f"**Tính cách chi tiết**: {personality}")

    if visual_anchor:
        lines.append(f"**Visual Anchor**: {visual_anchor}")

    # Always include snapshot supplement when visual_anchor is weak
    supplemental = []
    if va_is_weak and snapshot:
        supplemental = _load_supplemental_snapshots(character_id, con, limit=5)

    if supplemental:
        lines.append("")
        lines.append("## Ngoại hình từ snapshot (bổ sung)")
        seen_outfits: set[str] = set()
        seen_weapons: set[str] = set()
        for snap in supplemental:
            outfit = (snap.get("outfit") or "").strip()
            weapon = (snap.get("weapon") or "").strip()
            level = (snap.get("level") or "").strip()
            vfx = (snap.get("vfx_vibes") or "").strip()
            ch = snap.get("chapter_start", "?")
            parts: list[str] = [f"ch.{ch}"]
            if outfit and outfit not in seen_outfits:
                parts.append(f"Trang phục: {outfit}")
                seen_outfits.add(outfit)
            if weapon and weapon not in seen_weapons:
                parts.append(f"Vũ khí: {weapon}")
                seen_weapons.add(weapon)
            if level:
                parts.append(f"Cảnh giới: {level}")
            if vfx:
                parts.append(f"VFX: {vfx}")
            if len(parts) > 1:  # at least one real field beyond chapter
                lines.append("- " + " | ".join(parts))

    if not visual_anchor and snapshot:
        # Fallback: build visual anchor from snapshot fields
        snap_parts: list[str] = []
        if snapshot.get("physical_description"):
            snap_parts.append(snapshot["physical_description"])
        if snapshot.get("outfit"):
            snap_parts.append(f"Trang phục: {snapshot['outfit']}")
        if snapshot.get("weapon"):
            snap_parts.append(f"Vũ khí: {snapshot['weapon']}")
        if snapshot.get("vfx_vibes"):
            snap_parts.append(f"VFX: {snapshot['vfx_vibes']}")
        if snap_parts:
            lines.append(f"**Visual Anchor** (từ snapshot ch.{snapshot['chapter_start']}): {' | '.join(snap_parts)}")
        if snapshot.get("level"):
            lines.append(f"**Cảnh giới**: {snapshot['level']}")

    # --- Artifacts ---
    artifacts = _load_artifacts(character_id, con)
    key_artifacts = [a for a in artifacts if a.get("is_key_event")]
    if artifacts:
        lines.append("")
        lines.append("## Bảo vật / Vũ khí sở hữu")
        for a in artifacts:
            parts: list[str] = []
            if a.get("rarity"):
                parts.append(f"hiếm: {a['rarity']}")
            if a.get("material"):
                parts.append(f"chất liệu: {a['material']}")
            suffix = f" ({', '.join(parts)})" if parts else ""
            lines.append(f"### {a['name']}{suffix}")
            if a.get("visual_anchor"):
                lines.append(f"- Visual: {a['visual_anchor']}")
            if a.get("normal_state"):
                normal = a["normal_state"]
                active = a.get("active_state")
                line = f"- ch.{a['chapter_start']}: {normal}"
                if active:
                    line += f" / kích hoạt: {active}"
                lines.append(line)
            if a.get("condition"):
                cond_line = f"- Tình trạng: {a['condition']}"
                if a.get("vfx_color"):
                    cond_line += f" | VFX: {a['vfx_color']}"
                lines.append(cond_line)
            if a.get("is_key_event"):
                lines.append("- Sự kiện chủ chốt: có")

    # --- Relations ---
    relations = _load_relations(character_id, con)
    if relations:
        lines.append("")
        lines.append("## Quan hệ")
        for r in relations:
            desc = r.get("description") or ""
            lines.append(f"- {r['related_name']} (ch.{r['chapter_start']}): {desc}")

    # --- Footer ---
    coverage = _load_extraction_coverage(con)
    if coverage:
        lines.append("")
        lines.append("---")
        lines.append(f"> Dữ liệu: {coverage}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM derivation
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _derive_tags(markdown_text: str) -> dict[str, Any]:
    result = ollama_client.generate_json(
        prompt=markdown_text,
        system=_PROFILE_SYSTEM,
        temperature=0.2,
    )
    if isinstance(result, list) and result:
        return result[0]
    if isinstance(result, dict):
        return result
    raise ValueError(f"LLM returned unexpected type: {type(result)}")


def _sanitize_unknown(desc: str) -> str:
    """Enforce 1other, solo, androgynous prefix for unknown-gender characters."""
    tags = [t.strip() for t in desc.split(",") if t.strip()]
    tags = [t for t in tags if t not in ("1boy", "1girl", "1other", "solo", "androgynous")]
    return "1other, solo, androgynous, " + ", ".join(tags)


# ---------------------------------------------------------------------------
# Shared DB resolution helper
# ---------------------------------------------------------------------------


def _resolve_wiki_db(db_path: str) -> Optional[str]:
    """Return the DB path that actually contains wiki_characters.

    wiki tables and pipeline state tables (chapters/episodes) live in
    different files:
      - pipeline state: data/<slug>/<slug>.db  (settings.db_path)
      - wiki data:      data/<slug>.db         (flat path)
    Returns None when no wiki DB is found.
    """
    def _has_wiki(path: str) -> bool:
        if not Path(path).exists():
            return False
        try:
            con = sqlite3.connect(path)
            found = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='wiki_characters'"
            ).fetchone() is not None
            con.close()
            return found
        except Exception:
            return False

    if _has_wiki(db_path):
        return db_path

    flat = str(Path("data") / f"{settings.story_slug}.db")
    if _has_wiki(flat):
        logger.info("wiki_characters not in settings DB — using flat DB: {}", flat)
        return flat

    return None


def _resolve_character_dir(chars_dir: Path, char_id: str, char_name: str) -> Path:
    """Return canonical character folder path (slug from display name).

    No on-disk migration is performed; old folders can be cleaned manually.
    """
    canonical_dir = chars_dir / _slugify(char_name)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    return canonical_dir


# ---------------------------------------------------------------------------
# Per-episode profile builder (preferred: only builds what the episode needs)
# ---------------------------------------------------------------------------


def build_profiles_for_episode(
    character_names: list[str],
    force: bool = False,
) -> list[Character]:
    """Build profiles only for characters whose names appear in `character_names`.

    Matches against wiki_characters.name and aliases_json.
    Idempotent: skips characters that already have a .json profile.
    Raises RuntimeError if wiki DB is missing or no names matched.
    """
    if not character_names:
        raise RuntimeError(
            "build_profiles_for_episode called with empty character list — "
            "episode arc produced no character names (chapters may be empty)."
        )

    chars_dir = Path(settings.data_dir) / "characters"
    chars_dir.mkdir(parents=True, exist_ok=True)

    wiki_db = _resolve_wiki_db(settings.db_path)
    if wiki_db is None:
        raise RuntimeError(
            "wiki_characters table not found in any known DB path. "
            "Run wiki extraction before the pipeline."
        )

    built = skipped = failed = 0

    with _open_db(wiki_db) as con:
        active_filter = " WHERE is_deleted = 0" if _wiki_characters_has_is_delete(con) else ""

        # Build name/alias → character_id lookup
        rows = con.execute(
            f"SELECT character_id, name, aliases_json FROM wiki_characters{active_filter}"
        ).fetchall()
        name_to_id: dict[str, str] = {}
        for row in rows:
            name_to_id[row["name"].lower().strip()] = row["character_id"]
            for alias in json.loads(row["aliases_json"] or "[]"):
                name_to_id[alias.lower().strip()] = row["character_id"]

        # Resolve requested names → character_ids (preserve order, deduplicate)
        target_ids: list[str] = []
        seen: set[str] = set()
        for name in character_names:
            char_id = name_to_id.get(name.lower().strip())
            if char_id and char_id not in seen:
                seen.add(char_id)
                target_ids.append(char_id)
            elif not char_id:
                logger.debug("Character not matched in wiki | name={}", name)

        if not target_ids:
            raise RuntimeError(
                f"None of the episode characters matched wiki_characters: {character_names}. "
                "The arc summary likely contains placeholder names (empty chapter content). "
                "Ensure chapters are crawled and have content before running the LLM phase."
            )

        logger.info(
            "Building profiles for {}/{} episode characters",
            len(target_ids), len(character_names),
        )

        for char_id in target_ids:
            canonical_name_filter = (
                " AND is_deleted = 0" if _wiki_characters_has_is_delete(con) else ""
            )
            wiki_row = con.execute(
                f"SELECT name, gender FROM wiki_characters WHERE character_id=?{canonical_name_filter}",
                (char_id,),
            ).fetchone()
            char_name = wiki_row["name"] if wiki_row else char_id

            char_dir = _resolve_character_dir(chars_dir, char_id, char_name)
            json_path = char_dir / "profile.json"
            md_path = char_dir / "profile.md"
            raw_path = char_dir / "profile.raw.json"

            if json_path.exists() and not force:
                skipped += 1
                logger.debug("Profile exists, skipping | id={}", char_id)
                continue

            try:
                md = build_markdown(char_id, con)
                if md is None:
                    # No wiki data — try to infer from chapter context.
                    char_data = _load_wiki_character(char_id, con)
                    aliases = (char_data or {}).get("aliases_json") or []
                    context = _load_chapter_context(char_id, char_name, aliases, con)
                    if context is None:
                        logger.debug("No visual data or chapter context, skipping | id={}", char_id)
                        skipped += 1
                        continue

                    logger.info(
                        "No wiki visual data — inferring from chapter context | id={} snippets_chars={}",
                        char_id, len(context),
                    )
                    raw = ollama_client.generate_json(
                        prompt=(
                            f"Character name: {char_name}\n\n"
                            f"Story excerpts mentioning this character:\n\n{context}"
                        ),
                        system=_CONTEXT_INFER_SYSTEM,
                        temperature=0.3,
                    )
                    if isinstance(raw, list) and raw:
                        raw = raw[0]
                    if not isinstance(raw, dict):
                        logger.warning("Context inference returned unexpected type | id={}", char_id)
                        skipped += 1
                        continue
                else:
                    md_path.write_text(md, encoding="utf-8")
                    raw = _derive_tags(md)

                raw_path.write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                raw.setdefault("name", char_name)
                raw.setdefault("alias", [])
                raw.setdefault("gender", "unknown")
                raw.setdefault("description", "")
                raw.setdefault("relationships", {})

                char = Character(**raw)
                gender = char.gender
                if gender == "unknown":
                    char.description = _sanitize_unknown(char.description)
                elif gender in ("male", "female"):
                    char.description = _sanitize_description(char.description, gender)
                else:
                    char.gender = "unknown"
                    char.description = _sanitize_unknown(char.description)

                json_path.write_text(char.model_dump_json(indent=2), encoding="utf-8")
                logger.info("Built profile | id={} gender={}", char_id, char.gender)
                built += 1

            except Exception as exc:
                logger.warning("Failed to build profile | id={} error={}", char_id, exc)
                failed += 1

    logger.info(
        "Episode profile build done | built={} skipped={} failed={}",
        built, skipped, failed,
    )
    return _load_all_built(chars_dir)


# ---------------------------------------------------------------------------
# Full-catalog profile builder (use for initial seed or bulk rebuild)
# ---------------------------------------------------------------------------


def build_all_profiles(force: bool = False) -> list[Character]:
    """Build visual-anchor profiles for all characters in wiki_characters.

    Per character:
      1. Builds Markdown → saves <id>.md  (review before LLM)
      2. Calls LLM       → saves <id>.raw.json  (review LLM output)
      3. Validates + sanitizes → saves <id>.json (final)

    Idempotent: skips characters where <id>.json already exists (unless force=True).
    """
    chars_dir = Path(settings.data_dir) / "characters"
    chars_dir.mkdir(parents=True, exist_ok=True)

    db_path = settings.db_path
    if not Path(db_path).exists():
        # Fallback: try flat path (old DB location before slug scoping)
        flat = Path("data") / f"{settings.story_slug}.db"
        if flat.exists():
            db_path = str(flat)
            logger.warning("Using legacy DB path: {}", db_path)
        else:
            logger.error("DB not found at {} or {}", settings.db_path, flat)
            return []

    built = skipped = failed = 0

    wiki_db = _resolve_wiki_db(db_path)
    if wiki_db is None:
        logger.warning(
            "wiki_characters not found in {} or flat path — "
            "falling back to arc-based extraction.",
            db_path,
        )
        from llm.character_extractor import extract_all_characters
        return extract_all_characters()

    with _open_db(wiki_db) as con:
        active_filter = " WHERE is_deleted = 0" if _wiki_characters_has_is_delete(con) else ""

        cur = con.execute(
            f"SELECT character_id, name, gender FROM wiki_characters{active_filter} ORDER BY character_id"
        )
        all_chars = cur.fetchall()
        total = len(all_chars)
        logger.info("Found {} characters in wiki_characters", total)

        for row in all_chars:
            char_id: str = row["character_id"]
            char_name: str = row["name"]
            char_dir = _resolve_character_dir(chars_dir, char_id, char_name)
            json_path = char_dir / "profile.json"
            md_path = char_dir / "profile.md"
            raw_path = char_dir / "profile.raw.json"

            if json_path.exists() and not force:
                skipped += 1
                continue

            try:
                # Step 1: build Markdown anchor
                md = build_markdown(char_id, con)
                if md is None:
                    logger.debug("Skipped {} — no visual data", char_id)
                    skipped += 1
                    continue

                md_path.write_text(md, encoding="utf-8")
                logger.debug("Saved profile markdown | id={}", char_id)

                # Step 2: LLM derivation
                raw = _derive_tags(md)
                raw_path.write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                logger.debug("Saved raw LLM output | id={}", char_id)

                # Step 3: validate + sanitize
                raw.setdefault("name", char_name)
                raw.setdefault("alias", [])
                raw.setdefault("gender", "unknown")
                raw.setdefault("description", "")
                raw.setdefault("relationships", {})

                char = Character(**raw)

                gender = char.gender
                if gender == "unknown":
                    char.description = _sanitize_unknown(char.description)
                elif gender in ("male", "female"):
                    char.description = _sanitize_description(char.description, gender)
                else:
                    logger.warning("Unexpected gender '{}' for {} — treating as unknown", gender, char_id)
                    char.gender = "unknown"
                    char.description = _sanitize_unknown(char.description)

                json_path.write_text(char.model_dump_json(indent=2), encoding="utf-8")
                logger.info("Built profile | id={} gender={}", char_id, char.gender)
                built += 1

            except Exception as exc:
                logger.warning("Failed to build profile | id={} error={}", char_id, exc)
                failed += 1

    logger.info(
        "build_all_profiles done | built={} skipped={} failed={} total={}",
        built, skipped, failed, total,
    )
    return _load_all_built(chars_dir)


def _load_all_built(chars_dir: Path) -> list[Character]:
    # Delegate to load_all_characters so the is_delete active filter is applied
    # consistently. chars_dir is ignored here because load_all_characters already
    # resolves the correct directory from settings.
    from llm.character_extractor import load_all_characters
    return load_all_characters()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    force = "--force" in args
    char_ids = [a for a in args if not a.startswith("--")]

    if char_ids:
        # Build specific characters only
        db_path = settings.db_path
        if not Path(db_path).exists():
            flat = Path("data") / f"{settings.story_slug}.db"
            db_path = str(flat) if flat.exists() else db_path

        chars_dir = Path(settings.data_dir) / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)

        with _open_db(db_path) as con:
            for char_id in char_ids:
                char_dir = chars_dir / char_id
                char_dir.mkdir(parents=True, exist_ok=True)
                json_path = char_dir / "profile.json"
                md_path = char_dir / "profile.md"
                raw_path = char_dir / "profile.raw.json"

                if json_path.exists() and not force:
                    print(f"[skip] {char_id} — already built (use --force to rebuild)")
                    continue

                md = build_markdown(char_id, con)
                if md is None:
                    print(f"[skip] {char_id} — no visual data or not found")
                    continue

                md_path.write_text(md, encoding="utf-8")
                print(f"[md]   {md_path}")

                try:
                    raw = _derive_tags(md)
                    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[raw]  {raw_path}")

                    raw.setdefault("name", char_id)
                    raw.setdefault("alias", [])
                    raw.setdefault("gender", "unknown")
                    raw.setdefault("description", "")
                    raw.setdefault("relationships", {})

                    char = Character(**raw)
                    gender = char.gender
                    if gender == "unknown":
                        char.description = _sanitize_unknown(char.description)
                    elif gender in ("male", "female"):
                        char.description = _sanitize_description(char.description, gender)
                    else:
                        char.gender = "unknown"
                        char.description = _sanitize_unknown(char.description)

                    json_path.write_text(char.model_dump_json(indent=2), encoding="utf-8")
                    print(f"[json] {json_path}")
                except Exception as exc:
                    print(f"[fail] {char_id} — {exc}")
    else:
        profiles = build_all_profiles(force=force)
        print(f"\nTotal profiles loaded: {len(profiles)}")
