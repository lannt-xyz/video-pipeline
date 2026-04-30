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
from llm.client import image_prompt_client
from models.schemas import Character

# Alias used for the chapter-context fallback path. Kept distinct from
# image_prompt_client so we can swap providers per-path later if needed.
_context_infer_client = image_prompt_client


def _write_profile(json_path: Path, char: Character) -> None:
    """Write profile.json while preserving on-disk fields the model resets to None.

    `Character.model_dump_json()` writes default values for fields like
    `anchor_path` (and any future asset paths). Without preservation, a
    rebuild silently wipes already-generated asset references on disk.
    """
    # Whitelist of fields that may exist on disk but should not be lost on rebuild.
    _PRESERVE_FIELDS = ("anchor_path", "anchor_3q_path", "anchor_side_path")

    existing: dict[str, Any] = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    data = json.loads(char.model_dump_json(indent=2))
    for field in _PRESERVE_FIELDS:
        if existing.get(field) and not data.get(field):
            data[field] = existing[field]
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# System prompt (built lazily from settings.story_setting so swapping stories
# does not require code changes — abstraction principle from copilot-instructions)
# ---------------------------------------------------------------------------


def _build_profile_system() -> str:
    s = settings.story_setting
    forbidden = ", ".join(s.forbidden_visual_tags)
    return f"""You are a character appearance designer for AI image generation using Flux Dev (a natural-language image model).
Your job: read a Vietnamese character profile in Markdown and produce a rich, detailed English appearance description for that character.

The Markdown profile contains:
- **Visual Anchor** — the character's fixed, defining appearance (outfit, signature look). PRIMARY source.
- **Ngoại hình từ snapshot** — chapter-by-chapter appearance notes. Use to fill in missing details.
- **Tính cách / Tính cách chi tiết** — personality. Translate into VISUAL body language cues.
- **Bảo vật / Vũ khí** — weapons and artifacts. If "Sự kiện chủ chốt: có", add 1 representative item to description.
- **Quan hệ** — context only, do NOT extract appearance from here.

OUTPUT FORMAT — return a single JSON object (not an array):
{{
  "name": "full Vietnamese name",
  "alias": ["alternative names"],
  "gender": "male | female | unknown",
  "description": "20+ comma-separated English phrases — see rules below",
  "relationships": {{"other_character_name": "relationship"}}
}}

GENDER RULES:
- If profile says "Giới tính: male"   → gender = "male",    description starts with: young man (or adult man / elderly man depending on age)
- If profile says "Giới tính: female" → gender = "female",  description starts with: young woman (or adult woman / elderly woman depending on age)
- If profile says "Giới tính: unknown" → gender = "unknown", description starts with: androgynous figure, ambiguous gender
- CRITICAL: do NOT use Danbooru count tokens (1boy, 1girl, 1other, solo) — they are stripped by the pipeline. Use natural English phrases instead.

MANDATORY DESCRIPTION CATEGORIES (all required, in order):
1. GENDER + AGE — first phrase, always: "young man", "adult woman", "elderly man", etc.
2. ETHNICITY + SKIN TONE — "East Asian features", "fair skin", "olive complexion", "tan skin", "pale complexion", etc.
3. HAIR — color + length + texture + style (4 traits minimum): "black medium-length slightly tousled hair with natural layering"
4. EYES — color + shape + expression (3 traits minimum): "dark brown eyes, slightly hooded lids, sharp focused gaze"
5. FACE STRUCTURE — 3+ traits: jaw shape, cheekbone prominence, nose, brow: "strong angular jaw, high cheekbones, defined brow ridge, flat nose bridge"
6. BODY TYPE — build + height impression + posture: "lean athletic build, broad shoulders, upright composed posture"
7. PRIMARY OUTFIT — from Visual Anchor or snapshot, be specific: "crisp white button-up shirt with collar slightly open, dark tailored trousers, leather belt"
8. EXPRESSION + BODY LANGUAGE — translate personality into visuals: "calm measured expression, jaw set with quiet confidence, weight evenly distributed"
9. HANDS / CARRIED ITEMS — what character typically holds or carries: "talisman paper gripped between index and middle finger", "feng shui compass clipped at belt"
10. SIGNATURE FEATURE — 1 distinctive visual that makes this character recognizable: unique scar, accessory, posture habit, aura marker

DETAIL RULES:
- Minimum 20 phrases. Aim for 24-28 for a richer identity lock.
- Each phrase must be descriptive (3+ words). Single nouns like "talisman" or "black hair" are FORBIDDEN — expand to "yellowed talisman paper folded twice between fingers".
- PERSONALITY → VISUAL translation guide:
    điềm tĩnh / calm → "relaxed jaw, unhurried gaze, deliberate stillness in posture"
    kiêu ngạo / arrogant → "chin tilted slightly upward, half-lidded eyes, arms loosely crossed"
    quyết đoán / decisive → "set jaw, direct unblinking eye contact, squared shoulders"
    thông minh / intelligent → "slightly furrowed brow in thought, observant scanning gaze"
    quan tâm / caring → "slight softening around eyes, open non-aggressive body orientation"
- FORBIDDEN abstract tags: {forbidden}. Use visual equivalents only.
- Do NOT use weight syntax (tag:1.2) — Flux ignores it.
- For male: add masculine markers — strong jaw, defined cheekbones, flat nose bridge, short natural eyebrows. Forbidden hair styles: side ponytail, twin tails, pigtails.
- For female: at minimum 2 modesty tags — fully clothed, high collar, long sleeves, traditional attire, formal wear. NO bare shoulders / revealing tags.

SETTING = {s.genre_hint}.
Default outfit if Visual Anchor absent:
  - Young / student / hunter → {s.default_clothing_modern}.
  - Master / elder / spirit / ancient → {s.default_clothing_traditional}.

EXAMPLES:
Modern male daoist hunter (20 phrases):
"young man, East Asian features, fair skin with light tan, black medium-length slightly tousled hair, natural side part, dark brown eyes, slightly hooded lids, sharp focused gaze, strong angular jaw, high cheekbones, flat nose bridge, lean athletic build, broad shoulders, upright composed posture, crisp white button-up shirt collar slightly open, dark tailored trousers, leather belt, calm measured expression with jaw set, yellowed talisman paper folded between fingers, feng shui compass clipped at belt, slight upward chin tilt of quiet confidence"

Daoist elder male (20 phrases):
"elderly man, East Asian features, pale aged skin with visible wrinkles, long white hair tied in a low loose bun, sparse white eyebrows, deep-set dark eyes with gentle wisdom, prominent cheekbones on gaunt face, thin white beard reaching chest, frail slender frame, slightly stooped posture, layered white daoist robes with grey sash, worn wooden staff gripped in right hand, amber prayer beads looped at wrist, serene unhurried expression, lips pressed in knowing calm"

Modern female:
"young woman, East Asian features, fair porcelain skin, long straight black hair falling past shoulders, dark brown eyes with gentle expression, soft rounded jaw, slender build with graceful posture, white high-collar blouse, dark straight-leg trousers, modern casual wear, fully clothed, hands clasped in front, quiet observant gaze, small silver hairpin at left temple"
"""


def _build_context_infer_system() -> str:
    s = settings.story_setting
    forbidden = ", ".join(s.forbidden_visual_tags)
    return f"""You are a character appearance designer for AI image generation (Stable Diffusion / PonyXL).
You receive raw Vietnamese story excerpts mentioning a character. Infer their appearance from context clues.

OUTPUT — return a single JSON object:
{{
  "name": "full Vietnamese name as written in the text",
  "alias": [],
  "gender": "male | female | unknown",
  "description": "12+ comma-separated Danbooru tags — NO prose",
  "relationships": {{}}
}}

RULES:
- Infer gender from: pronouns (ông/anh/cậu/hắn → male; bà/cô/chị/nàng → female), role titles, family titles.
- Infer age/build from role (con trai = young man; vợ = adult woman; lão = elderly).
- MANDATORY description categories: hair color+style, eye expression, outfit (context-appropriate), body type, expression.
- SETTING = {s.genre_hint}. Default: {s.default_clothing_modern}.
- FORBIDDEN tags: {forbidden} — use visual equivalents only.
- CRITICAL: do NOT use Danbooru count tokens (1boy, 1girl, 1other, solo) — they are stripped by the pipeline. Use natural English phrases instead.
- Minimum 12 phrases. FIRST phrase MUST be a gender+age marker: "young man", "adult woman", "elderly man", "androgynous figure", etc.
- For male: add masculine face markers — strong jaw, defined cheekbones, short eyebrows.
- If no physical clues at all, generate plausible appearance for the role/age inferred from context."""

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


# ---------------------------------------------------------------------------
# Appearance / gender inference helpers
# ---------------------------------------------------------------------------

# Appearance vocabulary used to validate that a free-form Vietnamese string
# describes physical traits (vs. plot events). Lowercase, accent-preserving.
#
# Strong tokens: a single hit is enough — these are unambiguously about looks.
# Weak tokens:   common words that can appear in plot text too (e.g. "mặt"
#                in "mặt mọi người"); we require ≥2 hits before treating the
#                string as an appearance description.
_APPEARANCE_STRONG = (
    "tóc", "mắt", "râu", "ria", "ngũ quan", "tướng mạo", "diện mạo",
    "khuôn mặt", "mặt mũi", "vóc dáng", "thân hình", "thân thể",
    "vết sẹo", "hình xăm",
    "tuấn tú", "anh tuấn", "diễm lệ", "yêu kiều", "kiều diễm",
    "xinh đẹp", "mỹ nhân", "mỹ miều",
    "trang phục", "y phục", "quần áo", "hanfu", "đạo bào", "áo bào",
    "khôi giáp", "mặt nạ",
)
_APPEARANCE_WEAK = (
    "da", "thân", "vóc", "dáng", "mặt",
    "cao", "thấp", "lùn", "gầy", "béo", "mập",
    "đẹp", "xấu", "xinh", "tuấn", "diễm",
    "trắng", "đen", "vàng", "xanh", "đỏ", "tím", "nâu", "bạc", "hồng",
    "áo", "quần", "váy", "khăn", "mũ", "giày", "đai", "lụa",
    "tuổi", "trẻ", "trung niên", "già", "lão", "thiếu",
)


def _is_appearance_text(text: str) -> bool:
    """Return True when `text` describes physical appearance.

    Single hit on a strong-token list is enough; weak tokens require at
    least two distinct matches to overcome plot-text noise like
    "Bị ép buộc đi tiểu trước mặt mọi người" (only "mặt" matches, weak).
    """
    if not text:
        return False
    low = text.lower()
    if any(k in low for k in _APPEARANCE_STRONG):
        return True
    weak_hits = sum(1 for k in _APPEARANCE_WEAK if k in low)
    return weak_hits >= 2


# Gender heuristics — Vietnamese / Sino-Vietnamese cues. Multi-word phrases
# come first so longer matches win over their substrings.
_MALE_NAME_SUFFIXES = (
    "tiên sinh", "chân nhân", "đạo nhân", "hòa thượng", "đạo trưởng",
    "thiếu gia", "công tử", "lão gia", "đại nhân",
    "tử", "lão", "ông", "huynh", "đệ", "công", "vương", "đế", "hoàng",
    "lang", "phu", "sư phụ",
)
_FEMALE_NAME_SUFFIXES = (
    "phu nhân", "công chúa", "thái hậu", "tiên cô", "thiếu nữ",
    "cô nương", "nương tử",
    "nương", "thị", "tỷ", "muội", "cô", "bà", "nữ",
)

_MALE_TEXT_CUES = (
    "nam giới", "nam nhân", "đàn ông", "con trai", "chàng trai", "thiếu niên",
    "lão ông", "cụ ông", "anh tuấn", "tuấn tú", "vạm vỡ", "có râu",
    "cường tráng", "hảo hán", "hắn", " ông ", " anh ", " cậu ",
)
_FEMALE_TEXT_CUES = (
    "nữ giới", "nữ nhân", "đàn bà", "con gái", "thiếu nữ", "cô gái",
    "cô nương", "nương tử", "diễm lệ", "yêu kiều", "xinh đẹp", "mỹ nhân",
    "kiều diễm", "nàng", " bà ", " cô ", " chị ",
)


def _name_suffix_gender(name: str) -> Optional[str]:
    """Infer gender from a Sino-Vietnamese name's title/suffix words.

    Matches whole words at end of the name (preceded by space) or the entire
    name equals the suffix — avoids false positives like "Tử" inside "Tử Vi".
    Returns None when ambiguous.
    """
    if not name:
        return None
    low = name.strip().lower()
    for suf in _FEMALE_NAME_SUFFIXES:
        if low.endswith(" " + suf) or low == suf:
            return "female"
    for suf in _MALE_NAME_SUFFIXES:
        if low.endswith(" " + suf) or low == suf:
            return "male"
    return None


def _text_gender(*texts: Optional[str]) -> Optional[str]:
    """Tally male/female cue words across one or more free-form texts.

    Single-character cues like " ông " / " bà " are padded with whitespace
    so they only match as standalone words. Returns the gender with the
    higher count; ties → None.
    """
    blob = " ".join(t for t in texts if t).lower()
    if not blob.strip():
        return None
    blob = " " + blob + " "
    male = sum(blob.count(c) for c in _MALE_TEXT_CUES)
    female = sum(blob.count(c) for c in _FEMALE_TEXT_CUES)
    if male > female and male > 0:
        return "male"
    if female > male and female > 0:
        return "female"
    return None


def _infer_gender(
    name: str,
    aliases: list[str],
    visual_anchor: Optional[str],
    personality: Optional[str],
    traits: list[str],
    relations: list[dict],
    snapshot: Optional[dict] = None,
    supplemental: Optional[list[dict]] = None,
) -> Optional[str]:
    """Multi-signal Vietnamese gender inference.

    Priority — first decisive signal wins:
      1. Name / alias suffix titles.
      2. Cue-word tally in visual_anchor.
      3. Cue-word tally in personality + traits.
      4. Cue-word tally in relation descriptions.
      5. Cue-word tally in best snapshot + supplemental snapshot fields
         (outfit / physical_description). Catches cases like male-typical
         "quần tây", "áo sơ mi" outfits when DB gender is null.

    Returns None when no signal is decisive (caller falls back to "unknown").
    """
    for candidate in [name, *(aliases or [])]:
        if g := _name_suffix_gender(candidate):
            return g

    if g := _text_gender(visual_anchor):
        return g

    if g := _text_gender(personality, " ".join(traits or [])):
        return g

    if relations:
        rel_blob = " ".join((r.get("description") or "") for r in relations)
        if g := _text_gender(rel_blob):
            return g

    snap_texts: list[str] = []
    if snapshot:
        for k in ("physical_description", "outfit", "weapon"):
            v = snapshot.get(k)
            if v:
                snap_texts.append(str(v))
    for s in supplemental or []:
        for k in ("physical_description", "outfit", "weapon"):
            v = s.get(k)
            if v:
                snap_texts.append(str(v))
    if snap_texts:
        if g := _text_gender(" ".join(snap_texts)):
            return g

    return None


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
    """Load the highest visual_importance active snapshot for fallback.

    Filters out snapshots whose `physical_description` is event-text only
    (e.g. "Bị ép buộc đi tiểu trước mặt mọi người.") because the wiki
    extractor sometimes stores plot events in this field. Such text causes
    the downstream LLM to emit nonsense appearance tags.
    """
    cur = con.execute(
        """
        SELECT physical_description, outfit, weapon, vfx_vibes, level, chapter_start
        FROM wiki_snapshots
        WHERE character_id = ? AND is_active = 1
          AND (physical_description IS NOT NULL OR outfit IS NOT NULL)
        ORDER BY visual_importance DESC, chapter_start ASC
        LIMIT 5
        """,
        (character_id,),
    )
    candidates = [dict(r) for r in cur.fetchall()]
    if not candidates:
        return None

    for snap in candidates:
        phys = (snap.get("physical_description") or "").strip()
        outfit = (snap.get("outfit") or "").strip()
        # Treat physical_description as appearance only when it contains
        # at least one appearance keyword. Otherwise, scrub it so downstream
        # markdown does not present plot text as Visual Anchor.
        if phys and not _is_appearance_text(phys):
            snap["physical_description"] = None
            phys = ""
        # Snapshot is useful only if it carries SOME visual signal.
        if phys or outfit or (snap.get("weapon") or "").strip():
            return snap

    return None


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

    Returns None when there is no usable APPEARANCE signal — the caller
    should then either skip the character or fall back to chapter-context
    inference. Plot-only fields (traits / personality without any visual
    description) are NOT considered enough — they cause the LLM to fabricate.

    Appearance signals checked, in order:
      - `visual_anchor` containing real appearance keywords
      - best `wiki_snapshots` row with `physical_description` (passing
        `_is_appearance_text`), `outfit`, or `weapon`
      - any supplemental snapshot rows with outfit/weapon
    """
    char = _load_wiki_character(character_id, con)
    if char is None:
        logger.warning("Character not found in DB | id={}", character_id)
        return None

    visual_anchor: Optional[str] = char.get("visual_anchor")
    personality: Optional[str] = char.get("personality")
    traits: list[str] = char.get("traits_json") or []
    aliases: list[str] = char.get("aliases_json") or []
    relations = _load_relations(character_id, con)

    # _VA_WEAK_THRESHOLD: visual_anchor strings shorter than this are treated
    # as insufficient on their own — we'll pull supplemental snapshot data to
    # enrich the markdown so the LLM has something concrete to anchor on.
    _VA_WEAK_THRESHOLD = 80
    snapshot: Optional[dict] = _load_best_snapshot(character_id, con)
    supplemental = _load_supplemental_snapshots(character_id, con, limit=5)
    va_is_weak = not visual_anchor or len((visual_anchor or "").strip()) < _VA_WEAK_THRESHOLD

    # --- Appearance gate ---------------------------------------------------
    # Markdown is only useful to the LLM when at least one concrete visual
    # signal exists. Otherwise we'd ship plot-only data and the LLM would
    # hallucinate an appearance — exactly the bug we're fixing.
    has_va_appearance = bool(visual_anchor and _is_appearance_text(visual_anchor))
    has_snapshot_appearance = bool(
        snapshot and (
            (snapshot.get("physical_description") or "").strip()
            or (snapshot.get("outfit") or "").strip()
            or (snapshot.get("weapon") or "").strip()
        )
    )
    has_supplemental_appearance = any(
        (s.get("outfit") or "").strip() or (s.get("weapon") or "").strip()
        for s in supplemental
    )

    if not (has_va_appearance or has_snapshot_appearance or has_supplemental_appearance):
        logger.debug(
            "Skipping {} — no appearance signal (VA only contains plot/personality text)",
            character_id,
        )
        return None

    lines: list[str] = []

    # Helper: treat DB string "null" / "none" / empty as missing value.
    def _val(v) -> Optional[str]:
        s = (v or "").strip()
        return s if s and s.lower() not in ("null", "none", "n/a") else None

    # --- Header ---
    lines.append(f"# {char['name']}")
    lines.append("")

    # Gender — DB value first, then multi-signal Vietnamese inference.
    gender = _val(char.get("gender"))
    if gender is None:
        inferred = _infer_gender(
            name=char.get("name") or "",
            aliases=aliases,
            visual_anchor=_val(char.get("visual_anchor")),
            personality=_val(char.get("personality")),
            traits=traits,
            relations=relations,
            snapshot=snapshot,
            supplemental=supplemental,
        )
        if inferred:
            gender = inferred
            logger.debug(
                "Gender inferred | id={} name={} → {}",
                character_id, char.get("name"), gender,
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

    if aliases:
        lines.append(f"**Tên khác**: {', '.join(aliases)}")

    if traits:
        lines.append(f"**Tính cách**: {', '.join(traits)}")

    if personality:
        lines.append(f"**Tính cách chi tiết**: {personality}")

    if visual_anchor:
        lines.append(f"**Visual Anchor**: {visual_anchor}")

    # Supplemental snapshots — included whenever VA is weak, even if best
    # snapshot was rejected (event-only physical_description). Earlier code
    # gated this on `snapshot` truthiness which dropped useful outfit/weapon
    # rows for characters with sparse anchors.
    if va_is_weak and supplemental:
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
        # Fallback: build visual anchor from snapshot fields. `_load_best_snapshot`
        # has already null-ed `physical_description` if it was plot-text only,
        # so anything we read here is safe to surface as appearance.
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
            lines.append(
                f"**Visual Anchor** (từ snapshot ch.{snapshot['chapter_start']}): "
                + " | ".join(snap_parts)
            )
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

    # --- Relations (already loaded near the top for gender inference) ---
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
    result = image_prompt_client.generate_json(
        prompt=markdown_text,
        system=_build_profile_system(),
        temperature=0.2,
    )
    if isinstance(result, list) and result:
        return result[0]
    if isinstance(result, dict):
        return result
    raise ValueError(f"LLM returned unexpected type: {type(result)}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _derive_tags_from_context(char_name: str, context: str) -> dict[str, Any]:
    """Fallback path: infer profile from raw chapter excerpts when wiki has no
    structured visual data. Wrapped with the same retry budget as `_derive_tags`.
    """
    result = _context_infer_client.generate_json(
        prompt=(
            f"Character name: {char_name}\n\n"
            f"Story excerpts mentioning this character:\n\n{context}"
        ),
        system=_build_context_infer_system(),
        temperature=0.3,
    )
    if isinstance(result, list) and result:
        return result[0]
    if isinstance(result, dict):
        return result
    raise ValueError(f"Context-inference LLM returned unexpected type: {type(result)}")


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
            # Arc gave us only generic descriptors (e.g. "Nữ tử thi" — "female
            # corpse") that don't map to any wiki character. This is a normal
            # situation for episodes whose cast is dominated by anonymous
            # entities (ghosts, mobs). The downstream character extractor and
            # image phase both tolerate empty profile dirs, so we degrade
            # gracefully instead of blocking the pipeline.
            logger.warning(
                "No episode characters matched wiki_characters | names={} — "
                "skipping profile build (likely generic descriptors only).",
                character_names,
            )
            return []

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
                    raw = _derive_tags_from_context(char_name, context)
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

                _write_profile(json_path, char)
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

                _write_profile(json_path, char)
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
        # Build specific characters only.
        # Use _resolve_wiki_db so we hit the same DB path the bulk builder uses.
        db_path = _resolve_wiki_db(settings.db_path) or settings.db_path

        chars_dir = Path(settings.data_dir) / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)

        with _open_db(db_path) as con:
            for char_id in char_ids:
                # Resolve canonical folder via the same helper bulk builds use,
                # so CLI-built profiles land in the same directory and aren't
                # silently shadowed on the next bulk run.
                wiki_row = con.execute(
                    "SELECT name FROM wiki_characters WHERE character_id=?",
                    (char_id,),
                ).fetchone()
                char_name = wiki_row["name"] if wiki_row else char_id
                char_dir = _resolve_character_dir(chars_dir, char_id, char_name)
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

                    _write_profile(json_path, char)
                    print(f"[json] {json_path}")
                except Exception as exc:
                    print(f"[fail] {char_id} — {exc}")
    else:
        profiles = build_all_profiles(force=force)
        print(f"\nTotal profiles loaded: {len(profiles)}")
