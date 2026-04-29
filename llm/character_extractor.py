import json
import re
import unicodedata
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import ollama_client
from models.schemas import Character
from pipeline.state import StateDB

_EXTRACTOR_SYSTEM = """You are a character appearance designer for a photorealistic AI image model (Flux Dev).
Your job: read Vietnamese story context and produce a clean visual description for each character.

OUTPUT FORMAT — `description` is a comma-separated list of natural-English visual phrases.
NOT Danbooru tags. NO underscores in phrases. NO weights like (tag:1.2). NO quality words
(masterpiece, best quality, highly detailed, score_9, etc.). NO style words (anime, manga,
cartoon, illustration). NO count tokens (1boy, 1girl, solo, 2boys). The image model gets
those from the workflow — your job is purely visual identity.

MANDATORY content (10–14 phrases, in this order):
  1. Age + body archetype  — e.g. "young man in his twenties", "elderly woman in her sixties", "teenage girl"
  2. Build                 — e.g. "lean athletic build", "stocky build", "slender frame"
  3. Hair                  — color + length + style — e.g. "short black hair, side parted"
  4. Eyes                  — color + shape — e.g. "dark brown eyes, sharp narrow eyes"
  5. Face                  — distinctive features — e.g. "high cheekbones, light scar across left brow"
  6. Skin                  — tone — e.g. "fair skin", "tanned weathered skin"
  7. Outfit (top)          — concrete garment names, see rule 3
  8. Outfit (bottom)       — pants/skirt/robe details
  9. Accessories           — only if visually distinctive — e.g. "leather waist pouch, small jade pendant"
 10. Default expression    — e.g. "serious focused expression", "calm watchful expression"

RULE 1 — DO NOT use Danbooru shorthand. Write phrases the way a costume designer would.
  WRONG:  "long_black_hair, blue_sclera, casual_outfit, tactical_wear, monk_outfit"
  RIGHT:  "long black hair tied in a topknot, dark indigo eyes, plain dark grey changshan robe with wide sleeves"

RULE 2 — NO supernatural eye colors unless the character is explicitly a ghost or spirit.
  Living humans get: dark brown, black, hazel, grey. Never blue/red/glowing.

RULE 3 — Setting is modern urban ghost-hunter (Mao Son Troc Quy Nhan).
  Default outfits by role:
    - Living protagonists (ghost hunters, students):
        modern street clothes — "dark hooded jacket, slim cargo pants, leather boots".
        NO hanfu, NO armor, NO tactical military gear.
    - Daoist masters / elders:
        "Mao Shan taoist robe in deep indigo with bagua trim, wide cloth sash, cloth shoes".
        Buddhist monk robes are WRONG — characters are Taoist.
    - Ancient ghosts / spirits in flashbacks:
        period clothing — "faded burial robe in dull red lacquer, weathered hemp inner garment".

RULE 4 — Avoid feminising male characters.
  Forbidden on male: "long flowing hair", "twin tails", "side ponytail".
  Default male hair: "short black hair", "topknot", "tied back at nape".

RULE 5 — Female character safety.
  Forbidden on female: bare skin/cleavage/lingerie phrases, masculine build words
  ("muscular", "broad shoulders"), facial-hair phrases ("beard", "stubble").
  Required: every female description must explicitly state covering — e.g.
  "fully clothed in a long-sleeved blouse" or "long-sleeved hanfu with high collar".

RULE 6 — Strip abstract personality / aura words.
  FORBIDDEN: mysterious, ethereal, scholarly aura, spiritual energy, calculating,
  weathered feeling, exudes, symbolises, otherworldly, elegant aura, dangerous aura.
  REPLACE WITH visual cues: "serious expression", "calm watchful eyes", "deep frown lines".

EXAMPLES (use as style reference, do not copy verbatim):
  Male ghost hunter (modern):
    "young man in his twenties, lean athletic build, short black hair side-parted,
     dark brown eyes with sharp gaze, high cheekbones, fair skin, dark hooded jacket
     over a charcoal t-shirt, slim black cargo pants, scuffed leather boots,
     leather wrist strap holding a small talisman, focused determined expression"

  Comic-relief male:
    "young man in his twenties, stocky chubby build, round friendly face,
     messy short black hair, dark brown round eyes, light olive skin,
     bright blue casual hooded jacket, dark grey jeans, scuffed sneakers,
     cloth shoulder bag, easy cheerful smile"

  Mao Shan elder:
    "elderly man in his sixties, lean wiry build, long grey hair tied in a topknot,
     thin grey beard, weathered tanned skin, deep frown lines, sharp piercing dark eyes,
     dark indigo Mao Shan taoist robe with bagua trim, wide white cloth sash,
     black cloth shoes, wooden prayer beads on right wrist, calm authoritative expression"

  Modern female student:
    "young woman in her early twenties, slender frame, long straight black hair,
     dark almond-shaped eyes, fair skin, soft round face, fully clothed in a
     long-sleeved cream blouse buttoned to the throat, knee-length dark skirt,
     dark stockings, plain ballet flats, gentle attentive expression"

Return a JSON array:
[
  {
    "name": "string — full character name in Vietnamese",
    "alias": ["string — alternative names or nicknames"],
    "gender": "male or female",
    "description": "10–14 comma-separated natural-English phrases — NO Danbooru tags, NO weights, NO style/quality words",
    "relationships": {"other_character_name": "relationship description"}
  }
]"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _extract_raw(prompt: str) -> List[dict]:
    result = ollama_client.generate_json(
        prompt=prompt, system=_EXTRACTOR_SYSTEM, temperature=0.2
    )
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        # LLM returned a single character object directly
        if "name" in result:
            return [result]
        # LLM wrapped in a key
        for key in ("characters", "data", "result"):
            if key in result and isinstance(result[key], list):
                return result[key]
    return []


def extract_all_characters(force: bool = False) -> List[Character]:
    """Collect all character names from arc summaries, then use LLM to build
    appearance descriptions for any character not yet saved as a JSON file.
    Pass force=True to re-extract characters that already have JSON files."""
    summaries_dir = Path(settings.data_dir) / "summaries"
    arc_files = sorted(summaries_dir.glob("*-arc.json"))

    if not arc_files:
        logger.warning("No arc summary files found for character extraction")
        return []

    chars_dir = Path(settings.data_dir) / "characters"
    chars_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: collect unique names + full arc data for richer prompts
    all_names: list[str] = []
    all_arc_data: list[dict] = []
    for arc_file in arc_files[:10]:
        data = json.loads(arc_file.read_text(encoding="utf-8"))
        for name in data.get("characters_in_episode", []):
            if name not in all_names:
                all_names.append(name)
        all_arc_data.append(data)

    if not all_names:
        logger.warning("No character names found in arc summaries")
        return []

    logger.info("Found {} unique characters across arc summaries", len(all_names))

    # Step 2: skip characters that already have a JSON file (unless force)
    if force:
        target_names = all_names
    else:
        target_names = [n for n in all_names if not (chars_dir / _slugify(n) / "profile.json").exists()]
    if not target_names:
        logger.info("All characters already have JSON files — skipping extraction")
        return load_all_characters()

    # Build combined arc context for the prompt
    combined_summary = "\n\n---\n\n".join(
        d.get("arc_summary", "") for d in all_arc_data
    )
    all_key_events = [
        event
        for d in all_arc_data
        for event in d.get("key_events", [])
    ]

    # Load a few raw chapters for extra appearance hints (first 3 chapters only)
    db = StateDB()
    raw_chapters = ""
    for ch_num in range(1, 4):
        content = db.get_chapter_content(ch_num)
        if content:
            raw_chapters += f"\n\n=== Chương {ch_num} (trích) ===\n{content[:2000]}"

    saved: list[Character] = []
    for name in target_names:
        logger.info("Asking LLM for description of '{}'", name)

        # Filter key events that mention this character
        relevant_events = [e for e in all_key_events if name.split()[0] in e or name.split()[-1] in e]
        events_text = "\n".join(f"- {e}" for e in relevant_events[:5]) if relevant_events else "(no specific events found)"

        prompt = (
            f"Character to describe: {name}\n\n"
            f"Story arc summary:\n{combined_summary[:2000]}\n\n"
            f"Key events involving {name}:\n{events_text}\n\n"
            f"Raw chapter excerpts (for appearance clues):{raw_chapters}\n\n"
            f"Task: Generate a rich Danbooru tag description for {name}. "
            f"Minimum 12 tags. Infer appearance from their role if not explicitly stated."
        )
        raw_list = _extract_raw(prompt)
        if raw_list:
            try:
                char = Character(**raw_list[0])
            except Exception:
                char = _default_character(name)
        else:
            logger.warning("LLM did not describe '{}' — using default", name)
            char = _default_character(name)

        # Ensure the name matches exactly (LLM may romanize)
        char.name = name

        # Apply hard-coded gender overrides before any validation.
        if name in _GENDER_OVERRIDES:
            overridden = _GENDER_OVERRIDES[name]
            if char.gender != overridden:
                logger.info(
                    "Gender override applied | name={} llm={} → correct={}",
                    name, char.gender, overridden,
                )
                char.gender = overridden

        # Skip characters with ambiguous gender — not enough story context yet.
        # They will be picked up in a later episode when more context is available.
        if char.gender not in ("male", "female"):
            logger.warning(
                "Skipping '{}' — gender='{}' (insufficient story context)",
                name, char.gender,
            )
            continue

        # Post-process LLM output: enforce gender prefix, remove prose,
        # fix forbidden male hairstyles, trim excess weights.
        char.description = _sanitize_description(char.description, char.gender)

        char_path = chars_dir / _slugify(name) / "profile.json"
        char_path.parent.mkdir(parents=True, exist_ok=True)
        char_path.write_text(char.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved character | name={} gender={}", char.name, char.gender)
        saved.append(char)

    return load_all_characters()


# ---------- post-processing LLM output ----------

_PROSE_TAGS = {
    "mysterious aura", "calculating", "scholar's aura", "ethereal",
    "spiritual energy", "exudes", "symbolizing", "weathered feeling",
    "mysterious", "mythical", "enchanting", "magical presence",
    "otherworldly", "ancient wisdom", "scholar",
}

_MALE_FORBIDDEN_HAIR = {"side ponytail", "twin tails", "twintails", "pigtails"}

# Physical attributes that must not appear on female characters
_FEMALE_FORBIDDEN_PHYSICAL = {
    "thin beard", "beard", "stubble", "goatee", "mustache", "moustache",
    "facial hair", "sideburns", "short beard", "long beard", "thick beard",
}

# Booru/PonyXL-era tokens that the extractor was previously asked to emit.
# Removed because Flux Dev (T5 text encoder) prefers natural English and these
# tokens dilute attention or trigger the wrong style.
_BOORU_NOISE_TOKENS = {
    "1boy", "1girl", "2boys", "2girls", "solo",
    "score_9", "score_8", "score_7", "score_6", "score_5", "score_4",
    "score_9_up", "score_8_up", "score_7_up", "score_6_up",
    "masterpiece", "best quality", "highly detailed", "high quality",
    "ultra detailed", "absurdres", "8k",
}

# Style words that pull the photoreal Flux model toward illustration.
_STYLE_LEAK_TOKENS = {
    "anime", "anime style", "manga", "manga style", "cartoon",
    "illustration", "digital art", "concept art", "cel shading",
    "toon shading", "3d render", "cgi", "painting", "line art",
}

_DAOIST_ROLES = {"daoist", "daoist priest", "monk", "elder", "mao shan"}


def _sanitize_description(desc: str, gender: str) -> str:
    """Clean LLM-generated character description for Flux photoreal generation.

    The extractor system prompt asks for natural-English visual phrases (not
    Danbooru tags), but the LLM occasionally regresses to old habits. This
    pass enforces format hygiene:

    1. Strip leading/trailing whitespace from each phrase.
    2. Drop Danbooru count tokens (1boy/1girl/solo/2boys/...) — gender is
       conveyed by the prose itself ("young man", "elderly woman").
    3. Drop quality/score tokens (masterpiece, score_9, ...).
    4. Drop style-leak tokens (anime, illustration, ...).
    5. Drop abstract personality / aura phrases.
    6. Replace forbidden male hairstyles with "short black hair".
    7. Drop facial-hair phrases on female characters.
    8. Strip explicit weight notation `(phrase:1.2)` → `phrase`.
    9. Deduplicate while preserving order.
    """
    raw_phrases = [t.strip() for t in desc.split(",") if t.strip()]

    drop_set = (
        {p.lower() for p in _PROSE_TAGS}
        | _BOORU_NOISE_TOKENS
        | _STYLE_LEAK_TOKENS
    )
    if gender == "female":
        drop_set |= {p.lower() for p in _FEMALE_FORBIDDEN_PHYSICAL}

    cleaned: list[str] = []
    for phrase in raw_phrases:
        # Strip explicit weights — Flux ignores parenthesised weights and they
        # add noise to the prompt budget.
        bare = re.sub(r"\((.+?):\d+(?:\.\d+)?\)", r"\1", phrase).strip()
        bare_lower = bare.lower()

        if bare_lower in drop_set:
            continue
        # Substring match for style words inside longer phrases (e.g.
        # "soft anime-style face").
        if any(token in bare_lower for token in _STYLE_LEAK_TOKENS):
            continue
        # Replace forbidden male hairstyles wholesale.
        if gender == "male" and bare_lower in _MALE_FORBIDDEN_HAIR:
            cleaned.append("short black hair")
            continue
        cleaned.append(bare)

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for phrase in cleaned:
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(phrase)

    cleaned_desc = ", ".join(deduped)
    if cleaned_desc != desc:
        logger.debug(
            "Description sanitized | before={!r:.80} after={!r:.80}",
            desc, cleaned_desc,
        )
    return cleaned_desc


def _default_character(name: str) -> Character:
    """Fallback used when the LLM fails to describe a character.

    Uses the same natural-English format as the extractor system prompt so
    downstream consumers (anchor gen, Flux scene gen) see a consistent style.
    """
    return Character(
        name=name,
        gender="male",
        description=(
            "young man in his twenties, lean athletic build, "
            "short black hair side-parted, dark brown eyes with sharp gaze, "
            "high cheekbones, fair skin, dark hooded jacket, "
            "slim black cargo pants, leather boots, focused expression"
        ),
    )


def load_character(name: str) -> Character:
    char_path = (
        Path(settings.data_dir) / "characters" / _slugify(name) / "profile.json"
    )
    if not char_path.exists():
        raise FileNotFoundError(f"Character not found: {name}")
    return Character(**json.loads(char_path.read_text(encoding="utf-8")))


def _get_active_character_ids() -> "set[str] | None":
    """Return set of active character_ids (is_delete=0) from wiki DB.

    Returns None when the DB is unavailable or the column does not exist,
    which signals the caller to skip filtering (backward-compatible).
    Avoids circular imports by not importing from profile_builder.
    """
    import sqlite3 as _sqlite3

    db_path = Path(settings.db_path)
    if not db_path.exists():
        return None
    try:
        con = _sqlite3.connect(str(db_path))
        con.row_factory = _sqlite3.Row
        try:
            table_exists = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='wiki_characters'"
            ).fetchone()
            if not table_exists:
                return None
            col_names = {
                row["name"]
                for row in con.execute("PRAGMA table_info(wiki_characters)").fetchall()
            }
            if "is_delete" not in col_names:
                return None
            rows = con.execute(
                "SELECT character_id FROM wiki_characters WHERE is_delete = 0"
            ).fetchall()
            return {row["character_id"] for row in rows}
        finally:
            con.close()
    except Exception as exc:
        logger.debug("Could not resolve active character IDs from DB | error={}", exc)
        return None


def load_all_characters() -> List[Character]:
    chars_dir = Path(settings.data_dir) / "characters"
    if not chars_dir.exists():
        return []

    active_ids = _get_active_character_ids()

    characters = []
    for f in chars_dir.glob("*/profile.json"):
        char_id = f.parent.name
        if active_ids is not None and char_id not in active_ids:
            logger.debug("Skipping deleted character | id={}", char_id)
            continue
        try:
            characters.append(
                Character(**json.loads(f.read_text(encoding="utf-8")))
            )
        except Exception as exc:
            logger.warning("Failed to load character from {} | error={}", f, exc)
    return characters


_VN_SPECIAL = str.maketrans({"đ": "d", "Đ": "D"})


# Hard-coded gender overrides for characters the LLM frequently misidentifies.
# Key = exact Vietnamese name as it appears in arc summaries.
# Add entries here when you discover a new misidentification.
_GENDER_OVERRIDES: dict[str, str] = {
    "Chu Tĩnh Như": "female",
    "Liêu Thanh Thanh": "female",
    "Xảo Vân": "female",
    "Tiểu Mã": "male",
    "Mã Minh Lượng": "male",
    "Lý Đa": "male",
}


def _slugify(name: str) -> str:
    # Handle đ/Đ (does not decompose via NFD), then strip remaining diacritics
    name = name.translate(_VN_SPECIAL)
    normalized = unicodedata.normalize("NFD", name.lower())
    ascii_name = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return ascii_name.replace(" ", "_").replace("-", "_")
