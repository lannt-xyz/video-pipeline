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

_EXTRACTOR_SYSTEM = """You are a character appearance designer for AI image generation (Stable Diffusion / PonyXL Danbooru model).
Your job: read Vietnamese story context and produce a clean Danbooru tag list for each character.

OUTPUT FORMAT — description field MUST be comma-separated Danbooru tags ONLY. No sentences, no prose.

MANDATORY RULES:
1. Gender prefix comes FIRST — no exceptions:
     female → 1girl, solo
     male   → 1boy, solo
   This is CRITICAL. Read the story carefully. Do NOT assume gender from the name alone.
   If the story says a character is male, use 1boy. If female, use 1girl. Never swap.

2. Minimum 12 tags per character. Include these categories:
     - Hair: color + length + style  (e.g. short black hair, low bun)
     - Eyes: color + expression      (e.g. dark brown eyes, sharp eyes)
     - Face: notable features        (e.g. pale skin, wrinkled face, round face, strong jaw)
     - Outfit: clothes matching the story setting — see rule 4
     - Accessories: rings, weapons, talismans if relevant
     - Body type                     (e.g. slender, chubby build, athletic build)
     - Expression                    (e.g. serious expression, cheerful expression, looking at viewer)

3. WEIGHTS — use sparingly. Only add (tag:weight) for ONE OR TWO truly defining visual features.
   DO NOT add :1.2 or :1.3 to every tag. Over-weighting breaks PonyXL rendering.
   WRONG: (dark jacket:1.2), (dark pants:1.2), (talisman:1.1), (urban setting:1.3)
   RIGHT: dark jacket, dark pants, talisman in hand  — plain tags, no weights

4. Setting = modern urban ghost-hunter story (Mao Son Troc Quy Nhan).
   Default clothing by role (unless the story explicitly says otherwise):
     - Main characters (ghost hunters, students) → modern casual or tactical wear. NO hanfu.
     - Daoist masters, elders → daoist robes, hanfu is acceptable.
     - Ancient spirits/ghosts → traditional clothing acceptable.
   WRONG for a modern male ghost hunter: hanfu, ancient robes, scholar robe
   RIGHT for a modern male ghost hunter: jacket, dark pants, talisman in hand

5. Hair style — avoid feminizing male characters:
   FORBIDDEN on 1boy: side ponytail, twin tails, long flowing hair (unless story-explicit)
   Safe male styles: short hair, buzz cut, side part, low bun, tied back

6. Description must contain ONLY visual tags. Remove all abstract/personality words:
   FORBIDDEN: mysterious aura, mysterious, scholar, scholar's aura, calculating, weathered feeling, spiritual energy, symbolizing, exudes, ethereal, elegant aura, noble aura, cold aura, dangerous aura, manipulative, cunning aura
   REPLACE WITH visual equivalents: serious expression, sharp eyes, wrinkled face, cold expression, looking at viewer

7. Female character SAFETY rules — critical to prevent AI from generating monsters or males:
   - ALWAYS start female characters with: 1girl, solo
   - NEVER use tags that trigger male anatomy: masculine, muscular, broad shoulders, armor, horns, wings, claws
   - FORBIDDEN soft tags on 1girl: mysterious, scholar, ethereal, revealing, revealing outfit, bare shoulders
   - REQUIRED on every 1girl description: at least 2 of: fully clothed, high collar, long sleeves, traditional attire, modern casual wear, formal wear
   - If character wears traditional clothes: use "traditional chinese clothing, fully clothed, high collar"
   - If character wears modern clothes: use "blouse, dark skirt, modern casual wear" — no skin-exposure tags

EXAMPLES:
Modern male ghost hunter: "1boy, solo, short black hair, side part, dark brown eyes, sharp eyes, dark jacket, dark pants, talisman in hand, athletic build, serious expression, urban background"
Chubby comic-relief male: "1boy, solo, chubby build, round face, short black hair, messy hair, dark brown eyes, wide eyes, cheerful expression, blue casual jacket, dark pants, talisman in hand"
Daoist elder male: "1boy, solo, long white hair, low bun, white daoist robes, grey outer robe, thin beard, wrinkled face, wise gaze, wooden staff, prayer beads"
Modern female supporting: "1girl, solo, long black hair, straight hair, dark eyes, gentle expression, pale skin, slender, white blouse, dark skirt, casual modern wear, looking at viewer"

Return a JSON array:
[
  {
    "name": "string — full character name in Vietnamese",
    "alias": ["string — alternative names or nicknames"],
    "gender": "male or female",
    "description": "12+ comma-separated Danbooru tags — NO prose",
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

_DAOIST_ROLES = {"daoist", "daoist priest", "monk", "elder", "mao shan"}


def _sanitize_description(desc: str, gender: str) -> str:
    """Programmatically clean LLM-generated Danbooru tag string.

    Rules applied in order:
    1. Strip leading/trailing whitespace from each tag.
    2. Enforce gender prefix (1boy/1girl) as the very first tag.
    3. Remove tags in _PROSE_TAGS (abstract personality words).
    4. For male characters, replace forbidden hair tags with 'short hair'.
    5. Trim numeric weights: keep at most the 2 highest-weight tags;
       strip weight notation from the rest → plain tags.
    """
    tags = [t.strip() for t in desc.split(",") if t.strip()]

    # 1. Fix gender prefix — remove any stray 1girl/1boy/solo, re-prepend correct ones
    gender_prefix = "1boy" if gender == "male" else "1girl"
    tags = [t for t in tags if t not in ("1boy", "1girl", "solo")]
    tags = [gender_prefix, "solo"] + tags

    # 2. Remove prose tags (case-insensitive)
    lower_prose = {p.lower() for p in _PROSE_TAGS}
    tags = [t for t in tags if t.lower() not in lower_prose]

    # 3. For male characters, replace feminizing hairstyle tags
    if gender == "male":
        cleaned = []
        for t in tags:
            # Strip weight notation for comparison
            bare = re.sub(r"\((.+?):\d+\.\d+\)", r"\1", t).strip().lower()
            if bare in _MALE_FORBIDDEN_HAIR:
                logger.debug("Replaced forbidden male hair tag: {}", t)
                cleaned.append("short hair")
            else:
                cleaned.append(t)
        tags = cleaned

    # 4. For female characters, remove male physical attributes (beard etc.)
    if gender == "female":
        lower_forbidden = {p.lower() for p in _FEMALE_FORBIDDEN_PHYSICAL}
        before = len(tags)
        tags = [t for t in tags if t.lower() not in lower_forbidden]
        if len(tags) < before:
            logger.debug("Stripped {} male physical tags from female character", before - len(tags))

    # 4. Trim weights — keep only the 2 highest-weight (tag:N.N) items;
    #    strip weight notation from the rest.
    weighted = [(i, t) for i, t in enumerate(tags) if re.search(r":\d+\.\d+", t)]
    # Sort by weight value descending, keep top 2
    def _weight_val(item):
        m = re.search(r":(\d+\.\d+)", item[1])
        return float(m.group(1)) if m else 1.0
    weighted_sorted = sorted(weighted, key=_weight_val, reverse=True)
    keep_indices = {i for i, _ in weighted_sorted[:2]}
    result = []
    for i, t in enumerate(tags):
        if re.search(r":\d+\.\d+", t) and i not in keep_indices:
            # Strip weight from the tag but keep the tag itself
            plain = re.sub(r"\((.+?):\d+\.\d+\)", r"\1", t).strip()
            result.append(plain)
        else:
            result.append(t)

    # 5. Deduplicate preserving order
    seen: set = set()
    deduped = []
    for t in result:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    cleaned_desc = ", ".join(deduped)
    if cleaned_desc != desc:
        logger.debug("Description sanitized | before={!r:.80} after={!r:.80}", desc, cleaned_desc)
    return cleaned_desc


def _default_character(name: str) -> Character:
    """Fallback character with a generic male xianxia description."""
    return Character(
        name=name,
        gender="male",
        description=(
            f"{name}, 1boy, male, young xianxia cultivator, handsome face, "
            "sharp eyes, traditional chinese robes, long dark hair, "
            "full body portrait"
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
