import argparse
import hashlib
import sqlite3
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import settings
from pipeline.state import StateDB
from pipeline.validator import ValidationError, validator
from pipeline.vram_manager import VRAMConsumer, vram_manager

if TYPE_CHECKING:
    from image_gen.comfyui_client import ComfyUIClient
    from llm.gatekeeper import ReviewResult
    from models.schemas import EpisodeScript

PHASES = ["llm", "script_review", "images", "audio", "video", "validate"]


def setup_logging(episode_num: int = None) -> None:
    Path(settings.logs_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        f"{settings.logs_dir}/pipeline.log",
        rotation="100MB",
        level="DEBUG",
        encoding="utf-8",
    )
    if episode_num is not None:
        logger.add(
            f"{settings.logs_dir}/episode-{episode_num:03d}.log",
            level="DEBUG",
            encoding="utf-8",
        )


def _episode_chapter_range(episode_num: int) -> tuple[int, int]:
    start = (episode_num - 1) * settings.chapters_per_episode + 1
    end = min(episode_num * settings.chapters_per_episode, settings.total_chapters)
    return start, end


def run_llm(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.summarizer import summarize_episode
    from llm.scriptwriter import write_episode_script

    chapter_start, chapter_end = _episode_chapter_range(episode_num)

    if dry_run:
        logger.info("[dry-run] Skipping LLM | episode={}", episode_num)
        db.set_episode_status(episode_num, "SCRIPTED")
        return

    vram_manager.acquire(VRAMConsumer.OLLAMA)
    vram_manager.health_check_ollama()

    db.record_phase_start(episode_num, "llm")

    logger.info("Summarizing | episode={}", episode_num)
    summarize_episode(episode_num, chapter_start, chapter_end)
    db.set_episode_status(episode_num, "SUMMARIZED")

    # Distill arc key_events → English visual tags for thumbnail prompt.
    # Done here while summary model is warm; cached to disk for the images phase.
    from llm.summarizer import distill_thumbnail_tags
    distill_thumbnail_tags(episode_num)

    # Build profiles only for characters that appear in this episode (idempotent).
    from llm.profile_builder import build_profiles_for_episode
    from llm.summarizer import load_arc_overview
    arc = load_arc_overview(episode_num)
    if arc.characters_in_episode:
        logger.info(
            "Building character profiles for episode | episode={} chars={}",
            episode_num, arc.characters_in_episode,
        )
        build_profiles_for_episode(arc.characters_in_episode)
    else:
        logger.warning(
            "No characters detected in arc; skipping profile build | episode={}",
            episode_num,
        )

    # Phase 2: viral-moment extraction runs BEFORE the model switch — it shares
    # the summary LLM context to avoid an extra VRAM acquire. Behind feature flag.
    viral_moments = None
    if settings.retention.use_constraint_system:
        from llm.hook_extractor import extract_viral_moments
        logger.info("Extracting viral moments | episode={}", episode_num)
        moments = extract_viral_moments(episode_num)
        if moments:
            viral_moments = moments
        else:
            logger.warning(
                "Viral moment extraction returned empty; falling back to legacy hook | episode={}",
                episode_num,
            )

    # Evict summary model before loading script model to avoid VRAM pressure
    # when the two models are different (e.g. Gemma 4 → Mistral Small 3).
    if settings.effective_summary_model != settings.effective_script_model:
        logger.info(
            "Switching LLM model | unloading={} next={}",
            settings.effective_summary_model, settings.effective_script_model,
        )
        vram_manager.unload_model(settings.effective_summary_model)

    logger.info("Writing script | episode={}", episode_num)
    write_episode_script(episode_num, viral_moments=viral_moments)

    db.record_phase_done(episode_num, "llm")
    db.set_episode_status(episode_num, "SCRIPTED")


# Tags that describe outfit / accessories / background — excluded from identity (DNA) prompt
_OUTFIT_KEYWORDS = frozenset([
    "jacket", "pants", "robe", "robes", "dress", "blouse", "skirt", "shirt",
    "coat", "hanfu", "clothing", "wear", "outfit", "attire", "suit", "uniform",
    "apron", "talisman", "staff", "sword", "weapon", "beads", "in hand",
    "incense", "background", "setting", "lighting", "indoor", "outdoor",
    "standing", "sitting", "holding", "looking",
    # Count/group Danbooru tags — managed per-scene, not per-character
    "solo", "1boy", "1girl", "2boys", "2girls",
    # Profession/role tags — semantic, not physical identity
    "craftsman", "carpenter", "daoist master", "daoist priest",
    "ghost hunter", "cultivator", "monk", "elder",
    # Personality/trait tags — not visual
    "determined",
    # Framing/composition — scene-dependent
    "full body", "upper body",
])


def _find_character(llm_name: str, characters_map: dict):
    """Alias-aware lookup with substring containment fallback.

    Level 1 — exact canonical name match.
    Level 2 — exact alias match (e.g. "tôn tử" → Diệp Binh).
    Level 3 — canonical name is contained in llm_name, handles cases where
               LLM prepends a role/relation prefix (e.g. "Vợ Diệp Binh" → "Diệp Binh").
               Safe for Vietnamese: deterministic, no score threshold needed.
    """
    # Level 1+2: exact match on name or any alias
    for char_obj in characters_map.values():
        if llm_name == char_obj.name or llm_name in char_obj.alias:
            return char_obj
    # Level 3: canonical name is a substring of the LLM-provided name
    for char_obj in characters_map.values():
        if char_obj.name in llm_name:
            logger.debug(
                "Substring character match | llm_name={!r} → canonical={!r}",
                llm_name, char_obj.name,
            )
            return char_obj
    return None


def _extract_dna_tags(description: str) -> str:
    """Keep only identity tags (gender/hair/eyes/face/body) — drop outfit/accessory/background tags."""
    tags = [t.strip() for t in description.split(",") if t.strip()]
    dna = [t for t in tags if not any(kw in t.lower() for kw in _OUTFIT_KEYWORDS)]
    return ", ".join(dna)


_CLOTHING_KEYWORDS = frozenset([
    "jacket", "pants", "robe", "robes", "dress", "blouse", "skirt", "shirt",
    "coat", "hanfu", "clothing", "wear", "outfit", "attire", "suit", "uniform",
    "apron",
])


# Tags from PonyXL/Danbooru-style character descriptions that Flux either
# doesn't understand or that conflict with scene_prompt's own framing/composition
# directives. Stripped before injecting character appearance into Flux prompts.
_FLUX_UNFRIENDLY_DESC_TAGS = frozenset([
    # Danbooru count tags — already covered by gender_tags
    "1boy", "1girl", "2boys", "2girls", "solo",
    # PonyXL quality scoring tokens
    "score_9", "score_8", "score_7", "score_6", "score_5", "score_4",
    "score_9_up", "score_8_up", "score_7_up", "score_6_up",
    "masterpiece", "best quality", "highly detailed", "high quality",
    # Scene-dependent — must come from scene_prompt, not character description
    "looking at viewer", "looking away", "looking down", "looking up",
    "urban background", "indoor background", "outdoor background",
    "white background", "simple background", "plain background",
    "standing", "sitting", "kneeling",
    # Pony safety tags — workflow CLIP suffix already handles these
    "sfw", "fully clothed", "high collar", "long sleeves",
])


def _build_character_appearance_tags(char_anchor_pairs: list) -> str:
    """Build inline character appearance description for the Flux prompt.

    Flux uses T5-XXL which understands natural-language sentences much better
    than danbooru-style comma-soup tag lists. We emit a short prose sentence
    per character (max 2) listing identity tags (hair / eyes / face / body) and
    clothing tags pulled from `Character.description`, dropping Pony/Danbooru
    noise (score_X, 1boy, solo, looking at viewer, ...) and scene-dependent
    tags (expression, background).

    Important: we do NOT use A1111-style weight syntax like `(...:1.05)` —
    Flux/T5 ignores those and the literal parentheses become noise tokens.
    Identity strength comes from being placed early in the prompt instead.

    Returns "" when there are no usable tags so callers can skip the section.
    """
    if not char_anchor_pairs:
        return ""

    sentences: list[str] = []

    for char_obj, _ in char_anchor_pairs[:2]:
        if not char_obj or not char_obj.description:
            continue

        identity_tags: list[str] = []
        clothing_tags: list[str] = []
        seen_local: set[str] = set()

        for raw in char_obj.description.split(","):
            tag = raw.strip()
            if not tag:
                continue
            tag_lower = tag.lower()
            if tag_lower in _FLUX_UNFRIENDLY_DESC_TAGS:
                continue
            if tag_lower.endswith(" expression"):
                continue
            if tag_lower in seen_local:
                continue
            seen_local.add(tag_lower)

            if any(kw in tag_lower for kw in _CLOTHING_KEYWORDS):
                clothing_tags.append(tag)
            else:
                identity_tags.append(tag)

        if not identity_tags and not clothing_tags:
            continue

        # Cap each list to keep overall prompt length sane.
        identity_tags = identity_tags[:8]
        clothing_tags = clothing_tags[:6]

        gender_word = "woman" if (char_obj.gender or "").lower() == "female" else "man"
        name = (char_obj.name or "").strip()

        parts: list[str] = []
        subject = f"{name}, an East Asian {gender_word}" if name else f"an East Asian {gender_word}"
        if identity_tags:
            parts.append(f"{subject} with {', '.join(identity_tags)}")
        else:
            parts.append(subject)
        if clothing_tags:
            parts.append(f"wearing {', '.join(clothing_tags)}")

        sentences.append(", ".join(parts))

    if not sentences:
        return ""

    if len(sentences) == 1:
        return f"The shot features {sentences[0]}."
    return f"The shot features {sentences[0]}; alongside {sentences[1]}."


_SCENE_DETAIL_BOOST_TAGS = (
    "cinematic volumetric lighting",
    "eerie unsettling atmosphere",
    "ominous shadows creeping",
    "dark foreboding tone",
)

# Tags that belong in the workflow CLIP suffix, not in SCENE_PROMPT.
# Strip them from LLM output to avoid wasting tag positions.
_METADATA_STRIP_TAGS = frozenset([
    "sfw", "fully clothed", "high collar", "long sleeves",
    "anime style", "no text", "no watermarks",
    "masterpiece", "best quality", "highly detailed",
    *_SCENE_DETAIL_BOOST_TAGS,
])


def _strip_metadata_tags(prompt_text: str) -> str:
    """Remove metadata/style tags that now live in the workflow CLIP suffix."""
    tags = [t.strip() for t in prompt_text.split(",") if t.strip()]
    cleaned = [t for t in tags if t.lower() not in _METADATA_STRIP_TAGS]
    return ", ".join(cleaned)

# Framing tags injected into scene-only (no-character) prompts to prevent
# SDXL defaulting to portrait/close-up framing on a 9:16 canvas.
_SCENE_FRAMING_TAGS = "wide establishing shot, full scene view, environment focus, no characters in foreground"

# Negative tags that block portrait/close-up framing in scene-only shots.
_SCENE_ANTI_PORTRAIT_NEG = (
    ", (portrait:1.6), (close-up:1.6), (face focus:1.5), (headshot:1.5), "
    "(bust shot:1.4), (upper body focus:1.4), (extreme close up:1.6), "
    "(face close up:1.6), (zoomed in face:1.5), talking head"
)

_HOLDING_CONTEXT_KEYWORDS = frozenset([
    "holding", "in hand", "wielding", "brandishing", "weapon", "artifact",
    "relic", "sword", "blade", "staff", "talisman", "seal",
    "cầm", "nắm", "giơ", "vung", "pháp khí", "kiếm", "đao", "trượng", "phù", "ấn",
])

# Vietnamese/English keyword mapping to concise English visual tags for ComfyUI.
_ARTIFACT_TAG_HINTS = [
    (("kiếm", "sword"), ("ornate daoist sword", "engraved metal blade")),
    (("đao", "blade"), ("ritual curved blade",)),
    (("trượng", "staff"), ("carved taoist staff", "ritual staff in hand")),
    (("phù", "talisman", "bùa"), ("yellow talisman strips", "paper talisman in hand")),
    (("ấn", "seal"), ("daoist seal in hand", "engraved ritual seal")),
    (("chuông", "bell"), ("small ritual hand bell",)),
    (("hồ lô", "gourd"), ("daoist gourd flask",)),
    (("gương", "mirror", "bagua"), ("bagua mirror",)),
    (("la bàn", "la ban", "compass"), ("feng shui compass",)),
]

_ARTIFACT_MATERIAL_HINTS = [
    (("đồng", "bronze"), "aged bronze texture"),
    (("ngọc", "jade"), "jade inlays"),
    (("gỗ", "wood"), "weathered wood grain"),
    (("sắt", "iron", "steel"), "dark forged metal finish"),
]


def _enhance_scene_detail_tags(prompt_text: str) -> str:
    """Insert small, stable detail tags BEFORE style suffix for better CLIP attention."""
    base = prompt_text.strip()
    lower = base.lower()
    extra = [t for t in _SCENE_DETAIL_BOOST_TAGS if t not in lower]
    if not extra:
        return base
    tags = [t.strip() for t in base.split(",") if t.strip()]
    # Find style suffix boundary — boost tags must go BEFORE it.
    style_idx = len(tags)
    for idx, t in enumerate(tags):
        if "anime style" in t.lower():
            style_idx = idx
            break
    for i, etag in enumerate(extra):
        tags.insert(style_idx + i, etag)
    return ", ".join(tags)


def _compact_prompt_tags(tag_text: str, max_tags: int = 12) -> str:
    """Compact comma-separated tags to avoid prompt dilution.

    Keeps order, removes duplicates, and clips length to max_tags.
    """
    raw_tags = [t.strip() for t in tag_text.split(",") if t.strip()]
    compacted: list[str] = []
    seen: set[str] = set()

    for tag in raw_tags:
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        compacted.append(tag)
        if len(compacted) >= max_tags:
            break

    return ", ".join(compacted)


def _prompt_mentions_holding_context(prompt_text: str) -> bool:
    """Return True when prompt implies a character is holding/using a weapon/artifact."""
    lower = prompt_text.lower()
    return any(k in lower for k in _HOLDING_CONTEXT_KEYWORDS)


def _artifact_tags_from_text(text: str) -> list[str]:
    """Extract concise English visual tags from artifact text fields."""
    lower = text.lower()
    tags: list[str] = []

    for keywords, hint_tags in _ARTIFACT_TAG_HINTS:
        if any(k in lower for k in keywords):
            tags.extend(hint_tags)

    for keywords, material_tag in _ARTIFACT_MATERIAL_HINTS:
        if any(k in lower for k in keywords):
            tags.append(material_tag)

    if ("phát sáng" in lower) or ("hào quang" in lower) or ("glow" in lower):
        tags.append("glowing runic aura")
    if ("khắc" in lower) or ("rune" in lower) or ("chạm" in lower):
        tags.append("intricate rune engravings")

    # Dedupe while preserving order.
    unique: list[str] = []
    seen = set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _resolve_wiki_db_for_artifacts() -> Path | None:
    """Resolve DB path that contains wiki artifact tables."""
    candidates = [
        Path(settings.db_path),
        Path("data") / f"{settings.story_slug}.db",
    ]

    for db_path in candidates:
        if not db_path.exists():
            continue
        try:
            con = sqlite3.connect(str(db_path))
            tables = {
                row[0] for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            con.close()
        except Exception:
            continue

        required = {"wiki_characters", "wiki_artifacts", "wiki_artifact_snapshots"}
        if required.issubset(tables):
            return db_path

    return None


def _load_character_artifact_hints() -> dict[str, list[str]]:
    """Load per-character artifact visual hints from wiki DB.

    Returns mapping: character_name_lower -> list of English visual tags.
    """
    db_path = _resolve_wiki_db_for_artifacts()
    if db_path is None:
        return {}

    hints: dict[str, list[str]] = {}
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row

        char_cols = {
            r[1] for r in con.execute("PRAGMA table_info(wiki_characters)").fetchall()
        }
        char_filter = " WHERE c.is_deleted = 0" if "is_deleted" in char_cols else ""

        rows = con.execute(
            "SELECT c.name AS char_name, a.name AS artifact_name, "
            "COALESCE(a.material, '') AS material, COALESCE(a.visual_anchor, '') AS visual_anchor, "
            "COALESCE(s.normal_state, '') AS normal_state, COALESCE(s.active_state, '') AS active_state, "
            "COALESCE(s.is_key_event, 0) AS is_key_event, COALESCE(s.chapter_start, 0) AS chapter_start "
            "FROM wiki_characters c "
            "JOIN wiki_artifact_snapshots s ON s.owner_id = c.character_id "
            "JOIN wiki_artifacts a ON a.artifact_id = s.artifact_id"
            f"{char_filter} "
            "ORDER BY c.name, is_key_event DESC, chapter_start ASC"
        ).fetchall()
        con.close()
    except Exception as exc:
        logger.debug("Failed to load artifact hints from DB | err={}", exc)
        return {}

    seen_artifact_per_char: dict[str, set[str]] = {}

    for row in rows:
        char_name = str(row["char_name"]).strip()
        if not char_name:
            continue
        key = char_name.lower()

        artifact_name = str(row["artifact_name"] or "").strip().lower()
        seen_for_char = seen_artifact_per_char.setdefault(key, set())
        if artifact_name and artifact_name in seen_for_char:
            continue

        # Keep top 2 artifact entries per character to avoid overloading prompts.
        if len(seen_for_char) >= 2:
            continue

        raw = " | ".join([
            str(row["artifact_name"] or ""),
            str(row["material"] or ""),
            str(row["visual_anchor"] or ""),
            str(row["normal_state"] or ""),
            str(row["active_state"] or ""),
        ])
        tags = _artifact_tags_from_text(raw)
        if not tags:
            # Keep a safe fallback so "artifact in hand" is still explicit.
            tags = ["ritual artifact in hand"]

        existing = hints.setdefault(key, [])
        for t in tags:
            if t not in existing:
                existing.append(t)

        if artifact_name:
            seen_for_char.add(artifact_name)

    return hints


def _build_artifact_prompt_tags(
    prompt_text: str,
    char_anchor_pairs: list,
    artifact_hints_by_name: dict[str, list[str]] | None,
) -> str:
    """Build artifact detail tags for shot prompts when holding context is present."""
    if not artifact_hints_by_name or not _prompt_mentions_holding_context(prompt_text):
        return ""

    merged: list[str] = []
    for char_obj, _ in char_anchor_pairs[:2]:
        if not char_obj:
            continue
        for t in artifact_hints_by_name.get(char_obj.name.lower(), []):
            if t not in merged:
                merged.append(t)

    # Limit to avoid over-constraining composition.
    return ", ".join(merged[:6])


def _extract_clothing_tags(description: str) -> str:
    """Extract only clothing/outfit tags from a character description.

    Returns a weighted tag string like "(dark jacket, dark pants:1.2)" for
    injection into the positive prompt to lock character outfit.
    If no clothing tags found, returns empty string.
    """
    tags = [t.strip() for t in description.split(",") if t.strip()]
    clothing = [t for t in tags if any(kw in t.lower() for kw in _CLOTHING_KEYWORDS)]
    if not clothing:
        return ""
    return "(" + ", ".join(clothing) + ":1.2)"


_NEGATIVE_BASE = (
    # Hard NSFW block — must come first for PonyXL
    "nsfw, nudity, naked, nude, nipples, pussy, penis, genitals, "
    "underwear, lingerie, bikini, swimsuit, cleavage, navel, bare skin, "
    "undressing, topless, bottomless, lewd, ecchi, explicit, uncensored, "
    "(nsfw:1.5), (nudity:1.5), (naked:1.5), "
    # Monster / non-human anatomy block — prevents horns, armor, wings
    "(horn:1.5), (horns:1.5), (armor:1.5), (wings:1.5), (monster:1.5), "
    "(tail:1.3), (claws:1.3), (fangs:1.3), demon, beast, creature, "
    # Dangerous soft tags that PonyXL associates with NSFW
    "alluring, seductive, suggestive, provocative, erotic, sensual, "
    "bedroom eyes, revealing outfit, "
    # Flat / boring composition block — kills retention for horror shorts
    "(flat composition:1.4), (boring pose:1.4), (stock photo pose:1.4), "
    "(centered standing figure:1.3), (neutral expression:1.3), (empty scene:1.3), "
    "(generic portrait:1.3), (plain background:1.2), (static pose:1.2), "
    "bright daylight, cheerful atmosphere, warm sunlight, high key lighting, "
    # PonyDiffusion quality anti-tags
    "score_1, score_2, score_3, score_4, score_5, "
    # Text / watermark — prevents manga SFX, speech bubbles, captions
    "text, watermark, signature, username, logo, "
    "speech bubble, subtitle, caption, banner, label, "
    # General quality
    "lowres, bad anatomy, bad hands, error, "
    "worst quality, low quality, blurry, jpeg artifacts, "
    # Western architecture / Christian symbols — blocked by default (story is Chinese)
    "christian cross, crucifix, western tombstone, grave cross, western grave marker, "
    "gothic cross, church, chapel, cathedral, marble headstone, stone cross, "
    "western cemetery, roman architecture, european architecture, "
    "(cross:1.5), (crucifix:1.5), (church:1.4)"
)

_THUMBNAIL_LIGHTING_TAGS = (
    "bright cinematic lighting",
    "high key lighting",
    "dramatic key light from one side",
    "soft warm rim light",
    "vivid saturated colors",
    "punchy contrast",
    "clear well-lit subject face",
)

# Curiosity / "stop the scroll" composition tags. These do NOT change the
# scene content (the actual horror beat from the key shot is preserved) —
# they only enforce a poster-style framing that hints at a question:
# "what is happening?", "what is that thing?", "why is this person reacting?".
# Multiple cues are blended so the model picks at least one strong hook.
_THUMBNAIL_CURIOSITY_TAGS = (
    "movie poster composition",
    "central subject filling frame",
    "shocked widened eyes locking onto viewer",
    "mid-action frozen moment",
    "single mysterious focal object glowing softly",
    "hand reaching toward something unseen off-frame",
    "tension between subject and an unseen presence",
    "story-implying gesture",
)

# Filter: ONLY strip pure lighting modifiers that crush mobile readability.
# We deliberately keep concrete environment / object nouns (lantern, candle,
# fog, mist) — those carry episode identity and should appear, just lit
# more dramatically thanks to the lighting layer prepended below. Stripping
# everything dark also stripped the very subject of the episode and made all
# thumbnails look the same.
_THUMBNAIL_DARK_FILTER_KEYWORDS = frozenset([
    # Lighting modifiers only — no nouns
    "low light", "low-key", "low key", "dim", "dimly lit", "dimly-lit",
    "underexposed", "murky lighting", "muddy lighting", "pitch", "pitch-black",
    "void lit", "black void", "deep shadow", "heavy shadow", "shadowy",
    "rim-lighting silhouette", "silhouette lighting", "backlit only",
    "moonlit only", "candle only", "lantern only", "no fill light",
    "underlit", "unlit",
])

_THUMBNAIL_NEGATIVE = (
    "lowres, bad quality, blurry, underexposed, low key lighting, too dark, "
    "heavy shadows, murky colors, muddy colors, washed out, silhouette, "
    "backlit subject, face in shadow, night scene, "
    # Style guard — thumbnail must stay photoreal even though Flux drifts to anime
    "(anime:1.5), (manga:1.5), (cartoon:1.4), (illustration:1.4), "
    "(painting:1.3), (digital art:1.3), (3d render:1.3), (cgi:1.3), "
    "boring composition, empty background, no subject, generic stock photo"
)


def _extract_subject_nouns(scene_prompt: str, max_keep: int = 8) -> str:
    """Pull the concrete, content-bearing tokens out of a scene_prompt.

    A scene_prompt is typically a comma-separated list where the first ~6–10
    tokens describe the actual visual (person, action, location, key objects)
    and later tokens are mood / lighting / camera. For thumbnails we want the
    front of that list — that is where the episode's identity lives. Tags that
    are purely lighting modifiers are dropped so the prepended thumbnail-
    lighting layer wins.
    """
    tags = [t.strip() for t in scene_prompt.split(",") if t.strip()]
    kept: list[str] = []
    for tag in tags:
        lower = tag.lower()
        if any(k in lower for k in _THUMBNAIL_DARK_FILTER_KEYWORDS):
            continue
        kept.append(tag)
        if len(kept) >= max_keep:
            break
    return ", ".join(kept)


def _build_thumbnail_scene_prompt(
    scene_prompt: str,
    *,
    hook_scene_prompt: str | None = None,
    episode_title: str | None = None,
    arc_tags: str | None = None,
) -> str:
    """Build a thumbnail prompt that ties to episode content and induces curiosity.

    Inputs:
        scene_prompt:      key shot scene_prompt — the visual peak of the episode.
        hook_scene_prompt: optional shot-0 (hook) scene_prompt — adds the
                           "stop the scroll" element from the opening beat.
        episode_title:     optional Vietnamese title; not embedded in the
                           prompt directly (Flux understands English best),
                           but used to pick which curiosity cue is strongest
                           if heuristics in future builds need it.
        arc_tags:          English visual tags distilled from arc key_events;
                           placed after subject nouns to reinforce episode-
                           specific drama without displacing the lighting/
                           curiosity layers that drive thumbnail readability.

    Strategy:
        1. Lighting layer first (highest CLIP attention) — bright cinematic,
           NOT pure daylight, so it still feels like horror but is readable
           on a mobile feed.
        2. Curiosity composition layer — poster framing, shocked face, a
           single mysterious focal object. This is what creates the "what
           is happening?" feeling.
        3. Episode subject: the front-of-list nouns from the key shot
           (person, action, key objects, location). When a hook scene is
           provided we also blend in 3–4 hook nouns so the thumbnail
           visually rhymes with the very first shot of the video.
        4. Arc tags: condensed English visual tags from arc key_events —
           grounds the thumbnail in the episode's actual dramatic content.
    """
    _ = episode_title  # reserved for future title-aware variant selection

    subject = _extract_subject_nouns(scene_prompt, max_keep=8)
    hook_subject = (
        _extract_subject_nouns(hook_scene_prompt, max_keep=4)
        if hook_scene_prompt
        else ""
    )

    lighting = ", ".join(_THUMBNAIL_LIGHTING_TAGS)
    curiosity = ", ".join(_THUMBNAIL_CURIOSITY_TAGS)

    parts: list[str] = [lighting, curiosity, "wide cinematic shot"]
    if subject:
        parts.append(subject)
    if hook_subject and hook_subject != subject:
        parts.append(hook_subject)
    if arc_tags:
        parts.append(arc_tags)
    return ", ".join(parts)



def _resolve_char_anchor_pairs(shot_characters: list, characters_map: dict) -> list:
    """Resolve the first 2 characters of a shot to (char_obj, anchors) pairs.

    Returns only pairs where anchors exist on disk.
    Characters that don't match any known character are logged and skipped.
    """
    from image_gen.character_gen import get_anchor_paths

    pairs = []
    for name in shot_characters[:2]:
        char_obj = _find_character(name, characters_map)
        anchors = get_anchor_paths(char_obj.name if char_obj else name)
        if anchors:
            pairs.append((char_obj, anchors))
        else:
            logger.warning(
                "Anchor missing, skipping IPAdapter | char={} resolved={}",
                name, char_obj.name if char_obj else "NOT FOUND",
            )
    return pairs


def _scene_id_seed(scene_id: str, episode_num: int) -> int:
    """Deterministic seed from scene_id so all shots in the same location
    share the same background composition base."""
    key = f"{episode_num}:{scene_id}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


def _extract_env_tags(scene_prompt: str) -> str:
    """Extract location + lighting tags from a scene_prompt to use as
    a shared environment baseline for subsequent shots in the same scene_id.

    Heuristic: keep tags that contain location/lighting keywords and
    do NOT contain action/pose keywords. Returns at most 2 tags."""
    ACTION_WORDS = {"figure", "standing", "kneeling", "lunging", "running",
                    "sitting", "holding", "raising", "pose", "wielding",
                    "clasping", "reaching", "pointing", "crouching"}
    tags = [t.strip() for t in scene_prompt.split(",") if t.strip()]
    env_tags = [
        t for t in tags
        if not any(w in t.lower() for w in ACTION_WORDS)
    ]
    # Take first 2 environment tags (location + lighting)
    return ", ".join(env_tags[:2])


def _build_scene_env_baselines(shots) -> dict[str, str]:
    """Return {scene_id: env_baseline_tags} — baseline from the FIRST shot
    in each scene_id group for injection into subsequent shots."""
    seen: dict[str, str] = {}
    for shot in shots:
        sid = shot.scene_id
        if sid and sid not in seen:
            seen[sid] = _extract_env_tags(shot.scene_prompt)
    return seen


def _build_shot_image_params(
    prompt_text: str,
    char_anchor_pairs: list,
    seed: int,
    artifact_hints_by_name: dict[str, list[str]] | None = None,
    init_image_path: "Path | None" = None,
    denoise: float = 0.50,
    shot_subject: str = "person_action",
    scene_ref_image_path: "Path | None" = None,
) -> tuple[str, dict]:
    """Build (workflow_path, replacements) for a single shot/frame.

    Workflow routing:
      All shots (with or without characters) → flux_txt2img_scene
      NOTE: IPAdapter branches are temporarily disabled.
            Character identity is NOT enforced — Flux handles all generation.

    SCENE_PROMPT contains ONLY visual content (location, action, objects, mood).
    shot_subject: drives framing hints injected into the prompt.
    """
    wants_closeup = shot_subject not in ("person_action", "environment")

    # Strip leftover metadata tags that LLM might still include (from cached scripts).
    scene_text = _strip_metadata_tags(prompt_text)
    artifact_detail_tags = _build_artifact_prompt_tags(
        prompt_text, char_anchor_pairs, artifact_hints_by_name
    )

    # Gender count tags — still useful as prompt hints for Flux even without IPAdapter
    gender_tags = ""
    if len(char_anchor_pairs) >= 2:
        genders = [c.gender if c else "male" for c, _ in char_anchor_pairs]
        gender_tags = (
            "2boys" if all(g == "male" for g in genders)
            else "2girls" if all(g == "female" for g in genders)
            else "1boy, 1girl"
        )
    elif len(char_anchor_pairs) == 1:
        char_obj = char_anchor_pairs[0][0]
        gender_tags = "1girl, solo" if (char_obj and char_obj.gender == "female") else "1boy, solo"
    else:
        # No named characters — check if prompt has a human figure without explicit gender.
        # Default to male to prevent Flux from generating a naked female figure.
        _prompt_lower = scene_text.lower()
        _has_figure = any(
            kw in _prompt_lower
            for kw in ("figure", "silhouette", "man ", "male", "person", "character")
        )
        _is_explicitly_female = any(
            kw in _prompt_lower
            for kw in ("female", "woman", "girl", "corpse wearing", "ghost in", "ghost wearing")
        )
        if _has_figure and not _is_explicitly_female:
            gender_tags = "1boy"

    scene_prompt_parts = [p for p in [scene_text, artifact_detail_tags, gender_tags] if p]

    # Inject stable character appearance (hair/eyes/face/clothing) from the
    # character DB. Without IPAdapter (disabled for Flux), this is the only
    # mechanism keeping the same character looking consistent shot-to-shot.
    # IMPORTANT: placed at the FRONT so Flux/T5 attends to identity strongly
    # before reading the scene description. Natural-language sentence (not
    # tag-soup) because T5 understands prose better.
    appearance_tags = _build_character_appearance_tags(char_anchor_pairs)
    if appearance_tags:
        scene_prompt_parts.insert(0, appearance_tags)

    # Prepend wide-framing tags for pure environment shots (no character, no shock close-up)
    if not char_anchor_pairs and not wants_closeup:
        if _SCENE_FRAMING_TAGS not in (scene_prompt_parts[0] if scene_prompt_parts else ""):
            scene_prompt_parts.insert(0, _SCENE_FRAMING_TAGS)

    replacements: dict = {
        # Bumped from 30 → 40 because the natural-language character appearance
        # sentence consumes ~8-12 comma-separated chunks but is still one logical
        # block. We don't want it crowding out scene_prompt or gender_tags.
        "SCENE_PROMPT": _compact_prompt_tags(", ".join(scene_prompt_parts), max_tags=40),
        "SEED": seed,
        "WIDTH": settings.image_width,
        "HEIGHT": settings.image_height,
    }

    if init_image_path is not None:
        # Fan-out frame (fidx > 0): img2img from frame-0.
        workflow = "image_gen/workflows/img2img_scene.json"
        replacements["INIT_IMAGE"] = init_image_path
        replacements["DENOISE"] = denoise
        # img2img_scene still uses SDXL — needs NEGATIVE_PROMPT
        replacements["NEGATIVE_PROMPT"] = _NEGATIVE_BASE
    elif char_anchor_pairs and wants_closeup:
        # Close-up shot (wound, object, entity) WITH a character present.
        # Safe to use character Redux because the anchor portrait composition
        # matches the intended tight framing.
        anchor_path = char_anchor_pairs[0][1][0]  # (char_obj, [anchor_paths])[1][0]
        if scene_ref_image_path is not None:
            workflow = "image_gen/workflows/flux_txt2img_scene_dual_redux.json"
            replacements["ANCHOR_IMAGE"] = anchor_path
            replacements["REDUX_STRENGTH"] = settings.redux_strength
            replacements["SCENE_REF_IMAGE"] = scene_ref_image_path
            replacements["SCENE_REF_STRENGTH"] = settings.scene_ref_strength
        else:
            workflow = "image_gen/workflows/flux_txt2img_scene_redux.json"
            replacements["ANCHOR_IMAGE"] = anchor_path
            replacements["REDUX_STRENGTH"] = settings.redux_strength
    else:
        # Wide/medium scene shots (person_action, environment) — character appearance
        # is enforced via text prompt tags only. NOT using character Redux because
        # anchor images are portrait close-ups and would pull composition to portrait.
        if scene_ref_image_path is not None:
            workflow = "image_gen/workflows/flux_txt2img_scene_ref.json"
            replacements["SCENE_REF_IMAGE"] = scene_ref_image_path
            replacements["SCENE_REF_STRENGTH"] = settings.scene_ref_strength
        else:
            workflow = "image_gen/workflows/flux_txt2img_scene.json"

    return workflow, replacements


def run_images(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.scriptwriter import load_episode_script
    from llm.character_extractor import load_all_characters
    from image_gen.comfyui_client import comfyui_client
    from video.frame_decomposer import decompose_all_shots

    if dry_run:
        logger.info("[dry-run] Skipping images | episode={}", episode_num)
        db.set_episode_status(episode_num, "IMAGES_DONE")
        return

    vram_manager.acquire(VRAMConsumer.COMFYUI)
    vram_manager.health_check_comfyui()

    # Auto-generate anchors for any character that is missing them
    from image_gen.character_gen import generate_character_anchors
    generate_character_anchors(force=False)

    script = load_episode_script(episode_num)
    # Decompose shots into multi-frame structure
    script = script.model_copy(update={"shots": decompose_all_shots(script.shots)})
    images_dir = Path(settings.data_dir) / "images" / f"episode-{episode_num:03d}"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build name→Character map for gender resolution
    characters_map = {c.name: c for c in load_all_characters()}
    artifact_hints_by_name = _load_character_artifact_hints()

    # Build scene-level environment baselines for continuity:
    # shots sharing a scene_id inherit location+lighting tags from the first shot.
    scene_env_baselines = _build_scene_env_baselines(script.shots)

    # Pre-populate scene_first_image from already-generated images so that
    # re-runs (partial regeneration) still benefit from scene reference.
    scene_first_image: dict[str, Path] = {}
    for _pre_idx, _pre_shot in enumerate(script.shots):
        _sid = _pre_shot.scene_id
        if not _sid or _sid in scene_first_image:
            continue
        _pre_frames = _pre_shot.frames if _pre_shot.frames else [None]
        _first_path = (
            images_dir / f"shot-{_pre_idx:02d}-frame-00.png"
            if len(_pre_frames) > 1
            else images_dir / f"shot-{_pre_idx:02d}.png"
        )
        if _first_path.exists():
            scene_first_image[_sid] = _first_path

    db.record_phase_start(episode_num, "images")

    for idx, shot in enumerate(script.shots):
        # shot_subject=non-person forces scene-only workflow regardless of
        # whatever characters list happens to contain — the visual hero is
        # the corpse/wound/blood/entity/object, NOT a person standing next to it.
        subject_is_person = shot.shot_subject.value in ("person_action",)
        # Short-circuit: empty characters → scene-only workflow, skip anchor resolution
        char_anchor_pairs = (
            []
            if (not shot.characters or not subject_is_person)
            else _resolve_char_anchor_pairs(shot.characters, characters_map)
        )

        # Generate each frame for this shot
        frames = shot.frames if shot.frames else [None]
        for fidx, frame in enumerate(frames):
            # Use frame-aware path when multi-frame, legacy path when single frame
            if len(frames) > 1:
                output_path = images_dir / f"shot-{idx:02d}-frame-{fidx:02d}.png"
            else:
                output_path = images_dir / f"shot-{idx:02d}.png"

            if output_path.exists():
                logger.debug(
                    "Image exists, skipping | episode={} shot={} frame={}",
                    episode_num, idx, fidx,
                )
                continue

            # Use frame.scene_prompt (with camera_tag prepended) if available,
            # otherwise fall back to shot.scene_prompt
            prompt_text = frame.scene_prompt if frame else shot.scene_prompt

            # Continuity: prepend env baseline from the first shot in this scene_id
            # so SDXL sees consistent location+lighting tags across all shots in the scene.
            scene_id = shot.scene_id
            if scene_id and scene_id in scene_env_baselines:
                baseline = scene_env_baselines[scene_id]
                # Only prepend if baseline tags aren't already in this prompt
                if baseline and baseline.split(",")[0].strip() not in prompt_text:
                    prompt_text = baseline + ", " + prompt_text

            # All frames use txt2img. Consistency across frames is enforced by:
            #   - same seed (all frames of a shot share the same seed so SDXL starts
            #     from identical noise → same spatial layout / character placement)
            #   - same IPAdapter anchor (same character face across all frames)
            #   - same environment tags (location + lighting shared via scene_env_baselines)
            # Only the camera_tag prefix and per-frame action (from brief.actions) change,
            # so each frame shows the narration progression at a different zoom level.
            if scene_id:
                seed = _scene_id_seed(scene_id, episode_num)
            else:
                seed = episode_num * 10000 + idx * 100

            # Scene reference: first generated image of this scene_id used as
            # soft Redux background anchor for all subsequent shots in the scene.
            # Only apply on fidx==0 (first frame of shot) to avoid double-anchoring.
            scene_ref = scene_first_image.get(scene_id) if (scene_id and fidx == 0) else None

            workflow, replacements = _build_shot_image_params(
                prompt_text, char_anchor_pairs, seed, artifact_hints_by_name,
                shot_subject=shot.shot_subject.value,
                scene_ref_image_path=scene_ref,
            )

            comfyui_client.generate_image(workflow, replacements, output_path)
            logger.info(
                "Image generated | episode={} shot={} frame={} workflow={}",
                episode_num, idx, fidx, Path(workflow).stem,
            )

            # Record first frame of this shot as the scene reference for subsequent shots
            if scene_id and scene_id not in scene_first_image and fidx == 0:
                scene_first_image[scene_id] = output_path

    # Thumbnail for first key shot
    key_indices = [i for i, s in enumerate(script.shots) if s.is_key_shot]
    if key_indices:
        _generate_thumbnail(
            episode_num,
            script.shots[key_indices[0]],
            key_indices[0],
            hook_shot=script.shots[0] if script.shots else None,
            episode_title=script.title,
        )

    db.record_phase_done(episode_num, "images")
    db.set_episode_status(episode_num, "IMAGES_DONE")


def _load_thumbnail_arc_tags(episode_num: int) -> str:
    """Read cached thumbnail visual tags distilled from arc key_events."""
    cache_path = (
        Path(settings.data_dir)
        / "summaries"
        / f"episode-{episode_num:03d}-thumb-tags.txt"
    )
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()
    return ""


def _generate_thumbnail(
    episode_num: int,
    shot,
    shot_idx: int,
    *,
    hook_shot=None,
    episode_title: str | None = None,
) -> None:
    from image_gen.comfyui_client import comfyui_client

    thumbnail_path = (
        Path(settings.data_dir) / "thumbnails" / f"episode-{episode_num:03d}.png"
    )
    if thumbnail_path.exists():
        return

    arc_tags = _load_thumbnail_arc_tags(episode_num)
    thumbnail_scene_prompt = _build_thumbnail_scene_prompt(
        shot.scene_prompt,
        hook_scene_prompt=(hook_shot.scene_prompt if hook_shot else None),
        episode_title=episode_title,
        arc_tags=arc_tags or None,
    )

    comfyui_client.generate_image(
        workflow_path="image_gen/workflows/thumbnail_flux.json",
        replacements={
            "SCENE_PROMPT": thumbnail_scene_prompt,
            "NEGATIVE_PROMPT": _THUMBNAIL_NEGATIVE,
            "WIDTH": settings.thumbnail_width,
            "HEIGHT": settings.thumbnail_height,
            "SEED": episode_num * 1000 + shot_idx,
        },
        output_path=thumbnail_path,
    )

    # Post-process: brighten the base image and overlay Vietnamese title/CTA
    # using a font that supports diacritics (ComfyUI's bundled DrawText+ font
    # ShareTechMono lacks Vietnamese glyphs).
    _postprocess_thumbnail(
        thumbnail_path,
        title_text=f"Tập {episode_num}",
        cta_text="Link ở bio »",
    )
    logger.info("Thumbnail generated | episode={}", episode_num)


# Font candidates for Vietnamese text overlay, in priority order. The first
# path that exists on disk wins. NotoSans/DejaVuSans both support full
# Vietnamese diacritics.
_THUMBNAIL_TITLE_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
_THUMBNAIL_CTA_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _resolve_font(candidates: tuple[str, ...]) -> str | None:
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _postprocess_thumbnail(
    image_path: Path, title_text: str, cta_text: str
) -> None:
    """Brighten the base thumbnail and overlay Vietnamese title/CTA.

    Brightening lifts mid-tones since the source horror prompts often
    yield muddy outputs even after prompt filtering. Text overlay is done
    here (not in ComfyUI) because the bundled DrawText+ font has no
    Vietnamese diacritic support.
    """
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont

    img = Image.open(image_path).convert("RGB")
    # Brightness +18%, contrast +15%, saturation +15% — lifts dark mids
    # without blowing highlights.
    img = ImageEnhance.Brightness(img).enhance(1.18)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Color(img).enhance(1.15)

    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    title_font_path = _resolve_font(_THUMBNAIL_TITLE_FONT_CANDIDATES)
    cta_font_path = _resolve_font(_THUMBNAIL_CTA_FONT_CANDIDATES)

    if title_font_path is None or cta_font_path is None:
        logger.warning(
            "No Vietnamese font found, skipping thumbnail text overlay | "
            "image={}", image_path,
        )
        img.convert("RGB").save(image_path)
        return

    w, h = img.size
    # Scale font sizes relative to image height so 720p and other resolutions
    # render the same visual proportions.
    title_size = max(36, int(h * 0.10))
    cta_size = max(24, int(h * 0.065))
    title_font = ImageFont.truetype(title_font_path, title_size)
    cta_font = ImageFont.truetype(cta_font_path, cta_size)

    margin_x = int(w * 0.045)
    margin_y = int(h * 0.06)
    pad_x = int(title_size * 0.35)
    pad_y = int(title_size * 0.18)

    def _draw_text_with_box(
        text: str,
        font: ImageFont.FreeTypeFont,
        anchor_xy: tuple[int, int],
        text_color: tuple[int, int, int, int],
        box_color: tuple[int, int, int, int],
        align: str = "left",
        baseline: str = "top",
    ) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x, y = anchor_xy
        if align == "left":
            box_left = x - pad_x
        else:
            box_left = x - text_w - pad_x
        if baseline == "top":
            box_top = y - pad_y
        else:
            box_top = y - text_h - pad_y
        box_right = box_left + text_w + pad_x * 2
        box_bottom = box_top + text_h + pad_y * 2
        draw.rounded_rectangle(
            [box_left, box_top, box_right, box_bottom],
            radius=int(pad_y * 0.8),
            fill=box_color,
        )
        # Drop shadow for legibility on busy backgrounds.
        shadow_offset = max(2, int(title_size * 0.04))
        draw.text(
            (x - bbox[0] + shadow_offset, y - bbox[1] + shadow_offset),
            text,
            font=font,
            fill=(0, 0, 0, 200),
        )
        draw.text(
            (x - bbox[0], y - bbox[1]),
            text,
            font=font,
            fill=text_color,
        )

    # Title — top-left, white on semi-opaque dark plate
    _draw_text_with_box(
        title_text,
        title_font,
        (margin_x, margin_y),
        text_color=(255, 255, 255, 255),
        box_color=(0, 0, 0, 170),
        align="left",
        baseline="top",
    )

    # CTA — bottom-left, gold on semi-opaque dark plate
    cta_bbox = draw.textbbox((0, 0), cta_text, font=cta_font)
    cta_h = cta_bbox[3] - cta_bbox[1]
    cta_y = h - margin_y - cta_h
    _draw_text_with_box(
        cta_text,
        cta_font,
        (margin_x, cta_y),
        text_color=(255, 215, 0, 255),
        box_color=(0, 0, 0, 170),
        align="left",
        baseline="top",
    )

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    composed.save(image_path)


def run_audio(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.scriptwriter import load_episode_script
    from audio.tts import generate_episode_tts_sync
    from audio.mixer import mix_narration_with_bgm, find_bgm

    if dry_run:
        logger.info("[dry-run] Skipping audio | episode={}", episode_num)
        db.set_episode_status(episode_num, "AUDIO_DONE")
        return

    script = load_episode_script(episode_num)
    bgm_path = find_bgm()

    db.record_phase_start(episode_num, "audio")

    narration_paths = generate_episode_tts_sync(episode_num, script.shots)

    audio_dir = Path(settings.data_dir) / "audio" / f"episode-{episode_num:03d}"
    mix_failures: list[int] = []
    for idx, narr_path in enumerate(narration_paths):
        if not narr_path.exists():
            logger.error(
                "Narration file missing, skipping mix | episode={} shot={} path={}",
                episode_num, idx, narr_path,
            )
            mix_failures.append(idx)
            continue
        mixed_path = audio_dir / f"shot-{idx:02d}-mixed.aac"
        try:
            mix_narration_with_bgm(narr_path, mixed_path, bgm_path)
        except Exception as exc:
            logger.error(
                "Audio mix failed | episode={} shot={} error={}",
                episode_num, idx, exc,
            )
            mix_failures.append(idx)

    if mix_failures:
        logger.warning(
            "Audio mix had failures | episode={} failed_shots={}/{}",
            episode_num, len(mix_failures), len(narration_paths),
        )
        if len(mix_failures) == len(narration_paths):
            raise RuntimeError(
                f"All {len(narration_paths)} audio mix shots failed for episode {episode_num}."
            )

    db.record_phase_done(episode_num, "audio")
    db.set_episode_status(episode_num, "AUDIO_DONE")


def _probe_duration(path: Path) -> float:
    """Return audio/video file duration in seconds via ffprobe. Returns 0.0 on error."""
    import ffmpeg as _ffmpeg
    try:
        info = _ffmpeg.probe(str(path))
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


def run_video(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.scriptwriter import load_episode_script
    from video.assembler import assemble_shot_clips
    from video.editor import assemble_episode
    from video.frame_decomposer import decompose_all_shots

    if dry_run:
        logger.info("[dry-run] Skipping video | episode={}", episode_num)
        db.set_episode_status(episode_num, "VIDEO_DONE")
        return

    script = load_episode_script(episode_num)
    # Decompose shots into frames (same logic as run_images)
    script = script.model_copy(update={"shots": decompose_all_shots(script.shots)})
    audio_dir = Path(settings.data_dir) / "audio" / f"episode-{episode_num:03d}"
    audio_paths = [
        audio_dir / f"shot-{i:02d}-mixed.aac" for i in range(len(script.shots))
    ]

    # SYNC: actual audio duration is AUTHORITATIVE for clip length.
    # The scriptwriter's duration_sec is a BUDGET for narration planning only —
    # once TTS is rendered, clip length MUST equal mixed audio length so that:
    #   - no silent freeze at tail (viewer retention killer)
    #   - karaoke subtitle end aligns with clip end
    #   - shot-to-shot cross-fade happens the moment narration finishes
    # The mixed audio already contains tts_lead_in_sec + tail padding, so using
    # `actual` directly gives natural breath pacing without silent gaps.
    # Fall back to planned duration only when the audio file is missing / unreadable.
    for idx, (shot, audio_path) in enumerate(zip(script.shots, audio_paths)):
        if not audio_path.exists():
            logger.warning(
                "Mixed audio missing, keeping planned duration | episode={} shot={} planned={}s",
                episode_num, idx, shot.duration_sec,
            )
            continue
        actual = _probe_duration(audio_path)
        if actual <= 0:
            logger.warning(
                "Could not probe audio duration, keeping planned | episode={} shot={}",
                episode_num, idx,
            )
            continue
        drift = actual - shot.duration_sec
        if abs(drift) > 3.0:
            logger.warning(
                "Audio/plan drift >3s | episode={} shot={} planned={}s actual={:.2f}s drift={:+.2f}s",
                episode_num, idx, shot.duration_sec, actual, drift,
            )
        shot.duration_sec = actual
        shot.actual_audio_sec = actual

    db.record_phase_start(episode_num, "video")

    shot_clips = assemble_shot_clips(episode_num, script.shots, audio_paths)
    assemble_episode(episode_num, shot_clips, script.shots)

    db.record_phase_done(episode_num, "video")
    db.set_episode_status(episode_num, "VIDEO_DONE")


def run_script_review(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    """Phase: quality-review the generated script and auto-fix issues via LLM.

    When `retention.use_constraint_system` is on, runs the gatekeeper reviewer
    (constraint_validator → BLOCKING/WARNING tiers) with a bounded retry loop.
    Otherwise, runs the legacy rule-based + LLM-fix reviewer.
    """
    if dry_run:
        logger.info("[dry-run] Skipping script_review | episode={}", episode_num)
        db.set_episode_status(episode_num, "SCRIPT_REVIEWED")
        return

    db.record_phase_start(episode_num, "script_review")

    if settings.retention.use_constraint_system:
        _run_gatekeeper_review(episode_num)
    else:
        from llm.script_reviewer import review_episode_script

        _, report = review_episode_script(episode_num)

        if report.fixed_shots:
            logger.info(
                "Script review fixed {} shot(s) | episode={} fixed={}",
                len(report.fixed_shots), episode_num, report.fixed_shots,
            )
        else:
            logger.info(
                "Script review complete — no rewrites needed | episode={}",
                episode_num,
            )

    db.record_phase_done(episode_num, "script_review")
    db.set_episode_status(episode_num, "SCRIPT_REVIEWED")


def _run_gatekeeper_review(episode_num: int) -> None:
    """Phase 5: bounded-retry constraint reviewer.

    1. Run gatekeeper → blocking + warnings.
    2. If blocking and retries left → call regenerate_failed_shots, save, repeat.
    3. After max retries OR pass → save violations to script + log JSONL + persist.
    """
    from llm.gatekeeper import gatekeeper_review, log_violations_jsonl
    from llm.scriptwriter import (
        load_episode_script,
        regenerate_failed_shots,
    )

    script = load_episode_script(episode_num)
    max_retries = settings.retention.reviewer_max_retries
    known_chars = []  # gatekeeper computes proper-noun signals; chars optional

    result = gatekeeper_review(script, known_chars=known_chars)
    log_violations_jsonl(episode_num, result, attempt=0, final=(result.passed or max_retries == 0))

    if result.passed:
        logger.info("Gatekeeper PASS | episode={}", episode_num)
        _persist_violations(script, result, episode_num)
        return

    logger.warning(
        "Gatekeeper found {} BLOCKING + {} WARNING | episode={}\n{}",
        len(result.blocking), len(result.warnings), episode_num, result.summary(),
    )

    for attempt in range(1, max_retries + 1):
        try:
            script = regenerate_failed_shots(script, result.blocking, episode_num)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Regen failed | episode={} attempt={} err={}",
                episode_num, attempt, e,
            )
            # Ensure audit trail has a final=True entry even when regen crashes.
            log_violations_jsonl(episode_num, result, attempt=attempt, final=True)
            break

        result = gatekeeper_review(script, known_chars=known_chars)
        final = result.passed or attempt == max_retries
        log_violations_jsonl(episode_num, result, attempt=attempt, final=final)
        if result.passed:
            logger.info(
                "Gatekeeper PASS after retry={} | episode={}", attempt, episode_num,
            )
            break
        logger.warning(
            "Gatekeeper still failing after retry={} | episode={} blocking={}",
            attempt, episode_num, len(result.blocking),
        )

    # Graceful degrade: persist whatever we have. Pipeline continues even if
    # blocking violations remain — Phase 6 calibration will tighten thresholds.
    _persist_violations(script, result, episode_num)


def _persist_violations(script: "EpisodeScript", result: "ReviewResult", episode_num: int) -> None:
    """Save the (possibly regenerated) script + attach violation messages."""
    from llm.scriptwriter import _save_script_after_review

    script.constraint_violations = [
        f"[{v.severity.value.upper()}] shot={v.shot_index} {v.rule}: {v.message}"
        for v in result.all_violations
    ]
    _save_script_after_review(script, episode_num)


def run_validate(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    if dry_run:
        logger.info("[dry-run] Skipping validate | episode={}", episode_num)
        db.set_episode_status(episode_num, "VALIDATED")
        return

    try:
        validator.assert_episode(episode_num)
    except ValidationError:
        # Keep at VIDEO_DONE so operator can re-run --from-phase validate
        db.set_episode_status(episode_num, "VIDEO_DONE", error_msg="Validation failed")
        raise

    db.set_episode_status(episode_num, "VALIDATED")

    # Cleanup intermediate files — only after VALIDATED
    for subdir in ("images", "audio", "clips"):
        shutil.rmtree(
            Path(settings.data_dir) / subdir / f"episode-{episode_num:03d}",
            ignore_errors=True,
        )
    logger.info("Cleanup done | episode={}", episode_num)


_PHASE_RUNNERS = {
    "llm": run_llm,
    "script_review": run_script_review,
    "images": run_images,
    "audio": run_audio,
    "video": run_video,
    "validate": run_validate,
}


# ── Episode / pipeline entry points ───────────────────────────────────────────

def run_episode(
    episode_num: int,
    from_phase: str = "llm",
    dry_run: bool = False,
) -> None:
    db = StateDB()
    setup_logging(episode_num)
    chapter_start, chapter_end = _episode_chapter_range(episode_num)
    db.upsert_episode(episode_num, chapter_start, chapter_end)

    current_status = db.get_episode_status(episode_num)
    if from_phase != "llm" and current_status:
        db.reset_episode_to_phase(episode_num, from_phase)

    start_idx = PHASES.index(from_phase)

    logger.info(
        "Episode start | episode={} from_phase={} dry_run={}",
        episode_num, from_phase, dry_run,
    )

    for phase in PHASES[start_idx:]:
        runner = _PHASE_RUNNERS[phase]
        logger.info("Phase start | episode={} phase={}", episode_num, phase)
        try:
            runner(episode_num, db, dry_run=dry_run)
            logger.info("Phase done  | episode={} phase={}", episode_num, phase)
        except Exception as exc:
            logger.error(
                "Phase failed | episode={} phase={} error={}",
                episode_num, phase, str(exc),
            )
            db.set_episode_status(
                episode_num, "ERROR", error_msg=f"{phase}: {exc}"
            )
            # Always free VRAM on failure so the next run is not blocked
            try:
                vram_manager.release_all()
            except Exception as vram_exc:
                logger.warning("VRAM release failed during cleanup | error={}", vram_exc)
            raise


def run_pipeline(
    from_episode: int = 1,
    from_phase: str = "llm",
    dry_run: bool = False,
) -> None:
    db = StateDB()
    setup_logging()

    total = settings.total_episodes
    logger.info(
        "Pipeline start | total_episodes={} from={} dry_run={}",
        total, from_episode, dry_run,
    )

    for ep in range(from_episode, total + 1):
        try:
            run_episode(ep, from_phase=from_phase, dry_run=dry_run)

            if ep == 1:
                remaining = total - 1
                eta = db.estimate_eta(remaining)
                if eta:
                    logger.info(
                        "ETA for {} remaining episodes: {:.1f}h",
                        remaining, eta / 3600,
                    )
        except Exception as exc:
            logger.error(
                "Episode failed, continuing pipeline | episode={} error={}",
                ep, str(exc),
            )

        # Only first episode may start from non-llm phase
        from_phase = "llm"


# ── CLI entry point ────────────────────────────────────────────────────────────

def probe_images(episode_num: int, gen_shots: int = 0) -> None:
    """Dry-inspect character→scene matching for all shots of an episode.

    Prints a table showing: shot index, LLM character names, resolved canonical
    name, DNA tags that will be injected, and which workflow will be used.
    No ComfyUI calls are made unless gen_shots > 0, in which case only the
    first `gen_shots` shots are actually generated into data/probe/.
    """
    from llm.scriptwriter import load_episode_script
    from llm.character_extractor import load_all_characters

    try:
        script = load_episode_script(episode_num)
    except FileNotFoundError:
        logger.error("No script found for episode {} — run LLM phase first", episode_num)
        return

    characters_map = {c.name: c for c in load_all_characters()}

    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"

    print(f"\n{BOLD}Probe: episode {episode_num} — {len(script.shots)} shots{RESET}\n")

    for idx, shot in enumerate(script.shots):
        print(f"  {BOLD}Shot {idx:02d}{RESET}  dur={shot.duration_sec}s  key={shot.is_key_shot}")
        print(f"    scene  : {shot.scene_prompt[:90]}…" if len(shot.scene_prompt) > 90 else f"    scene  : {shot.scene_prompt}")

        char_anchor_pairs = []
        for name in shot.characters[:2]:
            char_obj = _find_character(name, characters_map)
            anchors = get_anchor_paths(char_obj.name if char_obj else name)
            if char_obj and anchors:
                dna = _extract_dna_tags(char_obj.description)
                print(f"    {GREEN}✓ {name!r} → {char_obj.name!r}{RESET}")
                print(f"      DNA  : {dna[:80]}…" if len(dna) > 80 else f"      DNA  : {dna}")
                print(f"      refs : {len(anchors)} anchor(s)")
                char_anchor_pairs.append((char_obj, anchors))
            elif char_obj:
                print(f"    {YELLOW}⚠ {name!r} → {char_obj.name!r} (NO anchor — run image phase anchor gen){RESET}")
            else:
                print(f"    {RED}✗ {name!r} → NOT FOUND (fallback: txt2img_scene){RESET}")

        if not shot.characters:
            print(f"    (no characters — scene-only shot)")

        # Workflow that would be selected
        n = len(char_anchor_pairs)
        if n >= 2:
            wf = "txt2img_ipadapter_dual"
        elif n == 1:
            _, anchors = char_anchor_pairs[0]
            wf = "txt2img_ipadapter_multiref" if len(anchors) > 1 else "txt2img_ipadapter"
        else:
            wf = "txt2img_scene (fallback)"
        print(f"    workflow: {wf}\n")

    if gen_shots <= 0:
        print(f"{BOLD}Tip:{RESET} add --probe-shots N to also generate the first N shots into data/probe/\n")
        return

    # ── Generate probe shots ──────────────────────────────────────────────────
    from image_gen.comfyui_client import comfyui_client

    probe_dir = Path(settings.data_dir) / "probe" / f"episode-{episode_num:03d}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    vram_manager.acquire(VRAMConsumer.COMFYUI)
    vram_manager.health_check_comfyui()

    print(f"{BOLD}Generating {gen_shots} probe shot(s) → {probe_dir}{RESET}\n")
    for idx, shot in enumerate(script.shots[:gen_shots]):
        out = probe_dir / f"shot-{idx:02d}.png"
        if out.exists():
            print(f"  Shot {idx:02d}: already exists, skipping")
            continue

        char_anchor_pairs = (
            []
            if (not shot.characters or shot.shot_subject.value != "person_action")
            else _resolve_char_anchor_pairs(shot.characters, characters_map)
        )
        seed = episode_num * 10000 + idx * 100
        workflow, replacements = _build_shot_image_params(
            shot.scene_prompt, char_anchor_pairs, seed,
            shot_subject=shot.shot_subject.value,
        )

        comfyui_client.generate_image(workflow, replacements, out)
        print(f"  Shot {idx:02d}: {GREEN}done{RESET} → {out}")

    print(f"\n{BOLD}Probe complete.{RESET} Open {probe_dir} to review.\n")


def _parse_episode_specs(specs: list[str]) -> list[int]:
    """Expand CLI episode specs into a deduped, ordered list of episode numbers.

    Accepts integers and inclusive ranges:
        ["1"]              -> [1]
        ["1", "2", "5"]    -> [1, 2, 5]
        ["1-3"]            -> [1, 2, 3]
        ["1", "3-5", "8"]  -> [1, 3, 4, 5, 8]
    """
    seen: set[int] = set()
    result: list[int] = []
    for raw in specs:
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_s, _, end_s = token.partition("-")
            try:
                start, end = int(start_s), int(end_s)
            except ValueError as exc:
                raise SystemExit(f"Invalid episode range: {token!r}") from exc
            if start > end:
                raise SystemExit(f"Invalid episode range (start > end): {token!r}")
            nums = range(start, end + 1)
        else:
            try:
                nums = [int(token)]
            except ValueError as exc:
                raise SystemExit(f"Invalid episode number: {token!r}") from exc
        for n in nums:
            if n < 1:
                raise SystemExit(f"Episode number must be >= 1, got {n}")
            if n not in seen:
                seen.add(n)
                result.append(n)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video Production Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--episode", type=str, nargs="+", metavar="N",
        help=(
            "Run one or more specific episodes (skips episodes not listed).\n"
            "Accepts: single (--episode 1), multiple (--episode 1 2 5),\n"
            "ranges (--episode 1-3), or mix (--episode 1 3-5 8)."
        ),
    )
    parser.add_argument(
        "--from-episode", type=int, default=1, metavar="N",
        help="Start full pipeline from episode N (default: 1)",
    )
    parser.add_argument(
        "--from-phase",
        default="llm",
        choices=PHASES,
        metavar="PHASE",
        help=f"Start from phase: {', '.join(PHASES)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log decisions without executing tasks",
    )
    parser.add_argument(
        "--probe-images", type=int, metavar="N",
        help=(
            "Inspect character→scene matching for episode N without generating images.\n"
            "Add --probe-shots K to also generate the first K shots into data/probe/."
        ),
    )
    parser.add_argument(
        "--probe-shots", type=int, default=0, metavar="K",
        help="Number of shots to actually generate during --probe-images (default: 0 = inspect only)",
    )
    args = parser.parse_args()

    if args.probe_images is not None:
        setup_logging(args.probe_images)
        probe_images(args.probe_images, gen_shots=args.probe_shots)
    elif args.episode:
        episodes = _parse_episode_specs(args.episode)
        if not episodes:
            parser.error("--episode produced an empty episode list")
        for ep in episodes:
            run_episode(ep, from_phase=args.from_phase, dry_run=args.dry_run)
    else:
        run_pipeline(
            from_episode=args.from_episode,
            from_phase=args.from_phase,
            dry_run=args.dry_run,
        )
