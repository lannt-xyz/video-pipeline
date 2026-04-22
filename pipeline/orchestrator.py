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

PHASES = ["llm", "images", "audio", "video", "validate"]


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

    # Evict summary model before loading script model to avoid VRAM pressure
    # when the two models are different (e.g. Gemma 4 → Mistral Small 3).
    if settings.effective_summary_model != settings.effective_script_model:
        logger.info(
            "Switching LLM model | unloading={} next={}",
            settings.effective_summary_model, settings.effective_script_model,
        )
        vram_manager.unload_model(settings.effective_summary_model)

    logger.info("Writing script | episode={}", episode_num)
    write_episode_script(episode_num)

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
    "anime style", "manhua art style", "no text", "no watermarks",
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
    # PonyDiffusion quality anti-tags
    "score_1, score_2, score_3, score_4, score_5, "
    # Text / watermark — prevents manga SFX, speech bubbles, captions
    "text, watermark, signature, username, logo, "
    "speech bubble, subtitle, caption, banner, label, "
    # General quality
    "lowres, bad anatomy, bad hands, error, "
    "worst quality, low quality, blurry, jpeg artifacts"
)

_THUMBNAIL_LIGHTING_TAGS = (
    "bright cinematic lighting",
    "high key lighting",
    "soft warm rim light",
    "vivid contrast",
    "clear subject face",
)

_THUMBNAIL_DARK_FILTER_KEYWORDS = frozenset([
    "dark", "dim", "night", "moonlight", "low light", "shadowy", "gloom",
])

_THUMBNAIL_NEGATIVE = (
    "lowres, bad quality, blurry, underexposed, low key lighting, too dark, "
    "heavy shadows, murky colors"
)


def _build_thumbnail_scene_prompt(scene_prompt: str) -> str:
    """Build a brighter thumbnail prompt from shot scene tags.

    Removes obviously dark lighting tags and appends high-visibility lighting tags
    so thumbnails stay readable on mobile feeds.
    """
    tags = [t.strip() for t in scene_prompt.split(",") if t.strip()]
    filtered = [
        t for t in tags
        if not any(k in t.lower() for k in _THUMBNAIL_DARK_FILTER_KEYWORDS)
    ]
    base = ", ".join(filtered)
    extras = ", ".join(_THUMBNAIL_LIGHTING_TAGS)
    return ", ".join(
        [p for p in [base, "wide cinematic shot", extras] if p]
    )



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
    denoise: float = 0.70,
) -> tuple[str, dict]:
    """Build (workflow_path, replacements) for a single shot/frame.

    All shots use txt2img_scene.json — no IPAdapter.
    SCENE_PROMPT contains ONLY visual content (location, action, objects, mood).
    Metadata tags (sfw, anime style, etc.) live in the workflow CLIP suffix.
    """
    has_female = any(c is not None and c.gender == "female" for c, _ in char_anchor_pairs)
    negative_base = (
        _NEGATIVE_BASE + ", (male:1.5), (masculine:1.3), 1boy"
        if has_female
        else _NEGATIVE_BASE
    )

    _ANTI_SPLIT_SINGLE = ", (2girls:1.5), (2boys:1.5), (multiple girls:1.5), (multiple boys:1.5), (split view:1.5), (grid:1.5), (collage:1.5), (multiple views:1.5)"

    # Strip leftover metadata tags that LLM might still include (from cached scripts).
    scene_text = _strip_metadata_tags(prompt_text)
    artifact_detail_tags = _build_artifact_prompt_tags(
        prompt_text, char_anchor_pairs, artifact_hints_by_name
    )

    # Gender count tags for character shots
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

    scene_prompt_parts = [p for p in [scene_text, artifact_detail_tags, gender_tags] if p]

    # Prepend framing tags for scene-only (no character) shots.
    if not char_anchor_pairs:
        if _SCENE_FRAMING_TAGS not in (scene_prompt_parts[0] if scene_prompt_parts else ""):
            scene_prompt_parts.insert(0, _SCENE_FRAMING_TAGS)

    neg = negative_base + _ANTI_SPLIT_SINGLE
    if not char_anchor_pairs:
        neg += _SCENE_ANTI_PORTRAIT_NEG

    replacements: dict = {
        "SCENE_PROMPT": _compact_prompt_tags(", ".join(scene_prompt_parts), max_tags=24),
        "NEGATIVE_PROMPT": neg,
        "SEED": seed,
    }

    if init_image_path is not None:
        workflow = "image_gen/workflows/img2img_scene.json"
        replacements["INIT_IMAGE"] = init_image_path
        replacements["DENOISE"] = denoise
    else:
        workflow = "image_gen/workflows/txt2img_scene.json"
        replacements["WIDTH"] = settings.image_width
        replacements["HEIGHT"] = settings.image_height

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

    db.record_phase_start(episode_num, "images")

    for idx, shot in enumerate(script.shots):
        # Short-circuit: empty characters → scene-only workflow, skip anchor resolution
        char_anchor_pairs = (
            []
            if not shot.characters
            else _resolve_char_anchor_pairs(shot.characters, characters_map)
        )

        # Generate each frame for this shot
        frames = shot.frames if shot.frames else [None]
        shot_frame0_path: "Path | None" = None  # fan-out: frames 1-3 all use frame-0 as init
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
                if fidx == 0:
                    shot_frame0_path = output_path
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

            # Continuity: shots in the same scene_id share the same seed base
            # so background composition stays consistent across the location.
            # Frames within the same shot also share the seed — only camera_tag differs,
            # which is enough to drive the zoom/pan progression.
            if scene_id:
                seed = _scene_id_seed(scene_id, episode_num)
            else:
                seed = episode_num * 10000 + idx * 100

            # Frame 0 = txt2img; frames 1+ = img2img fan-out from frame-0.
            # Fan-out (not chain) prevents artifact accumulation across the sequence.
            init_path = shot_frame0_path if fidx > 0 else None
            workflow, replacements = _build_shot_image_params(
                prompt_text, char_anchor_pairs, seed, artifact_hints_by_name,
                init_image_path=init_path,
            )

            comfyui_client.generate_image(workflow, replacements, output_path)
            if fidx == 0:
                shot_frame0_path = output_path
            logger.info(
                "Image generated | episode={} shot={} frame={} workflow={}",
                episode_num, idx, fidx, Path(workflow).stem,
            )

    # Thumbnail for first key shot
    key_indices = [i for i, s in enumerate(script.shots) if s.is_key_shot]
    if key_indices:
        _generate_thumbnail(episode_num, script.shots[key_indices[0]], key_indices[0])

    db.record_phase_done(episode_num, "images")
    db.set_episode_status(episode_num, "IMAGES_DONE")


def _generate_thumbnail(episode_num: int, shot, shot_idx: int) -> None:
    from image_gen.comfyui_client import comfyui_client

    thumbnail_path = (
        Path(settings.data_dir) / "thumbnails" / f"episode-{episode_num:03d}.png"
    )
    if thumbnail_path.exists():
        return

    thumbnail_scene_prompt = _build_thumbnail_scene_prompt(shot.scene_prompt)

    comfyui_client.generate_image(
        workflow_path="image_gen/workflows/thumbnail.json",
        replacements={
            "SCENE_PROMPT": thumbnail_scene_prompt,
            "NEGATIVE_PROMPT": _THUMBNAIL_NEGATIVE,
            "WIDTH": settings.thumbnail_width,
            "HEIGHT": settings.thumbnail_height,
            "SEED": episode_num * 1000 + shot_idx,
            "TITLE_TEXT": f"Tập {episode_num}",
            "CTA_TEXT": "Link bio 👇",
        },
        output_path=thumbnail_path,
    )
    logger.info("Thumbnail generated | episode={}", episode_num)


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
    for idx, narr_path in enumerate(narration_paths):
        mixed_path = audio_dir / f"shot-{idx:02d}-mixed.aac"
        mix_narration_with_bgm(narr_path, mixed_path, bgm_path)

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

    # Patch duration_sec: use max(script, audio) so clips are never shorter than
    # the scriptwriter's budget.  If TTS finishes early the zoompan animation
    # continues (with silence); if TTS is longer the clip extends to fit.
    for shot, audio_path in zip(script.shots, audio_paths):
        if audio_path.exists():
            actual = _probe_duration(audio_path)
            if actual > 0:
                shot.duration_sec = max(shot.duration_sec, actual)

    db.record_phase_start(episode_num, "video")

    shot_clips = assemble_shot_clips(episode_num, script.shots, audio_paths)
    assemble_episode(episode_num, shot_clips, script.shots)

    db.record_phase_done(episode_num, "video")
    db.set_episode_status(episode_num, "VIDEO_DONE")


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
            if not shot.characters
            else _resolve_char_anchor_pairs(shot.characters, characters_map)
        )
        seed = episode_num * 10000 + idx * 100
        workflow, replacements = _build_shot_image_params(
            shot.scene_prompt, char_anchor_pairs, seed
        )

        comfyui_client.generate_image(workflow, replacements, out)
        print(f"  Shot {idx:02d}: {GREEN}done{RESET} → {out}")

    print(f"\n{BOLD}Probe complete.{RESET} Open {probe_dir} to review.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video Production Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--episode", type=int, metavar="N",
        help="Run a single episode N end-to-end (or from --from-phase)",
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
        run_episode(args.episode, from_phase=args.from_phase, dry_run=args.dry_run)
    else:
        run_pipeline(
            from_episode=args.from_episode,
            from_phase=args.from_phase,
            dry_run=args.dry_run,
        )
