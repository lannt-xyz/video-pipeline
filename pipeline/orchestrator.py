import argparse
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import settings
from pipeline.state import StateDB
from pipeline.validator import ValidationError, validator
from pipeline.vram_manager import vram_manager

if TYPE_CHECKING:
    from image_gen.comfyui_client import ComfyUIClient

PHASES = ["crawl", "llm", "images", "audio", "video", "validate"]


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


# ── Phase runners ─────────────────────────────────────────────────────────────

def run_crawl(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from crawler.scraper import crawl_chapters
    from crawler.storage import save_chapter
    import asyncio

    chapter_start, chapter_end = _episode_chapter_range(episode_num)
    db.upsert_episode(episode_num, chapter_start, chapter_end)

    chapters_dir = Path(settings.data_dir) / "chapters"
    db_crawled = set(db.get_crawled_chapters(chapter_start, chapter_end))
    # Cross-check: chapter DB says crawled but file missing → re-crawl it
    actually_crawled = {
        n for n in db_crawled
        if (chapters_dir / f"chuong-{n:04d}.txt").exists()
    }
    missing_from_disk = db_crawled - actually_crawled
    if missing_from_disk:
        logger.warning(
            "{} chapters in DB but missing on disk, will re-crawl | chapters={}",
            len(missing_from_disk), sorted(missing_from_disk)[:10],
        )
    to_crawl = [n for n in range(chapter_start, chapter_end + 1) if n not in actually_crawled]

    if not to_crawl:
        logger.info("All chapters already crawled | episode={}", episode_num)
        db.set_episode_status(episode_num, "CRAWLED")
        return

    logger.info(
        "Crawling {} chapters | episode={} range={}-{}",
        len(to_crawl), episode_num, chapter_start, chapter_end,
    )

    if dry_run:
        logger.info("[dry-run] Skipping crawl | episode={}", episode_num)
        db.set_episode_status(episode_num, "CRAWLED")
        return

    db.record_phase_start(episode_num, "crawl")
    chapters = asyncio.run(
        crawl_chapters(to_crawl, on_fetched=lambda c: save_chapter(c, db))
    )
    db.record_phase_done(episode_num, "crawl")

    failed = [c for c in chapters if c.status == "ERROR"]
    if failed:
        logger.warning("{} chapters failed | episode={}", len(failed), episode_num)

    db.set_episode_status(episode_num, "CRAWLED")


def run_llm(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.summarizer import summarize_episode
    from llm.scriptwriter import write_episode_script

    chapter_start, chapter_end = _episode_chapter_range(episode_num)

    if dry_run:
        logger.info("[dry-run] Skipping LLM | episode={}", episode_num)
        db.set_episode_status(episode_num, "SCRIPTED")
        return

    vram_manager.unload_comfyui()
    vram_manager.health_check_ollama()

    db.record_phase_start(episode_num, "llm")

    logger.info("Summarizing | episode={}", episode_num)
    summarize_episode(episode_num, chapter_start, chapter_end)
    db.set_episode_status(episode_num, "SUMMARIZED")

    # Extract characters once from arc summaries (idempotent — skips existing JSONs)
    if episode_num == 1:
        from llm.character_extractor import extract_all_characters
        logger.info("Extracting characters from arc summaries")
        extract_all_characters()

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
    "mysterious aura, bedroom eyes, revealing outfit, "
    # PonyDiffusion quality anti-tags
    "score_1, score_2, score_3, score_4, score_5, "
    # Text / watermark — prevents manga SFX, speech bubbles, captions
    "text, watermark, signature, username, logo, "
    "speech bubble, subtitle, caption, banner, label, "
    # General quality
    "lowres, bad anatomy, bad hands, error, "
    "worst quality, low quality, blurry, jpeg artifacts"
)


def _generate_scene_fallback(
    comfyui_client: "ComfyUIClient",
    prompt_text: str,
    seed: int,
    output_path: Path,
) -> None:
    """Fallback: generate shot with txt2img_scene (no IPAdapter)."""
    comfyui_client.generate_image(
        "image_gen/workflows/txt2img_scene.json",
        {
            "SCENE_PROMPT": prompt_text,
            "NEGATIVE_PROMPT": _NEGATIVE_BASE,
            "WIDTH": settings.image_width,
            "HEIGHT": settings.image_height,
            "SEED": seed,
        },
        output_path,
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


def _build_shot_image_params(
    prompt_text: str,
    char_anchor_pairs: list,
    seed: int,
) -> tuple[str, dict]:
    """Build (workflow_path, replacements) for a single shot/frame.

    Uses _NEGATIVE_BASE with optional female-anatomy block.
    Injects DNA tags into single-character prompts to reinforce IPAdapter.
    """
    has_female = any(c is not None and c.gender == "female" for c, _ in char_anchor_pairs)
    negative_base = (
        _NEGATIVE_BASE + ", (male:1.5), (masculine:1.3), 1boy"
        if has_female
        else _NEGATIVE_BASE
    )

    # Anti-split tags: prevent PonyXL from tiling multiple figures into a grid
    _ANTI_SPLIT_DUAL = ", (split view:1.5), (grid:1.5), (collage:1.5), (multiple views:1.5), (4girls:1.5), (4boys:1.5)"
    _ANTI_SPLIT_SINGLE = ", (2girls:1.5), (2boys:1.5), (multiple girls:1.5), (multiple boys:1.5), (split view:1.5), (grid:1.5), (collage:1.5), (multiple views:1.5)"

    if len(char_anchor_pairs) >= 2:
        genders = [c.gender if c else "male" for c, _ in char_anchor_pairs]
        count_tag = (
            "2boys" if all(g == "male" for g in genders)
            else "2girls" if all(g == "female" for g in genders)
            else "1boy, 1girl"
        )
        workflow = "image_gen/workflows/txt2img_ipadapter_dual.json"
        replacements = {
            # Dual-char: IPAdapter anchors carry both visual identities;
            # text only needs count tag + scene (avoid DNA tag conflicts between chars)
            "SCENE_PROMPT": f"{count_tag}, {prompt_text}",
            "NEGATIVE_PROMPT": negative_base + _ANTI_SPLIT_DUAL,
            "WIDTH": settings.image_width,
            "HEIGHT": settings.image_height,
            "SEED": seed,
            "ANCHOR_PATH": char_anchor_pairs[0][1][0],
            "ANCHOR_PATH_2": char_anchor_pairs[1][1][0],
        }
    elif len(char_anchor_pairs) == 1:
        char_obj, anchors = char_anchor_pairs[0]
        gender_tag = "1girl" if (char_obj and char_obj.gender == "female") else "1boy"
        # Inject DNA tags (hair/eyes/face/body) to reinforce IPAdapter face match
        dna_text = _extract_dna_tags(char_obj.description) if char_obj else ""
        # Inject clothing tags with weight 1.2 to lock outfit matching anchor
        clothing_text = _extract_clothing_tags(char_obj.description) if char_obj else ""
        # Scene-first ordering: background/environment gets higher priority than character tags.
        # Wrapping scene in (…:1.2) prevents IPAdapter from crowding out the background.
        weighted_scene = f"({prompt_text}:1.2)" if prompt_text else ""
        parts = [p for p in [weighted_scene, gender_tag, "solo", clothing_text, dna_text] if p]
        scene_prompt = ", ".join(parts)
        workflow = "image_gen/workflows/txt2img_ipadapter.json"
        replacements = {
            "SCENE_PROMPT": scene_prompt,
            "NEGATIVE_PROMPT": negative_base + _ANTI_SPLIT_SINGLE,
            "WIDTH": settings.image_width,
            "HEIGHT": settings.image_height,
            "SEED": seed,
            "ANCHOR_PATH": anchors[0],
        }
        if len(anchors) > 1:
            replacements["ANCHOR_PATH_2"] = anchors[1]
            workflow = "image_gen/workflows/txt2img_ipadapter_multiref.json"
        if len(anchors) > 2:
            replacements["ANCHOR_PATH_3"] = anchors[2]
    else:
        workflow = "image_gen/workflows/txt2img_scene.json"
        replacements = {
            "SCENE_PROMPT": prompt_text,
            "NEGATIVE_PROMPT": negative_base + _ANTI_SPLIT_SINGLE,
            "WIDTH": settings.image_width,
            "HEIGHT": settings.image_height,
            "SEED": seed,
        }

    return workflow, replacements


def run_images(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.scriptwriter import load_episode_script
    from llm.character_extractor import load_all_characters
    from image_gen.comfyui_client import comfyui_client
    from image_gen.character_gen import generate_character_anchors
    from video.frame_decomposer import decompose_all_shots

    if dry_run:
        logger.info("[dry-run] Skipping images | episode={}", episode_num)
        db.set_episode_status(episode_num, "IMAGES_DONE")
        return

    vram_manager.unload_ollama()
    vram_manager.health_check_comfyui()

    # Ensure character anchors exist (idempotent)
    generate_character_anchors()

    script = load_episode_script(episode_num)
    # Decompose shots into multi-frame structure
    script = script.model_copy(update={"shots": decompose_all_shots(script.shots)})
    images_dir = Path(settings.data_dir) / "images" / f"episode-{episode_num:03d}"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build name→Character map for shared helpers
    characters_map = {c.name: c for c in load_all_characters()}

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
            seed = episode_num * 10000 + idx * 100 + fidx

            workflow, replacements = _build_shot_image_params(
                prompt_text, char_anchor_pairs, seed
            )

            try:
                comfyui_client.generate_image(workflow, replacements, output_path)
            except Exception as exc:
                err_msg = str(exc).lower()
                is_ipadapter = workflow != "image_gen/workflows/txt2img_scene.json"
                is_model_error = "model not found" in err_msg or "ipadapter" in err_msg

                if is_ipadapter and is_model_error:
                    logger.warning(
                        "IPAdapter model error, flushing ComfyUI VRAM and retrying | "
                        "episode={} shot={} frame={} error={}",
                        episode_num, idx, fidx, str(exc)[:200],
                    )
                    vram_manager.unload_comfyui()
                    try:
                        comfyui_client.generate_image(workflow, replacements, output_path)
                    except Exception:
                        logger.error(
                            "IPAdapter retry failed, falling back to txt2img_scene | "
                            "episode={} shot={} frame={}",
                            episode_num, idx, fidx,
                        )
                        _generate_scene_fallback(comfyui_client, prompt_text, seed, output_path)
                elif is_ipadapter:
                    logger.warning(
                        "IPAdapter workflow failed, falling back to txt2img_scene | "
                        "episode={} shot={} frame={} error={}",
                        episode_num, idx, fidx, str(exc)[:200],
                    )
                    _generate_scene_fallback(comfyui_client, prompt_text, seed, output_path)
                else:
                    raise
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

    comfyui_client.generate_image(
        workflow_path="image_gen/workflows/thumbnail.json",
        replacements={
            "SCENE_PROMPT": f"{shot.scene_prompt}, wide cinematic shot",
            "NEGATIVE_PROMPT": "lowres, bad quality, blurry",
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
    "crawl": run_crawl,
    "llm": run_llm,
    "images": run_images,
    "audio": run_audio,
    "video": run_video,
    "validate": run_validate,
}


# ── Episode / pipeline entry points ───────────────────────────────────────────

def run_episode(
    episode_num: int,
    from_phase: str = "crawl",
    dry_run: bool = False,
) -> None:
    db = StateDB()
    setup_logging(episode_num)

    current_status = db.get_episode_status(episode_num)
    if from_phase != "crawl" and current_status:
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
                vram_manager.unload_ollama()
            except Exception as vram_exc:
                logger.warning("Ollama unload failed during cleanup | error={}", vram_exc)
            try:
                vram_manager.unload_comfyui()
            except Exception as vram_exc:
                logger.warning("ComfyUI unload failed during cleanup | error={}", vram_exc)
            raise


def run_pipeline(
    from_episode: int = 1,
    from_phase: str = "crawl",
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

        # Only first episode may start from non-crawl phase
        from_phase = "crawl"


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

    vram_manager.unload_ollama()
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
        default="crawl",
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
