import argparse
import shutil
import sys
from pathlib import Path

from loguru import logger

from config.settings import settings
from pipeline.state import StateDB
from pipeline.validator import ValidationError, validator
from pipeline.vram_manager import vram_manager

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
    """Alias-aware lookup — match on canonical name or any alias."""
    for char_obj in characters_map.values():
        if llm_name == char_obj.name or llm_name in char_obj.alias:
            return char_obj
    return None


def _extract_dna_tags(description: str) -> str:
    """Keep only identity tags (gender/hair/eyes/face/body) — drop outfit/accessory/background tags."""
    tags = [t.strip() for t in description.split(",") if t.strip()]
    dna = [t for t in tags if not any(kw in t.lower() for kw in _OUTFIT_KEYWORDS)]
    return ", ".join(dna)


def run_images(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.scriptwriter import load_episode_script
    from llm.character_extractor import load_all_characters
    from image_gen.comfyui_client import comfyui_client
    from image_gen.character_gen import generate_character_anchors, get_anchor_paths
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

    # Build name→description map for reinforcing IP-Adapter with text tags
    characters_map = {c.name: c for c in load_all_characters()}

    _negative_base = (
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

    db.record_phase_start(episode_num, "images")

    for idx, shot in enumerate(script.shots):
        # Resolve character anchors once per shot (shared across all frames)
        char_anchor_pairs = []
        for name in shot.characters[:2]:
            char_obj = _find_character(name, characters_map)
            anchors = get_anchor_paths(char_obj.name if char_obj else name)
            if anchors:
                char_anchor_pairs.append((char_obj, anchors))
            else:
                logger.warning(
                    "Anchor missing, skipping IPAdapter for this char | episode={} shot={} char={}",
                    episode_num, idx, char_obj.name if char_obj else name,
                )

        # Gender-aware negative: block male anatomy on female characters
        has_female = any(c is not None and c.gender == "female" for c, _ in char_anchor_pairs)
        negative = (
            _negative_base + ", (male:1.5), (masculine:1.3), 1boy"
            if has_female
            else _negative_base
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
            # Unique seed per frame to ensure visual variety
            seed = episode_num * 10000 + idx * 100 + fidx

            if len(char_anchor_pairs) >= 2:
                genders = [c.gender if c else "male" for c, _ in char_anchor_pairs]
                count_tag = (
                    "2boys" if all(g == "male" for g in genders)
                    else "2girls" if all(g == "female" for g in genders)
                    else "1boy, 1girl"
                )
                workflow = "image_gen/workflows/txt2img_ipadapter_dual.json"
                replacements = {
                    "SCENE_PROMPT": f"{count_tag}, {prompt_text}",
                    "NEGATIVE_PROMPT": negative,
                    "WIDTH": settings.image_width,
                    "HEIGHT": settings.image_height,
                    "SEED": seed,
                    "ANCHOR_PATH": char_anchor_pairs[0][1][0],
                    "ANCHOR_PATH_2": char_anchor_pairs[1][1][0],
                }
            elif len(char_anchor_pairs) == 1:
                char_obj, anchors = char_anchor_pairs[0]
                gender_tag = "1girl" if (char_obj and char_obj.gender == "female") else "1boy"
                workflow = "image_gen/workflows/txt2img_ipadapter.json"
                replacements = {
                    "SCENE_PROMPT": f"{gender_tag}, {prompt_text}",
                    "NEGATIVE_PROMPT": negative,
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
                    "NEGATIVE_PROMPT": negative,
                    "WIDTH": settings.image_width,
                    "HEIGHT": settings.image_height,
                    "SEED": seed,
                }

            comfyui_client.generate_image(workflow, replacements, output_path)
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

    # Patch duration_sec with actual AAC duration so that zoompan and subtitle
    # timing match the real TTS length rather than the LLM-estimated value.
    for shot, audio_path in zip(script.shots, audio_paths):
        if audio_path.exists():
            actual = _probe_duration(audio_path)
            if actual > 0:
                shot.duration_sec = actual  # preserve exact float, no rounding

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
    args = parser.parse_args()

    if args.episode:
        run_episode(args.episode, from_phase=args.from_phase, dry_run=args.dry_run)
    else:
        run_pipeline(
            from_episode=args.from_episode,
            from_phase=args.from_phase,
            dry_run=args.dry_run,
        )
