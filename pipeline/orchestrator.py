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

    already = set(db.get_crawled_chapters(chapter_start, chapter_end))
    to_crawl = [n for n in range(chapter_start, chapter_end + 1) if n not in already]

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


def run_images(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.scriptwriter import load_episode_script
    from image_gen.comfyui_client import comfyui_client
    from image_gen.character_gen import generate_character_anchors, get_anchor_path

    if dry_run:
        logger.info("[dry-run] Skipping images | episode={}", episode_num)
        db.set_episode_status(episode_num, "IMAGES_DONE")
        return

    vram_manager.health_check_comfyui()
    vram_manager.unload_ollama()

    # Ensure character anchors exist (idempotent)
    generate_character_anchors()

    script = load_episode_script(episode_num)
    images_dir = Path(settings.data_dir) / "images" / f"episode-{episode_num:03d}"
    images_dir.mkdir(parents=True, exist_ok=True)

    negative = (
        "lowres, bad anatomy, bad hands, text, error,"
        " worst quality, low quality, blurry, score_1, score_2"
    )

    db.record_phase_start(episode_num, "images")

    for idx, shot in enumerate(script.shots):
        output_path = images_dir / f"shot-{idx:02d}.png"
        if output_path.exists():
            logger.debug("Image exists, skipping | episode={} shot={}", episode_num, idx)
            continue

        if shot.characters:
            anchor = get_anchor_path(shot.characters[0])
            workflow = "image_gen/workflows/txt2img_ipadapter.json"
            replacements = {
                "SCENE_PROMPT": shot.scene_prompt,
                "NEGATIVE_PROMPT": negative,
                "WIDTH": settings.image_width,
                "HEIGHT": settings.image_height,
                "SEED": episode_num * 1000 + idx,
                "ANCHOR_PATH": anchor,
            }
        else:
            workflow = "image_gen/workflows/txt2img_scene.json"
            replacements = {
                "SCENE_PROMPT": shot.scene_prompt,
                "NEGATIVE_PROMPT": negative,
                "WIDTH": settings.image_width,
                "HEIGHT": settings.image_height,
                "SEED": episode_num * 1000 + idx,
            }

        comfyui_client.generate_image(workflow, replacements, output_path)
        logger.info("Image generated | episode={} shot={}", episode_num, idx)

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


def run_video(episode_num: int, db: StateDB, dry_run: bool = False) -> None:
    from llm.scriptwriter import load_episode_script
    from video.assembler import assemble_shot_clips
    from video.editor import assemble_episode

    if dry_run:
        logger.info("[dry-run] Skipping video | episode={}", episode_num)
        db.set_episode_status(episode_num, "VIDEO_DONE")
        return

    script = load_episode_script(episode_num)
    audio_dir = Path(settings.data_dir) / "audio" / f"episode-{episode_num:03d}"
    audio_paths = [
        audio_dir / f"shot-{i:02d}-mixed.aac" for i in range(len(script.shots))
    ]

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
