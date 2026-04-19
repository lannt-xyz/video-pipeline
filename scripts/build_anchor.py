"""Standalone script: build visual-anchor profile + images for one or more characters.

Flow: wiki DB → Markdown → LLM tags → JSON profile → ComfyUI anchor images

Usage:
    uv run python scripts/build_anchor.py <character_id> [<character_id> ...] [--force]
    uv run python scripts/build_anchor.py diep_thieu_duong
    uv run python scripts/build_anchor.py diep_thieu_duong thanh_van_tu --force

Output files: data/<story_slug>/characters/
  <id>.md          — Markdown profile (review step)
  <id>.raw.json    — raw LLM output
  <id>.json        — validated profile with anchor_path
  <slug>/anchor.png, anchor_3q.png, anchor_side.png
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from config.settings import settings
from image_gen.character_gen import _generate_single_anchor
from llm.character_extractor import _sanitize_description
from llm.profile_builder import _derive_tags, _open_db, _sanitize_unknown, build_markdown
from models.schemas import Character


def build_anchor(char_id: str, chars_dir: Path, con, *, force: bool = False) -> bool:
    """Build profile + anchor images for a single character. Returns True on success."""
    char_dir = chars_dir / char_id
    char_dir.mkdir(parents=True, exist_ok=True)
    json_path = char_dir / "profile.json"
    md_path = char_dir / "profile.md"
    raw_path = char_dir / "profile.raw.json"

    # --- Step 1: Markdown profile ---
    if json_path.exists() and not force:
        logger.info("Profile exists, loading for image gen | id={}", char_id)
        char = Character(**json.loads(json_path.read_text(encoding="utf-8")))
    else:
        md = build_markdown(char_id, con)
        if md is None:
            logger.error("No visual data or character not found | id={}", char_id)
            return False
        md_path.write_text(md, encoding="utf-8")
        logger.info("Saved markdown | {}", md_path.name)

        # --- Step 2: LLM tag derivation ---
        raw = _derive_tags(md)
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved raw LLM output | {}", raw_path.name)

        # --- Step 3: Validate + sanitize ---
        raw.setdefault("name", char_id)
        raw.setdefault("alias", [])
        raw.setdefault("gender", "unknown")
        raw.setdefault("description", "")
        raw.setdefault("relationships", {})

        char = Character(**raw)
        if char.gender == "unknown":
            char.description = _sanitize_unknown(char.description)
        elif char.gender in ("male", "female"):
            char.description = _sanitize_description(char.description, char.gender)
        else:
            logger.warning("Unexpected gender '{}' — treating as unknown", char.gender)
            char.gender = "unknown"
            char.description = _sanitize_unknown(char.description)

        json_path.write_text(char.model_dump_json(indent=2), encoding="utf-8")
        logger.success("Profile built | id={} gender={}", char_id, char.gender)

    # --- Step 4: ComfyUI anchor images ---
    logger.info("Generating anchor images via ComfyUI | id={}", char_id)
    try:
        _generate_single_anchor(char, force=force)
    except Exception as exc:
        logger.error("ComfyUI anchor generation failed | id={} error={}", char_id, exc)
        return False

    return True


def main() -> None:
    args = sys.argv[1:]
    force = "--force" in args
    char_ids = [a for a in args if not a.startswith("--")]

    if not char_ids:
        print("Usage: uv run python scripts/build_anchor.py <character_id> [--force]")
        print("Example: uv run python scripts/build_anchor.py diep_thieu_duong")
        sys.exit(1)

    db_path = settings.db_path
    if not Path(db_path).exists():
        flat = Path("data") / f"{settings.story_slug}.db"
        db_path = str(flat) if flat.exists() else db_path

    chars_dir = Path(settings.data_dir) / "characters"
    chars_dir.mkdir(parents=True, exist_ok=True)

    ok_list: list[str] = []
    fail_list: list[str] = []

    with _open_db(db_path) as con:
        for char_id in char_ids:
            if build_anchor(char_id, chars_dir, con, force=force):
                ok_list.append(char_id)
            else:
                fail_list.append(char_id)

    print(f"\nDone: {len(ok_list)} ok, {len(fail_list)} failed")
    if fail_list:
        print("Failed:", ", ".join(fail_list))
        sys.exit(1)


if __name__ == "__main__":
    main()
