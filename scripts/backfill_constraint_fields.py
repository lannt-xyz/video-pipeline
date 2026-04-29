"""Backfill `energy_level`, `proper_nouns`, `exposition_ratio` on existing scripts.

Walks `data/{slug}/scripts/*.json` (or supplied glob), runs
`populate_shot_signals` on every shot whose constraint fields are still `None`
sentinels, and writes back. Idempotent.

Usage:
    uv run python scripts/backfill_constraint_fields.py            # dry-run, current story
    uv run python scripts/backfill_constraint_fields.py --apply    # write back
    uv run python scripts/backfill_constraint_fields.py --slug other-story --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Allow running as a script from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings  # noqa: E402
from llm.constraint_validator import populate_shot_signals  # noqa: E402
from models.schemas import EpisodeScript  # noqa: E402


def _load_known_chars(slug: str) -> List[str]:
    """Best-effort load character names from `data/{slug}/characters/*.json`."""
    chars_dir = Path("data") / slug / "characters"
    if not chars_dir.exists():
        return []
    names: List[str] = []
    for p in chars_dir.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("display_name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("display_name")
                        if isinstance(name, str) and name.strip():
                            names.append(name.strip())
        except (json.JSONDecodeError, OSError):
            continue
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=None, help="story slug (default: settings.story_slug)")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    parser.add_argument(
        "--scripts-dir", default=None,
        help="explicit scripts directory (overrides slug-derived path)",
    )
    args = parser.parse_args()

    settings = Settings()
    slug = args.slug or settings.story_slug

    if args.scripts_dir:
        scripts_dir = Path(args.scripts_dir)
    else:
        scripts_dir = Path("data") / slug / "scripts"

    if not scripts_dir.exists():
        print(f"[backfill] scripts dir not found: {scripts_dir}")
        return 0

    files = sorted(scripts_dir.glob("episode-*-script.json"))
    if not files:
        print(f"[backfill] no episode scripts found in {scripts_dir}")
        return 0

    known_chars = _load_known_chars(slug)
    print(f"[backfill] story={slug}  scripts={len(files)}  known_chars={len(known_chars)}  apply={args.apply}")

    total_shots = 0
    total_filled = 0
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        episode = EpisodeScript.model_validate(raw)
        per_file_filled = 0
        for shot in episode.shots:
            had_none = (
                shot.energy_level is None
                or shot.proper_nouns is None
                or shot.exposition_ratio is None
            )
            populate_shot_signals(shot, known_chars)
            if had_none:
                per_file_filled += 1
        total_shots += len(episode.shots)
        total_filled += per_file_filled

        if args.apply and per_file_filled > 0:
            path.write_text(
                json.dumps(episode.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tag = "WROTE"
        elif per_file_filled > 0:
            tag = "DRY  "
        else:
            tag = "SKIP "
        print(f"  [{tag}] {path.name}  filled={per_file_filled}/{len(episode.shots)}")

    print(f"[backfill] total: filled={total_filled}/{total_shots}  apply={args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
