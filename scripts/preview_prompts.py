#!/usr/bin/env python3
"""Preview the final ComfyUI workflow JSON for shots or character anchors.

Usage:
    # Preview all shots of episode 1
    python scripts/preview_prompts.py --episode 1

    # Preview specific shots + print positive prompt
    python scripts/preview_prompts.py --episode 1 --shots 0 2 5 --print

    # Preview anchor workflows for all characters
    python scripts/preview_prompts.py --anchors

    # Preview anchor for specific characters only
    python scripts/preview_prompts.py --anchors --chars "Diệp Binh" "Thanh Vân Tử"

Outputs resolved workflow JSONs to logs/comfyui_prompts/preview/
No ComfyUI calls — purely local resolution.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config.settings import settings
from image_gen.character_gen import _ANCHOR_WORKFLOW, _NEGATIVE, _slugify
from image_gen.comfyui_client import ComfyUIClient
from llm.character_extractor import load_all_characters
from llm.scriptwriter import load_episode_script
from pipeline.orchestrator import (
    _build_shot_image_params,
    _resolve_char_anchor_pairs,
)
from video.frame_decomposer import decompose_all_shots

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}", colorize=True)


def preview_anchors(char_filter: list[str] | None, print_json: bool) -> None:
    """Resolve and save anchor workflow JSONs for all (or filtered) characters."""
    characters = load_all_characters()
    if not characters:
        logger.error("No characters found — run LLM phase first")
        sys.exit(1)

    if char_filter:
        characters = [c for c in characters if c.name in char_filter]
        if not characters:
            logger.error("None of the requested characters found: {}", char_filter)
            sys.exit(1)

    client = ComfyUIClient()
    out_dir = Path("logs/comfyui_prompts/preview/anchors")
    out_dir.mkdir(parents=True, exist_ok=True)

    _VIEWS = [
        ("anchor.png",       "looking at viewer, front view"),
        ("anchor_3q.png",    "looking slightly to the side, three-quarter view"),
        ("anchor_side.png",  "looking to the side, profile view, side angle"),
    ]

    for character in characters:
        base_seed = int(hashlib.md5(character.name.encode()).hexdigest(), 16) % (2**32)
        char_slug = _slugify(character.name)

        for vi, (filename, angle_tags) in enumerate(_VIEWS):
            scene_prompt = (
                f"{character.description}, "
                f"close-up portrait, face focus, head and shoulders only, {angle_tags}, "
                "anime style, plain background, "
                "detailed face, high quality, masterpiece, best quality, ultra detailed"
            )
            replacements = {
                "SCENE_PROMPT": scene_prompt,
                "NEGATIVE_PROMPT": _NEGATIVE,
                "WIDTH": 768,
                "HEIGHT": 768,
                "SEED": base_seed + vi,
            }
            workflow = client._load_workflow(_ANCHOR_WORKFLOW, replacements)

            out_file = out_dir / f"{char_slug}_{Path(filename).stem}.json"
            out_file.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")

            logger.info(
                "anchor | char={} view={} | positive={}",
                character.name, filename,
                scene_prompt[:80] + ("…" if len(scene_prompt) > 80 else ""),
            )

            if print_json:
                print(f"\n{'='*60}")
                print(f"Anchor: {character.name} — {filename}")
                print("=" * 60)
                print(f"POSITIVE PROMPT:\n{scene_prompt}\n")

    print(f"\nSaved to: {out_dir}")
    print(f"Total files: {len(list(out_dir.glob('*.json')))}")


def preview(episode_num: int, shot_indices: list[int] | None, print_json: bool) -> None:
    try:
        script = load_episode_script(episode_num)
    except FileNotFoundError:
        logger.error("No script found for episode {} — run LLM phase first", episode_num)
        sys.exit(1)

    # Decompose into frames (same as run_images does)
    script = script.model_copy(update={"shots": decompose_all_shots(script.shots)})

    characters_map = {c.name: c for c in load_all_characters()}

    # Client instance — only used to call _load_workflow (no HTTP calls)
    client = ComfyUIClient()

    out_dir = Path("logs/comfyui_prompts/preview") / f"episode-{episode_num:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shots_to_preview = (
        [script.shots[i] for i in shot_indices if i < len(script.shots)]
        if shot_indices is not None
        else script.shots
    )
    indices_to_preview = (
        [i for i in shot_indices if i < len(script.shots)]
        if shot_indices is not None
        else list(range(len(script.shots)))
    )

    for idx, shot in zip(indices_to_preview, shots_to_preview):
        char_anchor_pairs = (
            []
            if not shot.characters
            else _resolve_char_anchor_pairs(shot.characters, characters_map)
        )

        frames = shot.frames if shot.frames else [None]
        for fidx, frame in enumerate(frames):
            prompt_text = frame.scene_prompt if frame else shot.scene_prompt
            seed = episode_num * 10000 + idx * 100 + fidx

            workflow_path, replacements = _build_shot_image_params(
                prompt_text, char_anchor_pairs, seed
            )

            # Resolve Path values to their uploaded names (same logic as generate_image)
            resolved: dict = {}
            for key, value in replacements.items():
                if isinstance(value, Path):
                    # Use same naming convention as upload_image
                    unique_name = f"{value.parent.name}_{value.name}" if value.parent.name else value.name
                    resolved[key] = unique_name
                else:
                    resolved[key] = value

            workflow = client._load_workflow(workflow_path, resolved)

            label = f"shot-{idx:02d}-frame-{fidx:02d}" if len(frames) > 1 else f"shot-{idx:02d}"
            out_file = out_dir / f"{label}.json"
            out_file.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")

            # Summary line
            wf_name = Path(workflow_path).stem
            char_names = shot.characters or ["(scene only)"]
            logger.info(
                "shot={:02d} frame={} | workflow={} | chars={} | prompt_preview={}",
                idx, fidx, wf_name, char_names,
                prompt_text[:80] + ("…" if len(prompt_text) > 80 else ""),
            )

            if print_json:
                print(f"\n{'='*60}")
                print(f"Shot {idx:02d} Frame {fidx} — {wf_name}")
                print("=" * 60)

                # Print only the positive prompt node for readability
                for node in workflow.values():
                    if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                        text = node["inputs"].get("text", "")
                        if "__NEGATIVE" not in text and "score_1" not in text[:30]:
                            print(f"POSITIVE PROMPT:\n{text}\n")
                            break

    print(f"\nSaved to: {out_dir}")
    print(f"Total files: {len(list(out_dir.glob('*.json')))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview ComfyUI prompts without running pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", type=int, metavar="N", help="Preview shots for episode N")
    group.add_argument("--anchors", action="store_true", help="Preview anchor workflows for characters")
    parser.add_argument(
        "--shots", type=int, nargs="+", metavar="K",
        help="(with --episode) Specific shot indices to preview (default: all)",
    )
    parser.add_argument(
        "--chars", type=str, nargs="+", metavar="NAME",
        help="(with --anchors) Specific character names to preview (default: all)",
    )
    parser.add_argument(
        "--print", dest="print_json", action="store_true",
        help="Also print positive prompts to stdout",
    )
    args = parser.parse_args()

    if args.anchors:
        preview_anchors(args.chars, args.print_json)
    else:
        preview(args.episode, args.shots, args.print_json)


if __name__ == "__main__":
    main()
