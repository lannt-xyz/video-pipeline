"""Generate a Markdown alignment report for an episode script.

The report maps each shot's narration → brief.actions → final scene_prompt →
characters → image path so reviewers can visually verify image-narration
alignment before or after image generation.

Usage:
    from pipeline.alignment_report import write_alignment_report
    write_alignment_report(script, episode_num)
"""
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings
from models.schemas import EpisodeScript


def write_alignment_report(script: EpisodeScript, episode_num: int) -> Path:
    """Write episode-XXX-alignment.md to the scripts directory.

    Returns the path to the written file.
    """
    data_dir = Path(settings.data_dir)
    scripts_dir = data_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    images_dir = data_dir / "images" / f"episode-{episode_num:03d}"
    report_path = scripts_dir / f"episode-{episode_num:03d}-alignment.md"

    lines: list[str] = [
        f"# Alignment Report — Episode {episode_num:03d}",
        f"**Story**: {settings.story_slug}  ",
        f"**Shots**: {len(script.shots)}",
        "",
        "| Shot | Duration | Characters (resolved) | Characters (raw LLM) | Narration | Brief actions | Final scene_prompt | Image |",
        "|------|----------|-----------------------|----------------------|-----------|--------------|-------------------|-------|",
    ]

    for i, shot in enumerate(script.shots):
        duration = f"{shot.duration_sec:.0f}s"
        chars = ", ".join(shot.characters) if shot.characters else "—"
        chars_raw = (
            ", ".join(shot.characters_raw)
            if shot.characters_raw is not None
            else "_not resolved_"
        )
        narration = _escape_md(shot.narration_text[:120])
        scene_prompt_short = _escape_md(shot.scene_prompt[:120])

        if shot.visual_brief and shot.visual_brief.actions:
            actions_md = "; ".join(
                _escape_md(a[:80]) for a in shot.visual_brief.actions
            )
        else:
            actions_md = "_no brief_"

        # Determine image path (frame-0 preferred, fallback to shot-only)
        image_cell = _image_cell(images_dir, i)

        lines.append(
            f"| {i} | {duration} | {chars} | {chars_raw} | {narration} "
            f"| {actions_md} | {scene_prompt_short} | {image_cell} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Shot Detail",
        "",
    ]

    for i, shot in enumerate(script.shots):
        lines += [
            f"### Shot {i}",
            f"**Narration**: {shot.narration_text}",
            "",
        ]

        if shot.visual_brief:
            brief = shot.visual_brief
            actions_detail = "\n".join(
                f"  - `{a}`" for a in brief.actions
            ) if brief.actions else "  - _none_"
            lines += [
                "**Visual Brief**:",
                f"- setting: `{brief.setting}`",
                f"- actions:",
                actions_detail,
                f"- mood_lighting: `{brief.mood_lighting}`",
                f"- key_objects: {brief.key_objects}",
                f"- subjects: {brief.subjects}",
                "",
            ]
        else:
            lines += ["**Visual Brief**: _not extracted_", ""]

        lines += [
            f"**Characters (resolved)**: {shot.characters}",
            f"**Characters (raw LLM)**: {shot.characters_raw}",
            f"**scene_id**: `{shot.scene_id or '—'}`",
            f"**camera_flow**: `{shot.camera_flow.value}`",
            "",
            f"**Final scene_prompt**:",
            f"```",
            shot.scene_prompt,
            f"```",
            "",
        ]

        # Per-frame detail
        if shot.frames:
            lines.append("**Frames**:")
            for fidx, frame in enumerate(shot.frames):
                image_cell = _image_path_str(images_dir, i, fidx, len(shot.frames))
                lines.append(f"- Frame {fidx} ({frame.camera_tag}) — `{image_cell}`")
                lines.append(f"  ```")
                lines.append(f"  {frame.scene_prompt}")
                lines.append(f"  ```")
            lines.append("")

        lines.append("---")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "Alignment report written | episode={} path={}",
        episode_num, report_path,
    )
    return report_path


def _escape_md(text: str) -> str:
    """Escape pipe characters so they don't break the Markdown table."""
    return text.replace("|", "\\|")


def _image_cell(images_dir: Path, shot_idx: int) -> str:
    """Return a short Markdown image reference or placeholder."""
    # Try multi-frame path first (frame-00)
    frame0 = images_dir / f"shot-{shot_idx:02d}-frame-00.png"
    if frame0.exists():
        return f"![f0]({frame0})"
    # Try single-frame path
    single = images_dir / f"shot-{shot_idx:02d}.png"
    if single.exists():
        return f"![img]({single})"
    return "_not generated_"


def _image_path_str(images_dir: Path, shot_idx: int, frame_idx: int, num_frames: int) -> str:
    if num_frames > 1:
        p = images_dir / f"shot-{shot_idx:02d}-frame-{frame_idx:02d}.png"
    else:
        p = images_dir / f"shot-{shot_idx:02d}.png"
    return str(p) if p.exists() else str(p) + " (not generated)"
