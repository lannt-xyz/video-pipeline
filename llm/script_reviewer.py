"""Script quality reviewer — runs after the LLM phase, before image generation.

Two-pass approach:
  Pass 1 (rule-based): fast heuristic checks — word count, hook length, shot count,
                        narration language, character name validity.
  Pass 2 (LLM fix): rewrite only the shots that failed Pass 1; others are untouched.

Entry point: review_episode_script(episode_num) -> EpisodeScript
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import script_client
from models.schemas import EpisodeScript, ShotScript

# ── Rule thresholds ──────────────────────────────────────────────────────────

_MIN_TOTAL_WORDS = 170          # TTS must fill ≥65s at ~2.5 wps (F5-TTS VI speed=1.15) — 8s margin over 57s validator floor
_MIN_HOOK_WORDS = 2             # shot 0 must have at least a sentence
_MAX_HOOK_WORDS = 10            # shot 0 is a 2-3s hook — keep it short
_MIN_SHOT_WORDS = 26            # shots 2+ (index ≥ 2) at 8s need ≥ 26 words at 2.5 wps with margin
_MIN_SHOT_COUNT = 8             # expected shot count
_MAX_SHOT_COUNT = 10

# Clues that a shot has a horror/atmosphere tag (Quick scan — not exhaustive)
_HORROR_TAGS = frozenset([
    "glow", "shadow", "mist", "fog", "moonlight", "candle", "flicker",
    "creep", "cold", "pale", "eerie", "ghostly", "dark", "ominous",
    "blood", "dread", "spectral", "spirit", "distorted", "dim",
])

# Vietnamese diacritic regex — if narration has almost no Vietnamese chars it's foreign-language
_VIET_DIACRITIC_RE = re.compile(
    r"[àáảãạăắặẳẵằâầấậẩẫđèéẻẽẹêềếệểễìíỉĩịòóỏõọôồốộổỗơờớợởỡùúủũụưừứựửữỳýỷỹỵÀÁẢÃẠĂẮẶẲẴẰÂẦẤẬẨẪĐÈÉẺẼẸÊỀẾỆỂỄÌÍỈĨỊÒÓỎÕỌÔỒỐỘỔỖƠỜỚỢỞỠÙÚỦŨỤƯỪỨỰỬỮỲÝỶỸỴ]"
)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ReviewIssue:
    shot_index: int     # 0-based index; -1 = episode-level issue
    issue_type: str     # e.g. "narration_too_short", "hook_too_long"
    description: str    # human-readable detail


@dataclass
class ReviewReport:
    episode_num: int
    issues: List[ReviewIssue] = field(default_factory=list)
    fixed_shots: List[int] = field(default_factory=list)   # indices that were rewritten

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    def summary(self) -> str:
        if not self.issues:
            return "No issues found."
        lines = [f"  [{i.shot_index}] {i.issue_type}: {i.description}" for i in self.issues]
        return "\n".join(lines)


# ── Pass 1: rule-based checks ─────────────────────────────────────────────────

def _rule_check(script: EpisodeScript) -> List[ReviewIssue]:
    issues: List[ReviewIssue] = []
    shots = script.shots

    # Episode-level: shot count
    if len(shots) < _MIN_SHOT_COUNT:
        issues.append(ReviewIssue(
            shot_index=-1,
            issue_type="shot_count_low",
            description=f"Expected {_MIN_SHOT_COUNT} shots, got {len(shots)}",
        ))
    elif len(shots) > _MAX_SHOT_COUNT:
        issues.append(ReviewIssue(
            shot_index=-1,
            issue_type="shot_count_high",
            description=f"Expected ≤{_MAX_SHOT_COUNT} shots, got {len(shots)}",
        ))

    # Episode-level: total narration
    total_words = sum(len(s.narration_text.split()) for s in shots)
    if total_words < _MIN_TOTAL_WORDS:
        issues.append(ReviewIssue(
            shot_index=-1,
            issue_type="total_words_low",
            description=f"Total narration={total_words} words < {_MIN_TOTAL_WORDS} minimum",
        ))

    for idx, shot in enumerate(shots):
        words = len(shot.narration_text.split())
        # Shot 0 (hook) — must be short and punchy
        if idx == 0 and words > _MAX_HOOK_WORDS:
            issues.append(ReviewIssue(
                shot_index=idx,
                issue_type="hook_too_long",
                description=f"Hook narration={words} words > {_MAX_HOOK_WORDS} max (hook must be ≤10 words)",
            ))
        # Shot 1 (second hook) — also kept short
        if idx == 1 and words > _MAX_HOOK_WORDS:
            issues.append(ReviewIssue(
                shot_index=idx,
                issue_type="hook_too_long",
                description=f"Second hook narration={words} words > {_MAX_HOOK_WORDS} max",
            ))
        # Shots 2+ — must have enough content to fill ~8s of TTS
        if idx >= 2 and words < _MIN_SHOT_WORDS:
            issues.append(ReviewIssue(
                shot_index=idx,
                issue_type="narration_too_short",
                description=f"Shot {idx+1} narration={words} words < {_MIN_SHOT_WORDS} minimum for 8s shot at 2.5 wps",
            ))
        # scene_prompt horror atmosphere check (soft warn — not auto-fixed)
        prompt_lower = shot.scene_prompt.lower()
        has_horror = any(tag in prompt_lower for tag in _HORROR_TAGS)
        if not has_horror:
            issues.append(ReviewIssue(
                shot_index=idx,
                issue_type="scene_prompt_no_horror",
                description=f"Shot {idx+1} scene_prompt has no horror/atmosphere tag",
            ))

        # Narration language check: hook is always in Vietnamese (except empty hooks)
        # Any shot's narration with no Vietnamese diacritics and ≥4 words is likely foreign-language
        narration_word_count = len(shot.narration_text.split())
        if narration_word_count >= 4:
            viet_chars = len(_VIET_DIACRITIC_RE.findall(shot.narration_text))
            ascii_word_chars = len(re.findall(r"[a-zA-Z]", shot.narration_text))
            # Flag if there are ASCII letters but essentially no Vietnamese diacritics
            if ascii_word_chars > 10 and viet_chars == 0:
                issues.append(ReviewIssue(
                    shot_index=idx,
                    issue_type="narration_language_wrong",
                    description=(
                        f"Shot {idx+1} narration appears to be non-Vietnamese "
                        f"(0 diacritic chars, {ascii_word_chars} ASCII chars): "
                        f"\"{shot.narration_text[:60]}\""
                    ),
                ))

    # Character name validation: load known characters and flag names not matching any
    _check_character_names(script, issues)

    return issues


def _check_character_names(script: EpisodeScript, issues: List[ReviewIssue]) -> None:
    """Flag character names in shots that don't match any known character in the data dir."""
    story_slug = settings.story_slug
    characters_file = Path(settings.data_dir) / story_slug / "characters.json"
    if not characters_file.exists():
        return  # No character file yet — skip check

    try:
        raw = json.loads(characters_file.read_text(encoding="utf-8"))
    except Exception:
        return

    # Collect all known names (primary + aliases if present)
    known_names: set[str] = set()
    char_list = raw if isinstance(raw, list) else raw.get("characters", [])
    for char in char_list:
        if isinstance(char, dict):
            if name := char.get("name"):
                known_names.add(name.strip())
            for alias in char.get("aliases", []):
                known_names.add(alias.strip())

    if not known_names:
        return

    for idx, shot in enumerate(script.shots):
        for char_name in shot.characters:
            char_name = char_name.strip()
            if not char_name:
                continue
            # Exact match
            if char_name in known_names:
                continue
            # Fuzzy: accept if char_name is a strict prefix/substring of a known name (handles short aliases)
            fuzzy_match = any(
                char_name in k or k in char_name for k in known_names
            )
            if not fuzzy_match:
                issues.append(ReviewIssue(
                    shot_index=idx,
                    issue_type="character_name_typo",
                    description=(
                        f"Shot {idx+1} character \"{char_name}\" not found in known characters. "
                        f"Known: {sorted(known_names)}"
                    ),
                ))


# ── Pass 2: LLM rewrite ───────────────────────────────────────────────────────

_FIX_SYSTEM = """You are a Vietnamese horror/supernatural short video scriptwriter reviewing and fixing shot scripts.
You will receive a list of shots that FAILED quality checks.

For each shot, rewrite ONLY the field that failed:
- "narration_too_short": expand narration_text to at least 26-36 words (3 full Vietnamese sentences, maintain horror/tension, same character/event, present-tense first-person narrator)
- "hook_too_long": shorten narration_text to AT MOST 10 words — keep the most shocking fragment only
- "scene_prompt_no_horror": add 1-2 specific horror/atmosphere tags to scene_prompt (comma-separated English tags, no sentences, no character names)
- "narration_language_wrong": rewrite narration_text entirely in Vietnamese — same event/mood, first-person narrator ("Tôi..."), keep ≤10 words if it is the hook shot (shot_index 0 or 1)

RULES:
- Do NOT change shots that are not in the input list
- Keep narration_text in Vietnamese, scene_prompt in English tags only
- narration_text: first-person narrator ("Tôi..."), present-tense tension, name specific actions/characters
- scene_prompt: comma-separated tags ONLY — no English sentences, no character names, no style/safety tags
- For narration_too_short: shots MUST reach 28+ words. Count words before returning.

Return JSON ARRAY (one object per input shot, same order):
[{"shot_index": 0, "narration_text": "...", "scene_prompt": "..."}, ...]
Return ONLY the JSON array — no markdown, no explanation."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=15),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
)
def _llm_fix_shots(shots: List[ShotScript], issues: List[ReviewIssue], episode_num: int) -> List[dict]:
    """Send only the flagged shots to LLM for rewrite. Returns list of {shot_index, narration_text, scene_prompt}."""
    # Deduplicate by shot_index; episode-level issues (index -1) are not sent
    fixable_types = {"narration_too_short", "hook_too_long", "scene_prompt_no_horror", "narration_language_wrong"}
    index_to_issues: dict[int, list[str]] = {}
    for issue in issues:
        if issue.shot_index >= 0 and issue.issue_type in fixable_types:
            index_to_issues.setdefault(issue.shot_index, []).append(issue.issue_type)

    if not index_to_issues:
        return []

    payload = []
    for shot_idx, issue_types in sorted(index_to_issues.items()):
        shot = shots[shot_idx]
        payload.append({
            "shot_index": shot_idx,
            "issue_types": issue_types,
            "narration_text": shot.narration_text,
            "scene_prompt": shot.scene_prompt,
        })

    prompt = (
        f"Episode {episode_num}: fix the following {len(payload)} shot(s) that failed quality checks.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    logger.info(
        "Sending {} shot(s) to LLM for quality fix | episode={} indices={}",
        len(payload), episode_num, sorted(index_to_issues.keys()),
    )

    result = script_client.generate_json(prompt=prompt, system=_FIX_SYSTEM, temperature=0.6)

    # Some local LLMs wrap the array in a single-key object even when the
    # prompt asks for a bare array (e.g. `{"shots": [...]}`). Unwrap any
    # such single-list wrapper before validation so we don't burn all 3
    # tenacity retries on a recoverable shape error.
    if isinstance(result, dict):
        list_values = [v for v in result.values() if isinstance(v, list)]
        if len(list_values) == 1:
            logger.debug(
                "Unwrapped LLM dict-wrapped fix array | top_keys={}",
                list(result.keys()),
            )
            result = list_values[0]
        elif "shot_index" in result and "narration_text" in result:
            # Single fix object returned without list wrapper — promote to list.
            logger.debug(
                "Promoted single-dict fix to list | shot_index={}",
                result.get("shot_index"),
            )
            result = [result]

    if not isinstance(result, list):
        raise ValueError(
            f"LLM returned non-list for shot fixes: type={type(result).__name__} "
            f"preview={str(result)[:200]!r}"
        )
    if len(result) != len(payload):
        raise ValueError(
            f"LLM returned {len(result)} fixes but expected {len(payload)}"
        )

    # Validate each fix has shot_index, narration_text, scene_prompt
    for fix in result:
        if not isinstance(fix, dict):
            raise ValueError(f"Fix item is not a dict: {fix!r}")
        if "shot_index" not in fix or "narration_text" not in fix or "scene_prompt" not in fix:
            raise ValueError(f"Fix item missing required fields: {fix!r}")

    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def review_episode_script(episode_num: int) -> tuple[EpisodeScript, ReviewReport]:
    """Run quality review on the generated script. Auto-fix issues via LLM.

    Returns the (possibly updated) script and a ReviewReport describing
    what was found and fixed.

    The updated script is saved in-place, overwriting the original.
    """
    from llm.scriptwriter import load_episode_script

    script = load_episode_script(episode_num)
    report = ReviewReport(episode_num=episode_num)

    # Pass 1 — rule-based
    issues = _rule_check(script)
    report.issues = issues

    if issues:
        logger.info(
            "Script review: {} issue(s) found | episode={}\n{}",
            len(issues), episode_num, report.summary(),
        )
    else:
        logger.info("Script review: PASS — no issues | episode={}", episode_num)
        return script, report

    # Separate fixable vs warn-only issues
    # character_name_typo = warn only (needs human correction in the JSON)
    _WARN_ONLY = {"shot_count_low", "shot_count_high", "total_words_low", "character_name_typo"}
    fixable = [i for i in issues if i.issue_type not in _WARN_ONLY]
    warn_only = [i for i in issues if i.issue_type in _WARN_ONLY]

    for w in warn_only:
        logger.warning(
            "Script review WARNING (not auto-fixed) | episode={} issue={}",
            episode_num, w.description,
        )

    if not fixable:
        logger.info("Script review: all issues are warnings — no LLM fix needed | episode={}", episode_num)
        return script, report

    # Pass 2 — LLM fix
    try:
        fixes = _llm_fix_shots(script.shots, fixable, episode_num)
    except Exception as exc:
        # tenacity wraps the underlying error in RetryError after exhausting
        # all attempts; surface the last real cause so the failure is
        # actually diagnosable from logs instead of an opaque "RetryError".
        from tenacity import RetryError
        if isinstance(exc, RetryError) and exc.last_attempt is not None:
            try:
                inner = exc.last_attempt.exception()
            except Exception:
                inner = None
            if inner is not None:
                logger.error(
                    "LLM shot fix failed after retries (proceeding with original script) | "
                    "episode={} cause={}: {}",
                    episode_num, type(inner).__name__, inner,
                )
                return script, report
        logger.error(
            "LLM shot fix failed (proceeding with original script) | episode={} error={}",
            episode_num, exc,
        )
        return script, report

    if not fixes:
        return script, report

    # Apply fixes to script
    shots = list(script.shots)
    for fix in fixes:
        idx: int = fix["shot_index"]
        if idx < 0 or idx >= len(shots):
            logger.warning("Fix has out-of-range shot_index={} — skipping", idx)
            continue

        old_narration = shots[idx].narration_text
        old_prompt = shots[idx].scene_prompt
        new_narration = fix.get("narration_text", old_narration).strip()
        new_prompt = fix.get("scene_prompt", old_prompt).strip()

        changed = False
        if new_narration and new_narration != old_narration:
            shots[idx] = shots[idx].model_copy(update={"narration_text": new_narration})
            changed = True
        if new_prompt and new_prompt != old_prompt:
            shots[idx] = shots[idx].model_copy(update={"scene_prompt": new_prompt})
            changed = True

        if changed:
            report.fixed_shots.append(idx)
            logger.info(
                "Shot {} fixed | episode={} words_before={} words_after={}",
                idx + 1, episode_num,
                len(old_narration.split()), len(shots[idx].narration_text.split()),
            )

    if not report.fixed_shots:
        logger.info("Script review: LLM returned no changes | episode={}", episode_num)
        return script, report

    updated_script = script.model_copy(update={"shots": shots})

    # Save in-place
    script_path = (
        Path(settings.data_dir)
        / "scripts"
        / f"episode-{episode_num:03d}-script.json"
    )
    script_path.write_text(updated_script.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Script review: {} shot(s) fixed and saved | episode={} fixed_indices={}",
        len(report.fixed_shots), episode_num, report.fixed_shots,
    )

    return updated_script, report
