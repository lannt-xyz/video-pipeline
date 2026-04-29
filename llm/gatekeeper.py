"""Phase 5 — Gatekeeper Reviewer.

Pure judge: runs `constraint_validator` checks against an EpisodeScript and
returns BLOCKING + WARNING violations. No auto-fix LLM call inside.

The orchestrator wraps this with a bounded retry loop that calls
`scriptwriter.regenerate_failed_shots` to rewrite blocking shots.

Design choices:
- Reviewer does NOT mutate the script — it observes only.
- Severity mapping: BLOCKING (must fix) vs WARNING (log only).
- Idempotent: same input → same violations list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from loguru import logger

from config.settings import settings
from llm.constraint_validator import (
    Severity,
    Violation,
    check_energy_monotony,
    check_exposition_density,
    check_hook_language,
    check_lore_before_curiosity,
    populate_shot_signals,
)
from models.schemas import EpisodeScript


@dataclass
class ReviewResult:
    passed: bool
    blocking: List[Violation] = field(default_factory=list)
    warnings: List[Violation] = field(default_factory=list)

    @property
    def all_violations(self) -> List[Violation]:
        return [*self.blocking, *self.warnings]

    def summary(self) -> str:
        if self.passed and not self.warnings:
            return "PASS — no violations."
        lines = []
        for v in self.blocking:
            lines.append(
                f"  [BLOCKING] shot={v.shot_index} {v.rule}: {v.message}"
            )
        for v in self.warnings:
            lines.append(
                f"  [WARNING]  shot={v.shot_index} {v.rule}: {v.message}"
            )
        return "\n".join(lines) if lines else "PASS"


def _violations_log_path() -> Path:
    return Path("logs") / "retention_violations.jsonl"


def gatekeeper_review(script: EpisodeScript, known_chars: List[str] | None = None) -> ReviewResult:
    """Run all constraint checks against the script. Returns BLOCKING + WARNING tiers.

    `known_chars` is optional; when provided, hook language check is unaffected
    but proper-noun extraction in `populate_shot_signals` benefits.
    """
    # Ensure each shot has signals computed (idempotent — skips if already set).
    chars = list(known_chars or [])
    for shot in script.shots:
        populate_shot_signals(shot, chars)

    blocking: List[Violation] = []
    warnings: List[Violation] = []

    # Hook language — BLOCKING (shot 0 must be Vietnamese).
    if script.shots:
        hook_v = check_hook_language(script.shots[0])
        if hook_v is not None:
            (blocking if hook_v.severity == Severity.BLOCKING else warnings).append(hook_v)

    # Exposition density — BLOCKING.
    threshold = settings.retention.max_exposition_ratio
    for v in check_exposition_density(script.shots, threshold=threshold):
        (blocking if v.severity == Severity.BLOCKING else warnings).append(v)

    # Lore-before-curiosity — WARNING.
    buffer_n = settings.retention.lore_curiosity_buffer_shots
    for v in check_lore_before_curiosity(script.shots, buffer=buffer_n):
        (blocking if v.severity == Severity.BLOCKING else warnings).append(v)

    # Energy monotony — WARNING.
    max_run = settings.retention.max_consecutive_same_energy
    for v in check_energy_monotony(script.shots, max_consec=max_run):
        (blocking if v.severity == Severity.BLOCKING else warnings).append(v)

    return ReviewResult(passed=not blocking, blocking=blocking, warnings=warnings)


def log_violations_jsonl(episode_num: int, result: ReviewResult, attempt: int, final: bool) -> None:
    """Append structured violation entry to logs/retention_violations.jsonl."""
    path = _violations_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "episode": episode_num,
        "attempt": attempt,
        "final": final,
        "blocking_count": len(result.blocking),
        "warning_count": len(result.warnings),
        "blocking": [
            {"shot": v.shot_index, "rule": v.rule, "msg": v.message}
            for v in result.blocking
        ],
        "warnings": [
            {"shot": v.shot_index, "rule": v.rule, "msg": v.message}
            for v in result.warnings
        ],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug(
        "Violations logged | episode={} attempt={} final={} blocking={} warnings={}",
        episode_num, attempt, final, len(result.blocking), len(result.warnings),
    )
