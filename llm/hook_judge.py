"""Phase 4 — Competitive hook selection with rubric-based LLM judge.

Pipeline:
1. `generate_hook_candidates(arc_text, n=3)` — 1 LLM call, 3 distinct variants
2. Pre-filter via `check_hook_language` (drop English candidates before judging)
3. `judge_candidates(...)` — 1 LLM call scoring all surviving candidates
4. Pick winner; retry once if winner.total_score < hook_min_score
5. Persist all candidates → `data/{slug}/hook_candidates/episode-NNN.json`

Behind `retention.use_constraint_system`. Caller falls back to legacy single-shot
hook when this returns None (graceful degrade).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from tenacity import RetryError, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import script_client as ollama_client
from llm.constraint_validator import check_hook_language
from models.schemas import HookCandidate, ShotScript

_HOOK_GEN_SYSTEM = """You are a TikTok/YouTube Shorts hook writer for Vietnamese supernatural-horror.

Generate EXACTLY 3 distinct hook candidates for shot 0 of an episode. Each emphasizes a DIFFERENT angle.

HARD RULES per candidate:
- `text`: ≤10 words, Vietnamese only (NO English).
- `text` must plant a question, not answer one.
- `text` must reference a CONCRETE visual moment (object, body part, action), not abstract dread.
- `visual_seed`: short English phrase (~5-12 words) describing the visual the hook should show.

ANTI-PATTERNS (these get rejected):
- Generic horror clichés: "bóng tối ập tới", "máu đã chảy", "cái chết đến gần"
- Setup / backstory: "Diệp Thiếu Dương bắt đầu hành trình..."
- Resolved statement: "Hắn đã chết." (no question, no curiosity)
- English: "The stench of death..." (REJECTED outright)

3 ANGLES — make each candidate target a DIFFERENT one:
A) Mid-action cut: "Hắn vừa chạm vào nắp quan tài thì—"
B) Visual paradox: "Đứa trẻ trong quan tài đang mỉm cười."
C) Off-screen sound + react: "Tiếng cào từ trong quan tài. Tôi không dám mở."

OUTPUT — JSON only:
{
  "candidates": [
    {"text": "<Vietnamese ≤10 words>", "visual_seed": "<English short phrase>"},
    {"text": "...", "visual_seed": "..."},
    {"text": "...", "visual_seed": "..."}
  ]
}"""


_HOOK_JUDGE_SYSTEM = """You are a hook quality judge for TikTok/YouTube Shorts.

For EACH candidate provided, score 3 dimensions on a 0.0-1.0 scale:

1. `curiosity_gap` — does this candidate make the viewer ask "what?" or "why?"
   - HIGH (0.8-1.0): plants a specific unanswered question that demands resolution
   - LOW (0.0-0.3): already answers itself, or asks nothing

2. `specificity` — is the imagery CONCRETE, not generic
   - HIGH: "Bàn tay nhỏ áp lên nắp quan tài" (specific body part + object)
   - LOW: "Bóng tối ập đến" (generic horror cliché)

3. `pattern_interrupt` — does it break the expected story-opening pattern
   - HIGH: cuts mid-action, opens on the consequence not the cause, paradox
   - LOW: standard "X bắt đầu hành trình..." setup

ANTI-BIAS DIRECTIVES (read these before scoring):
- PENALIZE candidates that "sound nice" but don't create curiosity gap.
- PENALIZE generic horror vocabulary even if grammatically perfect.
- PREFER candidates that withhold context — viewer should NEED to keep watching.
- The rationale MUST state: "this candidate plants the question: [...]". If you cannot complete that sentence, score curiosity_gap below 0.4.

OUTPUT — JSON only, scores per candidate (in input order):
{
  "scores": [
    {
      "curiosity_gap": 0.0-1.0,
      "specificity": 0.0-1.0,
      "pattern_interrupt": 0.0-1.0,
      "rationale": "<1-2 sentences>"
    },
    ...
  ]
}"""


def _hook_candidates_path(episode_num: int) -> Path:
    return (
        Path(settings.data_dir)
        / "hook_candidates"
        / f"episode-{episode_num:03d}.json"
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_not_exception_type(RetryError),
)
def _call_generator(arc_text: str, episode_num: int, hint: str = "") -> List[dict]:
    prompt = (
        f"Episode {episode_num} story context:\n{arc_text}\n\n"
        + (f"\nGUIDANCE: {hint}\n\n" if hint else "")
        + "Generate 3 distinct hook candidates following the rules. Return JSON only."
    )
    raw = ollama_client.generate_json(
        prompt=prompt, system=_HOOK_GEN_SYSTEM, temperature=0.85
    )
    if not isinstance(raw, dict):
        raise ValueError("hook generator returned non-dict")
    cands = raw.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise ValueError("hook generator returned no candidates list")
    return [c for c in cands if isinstance(c, dict)]


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_not_exception_type(RetryError),
)
def _call_judge(candidates: List[HookCandidate]) -> List[dict]:
    payload = [
        {"index": i, "text": c.text, "visual_seed": c.visual_seed}
        for i, c in enumerate(candidates)
    ]
    prompt = (
        "Score each candidate per the rubric.\n\n"
        f"Candidates (in order):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only."
    )
    raw = ollama_client.generate_json(
        prompt=prompt, system=_HOOK_JUDGE_SYSTEM, temperature=0.2
    )
    if not isinstance(raw, dict):
        raise ValueError("hook judge returned non-dict")
    scores = raw.get("scores")
    if not isinstance(scores, list):
        raise ValueError("hook judge returned no scores list")
    return scores


def _coerce_candidate(item: dict) -> Optional[HookCandidate]:
    text = str(item.get("text", "") or "").strip()
    if not text:
        return None
    return HookCandidate(
        text=text,
        visual_seed=str(item.get("visual_seed", "") or "").strip(),
    )


def _score_candidate(candidate: HookCandidate, score: dict) -> HookCandidate:
    """Apply judge scores + weighted total. Returns a new HookCandidate."""

    def _f(key: str) -> float:
        try:
            v = float(score.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))

    weights = settings.retention.hook_judge_weights
    cur = _f("curiosity_gap")
    spec = _f("specificity")
    pi = _f("pattern_interrupt")
    total = (
        weights.curiosity_gap * cur
        + weights.specificity * spec
        + weights.pattern_interrupt * pi
    )
    return candidate.model_copy(
        update={
            "curiosity_score": cur,
            "specificity_score": spec,
            "pattern_interrupt_score": pi,
            "total_score": total,
            "rationale": str(score.get("rationale", "") or "").strip(),
        }
    )


def _generate_and_score(
    arc_text: str, episode_num: int, hint: str = ""
) -> List[HookCandidate]:
    """Generate -> language pre-filter -> judge -> attach scores. May return []."""
    try:
        raw_cands = _call_generator(arc_text, episode_num, hint=hint)
    except Exception as e:  # noqa: BLE001
        logger.error("Hook generation failed | episode={} err={}", episode_num, e)
        return []

    candidates: List[HookCandidate] = []
    for item in raw_cands:
        c = _coerce_candidate(item)
        if c is None:
            continue
        # Pre-filter: language gate. Reject English candidates BEFORE waste judge call.
        probe = ShotScript(scene_prompt="x", narration_text=c.text)
        if check_hook_language(probe) is not None:
            logger.debug(
                "Hook candidate rejected by language pre-filter | episode={} text={!r}",
                episode_num, c.text,
            )
            continue
        candidates.append(c)

    if not candidates:
        logger.warning(
            "All hook candidates rejected by pre-filter | episode={}", episode_num
        )
        return []

    try:
        scores = _call_judge(candidates)
    except Exception as e:  # noqa: BLE001
        logger.error("Hook judge call failed | episode={} err={}", episode_num, e)
        return []

    scored: List[HookCandidate] = []
    for i, c in enumerate(candidates):
        score = scores[i] if i < len(scores) and isinstance(scores[i], dict) else {}
        scored.append(_score_candidate(c, score))
    return scored


def select_hook(
    arc_text: str, episode_num: int
) -> Optional[Tuple[HookCandidate, List[HookCandidate]]]:
    """Run the competitive hook selection pipeline.

    Returns (winner, all_candidates) or None on hard failure (caller falls back
    to legacy `_generate_hook_shot`).
    """
    threshold = settings.retention.hook_min_score
    all_candidates: List[HookCandidate] = []

    first_round = _generate_and_score(arc_text, episode_num)
    all_candidates.extend(first_round)

    best = max(first_round, key=lambda c: c.total_score) if first_round else None

    if best is not None and best.total_score < threshold:
        # Retry once with feedback on the weakest dimension across the round.
        weakest_dim = min(
            ("curiosity_gap", best.curiosity_score),
            ("specificity", best.specificity_score),
            ("pattern_interrupt", best.pattern_interrupt_score),
            key=lambda kv: kv[1],
        )[0]
        weak_examples = "; ".join(c.text for c in first_round[:3])
        hint = (
            f"Previous attempts scored low on {weakest_dim}. Examples: {weak_examples}. "
            f"Generate 3 NEW variants that specifically improve {weakest_dim}."
        )
        logger.info(
            "Hook below threshold {:.2f}; retrying once with hint | episode={}",
            threshold, episode_num,
        )
        second_round = _generate_and_score(arc_text, episode_num, hint=hint)
        all_candidates.extend(second_round)
        if second_round:
            new_best = max(second_round, key=lambda c: c.total_score)
            if new_best.total_score > best.total_score:
                best = new_best

    if best is None:
        return None

    _persist_candidates(episode_num, all_candidates, best)
    return best, all_candidates


def _persist_candidates(
    episode_num: int,
    all_candidates: List[HookCandidate],
    winner: HookCandidate,
) -> None:
    out_path = _hook_candidates_path(episode_num)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_num": episode_num,
        "winner_text": winner.text,
        "winner_score": winner.total_score,
        "threshold": settings.retention.hook_min_score,
        "candidates": [c.model_dump() for c in all_candidates],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Hook candidates persisted | episode={} count={} winner_score={:.3f} path={}",
        episode_num, len(all_candidates), winner.total_score, out_path,
    )
