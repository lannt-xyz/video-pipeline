"""Compute baseline retention metrics across existing scripts.

Reads `data/{slug}/scripts/*.json`, runs `populate_shot_signals` (in-memory,
no writes), aggregates exposition_ratio distribution, energy distribution,
and constraint violations. Writes report to `logs/baseline_violations.json`.

Used by Phase 1b to calibrate placeholder thresholds (e.g. set
`retention.max_exposition_ratio` to percentile-60 of the baseline distribution
rather than the v0 placeholder of 0.5).

Usage:
    uv run python scripts/retention_report.py
    uv run python scripts/retention_report.py --slug other-story
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings  # noqa: E402
from llm.constraint_validator import (  # noqa: E402
    Severity,
    collect_episode_violations,
    populate_shot_signals,
)
from models.schemas import EpisodeScript  # noqa: E402


def _load_known_chars(slug: str) -> List[str]:
    """Best-effort load character names from `data/{slug}/characters/*.json`."""
    import json as _json
    chars_dir = Path("data") / slug / "characters"
    if not chars_dir.exists():
        return []
    names: List[str] = []
    for p in chars_dir.glob("*.json"):
        try:
            obj = _json.loads(p.read_text(encoding="utf-8"))
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
        except (ValueError, OSError):
            continue
    return names


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--out", default="logs/baseline_violations.json")
    parser.add_argument("--max-exposition-ratio", type=float, default=0.6,
                        help="threshold for exposition_density rule (calibration knob)")
    args = parser.parse_args()

    settings = Settings()
    slug = args.slug or settings.story_slug
    scripts_dir = Path("data") / slug / "scripts"

    out = {
        "slug": slug,
        "scripts_dir": str(scripts_dir),
        "episodes_scanned": 0,
        "shots_scanned": 0,
        "exposition_ratio": {},
        "energy_distribution": {},
        "violations_by_rule": {},
        "violations_by_severity": {},
        "thresholds_used": {
            "max_exposition_ratio": args.max_exposition_ratio,
            "max_consecutive_same_energy": 2,
            "lore_curiosity_buffer_shots": 2,
        },
        "calibration_suggestion": {},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not scripts_dir.exists():
        print(f"[baseline] scripts dir not found: {scripts_dir}")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    files = sorted(scripts_dir.glob("episode-*-script.json"))
    if not files:
        print(f"[baseline] no episode scripts found in {scripts_dir}")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    known_chars = _load_known_chars(slug)

    expo_values: List[float] = []
    energy_counter: Counter[str] = Counter()
    rule_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()

    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        episode = EpisodeScript.model_validate(raw)
        for shot in episode.shots:
            populate_shot_signals(shot, known_chars)
            if shot.exposition_ratio is not None:
                expo_values.append(shot.exposition_ratio)
            if shot.energy_level is not None:
                energy_counter[shot.energy_level.value] += 1

        violations = collect_episode_violations(
            episode.shots,
            max_exposition_ratio=args.max_exposition_ratio,
            max_consecutive_same_energy=2,
            lore_curiosity_buffer_shots=2,
        )
        for v in violations:
            rule_counter[v.rule] += 1
            severity_counter[v.severity.value] += 1

        out["episodes_scanned"] += 1
        out["shots_scanned"] += len(episode.shots)

    if expo_values:
        out["exposition_ratio"] = {
            "n": len(expo_values),
            "mean": statistics.mean(expo_values),
            "median": statistics.median(expo_values),
            "p60": _percentile(expo_values, 0.6),
            "p75": _percentile(expo_values, 0.75),
            "p90": _percentile(expo_values, 0.9),
            "max": max(expo_values),
        }
        # Recommend p60 as the new max_exposition_ratio threshold.
        out["calibration_suggestion"]["max_exposition_ratio"] = round(_percentile(expo_values, 0.6), 3)

    out["energy_distribution"] = dict(energy_counter)
    out["violations_by_rule"] = dict(rule_counter)
    out["violations_by_severity"] = dict(severity_counter)

    # Phase 6 calibration aid: scan hook candidates persisted by Phase 4 to
    # detect if the rubric judge actually discriminates between candidates.
    # Low stddev (< 0.05) across many episodes → judge is biased toward one
    # score; raise temperature or revisit rubric prompt.
    hook_dir = Path("data") / slug / "hook_candidates"
    if hook_dir.exists():
        winner_scores: List[float] = []
        per_episode_stddev: List[float] = []
        for path in sorted(hook_dir.glob("episode-*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            cands = payload.get("candidates") or []
            scores = [
                float(c.get("total_score", 0.0) or 0.0)
                for c in cands
                if isinstance(c, dict)
            ]
            if scores:
                winner_scores.append(max(scores))
                if len(scores) >= 2:
                    per_episode_stddev.append(statistics.stdev(scores))
        if winner_scores:
            out["hook_judge"] = {
                "episodes": len(winner_scores),
                "winner_mean": round(statistics.mean(winner_scores), 3),
                "winner_min": round(min(winner_scores), 3),
                "candidate_score_stddev_mean": (
                    round(statistics.mean(per_episode_stddev), 3)
                    if per_episode_stddev
                    else None
                ),
                "discrimination_ok": bool(
                    per_episode_stddev
                    and statistics.mean(per_episode_stddev) >= 0.05
                ),
            }

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[baseline] wrote {out_path}")
    print(f"  episodes={out['episodes_scanned']}  shots={out['shots_scanned']}")
    print(f"  expo_ratio={out['exposition_ratio']}")
    print(f"  energy={out['energy_distribution']}")
    print(f"  violations_by_rule={out['violations_by_rule']}")
    print(f"  calibration_suggestion={out['calibration_suggestion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
