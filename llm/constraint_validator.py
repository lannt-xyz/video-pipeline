"""Pure-Python attention-constraint validators (no LLM, no I/O).

Phase 1b of the attention-constraint upgrade. All functions here are deterministic
and side-effect free — they read shot/episode data and return signals or violations
that the gatekeeper reviewer + orchestrator retry loop will act on.

Severity philosophy (see plans/attention-contrains-system-upgrade.md):
- BLOCKING: hard failure modes that ALWAYS hurt (English hook, exposition wall).
- WARNING: soft heuristics where occasional violations are acceptable for tu-tiên
  genre (lore density, monotony). Logged for Phase 6 calibration; do not gate.

All thresholds are placeholders v0; calibrate after baseline report runs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional

from models.schemas import CameraFlow, EnergyLevel, ShotScript, ShotSubject

# ---------------------------------------------------------------------------
# Severity + Violation
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    BLOCKING = "blocking"   # gatekeeper rejects script → orchestrator retries
    WARNING = "warning"     # logged only, does not gate


@dataclass(frozen=True)
class Violation:
    rule: str               # rule id, e.g. "hook_language"
    severity: Severity
    shot_index: Optional[int]   # None = episode-level
    message: str            # human-readable, in English (logs)


# ---------------------------------------------------------------------------
# Heuristic word/phrase lists (v0 — calibrate at Phase 6)
# ---------------------------------------------------------------------------

# Tension verbs at sentence start → strong signal of action/tension.
_TENSION_VERBS_START: frozenset[str] = frozenset({
    "nhìn", "nghe", "hét", "chạy", "đập", "mở", "chạm", "ngã", "cắt", "đốt",
    "đứng", "quay", "vung", "gọi", "túm", "đẩy", "kéo", "xé", "đâm", "lao",
    "vọt", "rít", "run", "trợn", "há", "lùi", "vỗ", "đỡ", "cản", "giật",
})

# Action verbs anywhere in sentence (mid-sentence tension signal).
_ACTION_VERBS_MID: frozenset[str] = frozenset({
    "đập", "vung", "chạy", "lao", "vọt", "đâm", "cắt", "xé", "túm",
    "giật", "kéo", "đẩy", "ngã", "trợn", "rít", "hét",
})

# Vietnamese pronouns/subjects often opening a tension sentence.
_PRONOUNS = frozenset({
    "tôi", "anh", "cô", "ông", "bà", "nó", "hắn", "y", "thằng", "lão", "gã",
})

# Sentence openings that almost always introduce exposition/lore.
_EXPOSITION_OPENINGS: tuple[str, ...] = (
    "theo ", "vì ", "do ", "bởi ", "đó là ", "loại ", "một loại ",
    "trong khi ", "thực ra ", "thật ra ", "vốn ", "vốn dĩ ",
)

# Pattern "X là [một] ..." — definition style.
_DEFINITION_PATTERN = re.compile(r"^\S+\s+là\s+(?:một\s+)?", re.UNICODE)

# Abstract nouns that signal abstract/expository content.
_ABSTRACT_NOUNS: frozenset[str] = frozenset({
    "loại", "bản chất", "hiện tượng", "nguyên nhân", "cổ thuật", "đạo pháp",
    "pháp thuật", "linh hồn", "tà thuật", "nghi thức", "truyền thuyết",
    "lịch sử", "nguồn gốc", "định nghĩa", "khái niệm", "phương pháp",
    "quy tắc", "luật lệ", "đạo lý", "chân lý",
})

# Narration keyword buckets for energy inference.
_NARR_KW_SHOCK = frozenset({"máu", "hét", "cắt", "đâm"})
_NARR_KW_HIGH = frozenset({"đập", "vung", "chạy", "lao", "vọt"})
_NARR_KW_MED = frozenset({"nói", "đứng", "quay"})
_NARR_KW_LOW = frozenset({"lặng", "mờ"})  # "nhìn xa" handled as bigram below

# Sentence terminators.
_SENT_SPLIT_RE = re.compile(r"[.!?]+\s*", re.UNICODE)

# Proper-noun candidate: 1-4 capitalized words (Vietnamese diacritics OK).
# Note: we cannot use a simple Unicode range like [A-ZÀ-Ỹ] because that block
# (U+00C0..U+1EF8) interleaves upper- and lowercase Vietnamese letters
# (e.g. lowercase đ U+0111 falls inside it). Instead we tokenize and check each
# leading character with `str.isupper()` which is locale-aware.
_TOKEN_RE = re.compile(r"[^\s,;:.!?\"'\(\)\[\]]+", re.UNICODE)

# Vietnamese diacritic detection: any char with combining mark or Vietnamese
# precomposed letter outside basic ASCII.
_VI_DIACRITIC_RE = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]")


# ---------------------------------------------------------------------------
# 3.1 — exposition_ratio
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _classify_sentence(sentence: str) -> str:
    """Return 'tension', 'expository', or 'neutral' based on multi-signal vote.

    Whichever bucket has ≥2 signals wins. Tie or <2 signals → neutral.
    """
    s = sentence.strip()
    s_lower = s.lower()
    tokens = s_lower.split()
    if not tokens:
        return "neutral"

    tension_signals = 0
    expo_signals = 0

    # Tension signal 1: starts with tension verb
    if tokens[0] in _TENSION_VERBS_START:
        tension_signals += 1
    # Tension signal 2: contains ? or !
    if "?" in s or "!" in s:
        tension_signals += 1
    # Tension signal 3: pronoun-or-name subject + action verb mid-sentence
    has_subject = tokens[0] in _PRONOUNS or (
        s[:1].isupper() and len(tokens) > 1 and not s_lower.startswith(_EXPOSITION_OPENINGS)
    )
    has_action_mid = any(t in _ACTION_VERBS_MID for t in tokens[1:])
    if has_subject and has_action_mid:
        tension_signals += 1

    # Expo signal 1: blacklist openings
    if s_lower.startswith(_EXPOSITION_OPENINGS):
        expo_signals += 1
    # Expo signal 2: "X là [một] ..." definition pattern
    if _DEFINITION_PATTERN.match(s):
        expo_signals += 1
    # Expo signal 3: contains abstract noun
    if any(noun in s_lower for noun in _ABSTRACT_NOUNS):
        expo_signals += 1
    # Expo signal 4: no concrete subject AND no action verb
    if not has_subject and not has_action_mid:
        expo_signals += 1

    if tension_signals >= 2 and tension_signals > expo_signals:
        return "tension"
    if expo_signals >= 2 and expo_signals > tension_signals:
        return "expository"
    return "neutral"


def compute_exposition_ratio(narration: str) -> float:
    """Fraction of expository sentences in narration.

    `ratio = expository / (expository + tension)`. Neutral sentences are excluded
    from denominator so a shot of pure neutral description doesn't count as
    expository. Returns 0.0 when no classifiable sentences exist.
    """
    sentences = _split_sentences(narration)
    if not sentences:
        return 0.0
    expo = 0
    tens = 0
    for s in sentences:
        label = _classify_sentence(s)
        if label == "expository":
            expo += 1
        elif label == "tension":
            tens += 1
    denom = expo + tens
    if denom == 0:
        return 0.0
    return expo / denom


# ---------------------------------------------------------------------------
# 3.2 — extract_proper_nouns
# ---------------------------------------------------------------------------


def _normalize_for_match(s: str) -> str:
    """Casefold + strip diacritics for char-name matching."""
    nfkd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.casefold().strip()


def _is_capitalized(w: str) -> bool:
    """True for title-case words: first char uppercase, rest lowercase (diacritics OK)."""
    if not w:
        return False
    if not w[0].isupper():
        return False
    return len(w) == 1 or w[1:].islower()


def extract_proper_nouns(
    narration: str,
    known_chars: Iterable[str],
) -> dict[str, List[str]]:
    """Split capitalized-token sequences into characters vs lore_terms.

    Sentence-initial proper-nouns (the first capitalized word of a sentence) are
    skipped to avoid false positives (every sentence start is capitalized).
    """
    known_norm = {_normalize_for_match(c) for c in known_chars if c.strip()}

    sentences = _split_sentences(narration)
    chars_found: List[str] = []
    lore_found: List[str] = []
    seen: set[str] = set()

    for sent in sentences:
        tokens = list(_TOKEN_RE.finditer(sent))
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            word = tok.group(0)
            if _is_capitalized(word):
                run = [word]
                run_start = tok.start()
                j = i + 1
                while j < len(tokens) and len(run) < 4:
                    nxt = tokens[j].group(0)
                    if _is_capitalized(nxt):
                        run.append(nxt)
                        j += 1
                    else:
                        break
                # Single capitalized word at sentence start = casing artifact; skip.
                if run_start == 0 and len(run) == 1:
                    i = j
                    continue
                term = " ".join(run)
                if term not in seen:
                    seen.add(term)
                    if _normalize_for_match(term) in known_norm:
                        chars_found.append(term)
                    else:
                        lore_found.append(term)
                i = j
            else:
                i += 1
    return {"characters": chars_found, "lore_terms": lore_found}


# ---------------------------------------------------------------------------
# 3.3 — infer_energy_level
# ---------------------------------------------------------------------------


def _bigram_low_signal(text_lower: str) -> bool:
    """Detect 'nhìn xa' phrase as LOW signal (verb 'nhìn' alone is too generic)."""
    return "nhìn xa" in text_lower


def _vote_subject(subject: ShotSubject) -> Optional[EnergyLevel]:
    if subject in (ShotSubject.WOUND, ShotSubject.CORPSE_FACE, ShotSubject.BLOODY_OBJECT):
        return EnergyLevel.SHOCK
    if subject == ShotSubject.SUPERNATURAL_ENTITY:
        return EnergyLevel.HIGH
    if subject == ShotSubject.RITUAL_OBJECT:
        return EnergyLevel.MED
    if subject == ShotSubject.ENVIRONMENT:
        return EnergyLevel.LOW
    # PERSON_ACTION decided after combined narration-keyword check.
    return None


def _vote_camera(flow: CameraFlow) -> EnergyLevel:
    if flow == CameraFlow.DETAIL_REVEAL:
        return EnergyLevel.SHOCK
    if flow == CameraFlow.WIDE_TO_CLOSE or flow == CameraFlow.CLOSE_TO_WIDE:
        return EnergyLevel.HIGH
    if flow in (CameraFlow.PAN_ACROSS, CameraFlow.STATIC_CLOSE):
        return EnergyLevel.MED
    return EnergyLevel.LOW  # STATIC_WIDE


def _vote_duration(d: float) -> EnergyLevel:
    if d <= 3:
        return EnergyLevel.SHOCK
    if d <= 5:
        return EnergyLevel.HIGH
    if d <= 8:
        return EnergyLevel.MED
    return EnergyLevel.LOW


def _vote_narration(text: str) -> Optional[EnergyLevel]:
    t = text.lower()
    tokens = set(re.findall(r"\w+", t, flags=re.UNICODE))
    if tokens & _NARR_KW_SHOCK:
        return EnergyLevel.SHOCK
    if tokens & _NARR_KW_HIGH:
        return EnergyLevel.HIGH
    if tokens & _NARR_KW_MED:
        return EnergyLevel.MED
    if (tokens & _NARR_KW_LOW) or _bigram_low_signal(t):
        return EnergyLevel.LOW
    return None


_ENERGY_RANK = {
    EnergyLevel.LOW: 0,
    EnergyLevel.MED: 1,
    EnergyLevel.HIGH: 2,
    EnergyLevel.SHOCK: 3,
}


def infer_energy_level(shot: ShotScript) -> EnergyLevel:
    """Vote across 4 signals; pick mode; tie-break by higher rank."""
    votes: List[EnergyLevel] = []

    subj_vote = _vote_subject(shot.shot_subject)
    narr_vote = _vote_narration(shot.narration_text)

    if subj_vote is not None:
        votes.append(subj_vote)
    elif shot.shot_subject == ShotSubject.PERSON_ACTION:
        # Upgrade to HIGH when narration carries an action keyword, else MED.
        if narr_vote is not None and _ENERGY_RANK[narr_vote] >= _ENERGY_RANK[EnergyLevel.HIGH]:
            votes.append(EnergyLevel.HIGH)
        else:
            votes.append(EnergyLevel.MED)

    votes.append(_vote_camera(shot.camera_flow))
    votes.append(_vote_duration(shot.duration_sec))
    if narr_vote is not None:
        votes.append(narr_vote)

    # Mode with higher-rank tie-break.
    counts: dict[EnergyLevel, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], _ENERGY_RANK[kv[0]]))
    return best[0]


# ---------------------------------------------------------------------------
# 3.4 — lore-before-curiosity (WARNING)
# ---------------------------------------------------------------------------


def _shot_has_tension(shot: ShotScript) -> bool:
    """Tension = HIGH/SHOCK energy OR low exposition OR explicit ?/! in narration."""
    if shot.energy_level in (EnergyLevel.HIGH, EnergyLevel.SHOCK):
        return True
    if shot.exposition_ratio is not None and shot.exposition_ratio < 0.3:
        return True
    if "?" in shot.narration_text or "!" in shot.narration_text:
        return True
    return False


def check_lore_before_curiosity(
    shots: List[ShotScript],
    buffer: int = 2,
) -> List[Violation]:
    violations: List[Violation] = []
    seen_lore: set[str] = set()
    for i, shot in enumerate(shots):
        lore = set(shot.proper_nouns or [])
        # In v0 we treat ALL proper_nouns as lore_terms because shot.proper_nouns
        # is the merged list. The reviewer-side path uses extract_proper_nouns
        # to get a split dict; here we keep it conservative.
        new_lore = lore - seen_lore
        if new_lore:
            tension_count = sum(1 for prev in shots[:i] if _shot_has_tension(prev))
            if tension_count < buffer:
                violations.append(
                    Violation(
                        rule="lore_before_curiosity",
                        severity=Severity.WARNING,
                        shot_index=i,
                        message=(
                            f"shot {i} introduces lore terms {sorted(new_lore)} but "
                            f"only {tension_count} tension shots preceded it "
                            f"(buffer={buffer})"
                        ),
                    )
                )
        seen_lore |= lore
    return violations


# ---------------------------------------------------------------------------
# 3.5 — energy monotony (WARNING)
# ---------------------------------------------------------------------------


def check_energy_monotony(
    shots: List[ShotScript],
    max_consec: int = 2,
) -> List[Violation]:
    if not shots:
        return []
    violations: List[Violation] = []
    run_level = shots[0].energy_level
    run_start = 0
    for i in range(1, len(shots)):
        lvl = shots[i].energy_level
        if lvl == run_level:
            continue
        run_len = i - run_start
        if run_len > max_consec and run_level is not None:
            violations.append(
                Violation(
                    rule="energy_monotony",
                    severity=Severity.WARNING,
                    shot_index=run_start,
                    message=(
                        f"shots {run_start}..{i - 1} ({run_len} in a row) all "
                        f"energy_level={run_level.value} > max {max_consec}"
                    ),
                )
            )
        run_level = lvl
        run_start = i
    # Tail run
    run_len = len(shots) - run_start
    if run_len > max_consec and run_level is not None:
        violations.append(
            Violation(
                rule="energy_monotony",
                severity=Severity.WARNING,
                shot_index=run_start,
                message=(
                    f"shots {run_start}..{len(shots) - 1} ({run_len} in a row) all "
                    f"energy_level={run_level.value} > max {max_consec}"
                ),
            )
        )
    return violations


# ---------------------------------------------------------------------------
# 3.6 — exposition density (BLOCKING)
# ---------------------------------------------------------------------------


def check_exposition_density(
    shots: List[ShotScript],
    threshold: float = 0.5,
) -> List[Violation]:
    violations: List[Violation] = []
    for i in range(1, len(shots)):
        prev = shots[i - 1].exposition_ratio
        cur = shots[i].exposition_ratio
        if prev is None or cur is None:
            continue
        if prev > threshold and cur > threshold:
            violations.append(
                Violation(
                    rule="exposition_density",
                    severity=Severity.BLOCKING,
                    shot_index=i,
                    message=(
                        f"shots {i - 1} and {i} both exposition_ratio "
                        f"({prev:.2f}, {cur:.2f}) > {threshold}"
                    ),
                )
            )
    return violations


# ---------------------------------------------------------------------------
# 3.7 — hook language (BLOCKING)
# ---------------------------------------------------------------------------


def check_hook_language(shot_0: ShotScript) -> Optional[Violation]:
    """Hook (shot 0) narration must contain Vietnamese diacritics."""
    if not _VI_DIACRITIC_RE.search(shot_0.narration_text):
        return Violation(
            rule="hook_language",
            severity=Severity.BLOCKING,
            shot_index=0,
            message=(
                f"hook narration has no Vietnamese diacritics: "
                f"{shot_0.narration_text[:80]!r}"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# 3.8 — populate_shot_signals
# ---------------------------------------------------------------------------


def populate_shot_signals(
    shot: ShotScript,
    known_chars: Iterable[str],
) -> ShotScript:
    """Fill `energy_level`, `proper_nouns`, `exposition_ratio` if None.

    Idempotent: existing non-None fields are preserved. Mutates in place AND
    returns the shot for convenient chaining.
    """
    if shot.exposition_ratio is None:
        shot.exposition_ratio = compute_exposition_ratio(shot.narration_text)
    if shot.proper_nouns is None:
        nouns = extract_proper_nouns(shot.narration_text, known_chars)
        # Merged list: chars + lore_terms. The split is recomputed where needed.
        shot.proper_nouns = nouns["characters"] + nouns["lore_terms"]
    if shot.energy_level is None:
        shot.energy_level = infer_energy_level(shot)
    return shot


# ---------------------------------------------------------------------------
# Episode-level aggregation helper
# ---------------------------------------------------------------------------


def collect_episode_violations(
    shots: List[ShotScript],
    *,
    max_exposition_ratio: float = 0.5,
    max_consecutive_same_energy: int = 2,
    lore_curiosity_buffer_shots: int = 2,
) -> List[Violation]:
    """Run all rules; return flat list of violations.

    Caller (gatekeeper reviewer) decides what to do with BLOCKING vs WARNING.
    """
    violations: List[Violation] = []
    if shots:
        hook_v = check_hook_language(shots[0])
        if hook_v:
            violations.append(hook_v)
    violations.extend(
        check_exposition_density(shots, threshold=max_exposition_ratio)
    )
    violations.extend(
        check_energy_monotony(shots, max_consec=max_consecutive_same_energy)
    )
    violations.extend(
        check_lore_before_curiosity(shots, buffer=lore_curiosity_buffer_shots)
    )
    return violations
