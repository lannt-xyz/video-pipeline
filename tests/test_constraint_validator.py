"""Unit tests for `llm.constraint_validator`.

Fixtures replicate the failure modes seen in episode-001 (English hook, lore
exposition wall) plus positive controls (Vietnamese hook, character names that
should not flag as lore).
"""

from __future__ import annotations

import pytest

from llm.constraint_validator import (
    Severity,
    check_energy_monotony,
    check_exposition_density,
    check_hook_language,
    check_lore_before_curiosity,
    collect_episode_violations,
    compute_exposition_ratio,
    extract_proper_nouns,
    infer_energy_level,
    populate_shot_signals,
)
from models.schemas import (
    CameraFlow,
    EnergyLevel,
    ShotScript,
    ShotSubject,
)


KNOWN_CHARS = ["Diệp Thiếu Dương", "Thanh Vân Tử", "Mao Sơn Đạo Trưởng"]


# ---------------------------------------------------------------------------
# Hook language (3.7)
# ---------------------------------------------------------------------------


def test_hook_language_english_is_blocking():
    shot = ShotScript(
        scene_prompt="x",
        narration_text="The stench of death filled the air as he opened the coffin.",
    )
    v = check_hook_language(shot)
    assert v is not None
    assert v.severity == Severity.BLOCKING
    assert v.rule == "hook_language"


def test_hook_language_vietnamese_passes():
    shot = ShotScript(
        scene_prompt="x",
        narration_text="Mùi tử khí xộc lên khi nắp quan tài bật mở.",
    )
    assert check_hook_language(shot) is None


# ---------------------------------------------------------------------------
# Exposition ratio (3.1)
# ---------------------------------------------------------------------------


def test_exposition_ratio_pure_lore_dump():
    # All four sentences are textbook expository (definition + abstract noun).
    text = (
        "Thi Du Cao là một loại thi độc cổ xưa. "
        "Loại độc này có nguồn gốc từ truyền thuyết. "
        "Theo cổ thuật, nó được dùng trong các nghi thức. "
        "Đó là một hiện tượng huyền bí."
    )
    ratio = compute_exposition_ratio(text)
    assert ratio >= 0.75


def test_exposition_ratio_pure_action_low():
    text = "Nó hét lên một tiếng. Anh lao tới đâm thẳng vào ngực hắn! Máu vọt ra."
    ratio = compute_exposition_ratio(text)
    assert ratio <= 0.2


def test_exposition_ratio_empty():
    assert compute_exposition_ratio("") == 0.0


# ---------------------------------------------------------------------------
# Proper nouns (3.2)
# ---------------------------------------------------------------------------


def test_extract_proper_nouns_splits_chars_and_lore():
    text = (
        "Diệp Thiếu Dương đứng nhìn quan tài. "
        "Bên trong là Thi Du Cao đã ngủ quên trăm năm. "
        "Mao Sơn Đạo Trưởng từng nói về nó."
    )
    out = extract_proper_nouns(text, KNOWN_CHARS)
    chars = set(out["characters"])
    lore = set(out["lore_terms"])
    assert "Diệp Thiếu Dương" in chars or "Mao Sơn Đạo Trưởng" in chars
    assert "Thi Du Cao" in lore
    # Character names must NOT be in lore.
    assert not (chars & lore)


def test_extract_proper_nouns_skips_sentence_initial_word():
    # "Anh" starts a sentence — should not be treated as a proper noun.
    text = "Anh nhìn xuống. Diệp Thiếu Dương không nói gì."
    out = extract_proper_nouns(text, KNOWN_CHARS)
    all_terms = out["characters"] + out["lore_terms"]
    assert "Anh" not in all_terms


# ---------------------------------------------------------------------------
# Energy inference (3.3)
# ---------------------------------------------------------------------------


def test_energy_corpse_face_is_shock():
    shot = ShotScript(
        scene_prompt="x",
        narration_text="Mặt xác trắng bệch.",
        duration_sec=3.0,
        camera_flow=CameraFlow.DETAIL_REVEAL,
        shot_subject=ShotSubject.CORPSE_FACE,
    )
    assert infer_energy_level(shot) == EnergyLevel.SHOCK


def test_energy_environment_long_static_is_low():
    shot = ShotScript(
        scene_prompt="x",
        narration_text="Sương mù phủ kín thung lũng, tất cả lặng như tờ.",
        duration_sec=10.0,
        camera_flow=CameraFlow.STATIC_WIDE,
        shot_subject=ShotSubject.ENVIRONMENT,
    )
    assert infer_energy_level(shot) == EnergyLevel.LOW


def test_energy_person_action_with_action_keyword_upgrades_to_high():
    shot = ShotScript(
        scene_prompt="x",
        narration_text="Anh lao tới, vung kiếm chém xuống.",
        duration_sec=5.0,
        camera_flow=CameraFlow.WIDE_TO_CLOSE,
        shot_subject=ShotSubject.PERSON_ACTION,
    )
    assert infer_energy_level(shot) == EnergyLevel.HIGH


# ---------------------------------------------------------------------------
# Exposition density (3.6) — BLOCKING
# ---------------------------------------------------------------------------


def _expo_shot(ratio: float) -> ShotScript:
    s = ShotScript(scene_prompt="x", narration_text=".")
    s.exposition_ratio = ratio
    return s


def test_exposition_density_two_consecutive_high_blocks():
    shots = [_expo_shot(r) for r in [0.1, 0.7, 0.8, 0.2]]
    vs = check_exposition_density(shots, threshold=0.6)
    assert any(v.severity == Severity.BLOCKING and v.rule == "exposition_density" for v in vs)
    # Only one violation pair in this fixture.
    assert len(vs) == 1
    assert vs[0].shot_index == 2


def test_exposition_density_alternating_passes():
    shots = [_expo_shot(r) for r in [0.7, 0.2, 0.7, 0.2]]
    assert check_exposition_density(shots, threshold=0.6) == []


# ---------------------------------------------------------------------------
# Energy monotony (3.5) — WARNING
# ---------------------------------------------------------------------------


def _energy_shot(level: EnergyLevel) -> ShotScript:
    s = ShotScript(scene_prompt="x", narration_text=".")
    s.energy_level = level
    return s


def test_energy_monotony_three_consecutive_warns():
    shots = [_energy_shot(l) for l in [
        EnergyLevel.MED, EnergyLevel.MED, EnergyLevel.MED, EnergyLevel.HIGH,
    ]]
    vs = check_energy_monotony(shots, max_consec=2)
    assert len(vs) == 1
    assert vs[0].severity == Severity.WARNING


def test_energy_monotony_alternating_no_warning():
    shots = [_energy_shot(l) for l in [
        EnergyLevel.MED, EnergyLevel.HIGH, EnergyLevel.MED, EnergyLevel.HIGH,
    ]]
    assert check_energy_monotony(shots, max_consec=2) == []


# ---------------------------------------------------------------------------
# Lore-before-curiosity (3.4) — WARNING
# ---------------------------------------------------------------------------


def test_lore_before_curiosity_warns_when_lore_too_early():
    # Two MED shots, then a shot introducing lore → no tension shots preceded.
    shots = [
        ShotScript(scene_prompt="x", narration_text="Anh đi tới.",
                   energy_level=EnergyLevel.MED, exposition_ratio=0.4,
                   proper_nouns=[]),
        ShotScript(scene_prompt="x", narration_text="Anh nhìn quanh.",
                   energy_level=EnergyLevel.MED, exposition_ratio=0.4,
                   proper_nouns=[]),
        ShotScript(scene_prompt="x", narration_text="Thi Du Cao chính là loại thi độc.",
                   energy_level=EnergyLevel.MED, exposition_ratio=0.7,
                   proper_nouns=["Thi Du Cao"]),
    ]
    vs = check_lore_before_curiosity(shots, buffer=2)
    assert len(vs) == 1
    assert vs[0].severity == Severity.WARNING
    assert vs[0].shot_index == 2


def test_lore_after_two_tension_shots_no_warning():
    shots = [
        ShotScript(scene_prompt="x", narration_text="Hắn hét lên!",
                   energy_level=EnergyLevel.HIGH, exposition_ratio=0.0,
                   proper_nouns=[]),
        ShotScript(scene_prompt="x", narration_text="Máu vọt ra.",
                   energy_level=EnergyLevel.SHOCK, exposition_ratio=0.0,
                   proper_nouns=[]),
        ShotScript(scene_prompt="x", narration_text="Đó chính là Thi Du Cao.",
                   energy_level=EnergyLevel.MED, exposition_ratio=0.5,
                   proper_nouns=["Thi Du Cao"]),
    ]
    assert check_lore_before_curiosity(shots, buffer=2) == []


# ---------------------------------------------------------------------------
# populate_shot_signals (3.8) — idempotent
# ---------------------------------------------------------------------------


def test_populate_shot_signals_idempotent():
    s = ShotScript(
        scene_prompt="x",
        narration_text="Anh lao tới, vung kiếm chém xuống. Máu vọt ra!",
        duration_sec=4.0,
        camera_flow=CameraFlow.WIDE_TO_CLOSE,
        shot_subject=ShotSubject.PERSON_ACTION,
    )
    populate_shot_signals(s, KNOWN_CHARS)
    snap = (s.energy_level, s.exposition_ratio, tuple(s.proper_nouns or []))
    populate_shot_signals(s, KNOWN_CHARS)
    snap2 = (s.energy_level, s.exposition_ratio, tuple(s.proper_nouns or []))
    assert snap == snap2


def test_populate_shot_signals_preserves_existing():
    s = ShotScript(scene_prompt="x", narration_text="Anh lao tới đâm.")
    s.energy_level = EnergyLevel.LOW  # bogus override
    s.exposition_ratio = 0.99
    s.proper_nouns = ["forced"]
    populate_shot_signals(s, KNOWN_CHARS)
    assert s.energy_level == EnergyLevel.LOW
    assert s.exposition_ratio == 0.99
    assert s.proper_nouns == ["forced"]


# ---------------------------------------------------------------------------
# Integration: episode-001 failure-mode replay
# ---------------------------------------------------------------------------


def test_episode_001_replay_collects_blocking_violations():
    """Replay of episode-001 failure modes: English hook + dual exposition wall."""
    shots = [
        # Shot 0: English hook → BLOCKING hook_language
        ShotScript(
            scene_prompt="x",
            narration_text="The stench of death filled the air.",
        ),
        # Filler shot
        ShotScript(scene_prompt="x", narration_text="Anh nhìn quanh."),
        # Shot 2: character intro — should NOT be flagged as lore
        ShotScript(
            scene_prompt="x",
            narration_text="Đứng cạnh là Thanh Vân Tử, ánh mắt lạnh lẽo.",
        ),
        # Shots 3-4: lore exposition wall → BLOCKING exposition_density
        ShotScript(
            scene_prompt="x",
            narration_text=(
                "Thi Du Cao là một loại thi độc cổ xưa. "
                "Loại độc này có bản chất là cổ thuật. "
                "Đó là một hiện tượng tà thuật."
            ),
            duration_sec=10.0,
        ),
        ShotScript(
            scene_prompt="x",
            narration_text=(
                "Theo truyền thuyết, Mao Sơn dạy đạo pháp cho đệ tử. "
                "Loại pháp thuật này có nguyên nhân từ cổ thuật. "
                "Đó là một định nghĩa khái niệm về tà thuật."
            ),
            duration_sec=10.0,
        ),
    ]
    for s in shots:
        populate_shot_signals(s, KNOWN_CHARS)

    vs = collect_episode_violations(
        shots,
        max_exposition_ratio=0.6,
        max_consecutive_same_energy=2,
        lore_curiosity_buffer_shots=2,
    )
    rules = {v.rule for v in vs if v.severity == Severity.BLOCKING}
    assert "hook_language" in rules
    assert "exposition_density" in rules

    # Character "Thanh Vân Tử" must NOT appear as a lore_term in shot 2.
    pn = extract_proper_nouns(shots[2].narration_text, KNOWN_CHARS)
    assert "Thanh Vân Tử" in pn["characters"]
    assert "Thanh Vân Tử" not in pn["lore_terms"]
