# Visual Brief Enrichment

## Status: Done

## Mục tiêu
Tách `_rewrite_scene_prompts_from_narration()` thành 2 bước rõ ràng:
- LLM chỉ làm extraction (hiểu nghĩa narration → structured brief)
- Python deterministic synthesis (format brief → ComfyUI tag list)

Thay thế call site bằng `_build_scene_prompts_from_narration()` wrapper.

## Roadmap

### Schema (schemas.py)
- [x] Thêm `ShotVisualBrief` model với 6 fields
- [x] Thêm `visual_brief: Optional[ShotVisualBrief] = None` vào `ShotScript`

### Backend (scriptwriter.py)
- [x] Hằng `_VISUAL_BRIEF_SYSTEM` — system prompt cho extraction pass
- [x] Hàm `_extract_visual_briefs(shots, episode_num)` — LLM batch call
- [x] Hàm `_synthesize_scene_prompt(brief, shot)` — deterministic Python, no LLM
- [x] Hàm `_synthesize_scene_prompts_from_briefs(shots, episode_num)` — iterate all shots
- [x] Hàm `_build_scene_prompts_from_narration(shots, episode_num)` — wrapper + fallback
- [x] Wire vào `write_episode_script()`: thay `_rewrite_scene_prompts_from_narration` bằng `_build_scene_prompts_from_narration`

### Testing & Verification
- [x] Unit test `_synthesize_scene_prompt()` — assert tag order, no duplicates, cap 16
- [x] Unit test `_extract_visual_briefs()` mock — hook shots skip, graceful fallback non-list
- [x] Unit test `_build_scene_prompts_from_narration()` — fallback khi 0 populated
- [x] Backward compat: load script JSON cũ không có `visual_brief` không raise error

## Acceptance Criteria
- [x] `ShotScript` với `visual_brief=None` load được từ JSON cũ không raise
- [x] `_synthesize_scene_prompt()` output đúng thứ tự: [composition+setting] → [action] → [key_objects] → [mood_lighting] → [subjects]
- [x] Hook shots (index ≤1 or duration_sec ≤3) giữ `visual_brief=None` sau extraction
- [x] Khi extracton fail 100%, `_build_scene_prompts_from_narration()` fallback về `_rewrite_scene_prompts_from_narration()` + log WARNING
- [x] `_rewrite_scene_prompts_from_narration()` vẫn còn trong code như private fallback
- [x] All tests pass

## Notes
- Branch: `feature/visual-brief-enrichment`
- `scene_prompt_client` dùng cho extraction (không phải `ollama_client`)
- Synthesis cap 16 tags (nhường buffer cho rule-based align pass)
- `key_objects` wrap `(tag:1.15)` — không phải 1.2
- Subject dedup dùng exact phrase substring check

## Implementation Summary

Implemented đúng theo plan:
- **Phase 1**: `ShotVisualBrief` Pydantic model mới trong `schemas.py`; `ShotScript.visual_brief: Optional[ShotVisualBrief] = None` backward-compatible.
- **Phase 2**: `_VISUAL_BRIEF_SYSTEM` prompt, `_extract_visual_briefs()` (LLM batch + retry), `_synthesize_scene_prompt()` (deterministic, no LLM), `_synthesize_scene_prompts_from_briefs()`, `_build_scene_prompts_from_narration()` wrapper với fallback.
- **Phase 3**: 1-line change trong `write_episode_script()` — thay `_rewrite_scene_prompts_from_narration` bằng `_build_scene_prompts_from_narration`.
- **Tests**: 20 unit tests mới covering schema backward-compat, synthesis tag order/dedup/cap, extraction hook skip/fallback/partial-failure, và wrapper fallback logic. 92 passed / 2 pre-existing failures (unrelated).

Known limitations:
- 2 test cũ (`test_write_episode_script_saves_json`, `test_write_episode_script_handles_shots_as_strings`) fail pre-existing do mock narration quá ngắn (24 words < 150) — không liên quan feature này.
- Branch: `feature/visual-brief-enrichment`
