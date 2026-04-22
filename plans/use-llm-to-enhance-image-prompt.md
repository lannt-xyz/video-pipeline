## Plan: Visual Brief Enrichment

**Status**: Draft

### Vấn đề gốc rễ

`_rewrite_scene_prompts_from_narration()` hiện tại bắt LLM làm **2 việc cùng lúc**: (1) hiểu nghĩa narration tiếng Việt, (2) format ra ComfyUI tag list chuẩn → hay ra tag đúng nhưng sai trọng tâm, bỏ sót hành động, không có foreground/background rõ.

**Giải pháp**: tách thành 2 bước rõ ràng — LLM chỉ làm việc hiểu nghĩa (extraction), Python làm việc format (synthesis). **Thay thế hoàn toàn** `_rewrite_scene_prompts_from_narration()`.

---

### Phase 1 — Schema (schemas.py)

Thêm model mới `ShotVisualBrief` trước `ShotScript`:

| Field | Type | Ví dụ |
|---|---|---|
| `subjects` | `list[str]` | `["hooded daoist figure", "kneeling young warrior"]` — role tags, max 2, không dùng tên nhân vật |
| `action` | `str` | `"figure prying open stone coffin lid with iron crowbar"` — cụ thể, observable |
| `setting` | `str` | `"dimly lit coffin shop with rows of dark wooden coffins"` |
| `key_objects` | `list[str]` | `["glowing talisman paper", "ritual candles", "iron chains on wall"]` — max 4 |
| `mood_lighting` | `str` | `"blood-red candle flame casting elongated shadows on stone wall"` |
| `composition` | `str` | `"medium close-up"` hoặc rỗng → sẽ được map từ `camera_flow` |

Sửa `ShotScript` — thêm 1 field optional backward-compatible:
```python
visual_brief: Optional[ShotVisualBrief] = None
# None ở scripts cũ → synthesis skip, scene_prompt giữ nguyên
```

---

### Phase 2 — Extraction Pass (scriptwriter.py)

#### 2a. Hằng `_VISUAL_BRIEF_SYSTEM`

System prompt **tập trung vào extraction**, không yêu cầu LLM viết tag:
- Input mỗi shot: chỉ `shot_index`, `narration_text`, `scene_id`, `characters` — **không gửi `scene_prompt` cũ** để tránh bias
- Output: JSON array `[{shot_index, subjects, action, setting, key_objects, mood_lighting, composition}]`
- Rule rõ cho từng field: `action` bắt buộc phải specific/observable (không chấp nhận "performing ritual"), `mood_lighting` bắt buộc horror-appropriate
- **`action` phải có verb cụ thể + hướng/thay đổi**: phải mô tả động từ và kết quả/hướng ("turning head sharply toward sound", "slamming fist onto table") — không chấp nhận "looking around", "standing", "mysterious gesture"
- **Blacklist pattern cho `action`**: không chấp nhận từ khóa `ritual`, `ceremony`, `pose`, `scene`, `performing`, `conducting` — LLM hay drifts sang "solemnly conducting ritual" hay "ceremonial stance" để lách rule cũ
- **`subjects` ordering**: `subjects[0]` BUỘC PHẢI là primary subject (người/vật thực hiện action). `subjects[1]` là secondary nếu có. Thứ tự này quan trọng: synthesis sẽ ưu tiên vị trí CLIP cho subjects[0]
- Fallback tốt hơn: nếu narration là discovery ("nhìn thấy", "phát hiện") → action = revealing moment; nếu là accusation ("chỉ mặt", "buộc tội") → action = accusatory gesture
- **Hallucination guard**: nếu narration thuần mô tả cảnh quan/không có hành động nhân vật rõ ràng → `action` phải là pose tĩnh hoặc chuyển động môi trường (ví dụ: `"figure crouching motionless behind stone pillar"`, `"leaves drifting through empty courtyard"`) — **không được** bịa hành động nhân vật không có trong narration
- **`mood_lighting` format**: ép output theo dạng "light source + color palette + effect" (ví dụ: `"dim amber candle light, teal shadow palette, volumetric fog drifting along floor"`) — không chấp nhận tag cảm xúc thuần túy như `"spooky lighting"` hay `"dark atmosphere"`

#### 2b. Hàm `_extract_visual_briefs(shots, episode_num) -> List[ShotScript]`

- **Client**: dùng `scene_prompt_client` (không phải `ollama_client`) — đây là client được configure riêng cho tác vụ translate narration → visual, qua `settings.effective_scene_prompt_model`
- Gửi tất cả shots trong 1 call (batch như `_rewrite_scene_prompts_from_narration()`)
- **Skip hook shots**: shots có `index <= 1` hoặc `duration_sec <= 3` giữ `visual_brief=None` (hook đã có `_HOOK_SYSTEM` riêng, narration ≤10 từ không đủ context để extract meaningful brief)
  ```python
  if idx <= 1 or shot.duration_sec <= 3:
      continue  # visual_brief stays None for hook shots
  ```
- Retry `@retry(stop=stop_after_attempt(3))` — consistent với pattern hiện có
- Parse từng item bằng `ShotVisualBrief(**item_dict)` — nếu 1 shot fail thì skip shot đó (giữ `visual_brief=None`), không abort toàn bộ
- Trả về shots với `visual_brief` được populate

#### 2c. Hàm private helper `_synthesize_scene_prompt(brief, shot) -> str`

Deterministic Python cho **1 shot**, **không có LLM**. Thứ tự tag cố định:

```
[composition + setting] → [action] → [key_objects[0..3]] → [mood_lighting] → [subjects nếu chưa có trong action]
```

- `composition` rỗng → map từ `camera_flow`: `STATIC_CLOSE`/`DETAIL_REVEAL` → `"medium close-up"`, `STATIC_WIDE` → `"wide establishing shot"`, `WIDE_TO_CLOSE` → để synthesis tự nhiên
- `key_objects` được wrap `(tag:1.15)` cho prop (không phải 1.2 — ở 1.2 object nhỏ có thể chiếm hết frame hoặc gây distortion vùng xung quanh tùy checkpoint), không wrap cho figure
- **Key objects dedup với setting**: nếu một object đã xuất hiện trong `setting` (substring check) → bỏ qua để tránh lãng phí tag budget
- **Subjects dedup**: chỉ skip subject nếu **exact phrase** của nó đã là substring của `action` (không dùng word-level match — vì "old man" sẽ bị drop nhầm khi action chứa từ "man"):
  ```python
  def _subject_already_in_action(subject: str, action: str) -> bool:
      return subject.lower().strip() in action.lower()
  ```
  Nếu không match exact phrase → append bình thường. Lặp lại nhẹ subject trong SD prompt giúp tăng convergence, không gây hại.
- **Tag priority khi cap 16** — thứ tự này đảm bảo các field quan trọng nhất không bị drop:
  1. `action` — **never drop**
  2. `setting` — **never drop**
  3. `subjects[0]` (primary) — **never drop**
  4. `mood_lighting` — **never drop**
  5. `key_objects` (fill remaining slots)
  6. `subjects[1]` (secondary) — drop nếu hết slot

#### 2d. Hàm `_synthesize_scene_prompts_from_briefs(shots, episode_num) -> List[ShotScript]`

Iterates tất cả shots, gọi `_synthesize_scene_prompt(brief, shot)` cho mỗi shot có `visual_brief is not None`; giữ nguyên `scene_prompt` cũ cho shot có `visual_brief=None`.

#### 2e. Hàm gộp `_build_scene_prompts_from_narration(shots, episode_num)`

Wrapper public, **thay thế** `_rewrite_scene_prompts_from_narration()` trong `write_episode_script()`. Logic:

```python
def _build_scene_prompts_from_narration(shots, episode_num):
    shots = _extract_visual_briefs(shots, episode_num)
    populated = sum(1 for s in shots if s.visual_brief is not None)
    if populated == 0:
        # Full extraction failure — fallback to old rewrite pass
        logger.warning(
            "Visual brief extraction yielded 0 results — falling back to rewrite pass | episode={}",
            episode_num,
        )
        return _rewrite_scene_prompts_from_narration(shots, episode_num)
    return _synthesize_scene_prompts_from_briefs(shots, episode_num)
```

`_rewrite_scene_prompts_from_narration()` **không xóa** — giữ lại là private fallback. Chỉ không được gọi trực tiếp từ `write_episode_script()` nữa.

---

### Phase 3 — Tích hợp (scriptwriter.py)

Thay đổi duy nhất trong `write_episode_script()`:

```
Trước: episode_shots = _rewrite_scene_prompts_from_narration(episode_shots, episode_num)
Sau:   episode_shots = _build_scene_prompts_from_narration(episode_shots, episode_num)
```

Thứ tự pass sau thay đổi (các pass khác **không đổi**):
```
_backfill_characters()                    ← unchanged
_build_scene_prompts_from_narration()     ← NEW (replaces _rewrite_...)
_align_scene_prompt_with_narration()      ← unchanged (safety net for missed tags)
_enforce_scene_continuity()               ← unchanged
```

Hook shots (shot 0, 1): vẫn qua `_generate_hook_shot()` riêng. `_extract_visual_briefs()` explicit skip những shot này (duration ≤ 3s) → `visual_brief=None` → `_synthesize_scene_prompts_from_briefs()` giữ nguyên `scene_prompt` từ `_HOOK_SYSTEM`.

---

### Phase 4 — Observability

- Sau extraction: log `brief.action`, `brief.key_objects`, `brief.mood_lighting` ở DEBUG level mỗi shot
- **Log missing fields**: log WARNING nếu `brief.action` rỗng, `brief.key_objects` rỗng, hoặc `brief.mood_lighting` rỗng; log summary sau batch: `"Visual brief quality | episode={} empty_action={} empty_objects={} empty_mood={}"`
- Sau synthesis: log old vs new `scene_prompt[:80]` ở DEBUG level
- `visual_brief` được serialize vào script JSON tự động (field của `ShotScript`) → inspect trực tiếp khi debug

---

### Relevant Files

- schemas.py — thêm `ShotVisualBrief`, field `visual_brief: Optional[ShotVisualBrief] = None` vào `ShotScript`
- scriptwriter.py — thêm `_VISUAL_BRIEF_SYSTEM`, `_extract_visual_briefs()`, `_synthesize_scene_prompt()`, `_synthesize_scene_prompts_from_briefs()`, `_build_scene_prompts_from_narration()`; **giữ** `_rewrite_scene_prompts_from_narration()` như private fallback
- test_llm.py — test synthesis và mock extraction
- test_pipeline.py — backward compat với script JSON cũ

**Không cần sửa**: orchestrator.py, frame_decomposer.py, workflows ComfyUI, `settings.yaml`.

---

### Verification

1. Load JSON script cũ không có `visual_brief` → `ShotScript(**data)` không raise
2. Unit test `_synthesize_scene_prompt()` với 1 brief điển hình → assert thứ tự tag đúng, không mất field, subjects không bị duplicate với action
3. Mock LLM response → assert `_extract_visual_briefs()` parse đúng, hook shots (index 0-1, duration ≤3s) giữ `visual_brief=None`, và graceful fallback khi LLM trả về non-list
4. Mock extraction trả về 0 results → assert `_build_scene_prompts_from_narration()` fallback về `_rewrite_scene_prompts_from_narration()` và log WARNING
5. Dry-run 1 episode, inspect script JSON: `visual_brief` có dữ liệu cho shots 2+, `scene_prompt` phản ánh narration (không cần chạy GPU)
6. **Metric đo được — Action Quality**: sau extraction, đếm % shots 2+ có `visual_brief.action` không chứa forbidden phrases và blacklist pattern (`"ritual"`, `"ceremony"`, `"pose"`, `"scene"`, `"performing"`, `"conducting"`) → target ≥ 90%
7. **Metric đo được — Keyword Preservation Rate**: % danh từ quan trọng trong `narration_text` (characters, key objects có trong `characters` list và `key_objects` expected) xuất hiện trong `final_scene_prompt` sau synthesis + alignment passes → target ≥ 80%
8. **Metric đo được — Visual Density Score**: avg số tags/shot sau synthesis (trước alignment pass) → target 10–14 tags (sweet spot CLIP; dưới 10 = thiếu detail, trên 14 = dilution)
9. Chạy image gen thử 2-3 shots có narration ngắn, so sánh ảnh trước/sau

---

### Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Extraction fail toàn batch (Ollama overload, JSON parse error) | Medium | Medium | Fallback về `_rewrite_scene_prompts_from_narration()` + log WARNING |
| Tag count vượt 24 sau khi `_align_scene_prompt_with_narration()` inject thêm | Medium | Low | Synthesis cap 16 tags; orchestrator `_compact_prompt_tags(max_tags=24)` là backstop |
| Hook shots bị overwrite bởi brief extraction | Low | High | Explicit skip `duration_sec <= 3` trong `_extract_visual_briefs()` |
| `subjects` bị duplicate với `action` gây CLIP over-weighting | Low | Low | `_subject_already_in_action()` dedup |
| `brief.setting` trùng với `scene_env_baselines` từ orchestrator | Low | Low | Orchestrator đã có guard kiểm tra substring trước khi prepend |

### Rollback Plan

Đổi 1 dòng trong `write_episode_script()` từ `_build_scene_prompts_from_narration(...)` về `_rewrite_scene_prompts_from_narration(...)`. Hàm cũ được giữ lại, không xóa.

### Decisions

- **Không xóa** `_rewrite_scene_prompts_from_narration()` — giữ là private fallback trong `_build_scene_prompts_from_narration()`
- **Client**: dùng `scene_prompt_client` (không phải `ollama_client`) cho extraction pass
- `_align_scene_prompt_with_narration()` (rule-based) giữ nguyên — vẫn là safety net inject tag bị sót
- `visual_brief` lưu trong script JSON để traceability; không cần file debug riêng
- Synthesis cap 16 tags (không phải 20) để nhường buffer cho rule-based pass
- Out of scope đợt này: frame-level brief, motif database, embedding validator