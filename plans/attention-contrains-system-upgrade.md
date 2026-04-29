# Plan: Attention-Constraint System Upgrade

> **Implementation status (live):**
> - [x] **Phase 1a** — Schema + Config (PR1)
> - [x] **Phase 1b** — Validator + Baseline (PR2) — 19/19 tests pass; baseline empty until first script run
> - [x] **Phase 2** — Hook Extraction stage (PR3) — `llm/hook_extractor.py` + orchestrator wiring + scriptwriter `viral_moments` param; 10/10 tests pass; behind `retention.use_constraint_system` flag
> - [x] **Phase 3** — Scriptwriter V2 + visual constraint (PR4 part 1) — `_SCRIPTWRITER_SYSTEM_V2` rule-list prompt; flag-based switch in `_write_raw`; `populate_shot_signals` after script generation; `prompt_version` tag; focal-point cap (≤2) in `_synthesize_scene_prompt`; SHOCK→close-up override in `decompose_shot`. All behind flag; 29/29 tests pass.
> - [x] **Phase 4** — Hook competitive selection (PR4 part 2) — `llm/hook_judge.py` (3 candidates → language pre-filter → rubric judge → weighted score → 1 retry on weak winner); persists to `data/{slug}/hook_candidates/episode-NNN.json`; integrated into `write_episode_script` behind flag with legacy fallback. 8 new tests, 37/37 total pass.
> - [x] **Phase 5** — Gatekeeper Reviewer + retry loop (PR5) — `llm/gatekeeper.py` (pure judge: BLOCKING/WARNING tiers using `constraint_validator`); `regenerate_failed_shots` in scriptwriter (anti-bias rewrite prompt); orchestrator `_run_gatekeeper_review` with bounded retry (`reviewer_max_retries`); JSONL audit log to `logs/retention_violations.jsonl`; graceful degrade. 4 new tests, 41/41 phase tests pass.
> - [~] **Phase 6** — Calibration + flip flag (PR6) — infrastructure ready: `retention_report.py` extended với hook score variance check (`discrimination_ok` if stddev ≥ 0.05). **Manual steps remaining (require real batch run):** (a) flip `use_constraint_system: true` on a 10-episode sample; (b) hand-rate hooks; (c) tune `hook_min_score`, `max_exposition_ratio`, `hook_judge_weights` based on report; (d) write ADR; (e) flip flag for full batch.

Chuyển toàn bộ retention logic từ "style guidance" sang **hard constraint system** có schema, có gatekeeper, có bounded retry. Không refactor pipeline architecture (orchestrator giữ nguyên), chỉ nâng cấp 4 stage hiện có + thêm 1 micro-stage hook scoring + thêm structured fields trong schema để enforce được.

## Bối cảnh phát hiện được

- Hook đã có (`_generate_hook_shot`, `_HOOK_SYSTEM`, budget 10 từ) nhưng **single-shot, không có scoring/ranking** — reviewer 2 nói đúng: "hook không bao giờ đủ mạnh" vì không có competitive selection.
- `script_reviewer.py` hiện là **advisory + best-effort patcher**, KHÔNG retry lại scriptwriter, KHÔNG block pipeline → constraint không bao giờ được enforce thật.
- Bug cụ thể trong `episode-001-script.json`: shot 0 narration là **tiếng Anh** ("The stench of death...") → reviewer có check Vietnamese diacritic nhưng là warn-only.
- Shots 3-4 dài 10s nhồi exposition lore ("Thi Du Cao là...", "mang nó lên Mao Sơn dạy đạo pháp...") — vi phạm rule "curiosity before lore" nhưng không có constraint nào catch.
- `summarizer.py` output là `ArcOverview` faithful summary → scriptwriter không có signal "moment nào đáng làm hook".
- `frame_decomposer.py` không có concept energy_level / focal point count → 2 shot liền nhau cùng "tone" không bị catch.

## TL;DR — 5 thay đổi cốt lõi

1. **Schema thêm structured signal** (energy, exposition ratio, proper noun, hook score) → có gì để measure
2. **Tách hook extraction khỏi summarizer** → 1 stage riêng generate `viral_moment` candidates (chỉ drive hook + 1-2 key shots, KHÔNG override narrative)
3. **Scriptwriter prompt → constraint list cứng** + per-shot validator function (không chỉ regex Vietnamese)
4. **Script reviewer → gatekeeper với bounded retry** (max 2) + degrade gracefully + retry directive **tăng tension** (không chỉ fix violation)
5. **Hook generator → competitive selection** (3 candidates → LLM judge với rubric → pick best)

## Severity Philosophy — tránh over-constraining

Quá nhiều rule BLOCKING ⇒ LLM tối ưu "pass test" thay vì "hấp dẫn". Giải pháp: chia 2 tier rõ ràng.

**HARD (BLOCKING — chặn pipeline, retry bắt buộc)** — chỉ những lỗi không thể chấp nhận:
- Hook không phải tiếng Việt
- Hook > 10 từ
- 2 shot exposition liên tiếp với `exposition_ratio > 0.6`
- Duration cap vi phạm (shot > 12s, episode total ngoài [55, 65])
- Total words < 200
- Character name typo (giữ legacy check)

**SOFT (WARNING — log, không block, không retry)**:
- Energy monotony
- Lore-before-curiosity (kể cả episode đầu arc)
- Single shot exposition cao (1 shot lẻ)
- Focal point cap exceeded (dùng truncate path thay vì retry)
- Banned opening detected (dùng làm hint cho prompt, không reject)

⇒ Mục tiêu: ≥90% episode pass gate ngay lần đầu. Retry chỉ dành cho "không xem được", không phải "chưa hoàn hảo".

---

## Layer × Phase Matrix

| Layer | Files | Phases |
|---|---|---|
| Schema | [models/schemas.py](../models/schemas.py) | P1 |
| Validator (pure Python) | [llm/constraint_validator.py](../llm/constraint_validator.py) **NEW** | P1 |
| LLM stages | [llm/hook_extractor.py](../llm/hook_extractor.py) **NEW**, [llm/hook_judge.py](../llm/hook_judge.py) **NEW**, [llm/scriptwriter.py](../llm/scriptwriter.py), [llm/script_reviewer.py](../llm/script_reviewer.py) | P2-P5 |
| Visual | [video/frame_decomposer.py](../video/frame_decomposer.py) | P3 |
| Pipeline | [pipeline/orchestrator.py](../pipeline/orchestrator.py) | P2, P5 |
| Config | [config/settings.yaml](../config/settings.yaml) | P1 |
| Tests | [tests/test_constraint_validator.py](../tests/test_constraint_validator.py) **NEW**, ... | P1, P3, P5 |
| Tooling | [scripts/retention_report.py](../scripts/retention_report.py) **NEW**, [scripts/backfill_constraint_fields.py](../scripts/backfill_constraint_fields.py) **NEW** | P1, P5 |

---

## State Persistence — Đã verify

- `pipeline/state.py` chỉ lưu **status enums** (`SCRIPTED`, `SCRIPT_REVIEWED`...), KHÔNG serialize `EpisodeScript` JSON vào SQLite.
- Script payload persist dưới dạng file: `data/{slug}/scripts/episode-NNN-script.json`.
- ⇒ Migration **chỉ cần xử lý JSON files**, không cần ALTER TABLE. Pydantic load JSON cũ với fields mới = default từ schema.
- Tuy nhiên, để phân biệt "field chưa compute" vs "default value thật", dùng `Optional[...] = None` cho fields auto-derived (`energy_level`, `exposition_ratio`, `proper_nouns`). Validator gặp `None` → tự compute và backfill.

---

## Phase 1: Schema & Constraint Foundation (blocker cho mọi thứ)

**Mục tiêu**: có structured fields + measurement primitives. Không có signal thì reviewer chỉ đoán.

### Phase 1a — Schema + Config (PR1, review nhanh)

1. Thêm vào `models/schemas.py`:
   - `EnergyLevel` enum: `low | med | high | shock`
   - `ShotScript.energy_level: Optional[EnergyLevel] = None` — sentinel `None` = chưa compute
   - `ShotScript.proper_nouns: Optional[List[str]] = None` — auto-derived
   - `ShotScript.exposition_ratio: Optional[float] = None` — auto-derived
   - `HookCandidate` model: `text: str`, `visual_seed: str`, `curiosity_score: float`, `specificity_score: float`, `pattern_interrupt_score: float`, `total_score: float`, `rationale: str`
   - `ViralMoment` model: `chapter_refs: List[int]`, `description: str`, `shock_factor: str`, `mystery_seed: str`
   - `EpisodeScript.hook_strength: Optional[float] = None`
   - `EpisodeScript.constraint_violations: List[str] = []` — log violations đã pass gatekeeper sau retry
   - `EpisodeScript.prompt_version: str = "v1"` — bump khi đổi prompt để DB trace được

2. Thêm `config/settings.yaml`:
   ```yaml
   retention:
     use_constraint_system: false   # feature flag — Phase 1 ship off, bật dần ở P3+
     max_exposition_ratio: 0.5      # placeholder — calibrate sau Phase 1b baseline
     max_consecutive_same_energy: 2
     lore_curiosity_buffer_shots: 2
     hook_min_score: 0.65
     reviewer_max_retries: 2
     hook_judge_weights:
       curiosity_gap: 0.5
       specificity: 0.25
       pattern_interrupt: 0.25
   ```

### Phase 1b — Validator + Baseline (PR2, không touch pipeline)

3. Tạo `llm/constraint_validator.py` (pure Python, không LLM):

   3.1 `compute_exposition_ratio(narration: str) -> float`
   - Tách câu theo `. ! ?`
   - **Multi-signal v0** (combine để giảm noise — câu tiếng Việt linh hoạt, single signal sẽ misclassify):
     - **Tension signals** (vote câu = "tension"):
       - Mở đầu bằng 1 trong whitelist 30 verbs (`nhìn, nghe, hét, chạy, đập, mở, chạm, ngã, cắt, đốt, đứng, quay, vung, gọi, túm, đẩy, kéo, xé, đâm, lao, vọt, rít, run, trợn, há, lùi, vỗ, đỡ, cản, giật`)
       - Có dấu `?` hoặc `!`
       - Có subject + action verb mạnh ở giữa câu (regex pronoun + verb whitelist)
     - **Exposition signals** (vote câu = "expository"):
       - Match blacklist openings (`X là `, `Theo `, `Vì `, `Đó là `, `Loại `, `Một loại `)
       - Chứa abstract noun (`loại, bản chất, hiện tượng, nguyên nhân, cổ thuật, đạo pháp` — danh sách 20 từ)
       - Không có subject cụ thể (không có pronoun đầu câu, không có character name)
       - Không có action verb mạnh trong câu
   - Câu được classify theo bucket có nhiều signal nhất (≥2 signal cùng tier). Ngang bằng → neutral.
   - `ratio = expository_count / max(1, expository_count + tension_count)`
   - Document: heuristic v0 multi-signal, calibrate ở Phase 6.

   3.2 `extract_proper_nouns(narration: str, known_chars: list[str]) -> dict[str, list[str]]`
   - Return `{"characters": [...], "lore_terms": [...]}`
   - Regex `[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){0,3}` (chuỗi 1-4 từ viết hoa)
   - Phân loại: nếu match `known_chars` (case-insensitive) → `characters`, ngược lại → `lore_terms`
   - **Lý do tách**: rule lore-before-curiosity chỉ apply cho `lore_terms`, không apply cho `characters` (episode đầu phải introduce nhân vật).

   3.3 `infer_energy_level(shot) -> EnergyLevel` — bảng mapping spec:

   | Signal | shock | high | med | low |
   |---|---|---|---|---|
   | `shot_subject` | `wound`, `corpse_face`, `bloody_object` | `supernatural_entity`, `person_action` (+ action keyword) | `ritual_object`, `person_action` (no keyword) | `environment` |
   | `camera_flow` | `detail_reveal` | `wide_to_close` | `pan_across`, `static_close` | `static_wide` |
   | `duration_sec` | ≤3 | ≤5 | 6-8 | ≥9 |
   | narration keywords | `máu, hét, cắt, đâm` | `đập, vung, chạy, lao, vọt` | `nói, đứng, quay` | `nhìn xa, lặng, mờ` |

   - Mỗi signal vote 1 bucket. Energy = mode (bucket có nhiều vote nhất). Tie-break ưu tiên cao hơn.

   3.4 `check_lore_before_curiosity(shots, buffer=2) -> list[Violation]`
   - Với mỗi shot có `lore_terms` mới (không xuất hiện ở shot trước):
     - Đếm số shot trước có `energy_level >= HIGH` HOẶC `exposition_ratio < 0.3` HOẶC narration chứa `?`/`!`
     - Nếu < `buffer` → **WARNING** violation (đã downgrade từ BLOCKING — xem Severity Philosophy)
   - Lý do SOFT: rule này dễ false-positive cho truyện tu tiên vốn lore-heavy. Log để Phase 6 calibrate, không block pipeline.

   3.5 `check_energy_monotony(shots, max_consec=2) -> list[Violation]` — WARNING nếu `> max_consec` shot liên tiếp cùng `energy_level`.

   3.6 `check_exposition_density(shots, threshold=0.6) -> list[Violation]` — **BLOCKING** nếu 2 shot liên tiếp `exposition_ratio > threshold` (cố ý giữ HARD vì đây là failure mode chính của episode-001).

   3.7 `check_hook_language(shot_0) -> Violation | None` — **BLOCKING** nếu shot 0 narration không có dấu tiếng Việt (catch hook tiếng Anh như episode-001).

   3.8 `populate_shot_signals(shot, known_chars) -> ShotScript` — gán energy/proper_nouns/exposition_ratio nếu `None`.

4. Tạo `scripts/backfill_constraint_fields.py`:
   - Quét `data/*/scripts/*.json`, load với schema mới, chạy `populate_shot_signals` cho mọi shot có sentinel `None`, save back.
   - Idempotent (chạy lại không đổi gì).
   - Dry-run flag mặc định bật, ghi diff log → ông review trước khi commit batch.

5. Tạo `tests/test_constraint_validator.py` với fixtures từ `episode-001-script.json`:
   - Hook shot 0 ("The stench of death...") → `check_hook_language` BLOCKING
   - Shots 3-4 (Thi Du Cao + Mao Sơn lore dump) → `check_exposition_density` BLOCKING + `check_lore_before_curiosity` WARNING
   - "Thanh Vân Tử" trong shot 2 → KHÔNG bị flag (là character, không phải lore term)

6. Chạy `scripts/retention_report.py` (sẽ tạo ở P5, có thể prototype version mini ở P1b) trên all existing scripts → **baseline numbers** cho exposition_ratio, energy distribution, hook violations. Output file `logs/baseline_violations.json`.

### Verification Phase 1
- [ ] Unit tests pass với fixtures từ episode-001
- [ ] Backfill dry-run trên `data/mao-son-troc-quy-nhan/scripts/` không lỗi
- [ ] Baseline report có số liệu cụ thể → cập nhật `retention.max_exposition_ratio` từ placeholder `0.5` thành value đã calibrate (ví dụ percentile 60 của baseline)
- [ ] Feature flag `use_constraint_system: false` → pipeline cũ chạy không thay đổi behavior

---

## Phase 2: Hook Extraction Stage Tách Riêng (PR3)

**Mục tiêu**: summarizer chỉ làm summary; chọn "viral moment" là job riêng với prompt bias hoàn toàn khác. Additive — feature flag off = pipeline cũ.

### Steps

7. Tạo `llm/hook_extractor.py`:
   - Input: `ArcOverview` + raw chapter chunks
   - Output: `List[ViralMoment]` (3-5 candidates)
   - Prompt bias: **không faithful**, cố ý chọn moment shocking/mysterious/twist; cho phép drop context; ưu tiên unanswered question
   - Persist: `data/{slug}/viral_moments/episode-NNN.json`

8. **Pipeline integration — DECIDED: sub-step trong `run_llm()`** (không thêm phase mới).
   - Lý do: phase `llm` đã holding LLM context qua VRAM manager; tách phase mới = acquire/release lại = waste.
   - Trong `pipeline/orchestrator.py::run_llm()`, thứ tự: `summarize → extract_viral_moments → scriptwriter(arc, viral_moments)`.
   - Behind feature flag `retention.use_constraint_system`: false = pass `viral_moments=None`, scriptwriter giữ behavior cũ.

9. Update `llm/scriptwriter.py` accept `viral_moments: Optional[List[ViralMoment]] = None`. Khi non-None:
   - Hook shot phải reference 1 viral_moment được chọn
   - **Tối đa 1-2 key_shot** align với `mystery_seed`
   - Còn lại của script GIỮ faithful theo `ArcOverview` từ summarizer
   - **Lý do constraint**: viral_moment cho phép "drop context" (Phase 2 step 7) — nếu drive toàn bộ narrative sẽ lệch tone story. Chỉ dùng để hook + tease, narrative chính vẫn từ summary.

### Verification Phase 2
- [ ] `viral_moments/episode-001.json` khác nội dung với `summaries/episode-001.json` (semantic diff, không phải copy)
- [ ] Manual review 5 episode đầu: ≥3/5 viral_moment đáng làm hook
- [ ] Feature flag off → output identical với baseline (regression test)

---

## Phase 3: Scriptwriter Constraint System (PR4, behind feature flag)

**Mục tiêu**: prompt là **rule list**, không phải style guidance. Mỗi rule có validator tương ứng.

### Steps

10. Lưu prompt cũ thành `_LEGACY_SCRIPTWRITER_SYSTEM` trong [llm/scriptwriter.py](../llm/scriptwriter.py). Tạo `_SCRIPTWRITER_SYSTEM_V2` với format:
    - **`[NARRATION RULES]`**: numbered hard limits — max 15 từ/câu, max 2 câu expository liên tiếp, banned openings (`X là một loại`, `Theo `, `Vì `), required tension verb trong shot 1
    - **`[STRUCTURE RULES]`**: shot count, durations, key_shot count
    - **`[LORE-CURIOSITY RULES]`**: lore term chỉ xuất hiện sau ≥2 shot tension; câu giới thiệu lore phải có hậu quả/mystery, không định nghĩa
    - **`[VISUAL-NARRATION ALIGNMENT]`**: scene_prompt phải match action chính của narration
    - **`[OUTPUT CONTRACT]`**: JSON example với constraint annotations
    - Anti-patterns kèm ví dụ ❌/✅ (lấy thẳng từ episode-001-script.json shots 3-4 làm ❌)
    - Bump `prompt_version` constant.

11. Switch giữa V1/V2 dựa trên `retention.use_constraint_system` flag. V2 default sau Phase 4 ổn định.

12. Sau khi scriptwriter trả JSON, gọi `populate_shot_signals()` cho mọi shot → fields auto-derived. Validator collect violations vào `episode.constraint_violations` (raw, chưa filter severity).

13. Visual constraint cho `video/frame_decomposer.py`:
    - Hard cap: `len(visual_brief.subjects) + len(visual_brief.key_objects) ≤ 2` cho focal points trong scene_prompt synthesis.
    - **Truncation priority** khi vượt cap: bỏ `key_objects` trước (giữ ưu tiên `subjects` vì chứa character anchor cho [image_gen/character_gen.py](../image_gen/character_gen.py)). Nếu sau khi bỏ hết `key_objects` vẫn > 2 subjects → giữ 2 subject đầu.
    - Energy-aware frame splitting: `shot.energy_level == SHOCK` → frame 0 phải là close-up (override `_FLOW_FRAMES` mapping).
    - Integration test: render 1 shot có character anchor sau truncate → confirm anchor reference vẫn match.

### Verification Phase 3
- [ ] Regenerate `episode-001` với V2 prompt → `avg(exposition_ratio)` giảm ≥30% so với baseline
- [ ] Hook shot 0 narration tiếng Việt, ≤10 từ
- [ ] Mọi shot có `energy_level` (không còn `None` sau populate)
- [ ] No ComfyUI anchor mismatch sau focal point truncate

---

## Phase 4: Hook Competitive Selection (PR4, đi cùng Phase 3)

**Mục tiêu**: hook hiện tại single-shot, no choice → chuyển sang generate-multiple + score + pick.

### Steps

14. Refactor `_generate_hook_shot()` trong [llm/scriptwriter.py](../llm/scriptwriter.py):
    - 1 LLM call sinh **3 candidates** (prompt: "produce 3 distinct variants emphasizing different angles")
    - Mỗi candidate: `text` (≤10 words), `visual_seed`
    - **Pre-filter trước khi judge**: chạy `check_hook_language()` reject candidate không có dấu tiếng Việt → tránh waste judge call.

15. Tạo `llm/hook_judge.py`:
    - Rubric prompt yêu cầu LLM trả JSON: `{curiosity_gap: 0-1, specificity: 0-1, pattern_interrupt: 0-1, rationale: str}` per candidate
    - **Rubric prompt directive (anti-bias)**:
      - **Ưu tiên** câu khiến người xem phải tự hỏi: "chuyện gì đang xảy ra?" / "sao lại như vậy?"
      - **Penalize** câu quá rõ nghĩa (đã trả lời câu hỏi của chính nó)
      - **Penalize** câu generic horror ("bóng tối ập đến", "máu đã chảy", "cái chết đang gần kề")
      - **Penalize** câu "nghe hay" nhưng không tạo curiosity gap
      - Rationale phải nêu rõ: candidate này tạo question gì trong đầu người xem?
    - `total_score = Σ weight_i × score_i` với weights từ `retention.hook_judge_weights` (config-driven)
    - Pick best; nếu `best.total_score < retention.hook_min_score`:
      - **Retry policy**: 1 lần. Generate 3 candidates **mới hoàn toàn** (không kế thừa), prompt thêm context: `"Previous attempts scored low on {weakest_dim}. Sample weak attempts: [list]. Generate 3 new variants fixing this dimension."`
      - Sau retry vẫn fail → chấp nhận best hiện tại + push WARNING vào `constraint_violations`
    - Trả về `(winner: HookCandidate, all_candidates: List[HookCandidate])`

16. Persist tất cả candidates + scores: `data/{slug}/hook_candidates/episode-NNN.json` (cho calibration Phase 6).

17. **Bias mitigation — score variance check**:
    - Sau mỗi 10 episode, compute `stddev(total_score across all candidates)` từ `hook_candidates/*.json`.
    - Nếu stddev < 0.1 → judge không discriminate → log ERROR + flag để Phase 6 calibrate (đổi rubric prompt hoặc swap judge model).
    - Implement check trong `scripts/retention_report.py`.

### Verification Phase 4
- [ ] 10 episode đầu: hook score stddev ≥ 0.15 (judge có discrimination)
- [ ] Manual review: ≥7/10 winner thật sự tốt hơn 2 loser còn lại
- [ ] Hook tiếng Anh không lọt qua pre-filter (confirm bằng adversarial test)

---

## Phase 5: Reviewer thành Gatekeeper với Bounded Retry (PR5)

**Mục tiêu**: chuyển reviewer từ advisory → enforcement với fail-safe. **Single ownership**: orchestrator điều phối retry, reviewer là pure judge.

### Steps

18. Refactor `llm/script_reviewer.py`:
    - **BỎ pass-2 LLM auto-fix** khỏi reviewer. Lý do: tránh trùng trách nhiệm regenerate giữa reviewer và orchestrator → blow-up token.
    - Reviewer = pure judge: input `EpisodeScript`, output `ReviewResult{passed: bool, blocking: List[Violation], warnings: List[Violation]}`.
    - Pass 1 dùng functions từ `constraint_validator.py` (không tự code lại regex).
    - **Severity tier**:
      - `BLOCKING`: hook fail (`check_hook_language`), `check_lore_before_curiosity` BLOCKING tier, `check_exposition_density`, total_words < floor, character name typo (existing)
      - `WARNING`: `check_energy_monotony`, lore-before-curiosity ở episode 1 exception, focal point cap exceeded
    - Giữ legacy advisory mode dưới `_LEGACY_review_episode_script()` cho rollback (feature flag off → dùng legacy).

19. Update `pipeline/orchestrator.py::run_script_review()`:
    ```python
    for attempt in range(retention.reviewer_max_retries + 1):
        result = review_episode_script(script)
        if result.passed:
            break
        if attempt == retention.reviewer_max_retries:
            # graceful degrade
            script.constraint_violations = [v.to_log() for v in result.blocking + result.warnings]
            log.warning(f"Episode {ep} failed after {attempt} retries; degrading.")
            break
        # regenerate ONLY failed shots, with neighbor context
        failed_indices = {v.shot_idx for v in result.blocking}
        script = regenerate_failed_shots(
            script,
            shot_indices=failed_indices,
            blocking_reasons=result.blocking,
            arc=arc,
            viral_moments=viral_moments,
        )
    ```
    - `regenerate_failed_shots()` là helper mới trong `scriptwriter.py`: rewrite per-shot với context = `(shot_before, shot_to_fix, shot_after, blocking_reasons)`.
    - **Retry directive (anti-mediocrity)**: prompt regenerate KHÔNG chỉ "fix the violation". Phải kèm positive directive:
      - `"Rewrite this shot to be MORE tense / MORE mysterious than original. Increase curiosity, don't just fix the listed issues."`
      - Nếu chỉ fix violation → LLM convergence về "trung bình nhưng pass rule" → mất impact.
      - Pass cả `arc.key_events` + adjacent shots làm context để LLM hiểu "điểm leo thang" của arc.
    - Tiết kiệm token + giữ coherence + tăng impact thay vì giảm.
    - Sau loop: KHÔNG fail pipeline. Episode tiếp tục sang phase `images` với `constraint_violations` log.

20. Tạo `scripts/retention_report.py`:
    - Quét all episodes (DB + JSON files), output bảng: `episode | hook_strength | avg_exposition_ratio | violation_count | hook_score_stddev`
    - Format: human-readable table + JSON (cho automation)
    - Flag episode "needs manual review" nếu: `violation_count > 3` HOẶC `hook_strength < 0.5` HOẶC `avg_exposition_ratio > 0.6`

21. Structured logging: violations log dưới dạng JSON line vào `logs/retention_violations.jsonl` với `{episode, shot_idx, rule, severity, value, threshold, timestamp}`. Phục vụ Phase 6.

### Verification Phase 5
- [ ] Episode-001 regenerate: pass gatekeeper (hook tiếng Anh fix sau retry 1, shots 3-4 lore dump fix)
- [ ] Adversarial test: inject 1 episode bad input cố tình → graceful degrade, không crash batch, có entry trong `retention_violations.jsonl`
- [ ] Token cost: 1 retry episode ≤ 1.5× cost của episode pass ngay (no blow-up)
- [ ] `retention_report.py` chạy được trên existing data, output đúng format

---

## Phase 6: Calibration Loop (manual, sau khi P1-P5 done)

22. Hand-label 10 episode đầu output: chấm 1-5 sao về "watchability" + ghi note.
23. Cross-reference với `logs/retention_violations.jsonl` + `hook_candidates/`: tìm correlation giữa metric vs hand-rating.
24. Adjust:
    - `retention.max_exposition_ratio` (siết nếu 5★ episode đều ≤ X)
    - `retention.hook_judge_weights` (re-weight nếu hand-rated winner ≠ judge winner)
    - Rubric prompt nếu judge bias (recommend Option B: swap sang model khác — xem Open Questions Q4)
25. Tài liệu hóa decisions vào [plans/architecture.md](architecture.md) (ADR mới).

**Đừng skip phase này** — không có ground truth thì rubric chỉ là guess.

---

## Relevant Files

| File | Status | Phase |
|---|---|---|
| [models/schemas.py](../models/schemas.py) | edit | P1a |
| [config/settings.yaml](../config/settings.yaml) | edit | P1a |
| [llm/constraint_validator.py](../llm/constraint_validator.py) | **NEW** | P1b |
| [scripts/backfill_constraint_fields.py](../scripts/backfill_constraint_fields.py) | **NEW** | P1b |
| [tests/test_constraint_validator.py](../tests/test_constraint_validator.py) | **NEW** | P1b |
| [llm/hook_extractor.py](../llm/hook_extractor.py) | **NEW** | P2 |
| [llm/scriptwriter.py](../llm/scriptwriter.py) | edit (V2 prompt, multi-candidate hook, regenerate_failed_shots) | P2-P5 |
| [llm/hook_judge.py](../llm/hook_judge.py) | **NEW** | P4 |
| [video/frame_decomposer.py](../video/frame_decomposer.py) | edit (focal point cap, energy-aware) | P3 |
| [llm/script_reviewer.py](../llm/script_reviewer.py) | edit (pure judge, severity tiers) | P5 |
| [pipeline/orchestrator.py](../pipeline/orchestrator.py) | edit (sub-step hook_extract, retry loop) | P2, P5 |
| [scripts/retention_report.py](../scripts/retention_report.py) | **NEW** | P5 |
| [llm/summarizer.py](../llm/summarizer.py) | KHÔNG đụng | — |

---

## Acceptance Criteria (toàn cục)

### Phase 1 done
- [ ] Unit tests pass với fixtures từ `episode-001-script.json` (hook tiếng Anh, lore dump, exposition density bị catch)
- [ ] `scripts/backfill_constraint_fields.py --dry-run` chạy clean trên all `data/*/scripts/`
- [ ] `logs/baseline_violations.json` có số liệu cụ thể để chốt thresholds
- [ ] Feature flag off → pipeline cũ behavior identical

### Phase 2 done
- [ ] `viral_moments/episode-001.json` ≠ `summaries/episode-001.json` semantic
- [ ] Manual: ≥3/5 viral_moment "đáng làm hook"

### Phase 3 done
- [ ] Episode-001 regenerate: `avg(exposition_ratio)` giảm ≥30% so baseline
- [ ] Mọi shot có `energy_level` populated
- [ ] No ComfyUI anchor regression sau focal point truncate

### Phase 4 done
- [ ] Hook score stddev ≥ 0.15 trên 10 episode (judge discriminates)
- [ ] Manual review: ≥7/10 winner đúng
- [ ] Hook tiếng Anh không lọt qua pre-filter

### Phase 5 done
- [ ] Episode-001 pass gatekeeper sau ≤2 retry
- [ ] Adversarial bad-input → graceful degrade, không crash
- [ ] 1 retry episode ≤ 1.5× token cost của pass-ngay episode
- [ ] `retention_report.py` output đúng format

### Phase 6 done (post-rollout)
- [ ] Hand-rate 10 episode mới: ≥7/10 đạt ≥3 sao
- [ ] Threshold values trong settings.yaml đã calibrate dựa trên data, không phải placeholder
- [ ] ADR ghi vào [plans/architecture.md](architecture.md)

### Production rollout gate
- [ ] 0 BLOCKING violation sau retry trên ≥90% episode trong batch test 20
- [ ] Avg `exposition_ratio` ≤ 0.4 (sau Phase 1 baseline calibrate threshold cụ thể)
- [ ] `retention.use_constraint_system: true` mặc định trong config

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Over-constraining → output "đúng luật nhưng nhạt"** | High | High | HARD/SOFT severity split (chỉ 6 rule BLOCKING); retry directive "increase tension", không chỉ "fix violation"; Phase 6 hand-rate phát hiện sớm nếu output convergence về mediocrity |
| Schema migration làm hỏng episode đã render | Low | High | State persistence chỉ lưu enum (đã verify); JSON schema dùng `Optional + None` sentinel; backfill dry-run trước khi commit batch |
| Outer + inner retry loop blow-up token | Med | Med | **Single ownership**: bỏ pass-2 LLM auto-fix khỏi reviewer; orchestrator điều phối; `regenerate_failed_shots` only on failed indices |
| Lore-before-curiosity false positives ở ep đầu arc | Med | Low | Đã downgrade thành SOFT/WARNING — không block, chỉ log. Vẫn tách `characters` vs `lore_terms` để report chính xác |
| LLM judge self-preference bias | High | High | Score variance check sau mỗi 10 episode (stddev ≥ 0.15); rubric prompt anti-bias directives (penalize "câu nghe hay", ưu tiên "câu tạo question"); Phase 6 manual calibration mandatory |
| Ollama không follow long rule list V2 prompt | Med | High | Feature flag `use_constraint_system`; legacy `_LEGACY_SCRIPTWRITER_SYSTEM` giữ trong code; `prompt_version` tag để DB trace |
| `_synthesize_scene_prompt` truncate phá ComfyUI anchor | Low | High | Truncate priority order (drop key_objects trước); integration test render 1 shot post-truncate; assertion anchor matches |
| Heuristic `compute_exposition_ratio` quá noisy cho tiếng Việt | Med | Med | Multi-signal v0 (verb whitelist + abstract noun + subject detect + punctuation); calibrate Phase 6; document là heuristic |
| Viral moment extractor "drop context" lệch tone narrative | Med | Med | Constraint scope: viral_moment chỉ drive hook + ≤2 key_shot, không override summary-driven narrative chính; manual review 5 episode đầu trước khi enable flag |
| Retry loop convergence về mediocrity ("pass rule but boring") | High | High | Retry prompt yêu cầu "increase tension/mystery" tích cực, không chỉ fix violation; max 2 retry hard cap; Phase 6 hand-rate detect early |

---

## Rollback Plan

1. **Feature flag**: `retention.use_constraint_system: false` trong [config/settings.yaml](../config/settings.yaml). Flip = quay về behavior cũ ngay, không restart pipeline.
2. **Legacy code paths**: giữ `_LEGACY_SCRIPTWRITER_SYSTEM`, `_LEGACY_review_episode_script()` ít nhất 2 sprint sau khi V2 default. Code phải comment rõ "DO NOT REMOVE — rollback path".
3. **Prompt version tracking**: `EpisodeScript.prompt_version` cho phép DB query "episodes generated by V1 vs V2" → diagnose regression nhanh.
4. **Per-episode revert**: nếu 1 episode V2 fail validation thực tế (not just gatekeeper), re-run với `--legacy` flag trên `main.py` (cần thêm flag).
5. **Data migration reversible**: `backfill_constraint_fields.py` chỉ ADD fields, không modify existing. Revert = ignore fields mới.

---

## Sequencing (PR plan)

| PR | Phase | Scope | Risk | Approve gate |
|---|---|---|---|---|
| PR1 | P1a | Schema + config (sentinel defaults, feature flag off) | Low | Schema review, no behavior change |
| PR2 | P1b | Validator + tests + backfill + baseline report | Low | All unit tests pass; baseline numbers chốt thresholds |
| PR3 | P2 | Hook extractor + sub-step trong `run_llm()` | Med | Manual review viral_moments 5 episode |
| PR4 | P3 + P4 | V2 scriptwriter prompt + multi-candidate hook + judge + visual constraint (behind flag) | High | Episode-001 regenerate pass acceptance; flag default ON sau verify |
| PR5 | P5 | Gatekeeper + retry + report | Med | Token cost test + adversarial graceful degrade test |
| PR6 | P6 | Manual calibration (no code, hoặc tune config + ADR) | Low | 7/10 hand-rated ≥3 sao |

---

## Decisions (chốt)

- **Pipeline architecture KHÔNG đổi**: hook_extract là sub-step trong `run_llm()`, không thêm phase mới (lý do: VRAM context).
- **Reviewer là pure judge**: orchestrator độc quyền điều phối retry. Bỏ pass-2 auto-fix khỏi reviewer.
- **Hook = competitive auto**, không manual (constraint: batch 100, 1 maintainer).
- **Reviewer fail KHÔNG block batch**: graceful degrade với log structured.
- **Summarizer giữ faithful**, viral extraction là stage RIÊNG.
- **Sentinel `Optional + None`** cho fields auto-derived (phân biệt "chưa compute" vs "default thật").
- **Out of scope**: BGM/SFX, A/B testing automation, thumbnail generation upgrade, ComfyUI workflow changes, Ollama model swap.

---

## Open Questions (cần ông quyết trước khi PR3-PR4)

1. **Reviewer ownership pattern**: plan đã chọn (A) reviewer pure judge. Confirm OK?
2. **Lore-before-curiosity tier**: đã downgrade thành SOFT/WARNING (không block) sau review 3. Confirm OK? Nếu muốn strict hơn cho lore truyện tu tiên thì có thể bump lên BLOCKING với episode 1 exception.
3. **Hook judge model**: Ollama có chạy được 2 model concurrently (deepseek + qwen) trong 8GB VRAM không, hay phải swap? Quyết định feasibility Option B trong Phase 6.
4. **Feature flag duration**: ông OK với 2-3 tuần dual code path (legacy + V2) hay cut-over hard sau PR4?
5. **Numeric thresholds**: chấp nhận chạy validator trên 5-10 episode hiện có **TRƯỚC** khi chốt threshold trong settings.yaml không? (Phase 1b output sẽ inform PR1 placeholders)
6. **Hand-label resource**: Phase 6 cần ông tự rate 10 episode. Có thể defer 1-2 tuần sau PR5 không, hay phải làm ngay sau merge?
7. **LLM-assisted energy override (deferred)**: reviewer 3 đề xuất LLM override `energy_level` thay vì pure heuristic. Hiện defer (cost: +1 LLM call × N shots × 100 episodes). Sau Phase 6, nếu hand-rated cho thấy heuristic energy mismatch perception → cân nhắc thêm trong v2 (out of current scope).
8. **Curiosity score global metric (rejected)**: reviewer 3 đề xuất `EpisodeScript.curiosity_score` từ LLM. Đã reject (cost + duplicate với hook judge bias). Component scores hiện tại đủ proxy. Nếu Phase 6 cho thấy không discriminate được "watchable vs not" → reconsider.

---

## Notes

- Discovery của codebase đã verify: state.py KHÔNG persist EpisodeScript JSON → migration đơn giản hơn dự kiến.
- `_HOOK_SYSTEM` hiện tại (10 từ budget) sẽ trở thành 1 trong 3 candidates ở Phase 4 — không vứt đi, tái dùng làm baseline.
- Hook visual anchoring qua `arc.key_events[0]` (trong `image-narration-mismatch-resolving.md` đã DONE) phải tích hợp với `viral_moments` — khi viral_moment override, anchor logic phải dùng `viral_moment.mystery_seed` thay vì `key_events[0]`. Cần handle ở Phase 2 step 9.
- **Triết lý sau review 3**: plan ưu tiên **"maximize curiosity"** thay vì **"minimize errors"**. Cụ thể: ít BLOCKING rule (6 rule HARD), retry với positive directive ("increase tension"), rubric judge anti-bias prompt. Mục đích: tránh trường hợp output đúng luật nhưng nhạt — system convergence về mediocrity là failure mode chính cần phòng.

