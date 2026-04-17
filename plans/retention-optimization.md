## Plan: Retention Optimization (Priority 1–4)

**TL;DR:** 3 thay đổi cụ thể trên 2 file, không break pipeline hiện tại. Ship được trong 1 ngày.

> **Phát hiện quan trọng:** Intro đã bị tắt từ trước — orchestrator (orchestrator.py) gọi `assemble_episode()` không truyền `intro_path`. Không cần làm gì ở đây.

---

### Phase A — LLM Script Prompt
**File:** scriptwriter.py

**Bước 1 — Hook rule:** Thêm vào `_SCRIPTWRITER_SYSTEM`:
- Shot 1 MUST mở bằng một **hành động cụ thể** hoặc **câu thoại/câu kể gây sốc** — cấm mở kiểu mô tả bối cảnh hoặc kể backstory.
- Hook phải giữ **thứ tự tuyến tính** — không được tease scene cuối rồi quay lại. Hook là scene đầu tiên của story, viết lại theo góc độ hành động/shock.
- Ví dụ pattern: "Một tiếng thét xé lòng vang lên..." / "Hắn mở nắp quan tài và..." (không phải "Diệp Thiếu Dương đứng trước ngôi miếu cổ")
- Thêm WRONG/RIGHT example cho hook (tương tự pattern scene_prompt đang có)

**Bước 2 — Cliffhanger rule:** Thêm vào `_SCRIPTWRITER_SYSTEM`:
- Shot 8 (final) MUST kết thúc ở **đỉnh điểm của sự tò mò** — câu hỏi chưa có đáp án, hành động bị dừng nửa chừng, hoặc revelation bị cắt ngay trước khi reveal.
- Forbidden: CTA "theo dõi tiếp", kết thúc trọn vẹn, giải thích xong hết.

**Bước 3 — Pacing hint trong prompt:** Thêm rule:
- `Shot 1 và Shot 2: duration_sec = 2 hoặc 3`
- `Shot 1 và Shot 2: narration_text không quá 12 từ` — TTS cần khớp với 2–3s, narration quá dài sẽ bị cắt cụt hoặc đọc không kịp.
- LLM được guide ngay từ prompt để output đúng cả hai giá trị.

---

### Phase B — Post-process Duration
**File:** scriptwriter.py

**Bước 4 — Carryover: force-regenerate hook shot**
- Khi `carryover` shots đủ để fill episode mà không cần gọi LLM (`len(all_shots) >= _MAX_SHOTS_PER_EPISODE`), shots[0] của episode là carryover — không có hook intent.
- **Fix:** Luôn gọi LLM để lấy 1 hook shot mới, replace `all_shots[0]` bằng hook shot đó. Hook shot này lấy từ `arc` của episode hiện tại (không dùng arc của episode trước).
- Cụ thể: tách riêng `_generate_hook_shot(arc_text, episode_num) -> ShotScript` — gọi LLM với prompt ngắn, chỉ yêu cầu 1 shot (hook), không ảnh hưởng carry-over logic còn lại.
- **Implementation note:** Trong `else` branch (carryover-only path), `arc` và `arc_text` hiện không được load. Cần gọi `load_arc_overview(episode_num)` + build `arc_text` trong cả 2 nhánh, hoặc refactor để load arc trước `if/else`. `_generate_hook_shot` nhận `arc_text` giống format đang dùng để prompt nhất quán.

**Bước 5 — Thêm `_normalize_hook_durations(shots)`:**
- `shots[0].duration_sec > 3` → set = 3
- `shots[1].duration_sec > 3` → set = 3

**Bước 6 — Activate `_normalize_duration()` + fix thứ tự gọi:**
- Hàm này đang là dead code — cần activate với thứ tự đúng để tránh xung đột.
- **Thứ tự bắt buộc** (operate trên `episode_shots`, không phải `new_shots`):
  1. `episode_shots = all_shots[:_MAX_SHOTS_PER_EPISODE]`
  2. `_normalize_duration(episode_shots, episode_num)` — cân bằng tổng ~60s
  3. `_normalize_hook_durations(episode_shots)` — enforce hook duration SAU CÙNG (tránh bị override)
  4. `_normalize_key_shots(episode_shots, episode_num)` — giữ nguyên vị trí hiện tại
- Thêm `logger.debug("Hook norm | shot0={}s shot1={}s", ...)` sau bước 3 để trace khi verify.

---

### Phase C — Subtitle Keyword Highlight
**File:** editor.py

**Bước 7 — Thêm 2 keyword list:**
- `_DANGER_KEYWORDS`: chết, xác, mộ, quỷ, máu, ám, tà, oan, hồn, thi hài, quan tài, phanh thây, cắn xé, gào thét, biến mất, sụp đổ...
- `_TWIST_KEYWORDS`: thật ra, không ngờ, bất ngờ, hóa ra, thực chất...

**Bước 8 — Wrap keywords trong `generate_ass()`:**
- Kết hợp `\kf` tag hiện có + ASS inline color. **Format đúng (tất cả tags trong 1 block):** `{\kf{dur}\c&H0000FF&}word{\r}`
- Danger = đỏ (`\c&H0000FF&` trong BGR), Twist = vàng (`\c&H00FFFF&`)
- Case-insensitive match trên từng word sau khi split, không break karaoke timing
- **Safety net:** wrap keyword-tagging logic trong try/except — nếu fail, fallback về `{\kf{dur}}word` (không màu) và log warning. Tránh hard fail toàn bộ video phase.
- Lưu ý: `\c` chỉ override PrimaryColour (phần đang spoken) — SecondaryColour (unspoken, màu vàng) không thay đổi. Behavior này được chấp nhận.

---

### Relevant files
- scriptwriter.py — `_SCRIPTWRITER_SYSTEM`, `write_episode_script()`, `_normalize_key_shots()`, `_normalize_duration()` (dead code → activate)
- editor.py — `generate_ass()`, `_WORDS_PER_SEGMENT`
- orchestrator.py — không cần sửa (intro đã tắt)

---

### Verification
1. Chạy `py --episode 2 --from-phase llm` → inspect script JSON: `shots[0].duration_sec ≤ 3`, `shots[1].duration_sec ≤ 3`, tổng ≈ 60s
2. Check `shots[0].narration_text` — phải là hook (shock/curiosity), giữ đúng scene đầu story, không phải backstory flat
3. Check `shots[7].narration_text` — phải kết thúc bằng cliffhanger/câu hỏi bỏ lửng
4. Chạy `py --episode 1 --from-phase video` (dùng episode-001 đã có script) → xem file data/scripts/episode-001.ass — verify màu đỏ/vàng xuất hiện ở đúng keyword (nhanh hơn là đợi generate LLM mới)
5. Play video output: xác nhận subtitle render đúng màu với ffmpeg

---

### Decisions
- **Scope là 2 file:** `scriptwriter.py` và `editor.py`. Không chạm schemas, assembler, orchestrator.
- **Sound effect bị loại:** quá phức tạp, không cùng batch.
- **`_normalize_duration()` dead code:** plan này activate bằng cách gọi đúng chỗ, đúng thứ tự.
- **Carryover + hook:** luôn force-regenerate 1 hook shot từ arc hiện tại, replace shots[0] của carryover.
- **Hook + story chronology:** giữ tuyến tính — hook là scene đầu story xử lý lại theo angle shock/mystery, không non-linear.
- **Rollback:** sửa lại `_SCRIPTWRITER_SYSTEM` và re-run `--from-phase llm` là đủ. Script JSON được lưu trên disk, không có side effect lên pipeline khác.
