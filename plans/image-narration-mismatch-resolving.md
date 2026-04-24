## Plan: Fix Image-Narration Mismatch (root cause)

**Status: ✅ ALL PHASES COMPLETE** (2026-04-23)

**TL;DR** — Pipeline có 4 nhóm root cause: (1) `shot.characters` LLM list sai chủ thể; (2) `visual_brief` chỉ extract 1 action nên multi-event narration bị mất; (3) hook shot dùng narration meta làm input → ảnh vô nghĩa; (4) Pass 3 rule-based + Pass 1 mood không có validator nên leak/violate horror tone. Refactor 2-pass alignment thành 4-pass + character resolution riêng + debug report.

### Bằng chứng cụ thể từ episode-001
- **Shot 1 hook**: narration `"Cái gì đang xảy ra đây?"` → brief `"figure looking around"` → ảnh generic, không hook
- **Shot 3**: thiếu Thiếu Dương trong characters; `scene_id="ruined_temple"` mâu thuẫn `setting="Diệp Đại Công's home"`; mood `"bright white light, clear visibility"` trái horror tone
- **Shot 5**: leak tag `"figure crouching and digging with shovel"` từ shot 4 (narration shot 5 không nói đào)
- **Shot 6**: narration 4 events liên tiếp, brief chỉ giữ "standing motionless in rain" → mất action chính
- **Shot 7-8**: `characters=[Diệp Đại Bảo]` (object) nhưng subject thực sự là "Thanh Vân Tử"

### Phases

| # | Phase | Status | Mục tiêu | Files chính |
|---|---|---|---|---|
| 1 | **Character Resolution Pass** | ✅ Done | LLM pass riêng phân biệt subject/object/visible; override `shot.characters`; saves `characters_raw` for debug | `llm/scriptwriter.py`, `models/schemas.py` |
| 2 | **Multi-Action Visual Brief** | ✅ Done | Đổi `action: str` → `actions: List[str]` (1-3 micro-events theo timeline); per-frame action injection | `models/schemas.py`, `llm/scriptwriter.py`, `video/frame_decomposer.py` |
| 3 | **Hook Shot Visual Anchoring** | ✅ Done | Hook shots nhận `arc.key_events[0]` làm `visual_context` để LLM extract visual từ event thực, không từ meta-narration | `llm/scriptwriter.py` |
| 4 | **Validators + Tag-Leak Isolation** | ✅ Done | (4a) reject mood vi phạm horror palette; (4b) warn scene_id/setting mismatch; (4c) skip action-rule injection khi brief đã có concrete action | `llm/scriptwriter.py` |
| 5 | **Debug Alignment Report** | ✅ Done | Sau script gen, dump `episode-XXX-alignment.md` — narration / actions / scene_prompt / characters / image path | `pipeline/alignment_report.py`, hook vào `llm/scriptwriter.py` |

### Verification (next steps)
1. Re-run pipeline cho `mao-son-troc-quy-nhan` ep1, mở `data/mao-son-troc-quy-nhan/scripts/episode-001-alignment.md` để check 8 shots
2. Test cases cụ thể: shot 1 hook không "looking around"; shot 3 mood không bright; shot 6 brief có 2-3 actions; shot 7-8 character[0]=Thanh Vân Tử
3. Unit tests trong `tests/test_scriptwriter.py`: character resolution với "A phát hiện B"; mood validator reject "bright white light"; multi-action extraction
4. Chấm điểm trực quan 1-5 mỗi shot, target trung bình ≥4

### Decisions
- **In scope**: refactor LLM/script layer (Pass 1-4 + report). Không động ComfyUI workflows.
- **Out of scope** (đã có plan riêng): anchor LoRA mismatch, DNA tag extraction, IPAdapter weight, InsightFace+anime risk.

### Further Considerations (locked)
1. **Per-frame audio sync** → monolithic TTS + char-ratio frame timing. Không split TTS.
2. **Character resolution fallback** → LLM-only với strict prompt. Fallback heuristic chỉ khi accuracy <85% sau đo lường.
3. **Max chars/shot** → cap 2 cứng; bystander xuống scene_prompt như generic role tag.
