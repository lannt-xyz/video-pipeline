## Plan: Fix Anchor-to-Video Quality Chain

**TL;DR**: Có **2 bug độc lập** khiến anchor không affect được video. Bug 1 là structural (LoRA mismatch giữa anchor workflow và chapter workflow). Bug 2 là weight có thể quá mạnh, gây distort composition.

---

### Root Cause Analysis

**Bug 1 — Structural (Critical)**: Anchor được gen bằng txt2img_scene.json (2 LoRAs), nhưng chapter shots dùng txt2img_ipadapter.json (4 LoRAs). `manhua_style_v2` ở strength **0.7** thay đổi toàn bộ aesthetic — character face, line weight, coloring. Kết quả: IPAdapter so sánh anchor "phong cách xianxia ink wash" với chapter shot "phong cách manhua" → extract features không match → nhân vật bị dị.

| | Anchor workflow (cũ) | Chapter workflow | Anchor workflow (sau fix) |
|---|---|---|---|
| ink_wash_painting | ✅ 0.6 | ✅ 0.6 | ✅ 0.6 |
| xianxia_fantasy | ✅ 0.5 | ✅ 0.5 | ✅ 0.5 |
| **manhua_style_v2** | ❌ không có | ✅ **0.7** | ✅ 0.7 |
| **dark_horror_atmosphere** | ❌ không có | ✅ 0.4 | ✅ present nhưng **strength=0.0** (neutral lighting cho IPAdapter) |

**Bug 2 — Weight quá mạnh (Medium)**: IPAdapter weight=0.8 + preset "PLUS (high strength)" + embed_scaling "K+V w/ C penalty" = quá aggressive. Với action shots (fight, crowd), IPAdapter ép composition của anchor (static portrait) vào scene động → nhân vật bị đơ, background bị kéo lệch.

---

### Các bước fix

**Phase 1 — Tạo `anchor_gen.json`** (file mới)
1. Copy txt2img_ipadapter.json, xóa nodes 6 (IPAdapterUnifiedLoader), 7 (LoadImage), 8 (IPAdapterAdvanced)
2. Reconnect KSampler node 12: `"model": ["8", 0]` → `"model": ["5", 0]` — đây là **chỗ duy nhất** cần sửa, CLIP nodes 9/10 vẫn trỏ `["5", 1]` đúng
3. Xóa placeholder `ANCHOR_PATH` khỏi `_meta.placeholders`
4. Set `dark_horror_atmosphere` LoRA (node 5): `strength_model: 0.0`, `strength_clip: 0.0` — anchor cần neutral lighting để IPAdapter encode face features sạch, không bị nhiễm color grading tối
5. Kết quả: anchor được gen với đúng aesthetic base như chapter shots (3 active LoRAs), lighting neutral

**Phase 2 — Update character_gen.py**
6. Đổi `_SCENE_WORKFLOW = "image_gen/workflows/txt2img_scene.json"` → `_ANCHOR_WORKFLOW = "image_gen/workflows/anchor_gen.json"`
7. Đổi tên constant trong `_generate_single_anchor()`: `workflow_path=_ANCHOR_WORKFLOW`
   - **Không cần** xóa `ANCHOR_PATH` khỏi replacements — `_generate_single_anchor()` chưa bao giờ có key này trong dict

**Phase 3 — Tune IPAdapter weight**
8. Trong txt2img_ipadapter.json node 8: giảm `weight` từ `0.8` → `0.65`
   - `lora_strength: 0.6` trong IPAdapterUnifiedLoader (node 6) **không đổi** — đây là internal LoRA của IPAdapter, khác với `weight` ở node 8
   - 0.65 đủ để lock face features nhưng cho scene composition tự do hơn với action shots

**Phase 4 — Regen anchors** *(bắt buộc sau deploy)*
9. Anchors cũ được gen bằng `txt2img_scene.json` (2 LoRAs) — style khác, **phải regen**
10. Sau khi deploy, chạy trước khi images phase:
    ```python
    from image_gen.character_gen import generate_character_anchors
    generate_character_anchors(force=True)
    ```
    Hoặc qua CLI: `python main.py --episode 1 --from-phase images` sẽ gọi `generate_character_anchors()` nhưng **không** force — phải gọi manual với `force=True`

---

**Relevant files**
- image_gen/workflows/anchor_gen.json — **file mới**: ipadapter workflow bỏ nodes 6/7/8, KSampler model→["5",0], horror LoRA strength=0.0
- image_gen/workflows/txt2img_ipadapter.json — giảm weight 0.8→0.65 trong node 8
- image_gen/character_gen.py — đổi constant `_SCENE_WORKFLOW` → `_ANCHOR_WORKFLOW`

**Verification**
1. Mở `anchor_gen.json` trong ComfyUI UI, kiểm tra graph: phải có đúng 4 LoRA nodes → KSampler, không có IPAdapter nodes
2. Gen lại 1 anchor với `--force`, so sánh style với chapter shot cùng nhân vật — phải cùng line weight và coloring
3. Gen 1 chapter shot với anchor mới, so sánh độ nhất quán face vs anchor cũ

---

**Scope exclusion**: Không thay đổi txt2img_scene.json (dùng cho non-character shots, không cần manhua LoRA). Không thay đổi TTS/audio/subtitle pipeline. 
