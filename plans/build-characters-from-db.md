## Plan: Build Character Profiles từ Wiki DB → Markdown → ComfyUI Tags

**TL;DR**: Thay flow cũ (arc JSON → LLM) bằng flow mới: đọc wiki tables từ SQLite → render Markdown anchor → LLM → Danbooru tags. File mới `llm/profile_builder.py`, orchestrator swap 2 dòng.

> **Scope**: step này chỉ build **visual anchor** — ngoại hình đặc trưng cố định. Mapping snapshot theo chapter cho từng scene xử lý ở phase image generation.

---

### Schema Reference

#### wiki_characters (689 rows) — nguồn chính
| Column | Dùng ở đâu |
|--------|-----------|
| `character_id` | PK, tên file output |
| `name` | Markdown header |
| `aliases_json` | JSON array → `**Tên khác**` |
| `traits_json` | JSON array → `**Tính cách**` |
| `personality` | `**Tính cách chi tiết**` |
| `visual_anchor` | `**Visual Anchor**` |
| `faction` | header `**Phe**` |
| `gender` | header + LLM gender hint |
| `remaster_version` | header `Dữ liệu v<N>` |

#### wiki_snapshots (2954 rows) — **không dùng ở bước này**
> Snapshot track sự thay đổi theo chương — input cho scene image generation (phase sau). Không đưa vào prompt anchor để tránh nhiễu.

#### wiki_relations (1742 rows) — context nhẹ
| Column | Dùng ở đâu |
|--------|-----------|
| `character_id` | filter |
| `related_name` | `## Quan hệ` — tên NV liên quan |
| `description` | mô tả quan hệ |
| `chapter_start` | `(ch.N)` ghi chú + sort |

#### wiki_batches (737 rows) — metadata footer
| Column | Dùng ở đâu |
|--------|-----------|
| `chapter_start/end` | coverage footer: `ch.1–3533` |
| `status` | `SUM(DONE)/COUNT(*)` → `737/737 batches` |
| `extraction_version` | `MAX(extraction_version)` → `v<N>` trong footer |

#### wiki_remaster_batches (0 rows — không dùng)
> Không dùng trong flow này.

#### wiki_artifacts (0 rows — sẵn sàng khi có data)
| Column | Dùng ở đâu |
|--------|-----------|
| `artifact_id` | PK |
| `name` | `## Bảo vật` section header |
| `rarity` | `(hiếm: <rarity>)` |
| `material` | `(chất liệu: <material>)` |
| `visual_anchor` | mô tả ngoại hình bảo vật — đưa vào anchor |
| `description` | mô tả chung |

#### wiki_artifact_snapshots (0 rows — sẵn sàng khi có data)
| Column | Dùng ở đâu |
|--------|-----------|
| `artifact_id` | FK → wiki_artifacts |
| `owner_id` | filter theo `character_id` |
| `chapter_start` | `ch.N` ghi chú |
| `normal_state` | trạng thái bình thường |
| `active_state` | trạng thái kích hoạt |
| `condition` | tình trạng (intact/damaged/broken) |
| `vfx_color` | màu hiệu ứng |
| `is_key_event` | `1` → hint LLM thêm tag visual của artifact vào anchor |
| `extraction_version` | dedup: `MAX(extraction_version)` |

---

### Phase 1 — DB Query Layer (4 hàm)

**1.** `_load_wiki_character(character_id, con)` — query `wiki_characters`, trả về tất cả columns

**2.** `_load_relations(character_id, con)` — query `wiki_relations WHERE character_id = ?`, sort by `chapter_start`

**3.** `_load_artifacts(character_id, con)` — lấy artifact của nhân vật ở **trạng thái gốc** (`MIN(chapter_start)` per artifact) với version cao nhất:
```sql
SELECT a.artifact_id, a.name, a.rarity, a.material, a.visual_anchor, a.description,
       s.chapter_start, s.normal_state, s.active_state, s.condition, s.vfx_color,
       s.is_key_event
FROM wiki_artifacts a
JOIN (
    SELECT artifact_id, MIN(chapter_start) AS first_ch
    FROM wiki_artifact_snapshots WHERE owner_id = ?
    GROUP BY artifact_id
) first ON a.artifact_id = first.artifact_id
JOIN (
    SELECT artifact_id, chapter_start, MAX(extraction_version) AS best_ver
    FROM wiki_artifact_snapshots WHERE owner_id = ?
    GROUP BY artifact_id, chapter_start
) ver ON a.artifact_id = ver.artifact_id AND ver.chapter_start = first.first_ch
JOIN wiki_artifact_snapshots s ON s.artifact_id = ver.artifact_id
    AND s.chapter_start = ver.chapter_start
    AND s.extraction_version = ver.best_ver
```
→ Trả về `[]` nếu tables rỗng (skip section trong Markdown)

**4.** `_load_extraction_coverage(con)` — aggregate `wiki_batches`:
```sql
SELECT MIN(chapter_start), MAX(chapter_end),
       COUNT(*) AS total,
       SUM(CASE WHEN status='DONE' THEN 1 ELSE 0 END) AS done,
       MAX(extraction_version) AS max_ver
FROM wiki_batches
```
→ Render footer: `> Dữ liệu: ch.1–3533 | 737/737 batches | v2`

---

### Phase 2 — Markdown Builder

**5.** `build_markdown(character_id, con) → str` — gọi 4 hàm query, render:

```
# Diệp Đại Bảo

**Giới tính**: male | **Phe**: Ghost-Hunter Sect | **Dữ liệu v1**
**Tên khác**: alias1, alias2
**Tính cách**: Dũng cảm, bảo vệ người yếu
**Tính cách chi tiết**: Luôn đặt người khác lên trên bản thân, không ngại hy sinh
**Visual Anchor**: Tóc ngắn, mắt sắc bén

## Bảo vật / Vũ khí sở hữu
### Dao Linh (hiếm: A, chất liệu: bạch kim)
- Visual: lưỡi dao phát sáng trắng khi kích hoạt
- Sự kiện chủ chốt: có

## Quan hệ
- Nhị tẩu (ch.1): Chị dâu, người được bảo vệ

---
> Dữ liệu: ch.1–3533 | 737/737 batches | v2
```
Bỏ qua field `None` hoàn toàn. Bỏ section `## Bảo vật` nếu `_load_artifacts` trả về `[]`.

Nếu `wiki_characters.visual_anchor = null`: không render dòng `**Visual Anchor**`. LLM tự suy ngoại hình từ `traits_json` + `personality` (ví dụ trait "Gian xảo" → `shifty eyes, smirk`). Nếu cả 3 đều null → skip character (không đủ data để build anchor).

Nếu `wiki_characters.gender = null`: render `**Giới tính**: unknown`. LLM dùng `1other, solo, androgynous` — vẽ trừu tượng, không phân biệt giới tính.

---

### Phase 3 — LLM Derivation

**6.** System prompt `_PROFILE_SYSTEM` — tái sử dụng Danbooru rules từ `_EXTRACTOR_SYSTEM`, thêm:
- `visual_anchor` từ `wiki_characters` là nguồn chính để suy ra tags
- Nếu artifact có `is_key_event=1`: thêm `visual_anchor` của artifact vào description tags
- Nếu `Giới tính: unknown`: output `"gender": "unknown"` và dùng `1other, solo, androgynous`

**7.** `_derive_tags(markdown_text) → dict` — `ollama_client.generate_json()`, tenacity 3 retries

**8.** `_sanitize_description()` — **import từ `character_extractor.py`**, không duplicate
- Thêm nhánh: nếu `gender == "unknown"` → prefix `1other, solo, androgynous`; bỏ qua rules chỉ áp dụng cho `male`/`female`

---

### Phase 4 — Entry Point

**9.** `build_all_profiles(force: bool = False) → list[Character]`
- **Invocation**: gọi từ orchestrator tại episode 1 (giống `extract_all_characters` cũ); **idempotent** — nếu `<id>.json` đã tồn tại thì skip
- Mở DB connection một lần từ `settings.db_path`
- Query tất cả `character_id` từ `wiki_characters` (không filter theo `visual_importance` — snapshot không còn được dùng)
- Per character (trong `try/except`): `build_markdown()` → lưu `<id>.md` → `_derive_tags()` → validate `Character(...)` → lưu `<id>.json`
  - Nếu exception: log warning + `continue`
- Log tổng kết cuối: `Built N / skipped M / failed K`
- Return `list[Character]`

---

### Phase 5 — Orchestrator Wiring

**10.** [pipeline/orchestrator.py](../pipeline/orchestrator.py) line 71–73: swap:
```python
# before
from llm.character_extractor import extract_all_characters
extract_all_characters()

# after
from llm.profile_builder import build_all_profiles
build_all_profiles()
```

---

### Relevant files
- `llm/profile_builder.py` — file MỚI, toàn bộ logic
- [llm/character_extractor.py](../llm/character_extractor.py) — giữ nguyên, chỉ import `_sanitize_description`
- [pipeline/orchestrator.py](../pipeline/orchestrator.py) — sửa 2 dòng
- [models/schemas.py](../models/schemas.py) — **không sửa**

### Verification
1. `build_all_profiles()` → kiểm tra `.md` + `.json` trong `data/<slug>/characters/`
2. Spot-check `.md`: có `Visual Anchor`, `Tính cách`, `Quan hệ`; **không có** section Ngoại hình theo chương
3. Spot-check `.json`: `description` bắt đầu `1boy, solo` hoặc `1girl, solo` hoặc `1other, solo`
4. `pytest tests/ -q` — 51/51 pass

### Decisions
- `wiki_snapshots` **không dùng** ở bước build anchor — dành cho scene image generation phase sau
- `min_importance` filter bỏ — không còn phụ thuộc snapshot; lấy toàn bộ `wiki_characters`
- `Character` model không thay đổi — Markdown là file `.md` riêng
- `extract_all_characters()` giữ nguyên — backward compat với tests
- `gender = null` → unisex (`1other, solo, androgynous`), không skip
- `build_all_profiles` idempotent, gọi từ orchestrator episode 1