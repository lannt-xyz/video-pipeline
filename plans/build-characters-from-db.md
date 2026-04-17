## Plan: Build Character Profiles từ Wiki DB → Markdown → ComfyUI Tags

**TL;DR**: Thay flow cũ (arc JSON → LLM) bằng flow mới: đọc **toàn bộ wiki tables** từ SQLite → render Markdown → LLM → Danbooru tags. File mới `llm/profile_builder.py`, orchestrator swap 2 dòng.

---

### Schema Reference

#### wiki_characters (689 rows)
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

#### wiki_snapshots (2954 rows)
| Column | Dùng ở đâu |
|--------|-----------|
| `character_id` | filter |
| `chapter_start` | section header |
| `is_active` | `1` → `## Ngoại hình`, `0` → `## Lịch sử ngoại hình` |
| `physical_description` | `- Ngoại hình:` |
| `outfit` | `- Trang phục:` |
| `weapon` | `- Vũ khí:` |
| `vfx_vibes` | `- VFX:` |
| `level` | `- Cấp độ:` |
| `visual_importance` | section header `(tầm quan trọng: X/10)` + filter entry point |
| `extraction_version` | dedup: `MAX(extraction_version)` per `(character_id, chapter_start)` |

#### wiki_relations (1742 rows)
| Column | Dùng ở đâu |
|--------|-----------|
| `character_id` | filter |
| `related_name` | `## Quan hệ` — tên NV liên quan |
| `description` | mô tả quan hệ |
| `chapter_start` | `(ch.N)` ghi chú + sort |

#### wiki_batches (737 rows)
| Column | Dùng ở đâu |
|--------|-----------|
| `chapter_start/end` | coverage footer: `ch.1–3533` |
| `status` | `SUM(DONE)/COUNT(*)` → `737/737 batches` |
| `extraction_version` | `MAX(extraction_version)` → `v<N>` trong footer |

#### wiki_remaster_batches (0 rows — không dùng)
> Không dùng trong flow này. Snapshot đã dedup bằng `MAX(extraction_version)` là đủ.

#### wiki_artifacts (0 rows — sẵn sàng khi có data)
| Column | Dùng ở đâu |
|--------|-----------|
| `artifact_id` | PK |
| `name` | `## Bảo vật` section header |
| `rarity` | `(hiếm: <rarity>)` |
| `material` | `(chất liệu: <material>)` |
| `visual_anchor` | mô tả ngoại hình bảo vật |
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
| `is_key_event` | `1` → hint LLM thêm tag visual của artifact |
| `extraction_version` | dedup: `MAX(extraction_version)` |

---

### Phase 1 — DB Query Layer (6 hàm)

**1.** `_load_wiki_character(character_id, con)` — query `wiki_characters`, trả về tất cả columns

**2.** `_load_best_snapshots(character_id, con)` — dedup `MAX(extraction_version)` per `(character_id, chapter_start)`, trả về tất cả (phân biệt `is_active` khi render):
```sql
SELECT s.* FROM wiki_snapshots s
INNER JOIN (
    SELECT character_id, chapter_start, MAX(extraction_version) AS best_ver
    FROM wiki_snapshots WHERE character_id = ?
    GROUP BY character_id, chapter_start
) best ON s.character_id = best.character_id
       AND s.chapter_start = best.chapter_start
       AND s.extraction_version = best.best_ver
ORDER BY s.chapter_start
```

**3.** `_load_relations(character_id, con)` — query `wiki_relations WHERE character_id = ?`, sort by `chapter_start`

**4.** `_load_artifacts(character_id, con)` — JOIN `wiki_artifacts + wiki_artifact_snapshots`, dedup `MAX(extraction_version)`:
```sql
SELECT a.artifact_id, a.name, a.rarity, a.material, a.visual_anchor, a.description,
       s.chapter_start, s.normal_state, s.active_state, s.condition, s.vfx_color,
       s.is_key_event, s.extraction_version
FROM wiki_artifacts a
JOIN (
    SELECT artifact_id, chapter_start, MAX(extraction_version) AS best_ver
    FROM wiki_artifact_snapshots WHERE owner_id = ?
    GROUP BY artifact_id, chapter_start
) dedup ON a.artifact_id = dedup.artifact_id
JOIN wiki_artifact_snapshots s ON s.artifact_id = dedup.artifact_id
    AND s.chapter_start = dedup.chapter_start
    AND s.extraction_version = dedup.best_ver
ORDER BY s.chapter_start
```
→ Trả về `[]` nếu tables rỗng (skip section trong Markdown)

**5.** `_load_extraction_coverage(con)` — aggregate `wiki_batches`:
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

**6.** `build_markdown(character_id, con) → str` — gọi 5 hàm query, render:

> **Token limit**: chỉ render **15 snapshots gần nhất** (sort `chapter_start DESC LIMIT 15`) để tránh "Lost in the Middle" với nhân vật chính có 50+ snapshots. `is_active` snapshots bị crop thì bỏ qua section `## Lịch sử ngoại hình` nếu rỗng.

```
# Diệp Đại Bảo

**Giới tính**: male | **Phe**: Ghost-Hunter Sect
**Tên khác**: alias1, alias2
**Tính cách**: Dũng cảm, bảo vệ người yếu
**Tính cách chi tiết**: Luôn đặt người khác lên trên bản thân, không ngại hy sinh
**Visual Anchor**: Tóc ngắn, mắt sắc bén

## Ngoại hình (theo chương)
### Chương 1 (tầm quan trọng: 7/10)
- Ngoại hình: quỳ xuống, run rẩy
- Vũ khí: dao găm

## Lịch sử ngoại hình
### Chương 50 (không còn active)
- Trang phục: áo cũ rách

## Bảo vật / Vũ khí sở hữu
### Dao Linh (hiếm: A, chất liệu: bạch kim)
- ch.120: bình thường / kích hoạt: phát sáng trắng
- Tình trạng: intact | VFX: ánh trắng
- Sự kiện chủ chốt: có

## Quan hệ
- Nhị tẩu (ch.1): Chị dâu, người được bảo vệ

---
> Dữ liệu: ch.1–3533 | 737/737 batches | v2
```
Bỏ qua field `None` hoàn toàn. Bỏ section `## Bảo vật` nếu `_load_artifacts` trả về `[]`.

Nếu `wiki_characters.gender = null`: render `**Giới tính**: unknown` và không render dòng giới tính trong header. LLM sẽ dùng Danbooru tags trung tính: **không dùng `1boy`/`1girl`**, thay bằng `1other, solo, androgynous` — vẽ nhân vật trừu tượng, không phân biệt giới tính.

---

### Phase 3 — LLM Derivation

**7.** System prompt `_PROFILE_SYSTEM` — tái sử dụng Danbooru rules từ `_EXTRACTOR_SYSTEM`, thêm:
- Đọc **snapshot `is_active=1` có `chapter_start` cao nhất** làm base appearance
- Nếu artifact có `is_key_event=1`: thêm `visual_anchor` của artifact vào description tags
- Nếu `Giới tính: unknown`: output `"gender": "unknown"` và dùng `1other, solo, androgynous` thay vì `1boy`/`1girl`

**8.** `_derive_tags(markdown_text) → dict` — `ollama_client.generate_json()`, tenacity 3 retries

**9.** `_sanitize_description()` — **import từ `character_extractor.py`**, không duplicate
- Thêm nhánh: nếu `gender == "unknown"` → thay prefix bằng `1other, solo, androgynous`; bỏ qua các rule chỉ áp dụng cho `male`/`female`

---

### Phase 4 — Entry Point

**10.** `build_all_profiles(min_importance: int = 6, force: bool = False) → list[Character]`
- **Invocation**: gọi từ orchestrator tại episode 1 (giống `extract_all_characters` cũ); **idempotent** — nếu `<id>.json` đã tồn tại thì skip, không re-call LLM
- Mở DB connection một lần từ `settings.db_path`
- Filter: `character_id` có ≥1 snapshot `visual_importance >= min_importance`
- Per character (trong `try/except`): `build_markdown()` → lưu `<id>.md` → `_derive_tags()` → validate `Character(...)` → lưu `<id>.json`
  - Nếu exception: log warning + `continue` (không abort toàn batch)
- Log tổng kết cuối: `Built N / skipped M / failed K`
- Return `list[Character]` (tất cả đã build, kể cả từ lần trước)

---

### Phase 5 — Orchestrator Wiring

**11.** [pipeline/orchestrator.py](../pipeline/orchestrator.py) line 71–73: swap:
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
1. `build_all_profiles(min_importance=6)` → kiểm tra `.md` + `.json` trong `data/<slug>/characters/`
2. Spot-check `.md`: snapshot v2 hiển thị, v1 bị loại nếu cùng chapter; `is_active=0` vào section lịch sử
3. Spot-check `.json`: `description` bắt đầu `1boy, solo` hoặc `1girl, solo`
4. `pytest tests/ -q` — 51/51 pass

### Decisions
- `Character` model không thay đổi — Markdown là file `.md` riêng
- `extract_all_characters()` giữ nguyên — backward compat với tests
- `wiki_artifacts` / `wiki_remaster_batches` rỗng hiện tại — code guard sẵn, không bị lỗi
- `_check_remaster_coverage` bị bỏ — không cần thiết, `MAX(extraction_version)` đủ
- `gender = null` → render unisex (`1other, solo, androgynous`), không skip
- `min_importance=6` default — loại nhân vật phụ không đáng tạo image
- `build_all_profiles` là standalone one-time per story (chạy từ orchestrator episode 1, idempotent)