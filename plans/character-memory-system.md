## Plan: Character Memory System (RAG + Evolution)

**TL;DR**: Nâng cấp `Character` từ flat JSON → layered evolution JSON. Xây `llm/character_memory.py` mới làm Crawl & Filter + LLM extraction. Tích hợp vào orchestrator với lazy-update trigger mỗi arc boundary.

---

### Phase 1 — Schema Evolution (schemas.py)

1. Thêm `EvolutionStage` Pydantic model với các fields: `chapter_range`, `outfit`, `props`, `mood`, `anchor_image` (Optional)
2. Update `Character` — giữ nguyên `description` làm **DNA base** (gender, face, hair — bất biến), thêm `evolution_stages: List[EvolutionStage] = []` (backward-compatible)

### Phase 2 — Character Memory Module (`llm/character_memory.py`) — **file mới**

3. `extract_character_passages(character, chapter_nums) → List[str]`
   - Load `data/chapters/chuong-{n:04d}.txt`
   - Keyword scan tên + aliases, lấy đoạn match ± N câu xung quanh
   - Dedup bằng `difflib.SequenceMatcher` (stdlib, không cần thư viện mới)
   - Cap ở `character_memory_max_passages` (default 30) để không nổ context

4. `extract_evolution_stage(character, passages, chapter_range) → Optional[EvolutionStage]`
   - Gọi `OllamaClient.generate_json()` với prompt: extract outfit/props/mood → JSON
   - Dùng lại retry logic từ client.py

5. `update_character_evolution(character_name, chapter_nums, chapter_range) → bool`
   - Orchestrate bước 3+4, merge stage vào JSON (replace nếu range đã có, append nếu mới)

### Phase 3 — Stage-Aware Image Generation (character_gen.py)

6. Thêm `get_stage_for_chapter(character, chapter_num) → Optional[EvolutionStage]` — parse "51-200" → range check
7. Update prompt building trong `run_images()`: merge `character.description` (DNA) + stage.outfit + stage.props + stage.mood

### Phase 4 — Config

8. Thêm 3 keys vào `settings.yaml` + `Settings` model:
   - `character_memory_context_sentences: 5`
   - `character_memory_arc_size: 50`
   - `character_memory_max_passages: 30`

### Phase 5 — Orchestrator Trigger (orchestrator.py)

9. Trong `run_llm()`, sau summarization: nếu `episode_num % arc_size == 1` → chạy `update_character_evolution()` cho tất cả characters trong arc đó

---

**Relevant files**: schemas.py, llm/character_memory.py *(mới)*, client.py, character_gen.py, orchestrator.py, settings.yaml, settings.py

---

**Verification**
1. Unit test `get_stage_for_chapter()`: chapter 1 → None, chapter 100 → stage "51-200", chapter 201 → None
2. Unit test `extract_character_passages()` với mock files: assert count ≤ max_passages
3. Integration test `update_character_evolution()` với mock Ollama → verify file JSON update đúng
4. Smoke test `Character.model_validate()` sau update không raise

---

**Scope exclusion**: Auto-regenerate anchor ảnh per evolution stage **không** nằm trong plan này (tốn VRAM, để phase sau).

**Lưu ý về context window**: 30 passages × ~200 tokens ≈ 6000 tokens input — an toàn với `llm_context_size: 16384` hiện tại.