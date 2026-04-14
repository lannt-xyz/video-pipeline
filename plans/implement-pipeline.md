# Video Production Pipeline — Full Implementation

## Status: In Progress

## Mục tiêu
Implement toàn bộ video production pipeline từ đầu theo `plans/initial.md` và `plans/architecture.md`. Output: 100 Shorts/TikTok video 60s từ truyện Mao Sơn Tróc Quỷ Nhân.

## Branch
`feature/implement-pipeline`

---

## Roadmap

### 1. Project Bootstrap
- [ ] Tạo cấu trúc thư mục (config/, crawler/, llm/, image_gen/, audio/, video/, pipeline/, models/, data/, assets/, db/, logs/, tests/)
- [ ] `config/settings.yaml` — story source, API endpoints, model, paths, flags
- [ ] `config/settings.py` — pydantic-settings wrapper
- [ ] `requirements.txt`
- [ ] `.gitignore`

### 2. Models
- [ ] `models/schemas.py` — ShotScript, EpisodeScript, Character, ChapterMeta (Pydantic BaseModel)

### 3. Crawler
- [ ] `crawler/scraper.py` — asyncio + httpx + BeautifulSoup; semaphore(1) rate-limit; tenacity retry (max 5, exp backoff); rotate User-Agent
- [ ] `crawler/storage.py` — lưu raw text `data/chapters/chuong-{N}.txt`; SQLite idempotent write

### 4. LLM Module
- [ ] `llm/client.py` — Ollama REST wrapper (generate, health_check) dùng httpx sync
- [ ] `llm/summarizer.py` — two-pass: Pass1 (5ch/chunk → chunk summary) + Pass2 (7 chunks → arc overview)
- [ ] `llm/character_extractor.py` — extract toàn bộ nhân vật từ tất cả summaries → `data/characters/{name}.json`
- [ ] `llm/scriptwriter.py` — arc overview → EpisodeScript JSON (8-10 shots, is_key_shot LLM-assigned); tenacity retry max 3

### 5. Image Generation Module
- [ ] `image_gen/comfyui_client.py` — submit_prompt, poll_result, health_check; SVD OOM → raise ComfyUIOutOfMemoryError
- [ ] `image_gen/character_gen.py` — generate anchor image per character (run-once guard); lưu `data/characters/{name}/anchor.png`
- [ ] `image_gen/workflows/txt2img_ipadapter.json` — SDXL + IPAdapter workflow template
- [ ] `image_gen/workflows/txt2img_scene.json` — SDXL scene-only workflow template
- [ ] `image_gen/workflows/svd_keyshot.json` — SVD workflow template
- [ ] `image_gen/workflows/thumbnail.json` — thumbnail 1280×720 + DrawText workflow

### 6. Audio Module
- [ ] `audio/tts.py` — asyncio + edge-tts; giọng vi-VN-HoaiMyNeural; batch generate per episode
- [ ] `audio/mixer.py` — FFmpeg amix: narration (fg) + BGM (bg, -15dB)

### 7. Video Module
- [ ] `video/assembler.py` — per-shot: zoompan (default) hoặc SVD clip; ThreadPoolExecutor(4) zoompan; serial NVENC encode; SVD fallback to zoompan on OOM
- [ ] `video/editor.py` — concat clips + xfade + SRT burn-in (narration_text sanitized) + intro 2s + outro CTA 3s; h264_nvenc

### 8. Pipeline Module
- [ ] `pipeline/state.py` — SQLite state machine; schema: chapters, episodes, episode_timings; ETA calculation
- [ ] `pipeline/vram_manager.py` — ollama stop/start; VRAM free assertion; health check ComfyUI + Ollama
- [ ] `pipeline/validator.py` — assert: duration 58-62s, resolution 720×1280, audio present, size>5MB, thumbnail exists
- [ ] `pipeline/orchestrator.py` — CLI argparse; main loop phase dispatch; --episode N, --from-phase PHASE, --dry-run
- [ ] `logs/` directory placeholder

### 9. Entry Point
- [ ] `main.py` — entry point, logger setup, delegate to orchestrator

### Testing & Verification
- [ ] `tests/test_crawler.py` — mock httpx, test parse + storage idempotency
- [ ] `tests/test_llm.py` — mock Ollama response, test JSON parse + schema validation
- [ ] `tests/test_video.py` — test sanitize narration, test validator assertions
- [ ] Smoke test: `python main.py --dry-run`

---

## Acceptance Criteria
- [ ] `python main.py --dry-run` chạy không crash, log setup OK
- [ ] `python main.py --episode 1 --from-phase crawl` chạy được (giả định services up)
- [ ] `pytest tests/ -v` tất cả tests pass
- [ ] `config/settings.py` fail fast nếu settings.yaml thiếu field bắt buộc
- [ ] Không có hardcoded URL, story slug, hoặc model name trong code
- [ ] FFmpeg command luôn dùng `-c:v h264_nvenc`
- [ ] Ollama và ComfyUI không chạy đồng thời (vram_manager enforce)
- [ ] narration_text được sanitize trước khi ghi SRT

## Notes
- SVD default OFF (`enable_svd: false`) — pipeline 15h thay vì 21h cho lần đầu
- LLM calls là sync (serial GPU), TTS là async, Crawler là async
- ComfyUI workflow JSON templates là parameterized (placeholders thay thế lúc runtime)
- story_slug abstract: swap story = chỉ sửa settings.yaml
- Pydantic model `ShotScript` dùng để validate LLM JSON output trực tiếp
- Branch: feature/implement-pipeline, base: main
