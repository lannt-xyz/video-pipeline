## Plan: Video Production Pipeline — Mao Sơn Tróc Quỷ Nhân (100 Shorts/TikTok)

**TL;DR:** Pipeline Python hoàn chỉnh, 6 phase tuần tự, từ crawl 3534 chương → LLM (Ollama) **tóm tắt & diễn giải lại (abstract)** → ComfyUI tạo ảnh anime 9:16 → Edge TTS narration → FFmpeg lắp ráp thành 100 video 60s cho Shorts/TikTok. Mục tiêu: tăng tương tác & bán hàng.

> **Về bản quyền**: nội dung được **diễn giải lại hoàn toàn** bằng LLM (không reproduce nguyên văn), tương tự review/commentary. Không đăng lại text gốc dưới bất kỳ hình thức nào.

---

### Phase 1 — Crawling
1. Khởi tạo project `video-pipeline/` với cấu trúc thư mục + `config/settings.yaml`
2. `crawler/scraper.py` — async crawler dùng **httpx + BeautifulSoup**: extract chapter title, nội dung, URL chương tiếp theo; rate-limit 1 req/s; **tenacity retry** (exponential backoff, max 5 lần) khi gặp lỗi mạng hoặc HTTP 429/5xx
3. `crawler/storage.py` — lưu raw text vào `data/chapters/chuong-{N}.txt` (chỉ dùng làm material cho LLM, không phân phối); đánh dấu trạng thái vào **SQLite** (`db/pipeline.db`)
4. Resume-safe: kiểm tra DB trước mỗi request, bỏ qua chapter đã crawl
5. Dùng URL pattern: `https://truyencv.io/truyen/mao-son-troc-quy-nhan/chuong-{N}/`, tổng ~3534 chương; rotate User-Agent header để giảm khả năng bị block

### Phase 2 — LLM Processing (Ollama)
6. `llm/summarizer.py` — gọi Ollama REST API (`http://localhost:11434`), model `qwen2.5:7b-instruct` Q8_0 (~7.7GB, vừa 8GB VRAM); **two-pass batching** để tránh tràn context window:
   - **Pass 1**: summarize từng nhóm 5 chương → chunk summary (~200 từ/chunk, 7 chunk/episode)
   - **Pass 2**: gộp 7 chunk summary → 1 arc overview + kịch bản episode
7. `llm/character_extractor.py` — extract tuyến nhân vật, quan hệ, ngoại hình → JSON character sheet (dùng làm prompt cho ComfyUI); chạy 1 lần qua toàn bộ chương đã summarize
8. `llm/scriptwriter.py` — từ arc overview 1 episode → viết kịch bản JSON gồm 8-10 shots; **LLM tự đánh dấu `is_key_shot`** dựa trên instruction trong system prompt ("đánh dấu 2-3 shot có cảnh hành động/cảm xúc cao trào nhất"); schema mỗi shot:
   ```json
   {
     "scene_prompt": "string (tiếng Anh, cho ComfyUI)",
     "narration_text": "string (tiếng Việt, cho TTS)",
     "duration_sec": 6,
     "is_key_shot": false,
     "characters": ["Diep Thieu Duong"]  // [] nếu shot cảnh thuần
   }
   ```
9. LLM output *luôn JSON schema cố định*, dùng structured output hoặc retry với **tenacity** (max 3 lần) nếu parse fail

### Phase 3 — Character & Image Generation (ComfyUI SDXL)
> ⚠️ **VRAM Strategy (8GB RTX 5060 Ti):** Phase 2 (Ollama) và Phase 3 (ComfyUI) chạy **tuần tự**, không song song. Sau khi Phase 2 xong, gọi `ollama stop` để unload model khỏi VRAM trước khi khởi động ComfyUI jobs.

10. `image_gen/character_gen.py` — generate ảnh "Anchor" reference cho 10 nhân vật chính (chạy 1 lần đầu tiên); lưu vào `data/characters/{name}/anchor.png` — đây là khuôn mặt gốc cho IPAdapter
11. `image_gen/comfyui_client.py` — gọi ComfyUI API (`http://localhost:8188`); workflow sử dụng:
    - **Checkpoint**: Pony Diffusion V6 XL
    - **LoRA stack** (xem bảng đề xuất bên dưới)
    - **IPAdapter**: inject ảnh Anchor vào mọi shot có `characters != []` → giữ khuôn mặt nhất quán 100%
    - Shot có `characters == []` → dùng `txt2img_scene.json`, không cần IPAdapter
    - Output: 720×1280 (9:16), PNG
12. Workflow JSON template được parameterized (`scene_prompt`, `character_anchor_path`, `lora_weights`, `seed`); seed chỉ dùng để reproduce — visual consistency chính do IPAdapter đảm bảo
13. Sinh ảnh nền/cảnh (shot `characters == []`) từ scene_prompt đơn thuần; dùng LoRA Ink Wash để tạo bầu không khí Manhua
14. **Thumbnail Generator** (step phụ): với shot đánh dấu `is_key_shot=true`, gen thêm 1 ảnh bìa 1280×720 (16:9) có Draw Text node (tiêu đề tập, số tập + CTA "Link bio 👇") → lưu vào `data/thumbnails/episode-{N:03d}.png`

### Phase 4 — Audio (Edge TTS + FFmpeg Mix)
> ℹ️ **Edge TTS yêu cầu kết nối internet** (gọi Microsoft Azure TTS cloud). Chạy parallel với Phase 3 được vì không dùng GPU.

15. `audio/tts.py` — dùng **edge-tts** Python lib, giọng `vi-VN-HoaiMyNeural`, generate MP3 narration per shot *(parallel với Phase 3)*
16. `audio/mixer.py` — mix narration (foreground) + BGM royalty-free (background, -15dB) bằng FFmpeg `amix`

### Phase 5 — Video Assembly (FFmpeg + NVENC)
17. `video/assembler.py` — per-shot rendering theo chiến lược hybrid:
    - **Shot thường** (7-8 shot/episode): ảnh + `zoompan` filter (Ken-Burns đơn giản) + narration → clip 6-7s
    - **Shot quan trọng** (`is_key_shot=true`, 2-3 shot/episode): gọi **SVD (Stable Video Diffusion)** qua ComfyUI API → SVD output ~25 frames @6fps (~4.2s); áp dụng `setpts=1.43*PTS` để slow-motion → đủ 6s
    - Encode bắt buộc dùng **`-c:v h264_nvenc`** (NVIDIA GPU encode, nhanh gấp ~10× so với libx264 CPU)
18. `video/editor.py` — concat 8-10 clips → 60s episode video; thêm:
    - fade in/out giữa shots (`xfade` filter)
    - subtitle SRT burn-in (từ narration text)
    - intro logo 2s + outro CTA 3s ("Theo dõi để xem tiếp + link bio bán hàng")
19. `pipeline/validator.py` — assertion sau mỗi episode: duration 58-62s, resolution 720×1280, audio present, file size >5MB; fail → re-queue episode
20. **Storage cleanup**: sau khi validator pass, xóa `data/images/episode-{N}/` và `data/audio/episode-{N}/`; giữ lại `data/scripts/` + `data/summaries/`
21. Output: `data/output/episode-{N:03d}.mp4` (720×1280, H.264 NVENC, AAC 44.1kHz, 30fps)

### Phase 6 — Orchestrator
22. `pipeline/orchestrator.py` — chạy toàn bộ pipeline, track tiến độ qua SQLite state machine (states: `CRAWLED → SUMMARIZED → SCRIPTED → IMAGES_DONE → AUDIO_DONE → VIDEO_DONE → VALIDATED`)
23. `pipeline/vram_manager.py` — gọi `ollama stop` trước Phase 3, gọi lại `ollama serve` trước Phase 2 khi resume
24. Structured logging với `loguru` → `logs/episode-{N:03d}.log` per episode + `logs/pipeline.log` tổng
25. Support `--episode N` để re-run 1 episode riêng lẻ; `--from-phase PHASE` để resume

---

**LoRA Stack đề xuất cho Pony V6 XL**
| LoRA | Mục đích | Weight |
|------|----------|--------|
| `[Pony] Manhua Style v2` | Lineart đậm, màu sắc Manhua Trung Quốc | 0.7 |
| `[Pony] Xianxia Fantasy` | Background cung điện, núi non, linh khí | 0.5 |
| `[Pony] Dark Horror Atmosphere` | Cảnh quỷ dị, màu tối, atmosphere Mao Sơn | 0.4 |
| `[Pony] Ink Wash Painting` | Shot cảnh thuần không nhân vật, phong cách thủy mặc | 0.6 |

> Tất cả download tại **CivitAI** → filter: *Pony*, *LoRA*, search theo tên. Đặt vào `ComfyUI/models/loras/`.

---

**Relevant files (to create)**
- `config/settings.yaml` — API endpoints, model names, paths, rate limits, VRAM strategy flags
- `crawler/scraper.py` + `crawler/storage.py`
- `llm/summarizer.py` + `llm/character_extractor.py` + `llm/scriptwriter.py`
- `image_gen/comfyui_client.py` + `image_gen/character_gen.py`
- `image_gen/workflows/txt2img_ipadapter.json` — ComfyUI workflow template cho shot có nhân vật
- `image_gen/workflows/txt2img_scene.json` — workflow template cho shot cảnh thuần
- `image_gen/workflows/svd_keyshot.json` — workflow template SVD cho key shots
- `image_gen/workflows/thumbnail.json` — workflow template thumbnail có text
- `audio/tts.py` + `audio/mixer.py`
- `video/assembler.py` + `video/editor.py`
- `pipeline/orchestrator.py` + `pipeline/state.py`
- `pipeline/vram_manager.py` — quản lý unload Ollama trước khi chạy ComfyUI
- `pipeline/validator.py` — assertion chất lượng output mỗi episode
- `db/pipeline.db` (auto-created)
- `logs/` (auto-created)
- `requirements.txt`

---

**Acceptance Criteria (tự động — `pipeline/validator.py`)**
- [ ] Video duration: 58–62s
- [ ] Resolution: 720×1280
- [ ] Audio track present, bitrate ≥ 128kbps
- [ ] File size > 5MB
- [ ] Thumbnail exists tại `data/thumbnails/episode-{N:03d}.png`
- [ ] Log file `logs/episode-{N:03d}.log` không có `ERROR`

**Verification (manual, lần đầu)**
1. Crawl test: chạy crawler cho chương 1-5, kiểm tra file `.txt` + DB row
2. LLM test: summarize 3 chương, validate JSON schema output (đặc biệt kiểm tra `characters` field và `is_key_shot`)
3. ComfyUI test: generate 1 ảnh 720×1280 với IPAdapter, kiểm tra khuôn mặt nhất quán với anchor
4. SVD test: gen 1 key shot clip, kiểm tra sau `setpts` ra đúng 6s
5. TTS test: generate audio cho 1 narration, kiểm tra Edge TTS kết nối và MP3 duration
6. FFmpeg test: assemble 1 shot clip (NVENC) → 1 episode video hoàn chỉnh
7. Full pipeline test: chạy episode 1 end-to-end, pass validator, kiểm tra `output/episode-001.mp4`

---

**Decisions**
- 35 chương/episode → 2-pass summarize (5 chương/chunk × 7 chunk → 1 arc summary) để giữ trong context window
- 8–10 shots/video × 6–7s/shot = ~60s; trong đó 2-3 shot là key shots dùng SVD
- SVD output (~4.2s) → `setpts=1.43*PTS` slow-motion → đủ 6s
- Output: 720×1280, **H.264 NVENC** (`h264_nvenc`), AAC 44.1kHz, 30fps
- BGM: pre-downloaded royalty-free, để trong `assets/music/`
- ComfyUI tại `http://localhost:8188`, Ollama tại `http://localhost:11434` — **không chạy đồng thời**
- **LLM model**: `qwen2.5:7b-instruct` Q8_0 (~7.7GB, primary, vừa 8GB VRAM)
- **Checkpoint**: Pony Diffusion V6 XL + LoRA stack (xem bảng LoRA)
- **Visual consistency**: IPAdapter dùng `characters` field từ shot schema
- **is_key_shot**: LLM tự đánh dấu theo instruction trong system prompt của `scriptwriter.py`
- **Error handling**: tenacity retry ở cả crawler và LLM calls; validator.py assertion sau mỗi episode
- **Thumbnail**: 1 ảnh bìa 1280×720 per episode + CTA text (ComfyUI Draw Text node)
- **Logging**: loguru → `logs/episode-{N:03d}.log` + `logs/pipeline.log`
- **Storage**: xóa intermediate files sau khi validator pass; giữ scripts + summaries
- **Mục tiêu**: tăng tương tác bán hàng → CTA rõ ràng trong outro + thumbnail
- **Nội dung**: abstract (diễn giải lại) không reproduce nguyên văn → giảm rủi ro bản quyền
- **Edge TTS**: cần internet (Microsoft Azure cloud); parallel với Phase 3 (không dùng GPU)

**VRAM Budget (RTX 5060 Ti 8GB)**
| Tác vụ | VRAM |  
|---|---|
| Ollama `qwen2.5:7b-instruct` Q8_0 | ~7.7GB |
| ComfyUI SDXL + IPAdapter | ~6-7GB |
| ComfyUI SVD | ~7-8GB |
| FFmpeg NVENC | ~1GB |
→ **Nguyên tắc**: chỉ 1 process nặng chạy tại 1 thời điểm; `vram_manager.py` enforce điều này.

**Risks & Mitigations**
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Bị block IP khi crawl | Medium | High | tenacity backoff + rotate User-Agent |
| LLM JSON parse fail | Medium | Medium | tenacity retry x3, fallback schema repair |
| ComfyUI OOM khi SVD | Medium | High | sequential VRAM management, fallback zoompan |
| SVD clip quá ngắn (<6s) | High | Low | `setpts=1.43*PTS` slow-motion đã xử lý |
| Edge TTS mất kết nối | Low | High | retry x3, cache MP3 đã gen |
| Khuôn mặt không nhất quán | Low | Medium | IPAdapter với Anchor image |
| Storage đầy (>20GB) | Medium | Medium | auto cleanup sau validator pass |