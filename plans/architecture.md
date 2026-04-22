## Architecture Decision Record — Video Production Pipeline

---

### Context & Ràng Buộc

| Chiều | Thực trạng |
|-------|-----------|
| **Hardware** | RTX 5060 Ti 8GB VRAM, on-premise, 1 máy |
| **Team** | 1 người, maintain lâu dài |
| **Batch size** | 100 videos, chạy 1 lần (không real-time) |
| **Integrations** | truyencv.io (HTTP), Ollama (localhost), ComfyUI (localhost), Edge TTS (Azure cloud) |
| **Scale horizon** | Có thể swap story source → pipeline phải abstract hóa source |

---

### Phase 2 — Bottleneck & Critical Path Analysis

#### Ước tính thời gian thực tế (RTX 5060 Ti)

| Phase | Tính toán | Thời gian | Ghi chú |
|-------|-----------|-----------|---------|
| Crawl 3534 chương | 3534 × 1s (rate-limit) | **~1 giờ** | I/O bound |
| LLM Phase 2 | 100 episodes × 9 calls × 30s avg | **~7.5 giờ** | Serial, VRAM-bound |
| Image gen SDXL | 900 ảnh × 12s (IPAdapter overhead) | **~3 giờ** | GPU-bound |
| SVD key shots | 250 shots × 75s avg | **~5-6 giờ** | GPU-bound, optional |
| TTS audio | 900 shots × 2s (network) | **~30 phút** | Parallel với image gen |
| FFmpeg NVENC | 100 episodes × 2 phút | **~3 giờ** | CPU nhẹ + NVENC |
| **Tổng (có SVD)** | | **~21 giờ** | |
| **Tổng (không SVD)** | | **~15 giờ** | |

**Critical Path:** `LLM (7.5h)` → `Image+SVD (8-9h)` → `Video (3h)` — **SVD là nút nghẽn lớn nhất sau LLM**.

#### Resource Contention Map

```
GPU VRAM 8GB:
  [Phase 2 LLM]    ████████ 7.7GB  ← full card, block everything else
  [Phase 3 SDXL]  ███████  6-7GB  ← safe, sau khi unload Ollama
  [Phase 3 SVD]   ████████ 7-8GB  ← gần full, OOM risk
  [Phase 5 NVENC] █        1GB    ← nhẹ

CPU:
  [FFmpeg zoompan] ████  CPU-bound, đơn luồng theo mặc định
  [FFmpeg concat]  █     nhẹ
```

**Assumption chưa kiểm chứng**:
1. ComfyUI workflow JSON templates chưa có — cần tạo thủ công và test trước khi run full pipeline
2. `qwen2.5:7b Q8_0` tốc độ inference thực tế trên 5060 Ti (~30-50 tok/s — chưa đo)
3. IPAdapter có sẵn trong ComfyUI installation chưa (`ComfyUI-IPAdapter-plus` custom node)

---

### Phase 3 — Kiến Trúc & Quyết Định Kỹ Thuật

---

#### [Quyết định 1] Workflow Orchestration

| Tiêu chí | Custom Python + SQLite | Prefect (local) | Celery + Redis |
|----------|----------------------|-----------------|----------------|
| Độ phức tạp setup | Low | Medium | High |
| RAM overhead | ~50MB | ~1GB (server) | ~500MB |
| Retry/resume | Manual (SQLite state) | Built-in | Built-in |
| UI giám sát | Không | Web UI có sẵn | Flower (thêm setup) |
| Phù hợp batch 1 máy | ✅ | ✅ | ❌ overkill |

**Decision**: Custom Python + SQLite  
**Lý do**: Pipeline là batch job tuần tự trên 1 máy, không cần distributed. Prefect cần chạy một server process riêng (~1GB RAM liên tục) chỉ để track state — VRAM đã căng, không nên thêm gánh nặng.  
**Trade-off chấp nhận**: Không có Web UI — business logger + SQLite query đủ để debug.  
**Consequence**: Tiết kiệm ~1GB RAM, giữ toàn bộ VRAM cho LLM/ComfyUI; code orchestrator phức tạp hơn Prefect ~30%.  
**Điều kiện review lại**: Khi cần chạy song song nhiều story hoặc nhiều máy → chuyển sang Prefect/Ray.

---

#### [Quyết định 2] Async Strategy per Layer

**Decision**: Phân tầng rõ ràng — không dùng async đồng bộ toàn bộ.

| Module | Strategy | Lý do |
|--------|----------|-------|
| `crawler/scraper.py` | `asyncio + httpx` | I/O bound; asyncio + semaphore(1) = rate-limit đẹp hơn `time.sleep` |
| `llm/*.py` | Sync | Ollama serial trên 1 GPU; async không tăng throughput |
| `image_gen/comfyui_client.py` | Sync + polling | ComfyUI WebSocket có nhưng polling đơn giản, latency không critical cho batch |  
| `audio/tts.py` | `asyncio + edge-tts` | Network I/O; async gen nhiều shots song song, không chiếm GPU |
| `video/assembler.py` | `ThreadPoolExecutor(4)` | Zoompan clips độc lập nhau, parallelism trên CPU; NVENC encode vẫn serial |

**Consequence**: TTS 900 shots chạy async batch → ~5 phút thay vì 30 phút sequential.

---

#### [Quyết định 3] SVD — Default OFF

**Decision**: `enable_svd: false` trong `config/settings.yaml`.  
**Lý do**: SVD thêm 5-6 giờ vào pipeline (25% tổng thời gian) cho cải thiện visual ở 2-3 shot/episode. Lần đầu chạy ưu tiên ra đủ 100 videos nhanh nhất để validate end-to-end flow.  
**Trade-off chấp nhận**: Key shots dùng zoompan — visual kém SVD nhưng ra hàng trước.  
**Consequence**: Pipeline rút ngắn từ 21h → 15h cho lần đầu. SVD có thể enable để re-render key shots cho episodes đã có, vì state machine cho phép re-run từng phase.  
**Điều kiện review lại**: Khi episode 1 đã validate xong, enable SVD cho batch tiếp.

---

#### [Quyết định 4] Dependency Management

**Decision**: `uv` + `requirements.txt`  
**Lý do**: uv là drop-in replacement cho pip, nhanh 10-100x khi resolve dependencies. Không cần pyproject.toml complexity (project 1 người). Conda không cần vì CUDA driver quản lý bởi ComfyUI venv riêng.  
**Consequence**: `uv pip install -r requirements.txt` đủ, tương thích 100% với pip.

---

#### [Quyết định 5] Config Management

**Decision**: `pydantic-settings` + YAML  
**Lý do**: Type validation bắt lỗi config **tại startup** (Fail Fast). Swap story = sửa 1 field `story_slug` trong settings.yaml, không sửa code. Pydantic cho autocomplete IDE, runtime error message rõ hơn `KeyError` từ plain dict.  

```python
# config/settings.py (auto-generated)
class Settings(BaseSettings):
    story_slug: str = "mao-son-troc-quy-nhan"  # → swap story ở đây
    base_url: str = "https://truyencv.io/truyen/{story_slug}/chuong-{n}/"
    total_chapters: int = 3534
    chapters_per_episode: int = 35
    ollama_url: str = "http://localhost:11434"
    comfyui_url: str = "http://localhost:8188"
    llm_model: str = "qwen2.5:7b-instruct:q8_0"
    enable_svd: bool = False  # ← OFF by default
    ...
```

---

#### [Quyết định 6] SQLite Schema (State Machine)

```sql
CREATE TABLE chapters (
    chapter_num INTEGER PRIMARY KEY,
    title TEXT, url TEXT, file_path TEXT,
    status TEXT DEFAULT 'PENDING',  -- PENDING | CRAWLED | ERROR
    crawled_at TIMESTAMP, error_msg TEXT
);

CREATE TABLE episodes (
    episode_num INTEGER PRIMARY KEY,
    chapter_start INTEGER, chapter_end INTEGER,
    status TEXT DEFAULT 'PENDING',
    -- PENDING|CRAWLED|SUMMARIZED|SCRIPTED|IMAGES_DONE|AUDIO_DONE|VIDEO_DONE|VALIDATED
    created_at TIMESTAMP, updated_at TIMESTAMP, error_msg TEXT
);

CREATE TABLE episode_timings (  -- để estimate ETA sau episode đầu
    episode_num INTEGER, phase TEXT,
    started_at TIMESTAMP, completed_at TIMESTAMP,
    duration_sec REAL
);
```

**episode_timings** là addition mới: sau episode 1 hoàn thành, orchestrator tính ETA cho toàn bộ 99 episodes còn lại.

#### [Quyết định 7] TTS Backend — Dual Strategy (Edge + Piper)

**Vấn đề gốc**: `edge-tts` gọi Microsoft Azure TTS cloud → **phụ thuộc internet**. Nếu mất kết nối hoặc Azure có sự cố, toàn bộ Phase 4 bị block. Không có fallback.

**Decision**: Hỗ trợ 2 backend, cấu hình qua `tts_backend` trong `settings.yaml`:

| Backend | Thư viện | Internet | GPU | Chất lượng tiếng Việt | Ghi chú |
|---------|----------|----------|-----|-----------------------|---------|
| `"edge"` | `edge-tts` | **Bắt buộc** (Azure) | Không | Rất tốt (HoaiMyNeural) | Default |
| `"piper"` | `piper-tts` | **Không cần** (local ONNX) | Không (CPU) | Tốt (vi_VN-vivos-medium) | Offline fallback |

**Lý do chọn Piper làm local backend**:
- CPU-only (ONNX Runtime) → không cạnh tranh VRAM với Ollama/ComfyUI
- Vẫn chạy parallel với Phase 3 (không ảnh hưởng GPU)
- Model nhỏ (~50-100MB), download từ HuggingFace một lần
- Voices `vi_VN-vivos-x_low` và `vi_VN-vivos-medium` đã có tiếng Việt

**Trade-off chấp nhận**: Chất lượng giọng đọc của piper thấp hơn HoaiMyNeural của edge-tts. Dùng `"piper"` khi không có internet hoặc muốn fully offline.

**Consequence**: `config/settings.py` validate fail-fast tại startup nếu backend = `"piper"` mà file model không tồn tại.

**Điều kiện review lại**: Nếu có local model tiếng Việt chất lượng tốt hơn (vd. Kokoro với Vietnamese fine-tune), swap backend mà không sửa code — chỉ thêm case mới trong `audio/tts.py`.

**Cách dùng**:
```yaml
# config/settings.yaml
tts_backend: "piper"          # chuyển sang local
piper_model_path: "models/piper/vi_VN-vivos-medium.onnx"
piper_config_path: "models/piper/vi_VN-vivos-medium.onnx.json"
```

Download model:
```bash
# vi_VN-vivos-medium (~70MB)
mkdir -p models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vivos/medium/vi_VN-vivos-medium.onnx -O models/piper/vi_VN-vivos-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vivos/medium/vi_VN-vivos-medium.onnx.json -O models/piper/vi_VN-vivos-medium.onnx.json
```

---



#### [RELIABILITY] ✅/⚠️
- ✅ tenacity retry crawler (max 5, exp backoff)
- ✅ tenacity retry LLM JSON parse (max 3)
- ⚠️ **ComfyUI health check còn thiếu** → thêm `comfyui_client.health_check()` trước Phase 3; nếu fail → abort với clear error thay vì hang
- ⚠️ **SVD OOM không có fallback** → nếu ComfyUI trả OOM error, auto-fallback sang zoompan cho shot đó

#### [PERFORMANCE] ✅/⚠️
- ✅ NVENC encode  
- ✅ VRAM sequential management
- ✅ TTS async (không trên critical path)
- ⚠️ **zoompan CPU-bound** → dùng `ThreadPoolExecutor(4)` cho clip render song song
- ⚠️ `episode_timings` chưa có → không estimate được ETA

#### [OBSERVABILITY] ✅/⚠️
- ✅ loguru per-episode log  
- ✅ SQLite state tracking  
- ⚠️ **Không có timing metrics** → `episode_timings` table cần thêm vào state machine

#### [SECURITY] ✅/⚠️
- ✅ Rate-limit + rotate User-Agent
- ✅ Endpoints configurable (không hardcode)
- ⚠️ **FFmpeg subtitle injection risk**: `narration_text` từ LLM → SRT → `drawtext` filter: nếu có ký tự `:` hoặc `'` chưa escape sẽ làm hỏng FFmpeg command → **cần sanitize narration_text trước khi ghi SRT** (strip/escape special chars)

#### [OPERABILITY] ✅
- ✅ `--episode N`, `--from-phase PHASE`
- ✅ cleanup sau validator pass
- ✅ `enable_svd: false` để skip phase nặng

#### [EVOLVABILITY] ✅/⚠️
- ✅ `story_slug` abstract → swap story = 1 config line
- ✅ ComfyUI workflow tách riêng JSON files
- ⚠️ **Character list hardcoded** → đưa vào settings.yaml: `characters: ["Diep Thieu Duong", ...]`

---

### Phase 5 — Output: Tech Stack & Cấu Trúc Source

#### Tech Stack

| Layer | Thư viện | Lý do chọn |
|-------|----------|-----------|
| HTTP crawl | `httpx[asyncio]` + `beautifulsoup4` | Async + lxml parser nhanh |
| Retry | `tenacity` | Decorator-based, clean API |
| LLM client | `httpx` → Ollama REST (tự viết, không dùng ollama-python SDK) | SDK thêm dependency không cần thiết, REST đủ |
| Config | `pydantic-settings` + `PyYAML` | Type-safe, Fail Fast |
| Database | `sqlite3` (stdlib) | Zero dependency, đủ cho 1 máy |
| TTS | `edge-tts` | Free, offline-ish, tiếng Việt tốt |
| Image gen | `httpx` → ComfyUI REST/WebSocket | self-hosted |
| Video | `ffmpeg-python` (wrapper) | Pythonic FFmpeg API |
| Logging | `loguru` | Structured, per-file sink dễ setup |
| Validation | `pydantic` (model cho shot schema) | JSON parse + validate trong 1 bước |
| Testing | `pytest` + `pytest-asyncio` | Standard |
| Package mgmt | `uv` | 10-100x nhanh hơn pip |

#### Cấu Trúc Source

```
video-pipeline/
├── config/
│   └── settings.yaml              # Story source, API endpoints, model, enable_svd
├── crawler/
│   ├── scraper.py                  # asyncio + httpx + tenacity; semaphore rate-limit
│   └── storage.py                  # SQLite write + file write; idempotent
├── llm/
│   ├── client.py                   # Ollama REST wrapper (generate, health_check)
│   ├── summarizer.py               # two-pass batching (5ch/chunk → arc overview)
│   ├── character_extractor.py      # global run → characters/*.json
│   └── scriptwriter.py             # arc → ShotScript (pydantic model, is_key_shot by LLM)
├── image_gen/
│   ├── comfyui_client.py           # submit_prompt, poll_result, health_check
│   ├── character_gen.py            # Anchor image gen (run once guard)
│   └── workflows/
│       ├── txt2img_ipadapter.json
│       ├── txt2img_scene.json
│       ├── svd_keyshot.json
│       └── thumbnail.json
├── audio/
│   ├── tts.py                      # async edge-tts, batch generate per episode
│   └── mixer.py                    # FFmpeg amix narration + BGM
├── video/
│   ├── assembler.py                # per-shot: zoompan|SVD + narration → clip
│   │                               # ThreadPoolExecutor(4) cho zoompan clips
│   └── editor.py                   # concat + xfade + SRT burn-in + intro/outro CTA
├── pipeline/
│   ├── orchestrator.py             # CLI (argparse), main loop, phase dispatch
│   ├── state.py                    # SQLite state machine + episode_timings
│   ├── vram_manager.py             # ollama stop/start; ComfyUI/Ollama health check
│   └── validator.py                # assert duration/resolution/audio/size/thumbnail
├── models/
│   └── schemas.py                  # pydantic: ShotScript, EpisodeScript, Character
├── data/
│   ├── chapters/                   # chuong-{N}.txt — LLM material only
│   ├── summaries/                  # episode-{N}-chunks.json, episode-{N}-arc.json
│   ├── scripts/                    # episode-{N}-script.json
│   ├── characters/                 # {name}/anchor.png + {name}.json
│   ├── images/                     # episode-{N}/shot-{K}.png (xóa sau validate)
│   ├── audio/                      # episode-{N}/shot-{K}.mp3 (xóa sau validate)
│   ├── thumbnails/                 # episode-{N:03d}.png (giữ lại)
│   └── output/                     # episode-{N:03d}.mp4 (final)
├── assets/
│   └── music/                      # royalty-free BGM .mp3
├── db/
│   └── pipeline.db                 # auto-created
├── logs/
│   ├── pipeline.log                # tổng
│   └── episode-{N:03d}.log        # per-episode
├── tests/
│   ├── test_crawler.py
│   ├── test_llm.py
│   ├── test_comfyui.py
│   └── test_video.py
├── requirements.txt
├── .gitignore
└── main.py                         # Entry: python main.py [--episode N] [--from-phase PHASE] [--dry-run]
```

#### Data Flow Diagram

```
                        truyencv.io
                             │ httpx async + tenacity
                             ▼
                     crawler/scraper.py
                             │ raw text + metadata
                             ▼
              crawler/storage.py ──── db/pipeline.db (state)
                             │          data/chapters/*.txt
                             │
                  [35 chương/episode batch]
                             │
                             ▼
              llm/summarizer.py ─── Ollama qwen2.5:7b Q8_0
              (Pass1: 5ch chunks)       (localhost:11434)
              (Pass2: arc synthesis)
                             │ arc_overview.json
                   ┌─────────┴──────────┐
                   ▼                    ▼
    llm/character_extractor.py   llm/scriptwriter.py
    ─── characters/*.json        ─── scripts/episode-{N}.json
                                     (is_key_shot per shot)
                                     (characters: [] per shot)
                                            │
                          ┌─────────────── vram_manager.py ──────┐
                          │ ollama stop                           │
                          ▼                                       │
          image_gen/character_gen.py                             │
          ─── data/characters/*/anchor.png (1st run only)       │
                          │                                       │
           ┌──────────────┼──────────────────────────┐          │
           ▼              ▼                           ▼          │
  characters != []   characters == []           is_key_shot     │
  txt2img_ipadapter  txt2img_scene              svd_keyshot     │
  + IPAdapter        + InkWash LoRA             (if enable_svd) │
           └──────────────┴──────────────────────────┘          │
                          │ data/images/episode-N/*.png          │
                          │                     ╔════════════════╝
                          │ (parallel)          ║ audio/tts.py (asyncio)
                          │                     ║ edge-tts → Azure TTS cloud
                          │                     ║ data/audio/episode-N/*.mp3
                          │                     ║ audio/mixer.py + BGM
                          ├─────────────────────╝
                          ▼
              video/assembler.py
              ThreadPoolExecutor(4): zoompan clips
              serial: SVD clips, NVENC encode
                          │ per-shot clips
                          ▼
              video/editor.py
              concat + xfade + SRT burn-in + CTA outro
              -c:v h264_nvenc
                          │
                          ▼
              pipeline/validator.py
              assert: duration 58-62s, 720×1280, audio, size>5MB
                          │ PASS
                          ├── cleanup images + audio
                          ▼
              data/output/episode-{N:03d}.mp4
              + data/thumbnails/episode-{N:03d}.png
```

#### Critical Path

```
CRAWL (1h) → LLM (7.5h) → IMAGE GEN (3h) + SVD (5-6h, optional) → VIDEO (3h)
                                    ↕ parallel
                              TTS AUDIO (0.5h)

Tổng không SVD: ~15 giờ  |  Tổng có SVD: ~21 giờ
```

#### Open Questions (Assumptions chưa kiểm chứng)

1. **ComfyUI custom nodes** `ComfyUI-IPAdapter-plus` và `ComfyUI-VideoHelperSuite` (cho SVD) đã install chưa?
2. **Pony V6 XL checkpoint** đã download tại `ComfyUI/models/checkpoints/` chưa?
3. **qwen2.5:7b Q8_0 inference speed** trên RTX 5060 Ti: cần đo thực tế, ảnh hưởng trực tiếp đến 7.5h LLM estimate
4. **truyencv.io có Cloudflare bot protection không?** — nếu có, httpx đơn thuần không đủ, cần `curl_cffi` hoặc Playwright

#### Review Triggers

| Khi nào | Action |
|---------|--------|
| Episode 1 hoàn thành → check `episode_timings` | Điều chỉnh `chapters_per_episode` nếu LLM chậm hơn/nhanh hơn dự kiến |
| Episode 1 pass validator | Enable `enable_svd: true` để thử SVD cho episode 2 |
| Swap sang story khác | Chỉ sửa `story_slug`, `base_url`, `total_chapters` trong settings.yaml |
| Scale lên nhiều story song song | Migrate orchestrator sang Prefect + thêm GPU thứ 2 | 

Completed: *Phase 3: Thiết kế kiến trúc + tech stack* (2/4)

ADR hoàn chỉnh. Tóm tắt những quyết định quan trọng nhất:

- **Orchestration**: Custom Python + SQLite — không Prefect, tiết kiệm ~1GB RAM VRAM đang căng
- **SVD default OFF** — pipeline 15h thay vì 21h cho lần đầu chạy; enable sau khi episode 1 validated
- **Async phân tầng**: chỉ crawler và TTS dùng asyncio; LLM và ComfyUI sync (serial GPU)
- **`episode_timings` table mới**: sau episode 1 có ETA chính xác cho 99 episodes còn lại
- **2 lỗ hổng cần fix trước khi code**: narration_text sanitize trước SRT (FFmpeg injection), ComfyUI health check trước Phase 3
- **4 assumptions cần kiểm chứng ngay**: IPAdapter/VideoHelper nodes, Pony V6 checkpoint, qwen2.5 real speed, Cloudflare protection 

