# Video Pipeline — Workspace Instructions

## Project Context

Batch pipeline Python sản xuất 100 video 60s (9:16 Shorts/TikTok) từ nguồn truyện online.  
Stack: Python · SQLite · Ollama (local LLM) · ComfyUI (local image gen) · Edge TTS · FFmpeg NVENC.  
Hardware: RTX 5060 Ti 8GB VRAM, on-premise, 1 người maintain.

Tài liệu tham khảo:
- `plans/initial.md` — plan chi tiết theo phase
- `plans/architecture.md` — ADR với reasoning chain đầy đủ

## Ngôn Ngữ

- **Code, config keys, file names, symbol names**: tiếng Anh
- **Comments trong code**: tiếng Anh
- **Giải thích, plan, tài liệu, chat responses**: tiếng Việt
- **LLM prompts cho ComfyUI (scene_prompt)**: tiếng Anh
- **LLM prompts cho TTS (narration_text)**: tiếng Việt

## Tech Stack Chuẩn (không thay thế nếu không có lý do rõ ràng)

| Mục đích | Thư viện |
|----------|----------|
| Config | `pydantic-settings` + `PyYAML` |
| JSON schema | `pydantic` (BaseModel) |
| HTTP | `httpx` (async hoặc sync tùy context) |
| Retry | `tenacity` |
| Logging | `loguru` |
| State DB | `sqlite3` (stdlib) |
| TTS | `edge-tts` |
| Video encode | FFmpeg với `-c:v h264_nvenc` |
| Package mgmt | `uv` |

## Abstraction Principle

Source story phải abstract: mọi thứ liên quan đến `story_slug`, `base_url`, `total_chapters` phải đọc từ `config/settings.yaml`, không hardcode trong code.  
Swap story = thay đổi config, không sửa code.
