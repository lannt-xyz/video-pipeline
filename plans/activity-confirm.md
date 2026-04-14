Có 3 tầng verification tùy theo infra nào đã sẵn sàng:

**Ngay bây giờ (không cần infra):**
```bash
# 1. Unit tests (mocks) — đã pass
.venv/bin/pytest tests/ -v

# 2. Config load + dry-run toàn pipeline
.venv/bin/python main.py --dry-run

# 3. Validate config parse đúng
.venv/bin/python -c "from config.settings import settings; print(settings.total_episodes, settings.get_chapter_url(1))"
```

**Khi có FFmpeg (không cần GPU):**
```bash
# Parse HTML thật từ truyencv.io
.venv/bin/python -c "
import asyncio
from crawler.scraper import crawl_chapters
chapters = asyncio.run(crawl_chapters([1, 2]))
print(chapters[0].title, len(chapters[0].content))
"
```

**Khi có Ollama + ComfyUI:**
```bash
# Chạy 1 episode nhỏ end-to-end
.venv/bin/python main.py --episode 1
# Xem log
tail -f logs/episode-001.log
```

**Kiểm tra output:**
```bash
# Sau khi chạy xong episode 1
ls -lh data/output/episode-001.mp4
ls -lh data/thumbnails/episode-001.png
ffprobe data/output/episode-001.mp4  # check duration/resolution
```

Cần check cụ thể tầng nào trước? Nếu đang setup Ollama hay ComfyUI thì tôi có thể viết script smoke-test từng service riêng lẻ. 

