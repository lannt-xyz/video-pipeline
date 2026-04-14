import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crawler.scraper import parse_chapter, crawl_chapters
from models.schemas import ChapterMeta


# ── parse_chapter ─────────────────────────────────────────────────────────────

class TestParseChapter:
    def test_extracts_title_and_content(self, sample_html_chapter):
        result = parse_chapter(sample_html_chapter, "http://example.com/ch1", 1)

        assert result.chapter_num == 1
        assert result.title == "Chương 1: Khởi Đầu"
        assert "Diep Thieu Duong" in result.content
        assert result.status == "CRAWLED"
        assert result.url == "http://example.com/ch1"

    def test_raises_on_missing_content_tag(self):
        html = "<html><body><h2>Title</h2><p>Short.</p></body></html>"
        with pytest.raises(ValueError, match="Cannot find chapter content"):
            parse_chapter(html, "http://example.com", 1)

    def test_raises_on_too_short_content(self):
        html = '<html><body><div class="chapter-content">Too short.</div></body></html>'
        with pytest.raises(ValueError, match="too short"):
            parse_chapter(html, "http://example.com", 1)

    def test_fallback_title_when_no_title_tag(self):
        html = '<html><body><div class="chapter-content">' + "A" * 200 + "</div></body></html>"
        result = parse_chapter(html, "http://example.com", 42)
        assert "42" in result.title

    def test_removes_script_tags_from_content(self):
        html = """<html><body>
        <div class="chapter-content">
            Real content here. """ + "A" * 200 + """
            <script>alert('xss')</script>
        </div></body></html>"""
        result = parse_chapter(html, "http://example.com", 1)
        assert "alert" not in result.content
        assert "xss" not in result.content


# ── crawl_chapters ─────────────────────────────────────────────────────────────

class TestCrawlChapters:
    def test_returns_chapter_meta_for_each_num(self, sample_html_chapter):
        """crawl_chapters should return one ChapterMeta per chapter number."""

        mock_response = MagicMock()
        mock_response.text = sample_html_chapter
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        with patch("crawler.scraper.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            results = asyncio.run(crawl_chapters([1, 2, 3]))

        assert len(results) == 3
        assert all(isinstance(r, ChapterMeta) for r in results)

    def test_marks_failed_chapter_as_error(self):
        """Network errors should produce ERROR status, not raise."""
        import httpx as _httpx

        async def mock_get(*args, **kwargs):
            raise _httpx.NetworkError("connection refused")

        with patch("crawler.scraper.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=_httpx.NetworkError("fail"))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            results = asyncio.run(crawl_chapters([99]))

        assert results[0].status == "ERROR"
        assert results[0].chapter_num == 99


# ── storage ────────────────────────────────────────────────────────────────────

class TestSaveChapter:
    def test_saves_file_and_calls_db(self, tmp_path):
        from crawler.storage import save_chapter

        chapter = ChapterMeta(
            chapter_num=1,
            title="Test",
            url="http://example.com",
            content="Content " * 50,
            status="CRAWLED",
        )

        mock_db = MagicMock()

        # Patch data_dir to tmp_path
        with patch("crawler.storage.settings") as mock_settings:
            mock_settings.data_dir = str(tmp_path)
            result = save_chapter(chapter, mock_db)

        assert result is True
        mock_db.upsert_chapter.assert_called_once()
        file_path = tmp_path / "chapters" / "chuong-0001.txt"
        assert file_path.exists()
        assert "Content" in file_path.read_text(encoding="utf-8")

    def test_does_not_save_error_chapter(self, tmp_path):
        from crawler.storage import save_chapter

        chapter = ChapterMeta(
            chapter_num=1,
            title="Bad",
            url="http://example.com",
            status="ERROR",
            error_msg="network fail",
        )
        mock_db = MagicMock()

        with patch("crawler.storage.settings") as mock_settings:
            mock_settings.data_dir = str(tmp_path)
            result = save_chapter(chapter, mock_db)

        assert result is False
        mock_db.set_chapter_status.assert_called_once_with(
            1, "ERROR", error_msg="network fail"
        )

    def test_idempotent_on_rerun(self, tmp_path):
        """Saving the same chapter twice should not raise."""
        from crawler.storage import save_chapter

        chapter = ChapterMeta(
            chapter_num=5,
            title="Dup",
            url="http://example.com",
            content="Body " * 50,
            status="CRAWLED",
        )
        mock_db = MagicMock()

        with patch("crawler.storage.settings") as mock_settings:
            mock_settings.data_dir = str(tmp_path)
            save_chapter(chapter, mock_db)
            result = save_chapter(chapter, mock_db)

        assert result is True
        assert mock_db.upsert_chapter.call_count == 2
