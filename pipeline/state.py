import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

from config.settings import settings

# Avoid DeprecationWarning for datetime in sqlite3 (Python 3.12+)
sqlite3.register_adapter(datetime, lambda d: d.isoformat())
sqlite3.register_converter("TIMESTAMP", lambda b: datetime.fromisoformat(b.decode()))

EPISODE_STATES = [
    "PENDING",
    "CRAWLED",
    "SUMMARIZED",
    "SCRIPTED",
    "IMAGES_DONE",
    "AUDIO_DONE",
    "VIDEO_DONE",
    "VALIDATED",
]

_PHASE_TO_RESET_STATUS = {
    "crawl": "PENDING",
    "llm": "CRAWLED",
    "images": "SCRIPTED",
    "audio": "IMAGES_DONE",
    "video": "AUDIO_DONE",
    "validate": "VIDEO_DONE",
}


class StateDB:
    """SQLite-backed state machine for the pipeline."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_num INTEGER PRIMARY KEY,
                    title       TEXT,
                    url         TEXT,
                    file_path   TEXT,
                    status      TEXT NOT NULL DEFAULT 'PENDING',
                    crawled_at  TIMESTAMP,
                    error_msg   TEXT
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    episode_num   INTEGER PRIMARY KEY,
                    chapter_start INTEGER,
                    chapter_end   INTEGER,
                    status        TEXT NOT NULL DEFAULT 'PENDING',
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg     TEXT
                );

                CREATE TABLE IF NOT EXISTS episode_timings (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_num   INTEGER NOT NULL,
                    phase         TEXT    NOT NULL,
                    started_at    TIMESTAMP,
                    completed_at  TIMESTAMP,
                    duration_sec  REAL,
                    UNIQUE(episode_num, phase)
                );
            """)

    # ── Chapters ──────────────────────────────────────────────────────────────

    def upsert_chapter(
        self,
        chapter_num: int,
        title: str,
        url: str,
        file_path: str,
        status: str,
        crawled_at: Optional[datetime] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chapters (chapter_num, title, url, file_path, status, crawled_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter_num) DO UPDATE SET
                    title=excluded.title,
                    url=excluded.url,
                    file_path=excluded.file_path,
                    status=excluded.status,
                    crawled_at=excluded.crawled_at,
                    error_msg=NULL
                """,
                (chapter_num, title, url, file_path, status, crawled_at),
            )

    def set_chapter_status(
        self, chapter_num: int, status: str, error_msg: Optional[str] = None
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE chapters SET status=?, error_msg=? WHERE chapter_num=?",
                (status, error_msg, chapter_num),
            )

    def get_chapter_status(self, chapter_num: int) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM chapters WHERE chapter_num=?", (chapter_num,)
            ).fetchone()
            return row["status"] if row else None

    def get_crawled_chapters(self, chapter_start: int, chapter_end: int) -> List[int]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT chapter_num FROM chapters
                WHERE chapter_num BETWEEN ? AND ? AND status='CRAWLED'
                """,
                (chapter_start, chapter_end),
            ).fetchall()
            return [r["chapter_num"] for r in rows]

    # ── Episodes ──────────────────────────────────────────────────────────────

    def upsert_episode(
        self, episode_num: int, chapter_start: int, chapter_end: int
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO episodes (episode_num, chapter_start, chapter_end)
                VALUES (?, ?, ?)
                ON CONFLICT(episode_num) DO NOTHING
                """,
                (episode_num, chapter_start, chapter_end),
            )

    def set_episode_status(
        self, episode_num: int, status: str, error_msg: Optional[str] = None
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET status=?, updated_at=CURRENT_TIMESTAMP, error_msg=?
                WHERE episode_num=?
                """,
                (status, error_msg, episode_num),
            )

    def get_episode_status(self, episode_num: int) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM episodes WHERE episode_num=?", (episode_num,)
            ).fetchone()
            return row["status"] if row else None

    def get_episode(self, episode_num: int) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM episodes WHERE episode_num=?", (episode_num,)
            ).fetchone()

    def get_episodes_by_status(self, status: str) -> List[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM episodes WHERE status=? ORDER BY episode_num", (status,)
            ).fetchall()

    def reset_episode_to_phase(self, episode_num: int, phase: str) -> None:
        """Set episode status to the state just before `phase` so it can re-run."""
        target = _PHASE_TO_RESET_STATUS.get(phase, "PENDING")
        self.set_episode_status(episode_num, target, error_msg=None)
        logger.info(
            "Reset episode {} to status={} for re-run from phase={}",
            episode_num, target, phase,
        )

    # ── Timings ───────────────────────────────────────────────────────────────

    def record_phase_start(self, episode_num: int, phase: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO episode_timings (episode_num, phase, started_at)
                VALUES (?, ?, ?)
                ON CONFLICT(episode_num, phase) DO UPDATE SET
                    started_at=excluded.started_at,
                    completed_at=NULL,
                    duration_sec=NULL
                """,
                (episode_num, phase, now),
            )

    def record_phase_done(self, episode_num: int, phase: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE episode_timings
                SET completed_at=?,
                    duration_sec=(julianday(?) - julianday(started_at)) * 86400
                WHERE episode_num=? AND phase=?
                """,
                (now, now, episode_num, phase),
            )

    def get_avg_phase_duration(self, phase: str) -> Optional[float]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT AVG(duration_sec) AS avg
                FROM episode_timings
                WHERE phase=? AND duration_sec IS NOT NULL
                """,
                (phase,),
            ).fetchone()
            return row["avg"] if row and row["avg"] is not None else None

    def estimate_eta(self, remaining_episodes: int) -> Optional[float]:
        """Estimate remaining time (seconds) based on completed episode timings."""
        phases = ["crawl", "llm", "images", "audio", "video"]
        total = 0.0
        for phase in phases:
            avg = self.get_avg_phase_duration(phase)
            if avg is None:
                return None
            total += avg
        return total * remaining_episodes
