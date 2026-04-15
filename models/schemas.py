from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, field_validator, model_validator


class ShotScript(BaseModel):
    """Schema for a single shot in an episode script."""

    scene_prompt: str          # English, for ComfyUI
    narration_text: str        # Vietnamese, for TTS
    duration_sec: int = 6
    is_key_shot: bool = False
    characters: List[str] = []  # [] = scene-only shot, no IPAdapter


class EpisodeScript(BaseModel):
    """Full script for one episode (8-10 shots)."""

    episode_num: int
    title: str
    shots: List[ShotScript]

    @model_validator(mode="after")
    def warn_shot_count(self) -> "EpisodeScript":
        n = len(self.shots)
        if not (8 <= n <= 10):
            # Script writer will normalize, just validate it's reasonable
            if n < 1 or n > 20:
                raise ValueError(f"shots count {n} is out of acceptable range [1, 20]")
        return self


class Character(BaseModel):
    """Character profile for ComfyUI image generation."""

    name: str
    alias: List[str] = []
    gender: str = "male"       # "male" | "female"
    description: str           # English appearance description for ComfyUI
    relationships: dict = {}   # { other_character_name: relationship_description }
    anchor_path: Optional[str] = None


class ChapterMeta(BaseModel):
    """Metadata and content for a single crawled chapter."""

    chapter_num: int
    title: str
    url: str
    content: Optional[str] = None
    status: str = "PENDING"    # PENDING | CRAWLED | ERROR
    crawled_at: Optional[datetime] = None
    error_msg: Optional[str] = None


class ChunkSummary(BaseModel):
    """Pass-1 summary for a group of chapters within one episode."""

    episode_num: int
    chunk_index: int
    chapter_start: int
    chapter_end: int
    summary: str


class ArcOverview(BaseModel):
    """Pass-2 arc synthesis for a full episode."""

    episode_num: int
    arc_summary: str
    key_events: List[str]
    characters_in_episode: List[str]
