from __future__ import annotations

from pydantic import BaseModel, Field


class ShotScript(BaseModel):
    """Schema for a single shot in an episode script.

    scene_prompt is in English (ComfyUI input).
    narration_text is in Vietnamese (TTS input).
    """

    scene_prompt: str = Field(..., description="English prompt for ComfyUI image generation")
    narration_text: str = Field(..., description="Vietnamese narration text for TTS")
    duration_sec: int = Field(default=6, ge=1, le=30)
    is_key_shot: bool = Field(default=False, description="LLM-flagged high-impact shot; triggers SVD when enabled")
    characters: list[str] = Field(default_factory=list, description="Character names present; empty = scene-only shot")


class EpisodeScript(BaseModel):
    """Validated output from scriptwriter.py for one episode."""

    episode_num: int
    arc_title: str
    shots: list[ShotScript]

    @property
    def shot_count(self) -> int:
        return len(self.shots)

    @property
    def key_shots(self) -> list[ShotScript]:
        return [s for s in self.shots if s.is_key_shot]


class Character(BaseModel):
    """Character sheet extracted by character_extractor.py."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    appearance: str = Field(default="", description="Physical description for ComfyUI prompt")
    anchor_image_path: str = Field(default="", description="Path to anchor PNG for IPAdapter")
