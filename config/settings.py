from pathlib import Path
from typing import Any, List, Tuple, Type

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource

_PROJECT_ROOT = Path(__file__).parent.parent
_YAML_FILE = str(_PROJECT_ROOT / "config" / "settings.yaml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=_YAML_FILE,
        yaml_file_encoding="utf-8",
        env_prefix="PIPELINE_",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    # Story key
    story_slug: str = "mao-son-troc-quy-nhan"

    # Episode planning
    total_chapters: int = 3534
    chapters_per_episode: int = 35

    # API endpoints
    ollama_url: str = "http://localhost:11434"
    comfyui_url: str = "http://localhost:8188"

    # LLM configuration
    llm_model: str = "qwen2.5:7b-instruct-q8_0"
    llm_timeout: int = 120
    llm_max_retries: int = 3
    llm_context_size: int = 16384

    # ComfyUI configuration
    comfyui_timeout: int = 300
    comfyui_poll_interval: int = 2

    # Image generation
    image_width: int = 720
    image_height: int = 1280
    thumbnail_width: int = 1280
    thumbnail_height: int = 720

    # Video
    fps: int = 30
    target_duration_sec: int = 60
    shot_duration_sec: int = 6

    # Audio
    tts_voice: str = "vi-VN-HoaiMyNeural"
    bgm_volume_db: int = -15

    # Multi-frame & motion
    frames_per_shot: int = 2
    crossfade_duration: float = 0.5
    shot_transition_duration: float = 0.3
    shot_transition_type: str = "dissolve"

    # Feature flags
    enable_svd: bool = False

    # Paths
    data_dir: str = "data/{story_slug}"
    db_path: str = "data/{story_slug}/{story_slug}.db"
    logs_dir: str = "logs"
    assets_dir: str = "assets"

    # LoRA weights
    lora_weights: dict = {}

    @field_validator("total_chapters")
    @classmethod
    def total_chapters_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("total_chapters must be positive")
        return v

    @field_validator("chapters_per_episode")
    @classmethod
    def chapters_per_episode_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chapters_per_episode must be positive")
        return v

    @model_validator(mode="after")
    def resolve_story_paths(self) -> "Settings":
        """Ensure data/db paths are always scoped by story_slug."""
        slug = self.story_slug.strip()
        if not slug:
            raise ValueError("story_slug must not be empty")

        data_tpl = self.data_dir.replace("{story_slug}", slug)
        data_path = Path(data_tpl)
        if data_path.name != slug:
            data_path = data_path / slug
        self.data_dir = str(data_path)

        db_tpl = self.db_path.replace("{story_slug}", slug)
        db_candidate = Path(db_tpl)
        db_name = f"{slug}.db"
        if db_candidate.name != db_name:
            self.db_path = str(data_path / db_name)
        else:
            self.db_path = str(data_path / db_candidate.name)

        return self

    @property
    def total_episodes(self) -> int:
        return (self.total_chapters + self.chapters_per_episode - 1) // self.chapters_per_episode


settings = Settings()
