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
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
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
            dotenv_settings,
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
    # Phase-specific models (fall back to llm_model when empty)
    summary_model: str = ""   # used by summarizer & character_extractor
    script_model: str = ""    # used by scriptwriter (narration_text generation)
    scene_prompt_model: str = ""  # used by scriptwriter (ComfyUI scene_prompt rewrite pass); falls back to script_model
    llm_timeout: int = 120

    # GitHub Models API (for script/scene-prompt generation)
    # Provider: "ollama" | "github" — controls which backend scriptwriter uses
    script_provider: str = "ollama"
    # Provider for ComfyUI image-prompt generation (scene_prompt + anchor tag derivation)
    # "ollama" uses scene_prompt_model; "github" uses github_model (same creds as script_provider)
    image_prompt_provider: str = "ollama"
    github_api_url: str = "https://models.inference.ai.azure.com"
    github_model: str = "gpt-4.1"
    github_rpm: int = 10  # proactive rate limit (requests/min); gpt-5=1, gpt-4.1=10
    # Max input tokens per request (system + user). gpt-4.1 free tier = 8000 total.
    # User content is truncated to (github_max_input_tokens - system_token_budget) before sending.
    github_max_input_tokens: int = 7500
    # GitHub token: set via env var PIPELINE_GITHUB_TOKEN (never hardcode)
    github_token: str = ""
    llm_max_retries: int = 3
    llm_context_size: int = 16384
    # Set false when script_model does not support Ollama constrained JSON decoding (e.g. gpt-oss)
    # Only affects the script client — summary/other clients use native format mode
    script_json_format: bool = True
    summary_json_format: bool = True  # set false for summary_model that ignores format:json
    llm_json_format: bool = True      # set false for llm_model that ignores format:json
    scene_prompt_json_format: bool = True  # set false for scene_prompt_model that ignores format:json
    ollama_json_format: bool = True  # deprecated alias kept for backward-compat; prefer script_json_format
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
    tts_lead_in_sec: float = 0.2
    tts_tail_padding_sec: float = 0.5

    # Multi-frame & motion
    frames_per_shot: int = 4
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
        """Pin data_dir and db_path to story_slug so swap = change config only."""
        slug = self.story_slug.strip()
        if not slug:
            raise ValueError("story_slug must not be empty")
        self.data_dir = str(Path("data") / slug)
        self.db_path = str(Path("data") / f"{slug}.db")
        return self

    @property
    def total_episodes(self) -> int:
        return (self.total_chapters + self.chapters_per_episode - 1) // self.chapters_per_episode

    @property
    def effective_summary_model(self) -> str:
        """Model used for summarize / character extraction phases."""
        return self.summary_model.strip() or self.llm_model

    @property
    def effective_script_model(self) -> str:
        """Model used for scriptwriting / narration generation phases."""
        return self.script_model.strip() or self.llm_model

    @property
    def effective_scene_prompt_model(self) -> str:
        """Model used for ComfyUI scene_prompt generation (narration-alignment rewrite pass)."""
        return self.scene_prompt_model.strip() or self.effective_script_model


settings = Settings()
