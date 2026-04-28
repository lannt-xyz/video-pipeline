from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource

_PROJECT_ROOT = Path(__file__).parent.parent
_YAML_FILE = str(_PROJECT_ROOT / "config" / "settings.yaml")


# ---------------------------------------------------------------------------
# Nested LLM config models
# ---------------------------------------------------------------------------

class OllamaLLMConfig(BaseModel):
    url: str = "http://localhost:11434"
    default_model: str = "mistral-small:22b"  # fallback for all phases using provider=ollama


class GitHubLLMConfig(BaseModel):
    api_url: str = "https://models.inference.ai.azure.com"
    model: str = "gpt-4.1"  # shared across all phases; per-phase override not needed
    rpm: int = 10
    max_input_tokens: int = 7000
    max_output_tokens: int = 4096  # explicit output token cap to prevent silent truncation

class GeminiLLMConfig(BaseModel):
    model: str = "gemini-2.0-flash"
    rpm: int = 15            # free tier: 15 RPM
    max_output_tokens: int = 8192

class PhaseConfig(BaseModel):
    provider: str = "ollama"  # "ollama" | "github"
    model: str = ""           # empty = use ollama.default_model (ignored when provider=github)
    json_format: bool = True


class PhasesConfig(BaseModel):
    summary: PhaseConfig = PhaseConfig(provider="ollama", model="", json_format=False)
    script: PhaseConfig = PhaseConfig(provider="ollama", model="", json_format=True)
    image_prompt: PhaseConfig = PhaseConfig(provider="ollama", model="", json_format=True)


class LLMConfig(BaseModel):
    timeout: int = 300
    max_retries: int = 3
    context_size: int = 32768
    ollama: OllamaLLMConfig = OllamaLLMConfig()
    github: GitHubLLMConfig = GitHubLLMConfig()
    gemini: GeminiLLMConfig = GeminiLLMConfig()
    phases: PhasesConfig = PhasesConfig()


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
    comfyui_url: str = "http://localhost:8188"

    # LLM — unified nested config
    llm: LLMConfig = LLMConfig()
    # GitHub token: set via env var PIPELINE_GITHUB_TOKEN (never hardcode)
    github_token: str = ""
    # Gemini API key: set via env var PIPELINE_GEMINI_API_KEY (never hardcode)
    gemini_api_key: str = ""

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

    # ------------------------------------------------------------------
    # Backward-compat shims — client.py and other modules use these
    # ------------------------------------------------------------------

    @property
    def llm_model(self) -> str:
        return self.llm.ollama.default_model

    @property
    def ollama_url(self) -> str:
        return self.llm.ollama.url

    @property
    def llm_timeout(self) -> int:
        return self.llm.timeout

    @property
    def llm_max_retries(self) -> int:
        return self.llm.max_retries

    @property
    def llm_context_size(self) -> int:
        return self.llm.context_size

    @property
    def github_api_url(self) -> str:
        return self.llm.github.api_url

    @property
    def github_model(self) -> str:
        return self.llm.github.model

    @property
    def github_rpm(self) -> int:
        return self.llm.github.rpm

    @property
    def github_max_input_tokens(self) -> int:
        return self.llm.github.max_input_tokens

    @property
    def github_max_output_tokens(self) -> int:
        return self.llm.github.max_output_tokens

    @property
    def gemini_model(self) -> str:
        return self.llm.gemini.model

    @property
    def gemini_rpm(self) -> int:
        return self.llm.gemini.rpm

    @property
    def gemini_max_output_tokens(self) -> int:
        return self.llm.gemini.max_output_tokens

    @property
    def summary_provider(self) -> str:
        return self.llm.phases.summary.provider

    @property
    def script_provider(self) -> str:
        return self.llm.phases.script.provider

    @property
    def image_prompt_provider(self) -> str:
        return self.llm.phases.image_prompt.provider

    @property
    def script_json_format(self) -> bool:
        return self.llm.phases.script.json_format

    @property
    def summary_json_format(self) -> bool:
        return self.llm.phases.summary.json_format

    @property
    def scene_prompt_json_format(self) -> bool:
        return self.llm.phases.image_prompt.json_format

    def _resolve_model(self, phase: "PhaseConfig") -> str:
        """Return phase model, falling back to ollama.default_model when empty."""
        return phase.model.strip() or self.llm.ollama.default_model

    @property
    def effective_summary_model(self) -> str:
        """Model used for summarize / character extraction phases."""
        return self._resolve_model(self.llm.phases.summary)

    @property
    def effective_script_model(self) -> str:
        """Model used for scriptwriting / narration generation phases."""
        return self._resolve_model(self.llm.phases.script)

    @property
    def effective_scene_prompt_model(self) -> str:
        """Model used for ComfyUI scene_prompt generation (narration-alignment rewrite pass)."""
        return self._resolve_model(self.llm.phases.image_prompt)


settings = Settings()
