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
    num_gpu: int = 999          # layers to offload to GPU; 999 = force all layers on GPU
    keep_alive: str = "5m"      # how long Ollama keeps model loaded after request (e.g. "5m", "-1" = forever, "0" = unload immediately)
    flash_attention: bool = True  # enable flash attention for lower VRAM + faster inference


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


class DeepSeekLLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    rpm: int = 60                       # generous default; DeepSeek paid tier has high limits
    max_output_tokens: int = 8192
    thinking: bool = False              # enable reasoning/thinking mode (deepseek-v4-pro)
    reasoning_effort: str = "medium"    # "low" | "medium" | "high" — only used when thinking=True


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
    deepseek: DeepSeekLLMConfig = DeepSeekLLMConfig()
    phases: PhasesConfig = PhasesConfig()


class HookJudgeWeights(BaseModel):
    curiosity_gap: float = 0.5
    specificity: float = 0.25
    pattern_interrupt: float = 0.25

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "HookJudgeWeights":
        total = self.curiosity_gap + self.specificity + self.pattern_interrupt
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"hook_judge_weights must sum to 1.0 (got {total:.4f}). "
                "Adjust curiosity_gap, specificity, or pattern_interrupt."
            )
        return self


class StorySettingConfig(BaseModel):
    """Story-level visual hints injected into character profile prompts.

    Lets profile_builder stay story-agnostic: swap story = swap config only.
    """

    # Short description of the story setting/era used in LLM system prompts.
    genre_hint: str = "modern urban ghost-hunter story"
    # Default clothing direction for ordinary/young characters.
    default_clothing_modern: str = "modern casual or tactical wear"
    # Default clothing direction for elder/master/spirit characters.
    default_clothing_traditional: str = "daoist robes or traditional attire"
    # Tags the LLM must NOT emit (abstract / non-visual). Replaced with concrete
    # visual equivalents downstream.
    forbidden_visual_tags: List[str] = [
        "mysterious aura", "mysterious", "scholar", "ethereal",
        "spiritual energy", "exudes", "symbolizing", "enchanting",
        "magical presence", "otherworldly", "ancient wisdom",
        "cunning aura", "dangerous aura", "noble aura", "cold aura",
    ]


class RetentionConfig(BaseModel):
    """Attention-constraint system tunables (Phase 1+).

    All thresholds are placeholders until Phase 1b baseline + Phase 6 calibration.
    `use_constraint_system` is the master feature flag — false = legacy behavior.
    """

    use_constraint_system: bool = False
    max_exposition_ratio: float = 0.5
    max_consecutive_same_energy: int = 2
    lore_curiosity_buffer_shots: int = 2
    hook_min_score: float = 0.65
    reviewer_max_retries: int = 2
    hook_judge_weights: HookJudgeWeights = HookJudgeWeights()

    @field_validator("max_exposition_ratio", "hook_min_score", mode="before")
    @classmethod
    def _clamp_ratio(cls, v: object) -> float:
        v = float(v)  # type: ignore[arg-type]
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value {v} out of range [0.0, 1.0].")
        return v


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
    # DeepSeek API key: set via env var PIPELINE_DEEPSEEK_API_KEY (never hardcode)
    deepseek_api_key: str = ""

    comfyui_timeout: int = 300
    comfyui_poll_interval: int = 2

    # Image generation
    image_width: int = 720
    image_height: int = 1280
    thumbnail_width: int = 1280
    thumbnail_height: int = 720
    redux_strength: float = 0.1
    # Scene reference Redux strength: first shot of scene used as soft background anchor
    # 0.15-0.25 recommended — lighter than char Redux to avoid over-constraining composition
    scene_ref_strength: float = 0.2

    # Video
    fps: int = 30
    target_duration_sec: int = 60
    shot_duration_sec: int = 6

    # Audio
    tts_provider: str = "edge"              # "edge" | "piper" | "f5tts"
    tts_voice: str = "vi-VN-HoaiMyNeural"   # used when tts_provider=edge
    piper_model: str = "vi_VN-vais1000-medium"  # used when tts_provider=piper
    piper_models_dir: str = "models/piper"   # local cache dir for .onnx files
    piper_speed: float = 1.1                 # atempo multiplier for post-processing
    f5tts_model_dir: str = "models/f5tts"    # local cache dir for F5-TTS VI checkpoint
    f5tts_ref_audio: str = "assets/ref_voice.wav"  # 6-30s Vietnamese reference audio
    f5tts_ref_text: str = ""                 # transcript of ref audio; empty = auto-transcribe
    f5tts_speed: float = 1.0
    f5tts_device: str = "cuda"  # "cuda" or "cpu"                 # generation speed multiplier
    bgm_volume_db: int = -15
    tts_lead_in_sec: float = 0.2
    tts_tail_padding_sec: float = 0.5

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

    # Retention / attention-constraint system (Phase 1+)
    retention: RetentionConfig = RetentionConfig()

    # Story-specific visual hints (used by character profile builder)
    story_setting: StorySettingConfig = StorySettingConfig()

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
    def ollama_num_gpu(self) -> int:
        return self.llm.ollama.num_gpu

    @property
    def ollama_keep_alive(self) -> str:
        return self.llm.ollama.keep_alive

    @property
    def ollama_flash_attention(self) -> bool:
        return self.llm.ollama.flash_attention

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
    def deepseek_base_url(self) -> str:
        return self.llm.deepseek.base_url

    @property
    def deepseek_model(self) -> str:
        return self.llm.deepseek.model

    @property
    def deepseek_rpm(self) -> int:
        return self.llm.deepseek.rpm

    @property
    def deepseek_max_output_tokens(self) -> int:
        return self.llm.deepseek.max_output_tokens

    @property
    def deepseek_thinking(self) -> bool:
        return self.llm.deepseek.thinking

    @property
    def deepseek_reasoning_effort(self) -> str:
        return self.llm.deepseek.reasoning_effort

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
