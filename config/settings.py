from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config/settings.yaml",
        yaml_file_encoding="utf-8",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: init kwargs > env vars > YAML file
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    # Story source — swap story here, never in code
    story_slug: str
    base_url: str
    total_chapters: int
    chapters_per_episode: int = 35

    # External services
    ollama_url: str = "http://localhost:11434"
    comfyui_url: str = "http://localhost:8188"
    llm_model: str = "qwen2.5:7b-instruct:q8_0"

    # Feature flags
    enable_svd: bool = False

    # TTS — "edge" uses Azure cloud (internet required); "piper" is fully local CPU
    tts_backend: Literal["edge", "piper"] = "edge"
    edge_voice: str = "vi-VN-HoaiMyNeural"
    piper_model_path: str = "models/piper/vi_VN-vivos-medium.onnx"
    piper_config_path: str = "models/piper/vi_VN-vivos-medium.onnx.json"

    # Audio output
    audio_format: str = "mp3"
    audio_bitrate: str = "128k"

    # BGM
    bgm_volume_db: int = -15
    assets_music_dir: str = "assets/music"

    # Paths
    data_dir: str = "data"
    db_path: str = "db/pipeline.db"
    logs_dir: str = "logs"

    @field_validator("tts_backend")
    @classmethod
    def validate_tts_backend(cls, v: str) -> str:
        if v not in ("edge", "piper"):
            raise ValueError(f"tts_backend must be 'edge' or 'piper', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_piper_paths_exist_when_selected(self) -> "Settings":
        """Fail fast at startup if piper is selected but model files are missing."""
        if self.tts_backend == "piper":
            model = Path(self.piper_model_path)
            config = Path(self.piper_config_path)
            if not model.exists():
                raise ValueError(
                    f"piper_model_path '{model}' not found. "
                    "Download the .onnx voice model from: "
                    "https://huggingface.co/rhasspy/piper-voices"
                )
            if not config.exists():
                raise ValueError(
                    f"piper_config_path '{config}' not found. "
                    "Download the .onnx.json config alongside the .onnx model."
                )
        return self


settings = Settings()  # type: ignore[call-arg]  # loaded from YAML
