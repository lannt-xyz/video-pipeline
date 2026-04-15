import time
from typing import Optional

import httpx
from loguru import logger

from config.settings import settings


class VRAMManager:
    """Manages mutual exclusion between Ollama and ComfyUI GPU usage."""

    def unload_ollama(self) -> None:
        """Force Ollama to release VRAM immediately via keep_alive=0 API call."""
        logger.info("Unloading Ollama to free VRAM...")
        try:
            # keep_alive=0 forces llama.cpp to evict the model from VRAM immediately
            resp = httpx.post(
                f"{settings.ollama_url}/api/generate",
                json={"model": settings.llm_model, "keep_alive": 0},
                timeout=15.0,
            )
            logger.debug("Ollama keep_alive=0 response | status={}", resp.status_code)
        except httpx.HTTPError as exc:
            logger.warning("Ollama unload API failed | error={}", exc)

        # Give GPU time to release VRAM
        time.sleep(2)
        logger.info("Ollama VRAM released")

    def unload_comfyui(self) -> None:
        """Ask ComfyUI to free all cached models from VRAM."""
        logger.info("Unloading ComfyUI models to free VRAM...")
        try:
            resp = httpx.post(
                f"{settings.comfyui_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=15.0,
            )
            logger.debug("ComfyUI /free response | status={}", resp.status_code)
        except httpx.HTTPError as exc:
            logger.warning("ComfyUI unload API failed | error={}", exc)

        time.sleep(2)
        logger.info("ComfyUI VRAM released")

    def load_ollama(self) -> None:
        """Start Ollama serve and wait until healthy."""
        logger.info("Starting Ollama service...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise RuntimeError("ollama CLI not found — cannot start Ollama")

        time.sleep(3)
        self.health_check_ollama(timeout=30)
        logger.info("Ollama started and healthy")

    def health_check_ollama(self, timeout: int = 30) -> bool:
        """Poll Ollama /api/tags until healthy. Raises RuntimeError on timeout."""
        url = f"{settings.ollama_url}/api/tags"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    logger.debug("Ollama health OK")
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise RuntimeError(
            f"Ollama not responding at {url} after {timeout}s"
        )

    def health_check_comfyui(self, timeout: int = 30) -> bool:
        """Poll ComfyUI /system_stats until healthy. Raises RuntimeError on timeout."""
        url = f"{settings.comfyui_url}/system_stats"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    logger.debug("ComfyUI health OK")
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise RuntimeError(
            f"ComfyUI not responding at {url} after {timeout}s"
        )


vram_manager = VRAMManager()
