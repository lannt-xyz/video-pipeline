import subprocess
import time
from typing import Optional

import httpx
from loguru import logger

from config.settings import settings


class VRAMManager:
    """Manages mutual exclusion between Ollama and ComfyUI GPU usage."""

    def unload_ollama(self) -> None:
        """Stop Ollama model to free VRAM before starting ComfyUI."""
        logger.info("Unloading Ollama to free VRAM...")
        try:
            result = subprocess.run(
                ["ollama", "stop", settings.llm_model],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning("ollama stop stderr: {}", result.stderr.strip())
        except FileNotFoundError:
            logger.warning("ollama CLI not found — skipping unload")
        except subprocess.TimeoutExpired:
            logger.warning("ollama stop timed out")

        # Give GPU time to release VRAM
        time.sleep(3)
        logger.info("Ollama unloaded")

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
