import subprocess
import time
from contextlib import contextmanager
from enum import Enum
from typing import Generator

import httpx
from loguru import logger

from config.settings import settings

# Minimum free VRAM (GiB) required before a new consumer is allowed to load.
# qwen2.5:14b Q4_K_M ≈ 8 GB — leave 1 GB headroom.
_VRAM_FREE_THRESHOLD_GIB = 9.0
# How long to wait for VRAM to drain after a release call.
_VRAM_WAIT_TIMEOUT_SEC = 30


class VRAMConsumer(str, Enum):
    """Exclusive VRAM consumers — only ONE may hold VRAM at any point in time.

    Add new consumers here when a new GPU-heavy service is integrated
    (e.g. CogVideoX, SVD standalone, etc.).
    """
    NONE = "none"
    OLLAMA = "ollama"
    COMFYUI = "comfyui"


class VRAMManager:
    """Enforces mutual exclusion across all GPU consumers.

    At most ONE consumer holds VRAM at any point in time.
    Calling acquire() automatically releases the previous holder first.

    Usage patterns:
        # Explicit acquire — consumer keeps VRAM until next acquire() or release_all()
        vram_manager.acquire(VRAMConsumer.OLLAMA)
        vram_manager.health_check_ollama()
        run_llm_phase()

        # Context manager — releases automatically on exit (use for isolated blocks)
        with vram_manager.using(VRAMConsumer.COMFYUI):
            vram_manager.health_check_comfyui()
            generate_images()

        # Flush without changing owner (e.g. clear ComfyUI cache mid-phase and retry)
        vram_manager.flush()

        # Error cleanup — always release no matter who holds VRAM
        vram_manager.release_all()
    """

    def __init__(self) -> None:
        self._current: VRAMConsumer = VRAMConsumer.NONE

    # ---------------------------------------------------------------------------
    # Public acquisition API
    # ---------------------------------------------------------------------------

    def acquire(self, consumer: VRAMConsumer) -> None:
        """Give exclusive VRAM to *consumer*.

        If a different consumer currently holds VRAM it is released first.
        No-op when the requested consumer already owns VRAM.
        """
        if self._current == consumer:
            logger.debug("VRAM already held by '{}' — no-op", consumer.value)
            return

        if self._current != VRAMConsumer.NONE:
            logger.info(
                "VRAM handoff: '{}' → '{}'", self._current.value, consumer.value
            )
            self._release(self._current)

        self._current = consumer
        logger.info("VRAM acquired | owner='{}'", consumer.value)

    def release_all(self) -> None:
        """Release VRAM from whoever currently holds it.

        Safe to call even when VRAM is already free.
        Use in error-cleanup blocks so the next run is never blocked.
        """
        if self._current == VRAMConsumer.NONE:
            return
        logger.info("Releasing all VRAM | current_owner='{}'", self._current.value)
        self._release(self._current)
        self._current = VRAMConsumer.NONE

    def flush(self) -> None:
        """Flush the current consumer's model cache without changing ownership.

        Use when a mid-phase cache clear is needed (e.g. ComfyUI IPAdapter cache
        flush before a retry) while the consumer should remain the VRAM owner.
        No-op when no consumer holds VRAM.
        """
        if self._current == VRAMConsumer.NONE:
            return
        logger.info("Flushing VRAM cache | owner='{}'", self._current.value)
        self._release(self._current)
        # Re-assert ownership — flush does not relinquish VRAM slot.
        logger.info("VRAM flush done, ownership retained | owner='{}'", self._current.value)

    @contextmanager
    def using(self, consumer: VRAMConsumer) -> Generator[None, None, None]:
        """Context manager wrapper around acquire / release_all.

        Prefer this for self-contained blocks that should not leave VRAM
        allocated after the block ends.
        """
        self.acquire(consumer)
        try:
            yield
        finally:
            self.release_all()

    # ---------------------------------------------------------------------------
    # Intra-phase helpers (do NOT change VRAM ownership)
    # ---------------------------------------------------------------------------

    def unload_model(self, model_name: str) -> None:
        """Evict a specific Ollama model from VRAM (keep_alive=0).

        Use between LLM sub-phases when switching models (e.g. summary→script)
        so the previous model doesn't OOM the next one.
        No-op when model_name is empty.
        """
        if not model_name:
            return
        logger.info("Evicting Ollama model from VRAM | model={}", model_name)
        try:
            httpx.post(
                f"{settings.ollama_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            logger.warning("Ollama model evict failed | model={} error={}", model_name, exc)
        time.sleep(1)

    # ---------------------------------------------------------------------------
    # Health checks — probe service readiness, do NOT affect VRAM ownership
    # ---------------------------------------------------------------------------

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
        raise RuntimeError(f"Ollama not responding at {url} after {timeout}s")

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
        raise RuntimeError(f"ComfyUI not responding at {url} after {timeout}s")

    # ---------------------------------------------------------------------------
    # Internal release dispatch
    # ---------------------------------------------------------------------------

    def _release(self, consumer: VRAMConsumer) -> None:
        """Dispatch to the appropriate hardware-level unload routine."""
        _handlers = {
            VRAMConsumer.OLLAMA: self._release_ollama,
            VRAMConsumer.COMFYUI: self._release_comfyui,
        }
        handler = _handlers.get(consumer)
        if handler:
            handler()
        else:
            logger.debug("No release handler registered for consumer='{}'", consumer.value)
        self._wait_vram_free()

    def _wait_vram_free(
        self,
        threshold_gib: float = _VRAM_FREE_THRESHOLD_GIB,
        timeout_sec: int = _VRAM_WAIT_TIMEOUT_SEC,
    ) -> None:
        """Poll nvidia-smi until free VRAM >= threshold_gib or timeout.

        Logs a warning when the threshold is never reached; does NOT raise so
        the pipeline can attempt to continue rather than hard-failing.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            free_gib = self._free_vram_gib()
            if free_gib is None:
                # nvidia-smi unavailable — cannot assert, skip
                logger.debug("nvidia-smi unavailable, skipping VRAM assertion")
                return
            if free_gib >= threshold_gib:
                logger.info(
                    "VRAM clear | free={:.1f} GiB threshold={:.1f} GiB",
                    free_gib, threshold_gib,
                )
                return
            logger.debug(
                "Waiting for VRAM to drain | free={:.1f} GiB threshold={:.1f} GiB",
                free_gib, threshold_gib,
            )
            time.sleep(3)
        free_gib = self._free_vram_gib() or 0.0
        logger.warning(
            "VRAM did not reach threshold after {}s — proceeding anyway "
            "| free={:.1f} GiB threshold={:.1f} GiB. "
            "Model may run in CPU/GPU split mode.",
            timeout_sec, free_gib, threshold_gib,
        )

    @staticmethod
    def _free_vram_gib() -> float | None:
        """Return free VRAM in GiB via nvidia-smi. Returns None when unavailable."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            free_mib = float(out.decode().strip().splitlines()[0])
            return free_mib / 1024.0
        except Exception:
            return None

    def _release_ollama(self) -> None:
        """Force Ollama to release VRAM: unload every currently-loaded model."""
        logger.info("Unloading all Ollama models from VRAM...")

        running_models: list[str] = []
        try:
            resp = httpx.get(f"{settings.ollama_url}/api/ps", timeout=10.0)
            resp.raise_for_status()
            running_models = [
                entry.get("name") or entry.get("model", "")
                for entry in resp.json().get("models", [])
            ]
            running_models = [m for m in running_models if m]
        except httpx.HTTPError as exc:
            logger.warning(
                "Could not query /api/ps — falling back to configured model names | error={}", exc
            )

        if not running_models:
            candidates = {
                settings.llm_model,
                settings.effective_summary_model,
                settings.effective_script_model,
            }
            running_models = [m for m in candidates if m]

        unloaded = 0
        for model_name in running_models:
            try:
                httpx.post(
                    f"{settings.ollama_url}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=15.0,
                )
                unloaded += 1
                logger.debug("Unloaded Ollama model | model={}", model_name)
            except httpx.HTTPError as exc:
                logger.warning(
                    "Failed to unload Ollama model | model={} error={}", model_name, exc
                )

        time.sleep(2)
        logger.info("Ollama VRAM released | models_unloaded={}", unloaded)

    def _release_comfyui(self) -> None:
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


vram_manager = VRAMManager()
