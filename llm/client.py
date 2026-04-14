import json
from typing import Any, Optional

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings


class OllamaClient:
    """Thin httpx wrapper for Ollama REST API. No ollama-python SDK."""

    def __init__(self) -> None:
        self.base_url = settings.ollama_url
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout

    def health_check(self) -> bool:
        """Raises RuntimeError if Ollama is not reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama not available at {self.base_url}: {exc}"
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        response_format: Optional[str] = None,
    ) -> str:
        """Call /api/generate and return raw text response."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": settings.llm_context_size,
            },
        }
        if system:
            payload["system"] = system
        if response_format == "json":
            payload["format"] = "json"

        logger.debug(
            "Calling Ollama | model={} prompt_len={}", self.model, len(prompt)
        )
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Any:
        """Call Ollama and return parsed JSON. Raises json.JSONDecodeError on bad output."""
        raw = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            response_format="json",
        )
        return json.loads(raw)


ollama_client = OllamaClient()
