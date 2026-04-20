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

    def __init__(self, model: str | None = None) -> None:
        self.base_url = settings.ollama_url
        self.model = model or settings.llm_model
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

        stripped = self._strip_markdown_fences(raw)
        if not stripped:
            logger.warning("Ollama returned empty response — will retry")
            raise json.JSONDecodeError("Empty response from Ollama", "", 0)

        try:
            return self._parse_json_with_fallbacks(stripped)
        except json.JSONDecodeError as decode_err:
            logger.warning("Ollama returned non-JSON (first 200 chars): {!r}", stripped[:200])

            repaired = self._repair_json_text(stripped)
            if repaired:
                try:
                    return self._parse_json_with_fallbacks(repaired)
                except json.JSONDecodeError:
                    logger.warning(
                        "JSON repair attempt failed (first 200 chars): {!r}", repaired[:200]
                    )

            raise decode_err

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        inner = lines[1:] if lines and lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        start = -1
        depth = 0
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if start < 0:
                if ch in "[{":
                    start = idx
                    depth = 1
                continue

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch in "[{":
                depth += 1
                continue

            if ch in "]}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1].strip()

        return None

    @staticmethod
    def _try_json_loads(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # strict=False tolerates unescaped control chars in strings.
            return json.loads(text, strict=False)

    def _parse_json_with_fallbacks(self, text: str) -> Any:
        candidates: list[str] = [text.strip()]

        extracted = self._extract_first_json_object(text)
        if extracted and extracted not in candidates:
            candidates.append(extracted)

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            sliced = text[first_brace : last_brace + 1].strip()
            if sliced and sliced not in candidates:
                candidates.append(sliced)

        last_error: Optional[json.JSONDecodeError] = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return self._try_json_loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("No JSON candidate produced", text, 0)

    def _repair_json_text(self, malformed: str) -> Optional[str]:
        compact = malformed.strip()
        if not compact:
            return None

        repair_prompt = (
            "Rewrite the malformed JSON below into valid strict JSON. "
            "Do not add commentary. Return only one JSON object with the same keys and semantics.\n\n"
            f"{compact[:12000]}"
        )
        try:
            return self.generate(
                prompt=repair_prompt,
                system=(
                    "You repair malformed JSON. Output exactly one valid JSON object. "
                    "No markdown fences, no extra text."
                ),
                temperature=0.0,
                response_format="json",
            )
        except (httpx.HTTPError, httpx.TimeoutException):
            return None


ollama_client = OllamaClient()
# Phase-specific clients — model falls back to llm_model when the phase fields are empty.
summary_client = OllamaClient(model=settings.effective_summary_model)
script_client = OllamaClient(model=settings.effective_script_model)
