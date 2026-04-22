import json
import threading
import time
from typing import Any, Optional

import httpx
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings


class OllamaClient:
    """Thin httpx wrapper for Ollama REST API. No ollama-python SDK."""

    def __init__(self, model: str | None = None, json_format: bool = True) -> None:
        self.base_url = settings.ollama_url
        self.model = model or settings.llm_model
        self.timeout = settings.llm_timeout
        # When False, skip Ollama format:json flag and inject prompt instruction instead.
        # Use for models that ignore constrained decoding (e.g. gpt-oss family).
        self._json_format = json_format

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
        # When constrained JSON decoding is disabled, inject explicit instruction into prompt.
        actual_prompt = prompt
        if not self._json_format:
            actual_prompt = (
                prompt.rstrip()
                + "\n\nIMPORTANT: Respond with ONLY a valid JSON object. "
                "No explanation, no markdown fences, no extra text — pure JSON only."
            )

        raw = self.generate(
            prompt=actual_prompt,
            system=system,
            temperature=temperature,
            response_format="json" if self._json_format else None,
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


ollama_client = OllamaClient(json_format=settings.llm_json_format)
# Phase-specific clients — model falls back to llm_model when the phase fields are empty.
summary_client = OllamaClient(model=settings.effective_summary_model, json_format=settings.summary_json_format)
# scene_prompt_client: used for ComfyUI scene_prompt narration-alignment rewrite pass.
# Falls back to script_model → llm_model when scene_prompt_model is empty.
scene_prompt_client = OllamaClient(
    model=settings.effective_scene_prompt_model,
    json_format=settings.scene_prompt_json_format,
)


def _github_retry_wait(retry_state) -> float:  # type: ignore[type-arg]
    """Custom tenacity wait: respect Retry-After on 429, else exponential backoff."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = int(exc.response.headers.get("retry-after", 65))
        logger.warning(
            "GitHub Models rate limit (429) — waiting {}s before retry", retry_after
        )
        return float(retry_after)
    # Exponential backoff for transient errors (5xx, timeout)
    attempt = retry_state.attempt_number
    return min(2.0 ** attempt * 2, 30.0)


class GitHubLLMClient:
    """Thin httpx wrapper for GitHub Models API (OpenAI-compatible chat completions)."""

    def __init__(self) -> None:
        self.base_url = settings.github_api_url.rstrip("/")
        self.model = settings.github_model
        self.timeout = settings.llm_timeout
        self._token = settings.github_token
        # Proactive rate limiting: min interval between requests
        self._min_interval: float = 60.0 / max(settings.github_rpm, 1)
        self._last_call_at: float = 0.0
        self._rate_lock = threading.Lock()
        if not self._token:
            logger.warning(
                "PIPELINE_GITHUB_TOKEN is not set — GitHub Models API calls will fail."
            )

    def health_check(self) -> bool:
        """Validates token is present; raises RuntimeError if missing."""
        if not self._token:
            raise RuntimeError("GitHub token not configured (PIPELINE_GITHUB_TOKEN).")
        return True

    @retry(
        stop=stop_after_attempt(4),
        wait=_github_retry_wait,
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        response_format: Optional[str] = None,
    ) -> str:
        """Call /chat/completions and return assistant message content."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        # Proactive throttle — sleep before sending if needed
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_call_at
            wait = self._min_interval - elapsed
            if wait > 0:
                logger.debug("GitHub rate throttle: sleeping {:.1f}s", wait)
                time.sleep(wait)
            self._last_call_at = time.monotonic()

        logger.debug(
            "Calling GitHub Models API | model={} prompt_len={}", self.model, len(prompt)
        )
        if not self._token:
            raise RuntimeError(
                "GitHub token not set. Export PIPELINE_GITHUB_TOKEN env var."
            )
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        # Do not retry on 4xx client errors except 429 (rate limit)
        if resp.status_code not in (429,) and 400 <= resp.status_code < 500:
            logger.error(
                "GitHub Models API client error {} — not retrying: {}",
                resp.status_code, resp.text[:300],
            )
            resp.raise_for_status()  # raises immediately, no retry
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Any:
        """Call GitHub Models API and return parsed JSON."""
        raw = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            response_format="json",
        )

        stripped = OllamaClient._strip_markdown_fences(raw)
        if not stripped:
            logger.warning("GitHub Models API returned empty response — will retry")
            raise json.JSONDecodeError("Empty response from GitHub Models API", "", 0)

        return json.loads(stripped)


def get_script_client() -> "OllamaClient | GitHubLLMClient":
    """Factory: return the correct LLM client for the scriptwriting phase."""
    if settings.script_provider == "github":
        return GitHubLLMClient()
    return OllamaClient(
        model=settings.effective_script_model,
        json_format=settings.script_json_format,
    )


# Module-level instance for backward-compat imports (scriptwriter uses this)
script_client = get_script_client()
