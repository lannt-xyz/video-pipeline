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
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": settings.llm_context_size,
                "num_gpu": settings.ollama_num_gpu,
                "flash_attention": settings.ollama_flash_attention,
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





_GITHUB_429_MAX_WAIT = 120  # seconds — if retry-after exceeds this, fail fast instead of sleeping


def _github_retry_wait(retry_state) -> float:  # type: ignore[type-arg]
    """Custom tenacity wait: respect Retry-After on 429, else exponential backoff.

    If retry-after > _GITHUB_429_MAX_WAIT the quota is exhausted for this session
    (GitHub returns values like 76000s when daily token limit is hit).
    Raise immediately so the pipeline fails fast instead of sleeping for hours.
    """
    exc = retry_state.outcome.exception()
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = int(exc.response.headers.get("retry-after", 65))
        if retry_after > _GITHUB_429_MAX_WAIT:
            raise RuntimeError(
                f"GitHub Models quota exhausted — retry-after={retry_after}s "
                f"(>{_GITHUB_429_MAX_WAIT}s threshold). "
                "Switch provider or wait until quota resets."
            )
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
        # Truncate user prompt to stay within github_max_input_tokens.
        # Reserve ~500 tokens for system prompt overhead; Vietnamese ~2 chars/token.
        system_token_budget = len(system) // 2 if system else 0
        max_user_tokens = max(settings.github_max_input_tokens - system_token_budget - 200, 1000)
        max_user_chars = max_user_tokens * 2  # Vietnamese: ~2 chars per token
        if len(prompt) > max_user_chars:
            logger.warning(
                "GitHub prompt truncated | original={} chars → {} chars (max_user_tokens={})",
                len(prompt), max_user_chars, max_user_tokens,
            )
            prompt = prompt[:max_user_chars]

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": settings.github_max_output_tokens,
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
        data = resp.json()
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        content = choice["message"]["content"]
        if finish_reason == "length":
            logger.warning(
                "GitHub Models API output truncated (finish_reason=length) | "
                "model={} output_len={} max_tokens={}",
                self.model, len(content), settings.github_max_output_tokens,
            )
            raise httpx.HTTPStatusError(
                "Output truncated (finish_reason=length) — increase github.max_output_tokens",
                request=resp.request,
                response=resp,
            )
        return content

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Any:
        """Call GitHub Models API and return parsed JSON. Tries to repair malformed JSON if needed."""
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

        try:
            return json.loads(stripped)
        except json.JSONDecodeError as decode_err:
            logger.warning("GitHub Models API returned non-JSON (first 200 chars): {!r}", stripped[:200])
            # Try to repair using OllamaClient logic for consistency
            repaired = OllamaClient._repair_json_text(self, stripped)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    logger.warning("JSON repair attempt failed (first 200 chars): {!r}", repaired[:200])
            raise decode_err


def _gemini_retry_wait(retry_state) -> float:  # type: ignore[type-arg]
    """Custom tenacity wait for Gemini: parse retryDelay from body → Retry-After header → backoff."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        # 1. Gemini often embeds retryDelay in the JSON body error details
        wait: float = 0.0
        try:
            body = exc.response.json()
            for detail in body.get("error", {}).get("details", []):
                delay_str: str = detail.get("retryDelay", "")
                if delay_str.endswith("s"):
                    wait = float(delay_str[:-1])
                    break
        except Exception:
            pass
        # 2. Fall back to Retry-After header
        if not wait:
            wait = float(exc.response.headers.get("retry-after", 0))
        # 3. Default: one full minute window + small buffer
        if not wait:
            wait = 62.0
        logger.warning("Gemini rate limit (429) — waiting {:.0f}s before retry", wait)
        return wait
    attempt = retry_state.attempt_number
    return min(2.0 ** attempt * 2, 30.0)


class GeminiLLMClient:
    """Thin httpx wrapper for Google Gemini REST API (v1beta generateContent)."""

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self) -> None:
        self.model = settings.gemini_model
        self.timeout = settings.llm_timeout
        self._api_key = settings.gemini_api_key
        self._min_interval: float = 60.0 / max(settings.gemini_rpm, 1)
        self._last_call_at: float = 0.0
        self._rate_lock = threading.Lock()
        if not self._api_key:
            logger.warning("PIPELINE_GEMINI_API_KEY is not set — Gemini API calls will fail.")

    def health_check(self) -> bool:
        """Validates API key is present; raises RuntimeError if missing."""
        if not self._api_key:
            raise RuntimeError("Gemini API key not configured (PIPELINE_GEMINI_API_KEY).")
        return True

    @retry(
        stop=stop_after_attempt(4),
        wait=_gemini_retry_wait,
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
        """Call Gemini generateContent and return text response."""
        if not self._api_key:
            raise RuntimeError("Gemini API key not set. Export PIPELINE_GEMINI_API_KEY env var.")

        contents: list[dict] = [{"role": "user", "parts": [{"text": prompt}]}]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": settings.gemini_max_output_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if response_format == "json":
            payload["generationConfig"]["responseMimeType"] = "application/json"

        # Proactive rate throttle
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_call_at
            wait = self._min_interval - elapsed
            if wait > 0:
                logger.debug("Gemini rate throttle: sleeping {:.1f}s", wait)
                time.sleep(wait)
            self._last_call_at = time.monotonic()

        logger.debug("Calling Gemini API | model={} prompt_len={}", self.model, len(prompt))
        resp = httpx.post(
            f"{self._BASE_URL}/models/{self.model}:generateContent?key={self._api_key}",
            json=payload,
            timeout=self.timeout,
        )
        # Do not retry on 4xx except 429
        if resp.status_code not in (429,) and 400 <= resp.status_code < 500:
            logger.error(
                "Gemini API client error {} — not retrying: {}", resp.status_code, resp.text[:300]
            )
            resp.raise_for_status()
        resp.raise_for_status()

        data = resp.json()
        # Handle safety blocks
        if not data.get("candidates"):
            finish = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
            raise RuntimeError(f"Gemini response blocked: {finish}")

        candidate = data["candidates"][0]
        finish_reason = candidate.get("finishReason", "STOP")
        if finish_reason == "MAX_TOKENS":
            content_so_far = "".join(
                p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
            )
            logger.warning(
                "Gemini output truncated (MAX_TOKENS) | model={} output_len={} max_tokens={}",
                self.model, len(content_so_far), settings.gemini_max_output_tokens,
            )
            raise httpx.HTTPStatusError(
                "Output truncated (MAX_TOKENS) — increase gemini.max_output_tokens",
                request=resp.request,
                response=resp,
            )
        return "".join(p.get("text", "") for p in candidate["content"]["parts"])

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Any:
        """Call Gemini API and return parsed JSON."""
        raw = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            response_format="json",
        )
        stripped = OllamaClient._strip_markdown_fences(raw)
        if not stripped:
            logger.warning("Gemini returned empty response — will retry")
            raise json.JSONDecodeError("Empty response from Gemini", "", 0)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as decode_err:
            logger.warning("Gemini returned non-JSON (first 200 chars): {!r}", stripped[:200])
            repaired = OllamaClient._repair_json_text(self, stripped)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    logger.warning("JSON repair attempt failed (first 200 chars): {!r}", repaired[:200])
            raise decode_err


def _deepseek_retry_wait(retry_state) -> float:  # type: ignore[type-arg]
    """Custom tenacity wait for DeepSeek: respect Retry-After on 429, else exponential backoff."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = float(exc.response.headers.get("retry-after", 30))
        logger.warning("DeepSeek rate limit (429) — waiting {:.0f}s before retry", retry_after)
        return retry_after
    attempt = retry_state.attempt_number
    return min(2.0 ** attempt * 2, 30.0)


class DeepSeekLLMClient:
    """Thin httpx wrapper for DeepSeek API (OpenAI-compatible chat completions).

    Supports thinking mode via `extra_body={"thinking": {"type": "enabled"}}` and
    `reasoning_effort` parameter — controlled by settings.llm.deepseek.thinking.
    """

    def __init__(self) -> None:
        self.base_url = settings.deepseek_base_url.rstrip("/")
        self.model = settings.deepseek_model
        self.timeout = settings.llm_timeout
        self._api_key = settings.deepseek_api_key
        self._thinking = settings.deepseek_thinking
        self._reasoning_effort = settings.deepseek_reasoning_effort
        self._min_interval: float = 60.0 / max(settings.deepseek_rpm, 1)
        self._last_call_at: float = 0.0
        self._rate_lock = threading.Lock()
        if not self._api_key:
            logger.warning(
                "PIPELINE_DEEPSEEK_API_KEY is not set — DeepSeek API calls will fail."
            )

    def health_check(self) -> bool:
        if not self._api_key:
            raise RuntimeError("DeepSeek API key not configured (PIPELINE_DEEPSEEK_API_KEY).")
        return True

    @retry(
        stop=stop_after_attempt(4),
        wait=_deepseek_retry_wait,
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
        if not self._api_key:
            raise RuntimeError("DeepSeek API key not set. Export PIPELINE_DEEPSEEK_API_KEY env var.")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": settings.deepseek_max_output_tokens,
            "stream": False,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        if self._thinking:
            # OpenAI-SDK extra_body fields are passed at top level for raw HTTP.
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = self._reasoning_effort

        # Proactive throttle
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_call_at
            wait = self._min_interval - elapsed
            if wait > 0:
                logger.debug("DeepSeek rate throttle: sleeping {:.1f}s", wait)
                time.sleep(wait)
            self._last_call_at = time.monotonic()

        logger.debug(
            "Calling DeepSeek API | model={} thinking={} prompt_len={}",
            self.model, self._thinking, len(prompt),
        )
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        # Do not retry on 4xx except 429
        if resp.status_code not in (429,) and 400 <= resp.status_code < 500:
            logger.error(
                "DeepSeek API client error {} — not retrying: {}",
                resp.status_code, resp.text[:300],
            )
            resp.raise_for_status()
        resp.raise_for_status()

        data = resp.json()
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        content = choice["message"]["content"] or ""
        if finish_reason == "length":
            logger.warning(
                "DeepSeek output truncated (finish_reason=length) | "
                "model={} output_len={} max_tokens={}",
                self.model, len(content), settings.deepseek_max_output_tokens,
            )
            raise httpx.HTTPStatusError(
                "Output truncated (finish_reason=length) — increase deepseek.max_output_tokens",
                request=resp.request,
                response=resp,
            )
        return content

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Any:
        """Call DeepSeek API and return parsed JSON. Tries to repair malformed JSON if needed."""
        raw = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            response_format="json",
        )
        stripped = OllamaClient._strip_markdown_fences(raw)
        if not stripped:
            logger.warning("DeepSeek returned empty response — will retry")
            raise json.JSONDecodeError("Empty response from DeepSeek", "", 0)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as decode_err:
            logger.warning("DeepSeek returned non-JSON (first 200 chars): {!r}", stripped[:200])
            repaired = OllamaClient._repair_json_text(self, stripped)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    logger.warning("JSON repair attempt failed (first 200 chars): {!r}", repaired[:200])
            raise decode_err


_LLMClient = "OllamaClient | GitHubLLMClient | GeminiLLMClient | DeepSeekLLMClient"


def _build_provider_client(provider: str, *, ollama_model: str, ollama_json: bool):
    """Resolve provider name → client instance. Falls back to Ollama for unknown providers."""
    if provider == "github":
        return GitHubLLMClient()
    if provider == "gemini":
        return GeminiLLMClient()
    if provider == "deepseek":
        return DeepSeekLLMClient()
    return OllamaClient(model=ollama_model, json_format=ollama_json)


def get_summary_client() -> "OllamaClient | GitHubLLMClient | GeminiLLMClient | DeepSeekLLMClient":
    """Factory: return the correct LLM client for the summary/character-extraction phase."""
    return _build_provider_client(
        settings.summary_provider,
        ollama_model=settings.effective_summary_model,
        ollama_json=settings.summary_json_format,
    )


def get_script_client() -> "OllamaClient | GitHubLLMClient | GeminiLLMClient | DeepSeekLLMClient":
    """Factory: return the correct LLM client for the scriptwriting phase."""
    return _build_provider_client(
        settings.script_provider,
        ollama_model=settings.effective_script_model,
        ollama_json=settings.script_json_format,
    )


def get_image_prompt_client() -> "OllamaClient | GitHubLLMClient | GeminiLLMClient | DeepSeekLLMClient":
    """Factory: return the correct LLM client for ComfyUI image-prompt generation.

    Controls two tasks together:
    - scene_prompt rewrite pass (scriptwriter)
    - anchor character tag derivation (profile_builder)
    """
    return _build_provider_client(
        settings.image_prompt_provider,
        ollama_model=settings.effective_scene_prompt_model,
        ollama_json=settings.scene_prompt_json_format,
    )


# Module-level instances — provider-aware (ollama | github) via factories above.
summary_client = get_summary_client()
script_client = get_script_client()
image_prompt_client = get_image_prompt_client()
# Aliases for backward-compat imports
scene_prompt_client = image_prompt_client   # scriptwriter uses this name
ollama_client = image_prompt_client         # character_extractor uses this name (image-prompt phase)
