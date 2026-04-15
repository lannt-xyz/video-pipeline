import json
import time
import uuid
from pathlib import Path
from typing import Any, List

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings


class ComfyUIOutOfMemoryError(RuntimeError):
    """Raised when ComfyUI reports an out-of-memory error."""


class ComfyUINodeNotFoundError(RuntimeError):
    """Raised when a required custom node is not installed in ComfyUI."""


class ComfyUIError(RuntimeError):
    """Raised on non-OOM ComfyUI errors."""


class ComfyUIClient:
    """Sync client for ComfyUI REST API with polling."""

    def __init__(self) -> None:
        self.base_url = settings.comfyui_url
        self.timeout = settings.comfyui_timeout
        self.poll_interval = settings.comfyui_poll_interval
        self._client_id = str(uuid.uuid4())

    def health_check(self) -> bool:
        """Raises RuntimeError if ComfyUI is not available."""
        try:
            resp = httpx.get(f"{self.base_url}/system_stats", timeout=5.0)
            resp.raise_for_status()
            logger.debug("ComfyUI health check OK")
            return True
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"ComfyUI not available at {self.base_url}: {exc}"
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=5, max=30),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def submit_prompt(self, workflow: dict) -> str:
        """Submit a workflow prompt. Returns prompt_id."""
        payload = {"prompt": workflow, "client_id": self._client_id}
        resp = httpx.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=30.0,
        )
        if resp.status_code >= 400:
            logger.error(
                "ComfyUI rejected workflow | status={} body={}",
                resp.status_code,
                resp.text[:2000],
            )
            try:
                err_body = resp.json()
                err_type = err_body.get("error", {}).get("type", "")
                if err_type == "missing_node_type":
                    missing = err_body["error"]["message"]
                    raise ComfyUINodeNotFoundError(missing)
            except (ValueError, KeyError):
                pass
        resp.raise_for_status()
        prompt_id: str = resp.json()["prompt_id"]
        logger.debug("Submitted prompt | id={}", prompt_id)
        return prompt_id

    def poll_result(self, prompt_id: str) -> dict:
        """Poll /history/{prompt_id} until complete. Raises on OOM or timeout."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(
                    f"{self.base_url}/history/{prompt_id}", timeout=10.0
                )
                resp.raise_for_status()
                history = resp.json()
                if prompt_id in history:
                    entry = history[prompt_id]
                    if "error" in entry:
                        error_msg = str(entry["error"])
                        if "out of memory" in error_msg.lower() or "cuda out" in error_msg.lower():
                            raise ComfyUIOutOfMemoryError(
                                f"ComfyUI OOM for prompt {prompt_id}: {error_msg}"
                            )
                        raise ComfyUIError(
                            f"ComfyUI error for prompt {prompt_id}: {error_msg}"
                        )
                    return entry
            except (httpx.HTTPError, KeyError) as exc:
                logger.debug(
                    "Poll attempt failed | prompt_id={} error={}", prompt_id, exc
                )
            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"ComfyUI prompt {prompt_id} timed out after {self.timeout}s"
        )

    def get_file(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        """Download a generated file (image or video) from ComfyUI."""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        }
        resp = httpx.get(f"{self.base_url}/view", params=params, timeout=30.0)
        resp.raise_for_status()
        return resp.content

    def upload_image(self, path: Path) -> str:
        """Upload a local image to ComfyUI /upload/image. Returns the filename as known to ComfyUI."""
        with open(path, "rb") as f:
            resp = httpx.post(
                f"{self.base_url}/upload/image",
                files={"image": (path.name, f, "image/png")},
                data={"type": "input", "overwrite": "true"},
                timeout=30.0,
            )
        resp.raise_for_status()
        uploaded_name: str = resp.json()["name"]
        logger.debug("Uploaded image to ComfyUI | local={} remote={}", path, uploaded_name)
        return uploaded_name

    def generate_image(
        self,
        workflow_path: str,
        replacements: dict[str, Any],
        output_path: Path,
    ) -> Path:
        """Load workflow, apply replacements, submit, poll, download output file."""
        # Upload any Path values so ComfyUI LoadImage can reference them by name
        resolved: dict[str, Any] = {}
        for key, value in replacements.items():
            if isinstance(value, Path):
                resolved[key] = self.upload_image(value)
            else:
                resolved[key] = value
        workflow = self._load_workflow(workflow_path, resolved)
        prompt_id = self.submit_prompt(workflow)
        result = self.poll_result(prompt_id)

        files = self._extract_output_files(result)
        if not files:
            raise ComfyUIError(
                f"No output files in ComfyUI result for prompt {prompt_id}"
            )

        file_info = files[0]
        file_bytes = self.get_file(
            file_info["filename"],
            file_info.get("subfolder", ""),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(file_bytes)
        logger.info("File downloaded | path={}", output_path)
        return output_path

    def _load_workflow(self, workflow_path: str, replacements: dict) -> dict:
        """Load workflow JSON and replace __KEY__ placeholders.
        - Numeric/bool values replace "\"__KEY__\"" (with quotes) → bare value
        - String values replace __KEY__ inline within string fields
        """
        path = Path(workflow_path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow template not found: {workflow_path}")

        content = path.read_text(encoding="utf-8")

        for key, value in replacements.items():
            quoted = f'"__{key}__"'
            inline = f"__{key}__"
            if isinstance(value, bool):
                content = content.replace(quoted, "true" if value else "false")
            elif isinstance(value, (int, float)):
                content = content.replace(quoted, str(value))
            else:
                # String: JSON-escape the value to prevent breaking json.loads
                # e.g. scene_prompt from LLM may contain double quotes
                escaped = json.dumps(str(value))[1:-1]  # strip surrounding quotes
                content = content.replace(inline, escaped)

        workflow = json.loads(content)
        # Strip metadata keys (e.g. "_meta") — ComfyUI rejects non-node top-level keys
        return {k: v for k, v in workflow.items() if not k.startswith("_")}

    def _extract_output_files(self, history_entry: dict) -> List[dict]:
        """Extract all output image/video file references from a history entry."""
        files: List[dict] = []
        outputs = history_entry.get("outputs", {})
        for node_output in outputs.values():
            for key in ("images", "gifs", "videos"):
                if key in node_output:
                    files.extend(node_output[key])
        return files


comfyui_client = ComfyUIClient()
