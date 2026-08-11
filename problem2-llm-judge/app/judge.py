import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class JudgeClient:
    """
    LLM Judge Client communicating with local Ollama API.
    Independently configured via JUDGE_MODEL and OLLAMA_BASE_URL.
    Logs prompts and raw responses for auditability.
    """

    def __init__(
        self,
        judge_model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        audit_log_path: Optional[Path] = None,
    ):
        self.judge_model = judge_model or settings.judge_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.temperature = temperature if temperature is not None else settings.judge_temperature
        self.timeout = timeout if timeout is not None else settings.judge_timeout

        self.audit_log_path = audit_log_path or (settings.results_dir / "audit.log")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.total_calls = 0
        self.total_eval_tokens = 0
        self.total_prompt_tokens = 0

    def generate_judge_response(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        """
        Send judging prompt to Ollama LLM service.
        Returns tuple of (raw_response_text, latency_ms).
        Logs prompt and response to audit log.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.judge_model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        self.total_calls += 1
        start_time = time.perf_counter()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            raw_text = data.get("response", "")

            # Track tokens if provided by Ollama
            self.total_prompt_tokens += data.get("prompt_eval_count", 0)
            self.total_eval_tokens += data.get("eval_count", 0)

            # Audit log prompt & raw response
            self._log_audit(system_prompt, user_prompt, raw_text, elapsed_ms)

            return raw_text, elapsed_ms

        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Ollama HTTP error {exc.response.status_code}: {exc.response.text}"
            logger.error(err_msg)
            self._log_audit(system_prompt, user_prompt, f"ERROR: {err_msg}", elapsed_ms)
            raise RuntimeError(err_msg) from exc
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Ollama connection error: {exc}"
            logger.error(err_msg)
            self._log_audit(system_prompt, user_prompt, f"ERROR: {err_msg}", elapsed_ms)
            raise RuntimeError(err_msg) from exc

    def _log_audit(self, system_prompt: str, user_prompt: str, raw_response: str, latency_ms: float):
        """Append prompt and raw response to audit log for replayability."""
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"JUDGE MODEL: {self.judge_model} | LATENCY: {latency_ms:.2f} ms\n")
                f.write("-" * 80 + "\n")
                f.write(f"[SYSTEM PROMPT]\n{system_prompt}\n")
                f.write("-" * 80 + "\n")
                f.write(f"[USER PROMPT]\n{user_prompt}\n")
                f.write("-" * 80 + "\n")
                f.write(f"[RAW RESPONSE]\n{raw_response}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as exc:
            logger.warning(f"Failed to write audit log: {exc}")
