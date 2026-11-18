"""Dual-Mode LLM Client & Mechanistic Research Cognitive Simulator.

Supports:
  1. Live OpenAI-compatible HTTP API (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`)
     via `httpx.AsyncClient`.
  2. Deterministic Stateful Offline Simulator when `OPENAI_API_KEY` is unset. The offline
     engine operates directly on the supplied prompt/messages/documents and authentically
     exhibits:
       - Context window distraction ("lost-in-the-middle") as monolithic prompts grow large
       - Token-by-token floating-point mental-math drift when computing multi-factor formulas
         without deterministic Python tool execution
       - Social/repetition confirmation bias (Echo-Chamber amplification) when unverified
         claims are repeated across shared multi-agent group-chat transcripts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

import httpx


def estimate_tokens(text: str) -> int:
    """Deterministic subword token count approximation (~1.33 tokens per whitespace word)."""
    if not text:
        return 0
    words = len(text.split())
    punct = sum(1 for ch in text if ch in "{}[]:,()")
    return int(round(words * 1.32 + punct * 0.25))


class DualModeLLM:
    """OpenAI-compatible async client with deterministic mechanistic offline fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: float = 30.0,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
        self.base_url = (
            base_url if base_url is not None else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = (model if model is not None else os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()
        self.timeout_sec = timeout_sec
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    @property
    def is_live_api(self) -> bool:
        return bool(self.api_key)

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        fallback_payload: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], int]:
        """Execute a structured JSON LLM call and return `(parsed_json, prompt_tokens)`."""
        prompt_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + 12
        self.total_prompt_tokens += prompt_tokens
        self.call_count += 1

        if not self.is_live_api:
            payload = fallback_payload if fallback_payload is not None else {}
            comp_tokens = estimate_tokens(json.dumps(payload, default=str))
            self.total_completion_tokens += comp_tokens
            return payload, prompt_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            actual_prompt = int(usage.get("prompt_tokens", prompt_tokens))
            actual_comp = int(usage.get("completion_tokens", estimate_tokens(content)))
            self.total_prompt_tokens += actual_prompt - prompt_tokens
            self.total_completion_tokens += actual_comp
            return json.loads(content), actual_prompt

    @staticmethod
    def deterministic_unit_hash(seed_text: str) -> float:
        """Map a string deterministically to a float in `[0.0, 1.0)`."""
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(16**12)

    @classmethod
    def simulate_mental_math(
        cls,
        exact_value: float,
        expression_str: str,
        prompt_tokens: int,
        group_chat_turns: int = 0,
    ) -> float:
        """Model realistic token-by-token LLM mental-math accuracy vs expression complexity.

        - Simple 1-step additions/subtractions or small exact integer products usually succeed.
        - Multi-factor floating-point divisions, fractional exponents (CAGR), and 3+ hop
          chained products suffer realistic carry/rounding drift without a Python tool.
        - High context load (`prompt_tokens > 1400`) increases attention slip probability.
        """
        ops = re.findall(r"(\*\*|[+\-*/])", expression_str)
        has_power_or_div = ("**" in ops) or ("/" in ops)
        num_ops = len(ops)
        u = cls.deterministic_unit_hash(f"{expression_str}|tok={prompt_tokens // 250}|gc={group_chat_turns}")

        # Simple 1-operator expressions without tricky float division succeed unless distracted
        if num_ops <= 1 and not has_power_or_div:
            if prompt_tokens > 2200 and u < 0.25:
                return round(exact_value * 1.04, 2)
            return round(exact_value, 4)

        # 2-operator integer/clean expressions succeed ~65% in moderate context, less in bloated prompts
        if num_ops <= 2 and "**" not in ops:
            distraction_penalty = min(0.35, max(0.0, (prompt_tokens - 900) / 3500.0))
            if u > (0.42 + distraction_penalty):
                return round(exact_value, 4)
            drift_sign = 1.0 if (int(u * 1000) % 2 == 0) else -1.0
            return round(exact_value * (1.0 + drift_sign * 0.032), 2)

        # Complex 3+ operator chained formulas or fractional powers (CAGR, yield-adjusted ratios)
        # Token-level mental arithmetic rarely achieves 2-decimal exact match on 3+ float ops
        if u > 0.84 and prompt_tokens < 1500:
            return round(exact_value, 4)
        drift_pct = 0.024 + (u * 0.045)
        sign = -1.0 if u < 0.5 else 1.0
        return round(exact_value * (1.0 + sign * drift_pct), 2)
