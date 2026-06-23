"""
LLM Router — unified adapter for OpenAI and Google Gemini.
All public methods are synchronous (Streamlit-friendly).

Uses the modern `google-genai` SDK instead of the deprecated
`google.generativeai` package.
"""
from typing import Optional
import time

import config


class LLMRouter:
    """Route generation calls to OpenAI or Gemini based on model name."""

    def __init__(self):
        self._openai_client = None
        self._gemini_client = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        model_key: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1200,
        temperature: float = 0.7,
        retries: int = 3,
        timeout: int = 60,
    ) -> str:
        """
        Generate text using the specified model.

        model_key: key from config.ALL_MODELS (e.g. "GPT-4o", "Gemini 1.5 Flash")
                   OR a raw model id string (e.g. "gpt-4o-mini")
        Returns the generated text string.
        """
        model_id = config.ALL_MODELS.get(model_key, model_key)

        if self._is_openai(model_id):
            return self._openai_generate(model_id, system_prompt, user_message, max_tokens, temperature, retries, timeout)
        elif self._is_gemini(model_id):
            return self._gemini_generate(model_id, system_prompt, user_message, max_tokens, temperature, retries)
        else:
            raise ValueError(f"Unknown model: {model_key!r}")

    def list_available_models(self) -> list:
        """Return model display-names whose API key is configured."""
        available = []
        if config.OPENAI_API_KEY:
            available += list(config.OPENAI_MODELS.keys())
        if config.GEMINI_API_KEY:
            available += list(config.GEMINI_MODELS.keys())
        return available

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _is_openai(self, model_id: str) -> bool:
        return model_id.startswith("gpt-") or model_id.startswith("o1")

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=30,
            )
        return self._openai_client

    def _openai_generate(
        self,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        retries: int,
        timeout: int,
    ) -> str:
        client = self._get_openai_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "quota" in err:
                    wait = 2 ** attempt * 10  # 10s, 20s, 40s
                    time.sleep(wait)
                elif attempt == retries - 1:
                    raise
                else:
                    time.sleep(2)
        return ""

    # ── Google Gemini (new google-genai SDK) ───────────────────────────────────

    def _is_gemini(self, model_id: str) -> bool:
        return "gemini" in model_id.lower()

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        return self._gemini_client

    def _gemini_generate(
        self,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        retries: int,
    ) -> str:
        from google.genai import types

        client = self._get_gemini_client()

        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        for attempt in range(retries):
            try:
                resp = client.models.generate_content(
                    model=model_id,
                    contents=user_message,
                    config=gen_config,
                )
                text = resp.text or ""
                return text
            except Exception as e:
                err = str(e)
                err_lower = err.lower()
                if "quota" in err_lower or "429" in err_lower or "resource" in err_lower:
                    # Try to extract retry delay from error message
                    import re as _re
                    m = _re.search(r"retry in (\d+\.?\d*)s", err, _re.IGNORECASE)
                    if m:
                        wait = float(m.group(1)) + 2
                    else:
                        wait = min(15 * (2 ** attempt), 120)  # 15s, 30s, 60s, max 120s
                    print(f"[LLMRouter] Rate limit hit ({model_id}), waiting {wait:.0f}s... (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                elif attempt == retries - 1:
                    raise RuntimeError(f"Gemini API error after {retries} attempts: {err[:300]}")
                else:
                    time.sleep(3)
        return ""
