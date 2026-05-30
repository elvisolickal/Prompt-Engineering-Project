"""
LLM Router — unified adapter for OpenAI and Google Gemini.
All public methods are synchronous (Streamlit-friendly).
"""
from typing import Optional
import time

import config


class LLMRouter:
    """Route generation calls to OpenAI or Gemini based on model name."""

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        model_key: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1200,
        temperature: float = 0.7,
        retries: int = 3,
    ) -> str:
        """
        Generate text using the specified model.

        model_key: key from config.ALL_MODELS (e.g. "GPT-4o", "Gemini 1.5 Flash")
                   OR a raw model id string (e.g. "gpt-4o-mini")
        Returns the generated text string.
        """
        model_id = config.ALL_MODELS.get(model_key, model_key)

        if self._is_openai(model_id):
            return self._openai_generate(model_id, system_prompt, user_message, max_tokens, temperature, retries)
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

    def _openai_generate(
        self,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        retries: int,
    ) -> str:
        import openai
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

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
            except openai.RateLimitError:
                wait = 2 ** attempt * 5
                time.sleep(wait)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2)
        return ""

    # ── Google Gemini ─────────────────────────────────────────────────────────

    def _is_gemini(self, model_id: str) -> bool:
        return "gemini" in model_id.lower()

    def _gemini_generate(
        self,
        model_id: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        retries: int,
    ) -> str:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)

        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        # Combine system + user for Gemini (system_instruction parameter)
        model_obj = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system_prompt if system_prompt else None,
            generation_config=generation_config,
        )

        for attempt in range(retries):
            try:
                resp = model_obj.generate_content(user_message)
                return resp.text or ""
            except Exception as e:
                err = str(e).lower()
                if "quota" in err or "429" in err:
                    time.sleep(2 ** attempt * 5)
                elif attempt == retries - 1:
                    raise
                else:
                    time.sleep(2)
        return ""
