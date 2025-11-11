import os
import asyncio
import dirtyjson
import json
from typing import AsyncGenerator, List, Optional
from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

# ✅ Stable Gemini import for google-generativeai==0.8.5+
import google.generativeai as genai

# ✅ Anthropic imports
from anthropic import AsyncAnthropic

# ✅ Internal imports
from enums.llm_provider import LLMProvider
from utils.get_env import (
    get_anthropic_api_key_env,
    get_google_api_key_env,
    get_openai_api_key_env,
)


# ==========================================================
# ✅ Universal LLMClient (Gemini 2.0-exp + OpenAI + Anthropic)
# ==========================================================
class LLMClient:
    """
    Unified LLM client compatible with Presenton’s backend.
    Supports Gemini 2.0 Experimental, OpenAI, and Anthropic.
    Fully backward-compatible with Presenton calls.
    """

    def __init__(self):
        # Load all available API keys
        self.google_api_key = get_google_api_key_env()
        self.openai_api_key = get_openai_api_key_env()
        self.anthropic_api_key = get_anthropic_api_key_env()

        # Configure Gemini globally once
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)

        # ✅ Default Gemini model
        self.gemini_model_name = "gemini-2.0-exp"

    # ✅ Presenton expects this callable method
    def enable_web_grounding(self) -> bool:
        """Compatibility placeholder — Presenton checks this before grounding web data."""
        return False

    # ----------------------------------------------------------
    # 🧠 Universal generation method
    # ----------------------------------------------------------
    async def generate(self, prompt: str, provider: str = "google", model: Optional[str] = None):
        """Generate text from the selected LLM provider."""
        try:
            if provider == "google":
                model_name = model or self.gemini_model_name
                model_instance = genai.GenerativeModel(model_name)
                response = await asyncio.to_thread(model_instance.generate_content, prompt)
                return response.text

            elif provider == "openai":
                client = AsyncOpenAI(api_key=self.openai_api_key)
                response = await client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

            elif provider == "anthropic":
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                response = await client.messages.create(
                    model=model or "claude-3-opus-20240229",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"❌ LLMClient.generate error ({provider}):", str(e))
            raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")

    # ----------------------------------------------------------
    # ⚡ stream_structured — compatibility with Presenton
    # ----------------------------------------------------------
    async def stream_structured(
        self,
        prompt: str,
        response_model=None,
        stream: bool = False,
        **kwargs,  # ✅ allows model, provider, etc.
    ):
        """
        Presenton expects this method for real-time structured responses.
        Gemini 2.0 doesn’t expose native structured streaming,
        so we emulate it via async generator.
        """
        try:
            model = kwargs.get("model", None)
            provider = kwargs.get("provider", "google")

            result = await self.generate(prompt, provider=provider, model=model)
            if isinstance(result, str):
                yield {"text": result}
            else:
                yield result

        except Exception as e:
            print("⚠️ stream_structured fallback failed:", e)
            raise HTTPException(status_code=500, detail=f"stream_structured failed: {str(e)}")

