# servers/fastapi/services/llm_client.py
import os
import asyncio
import dirtyjson
import json
from typing import AsyncGenerator, List, Optional, Any
from fastapi import HTTPException

# OpenAI async client (if used)
try:
    from openai import AsyncOpenAI
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
except Exception:
    AsyncOpenAI = None
    OpenAIChatCompletionChunk = None

# Gemini / Google stable import
import google.generativeai as genai

# Anthropic async client (if used)
try:
    from anthropic import AsyncAnthropic
except Exception:
    AsyncAnthropic = None

# Internal helpers
from enums.llm_provider import LLMProvider  # if you use this enum elsewhere
from utils.get_env import (
    get_anthropic_api_key_env,
    get_google_api_key_env,
    get_openai_api_key_env,
)


class LLMClient:
    """
    Unified LLM client compatible with Presenton’s backend.
    Supports Gemini 2.0 Experimental (default), OpenAI, and Anthropic.
    Provides a robust `stream_structured` that accepts arbitrary kwargs
    (model, provider, response_model, etc.) so Presenton calls won't fail.
    """

    def __init__(self):
        # load keys from environment helpers
        self.google_api_key = get_google_api_key_env()
        self.openai_api_key = get_openai_api_key_env()
        self.anthropic_api_key = get_anthropic_api_key_env()

        # configure google generative API globally if key present
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
            except Exception as e:
                # don't crash init — surface later on first call
                print("Warning: failed to configure genai:", e)

        # default gemini model name (your environment uses gemini-2.0-exp)
        self.gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-exp")

    # compatibility placeholder — some code expects this attribute/method
    def enable_web_grounding(self) -> bool:
        # If you later want to enable, return True and implement grounding support
        return False

    # Helper: normalize provider argument (accepts enum or string)
    def _normalize_provider(self, provider: Any) -> str:
        if provider is None:
            return "google"
        if isinstance(provider, str):
            return provider.lower()
        # if provider is enum (LLMProvider) or similar
        try:
            return str(provider).lower()
        except Exception:
            return "google"

    # Primary generation method
    async def generate(self, prompt: str, provider: Optional[Any] = "google", model: Optional[str] = None, **kwargs):
        """
        Generate text from selected LLM provider.
        - prompt: text prompt
        - provider: "google" | "openai" | "anthropic" (or enum)
        - model: model name to use (overrides default)
        - kwargs: passed through for future options
        """
        provider_norm = self._normalize_provider(provider)
        try:
            if provider_norm.startswith("google"):
                model_name = model or self.gemini_model_name
                # gemini model instance
                model_instance = genai.GenerativeModel(model_name)
                # model.generate_content is synchronous in many builds - run in thread
                response = await asyncio.to_thread(model_instance.generate_content, prompt)
                # response may be object with .text or nested structure
                if hasattr(response, "text"):
                    return response.text
                # sometimes response.content or response.output exists
                if hasattr(response, "output") and response.output:
                    # try to join text parts
                    try:
                        # support multiple content parts
                        if hasattr(response.output, "items"):
                            return "".join(p.text for p in response.output.items if hasattr(p, "text"))
                    except Exception:
                        pass
                # fallback: string-cast
                return str(response)

            elif provider_norm.startswith("openai"):
                if AsyncOpenAI is None:
                    raise RuntimeError("OpenAI client not installed")
                client = AsyncOpenAI(api_key=self.openai_api_key)
                model_name = model or kwargs.get("model") or "gpt-4o-mini"
                # using chat completions path used earlier
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                # be defensive about structure
                try:
                    return response.choices[0].message.content
                except Exception:
                    return str(response)

            elif provider_norm.startswith("anthropic"):
                if AsyncAnthropic is None:
                    raise RuntimeError("Anthropic client not installed")
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                model_name = model or kwargs.get("model") or "claude-3-opus-20240229"
                response = await client.messages.create(
                    model=model_name,
                    max_tokens=kwargs.get("max_tokens", 500),
                    messages=[{"role": "user", "content": prompt}],
                )
                # response.content may be list-like
                try:
                    # structure differs across clients; try common patterns
                    if hasattr(response, "content") and isinstance(response.content, list):
                        return response.content[0].text
                    if hasattr(response, "text"):
                        return response.text
                except Exception:
                    pass
                return str(response)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except HTTPException:
            # re-raise unchanged
            raise
        except Exception as e:
            print(f"❌ LLMClient.generate error ({provider_norm}): {e}")
            raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")

    # Compatibility streaming method
    async def stream_structured(self, *args, **kwargs):
        """
        Backwards-compatible structured streaming wrapper.
        Accepts either:
            - stream_structured(prompt, response_model=..., model=..., provider=...)
            - stream_structured(prompt="...", model=..., provider=...)
            - stream_structured(model=..., provider=..., prompt="...")
            - stream_structured(...) with any extra kwargs
        Yields one item with {"text": "..."} as a fallback since Gemini 2.0-exp
        doesn't provide a structured streaming API here.
        """
        # Extract prompt from positional args or kwargs
        prompt = None
        if len(args) > 0:
            # if first arg is not None and is string, treat as prompt
            try:
                if isinstance(args[0], str):
                    prompt = args[0]
            except Exception:
                prompt = None

        if prompt is None:
            # try kwargs
            prompt = kwargs.get("prompt") or kwargs.get("input") or kwargs.get("text")

        if not prompt:
            # Presenton sometimes calls stream_structured without prompt -> raise helpful error
            raise HTTPException(status_code=400, detail="LLMClient.stream_structured() missing required 'prompt'")

        # accept model/provider from kwargs
        model = kwargs.get("model") or kwargs.get("model_name") or None
        provider = kwargs.get("provider") or kwargs.get("llm_provider") or "google"

        # If present, handle 'response_model' or other args - we ignore for fallback
        try:
            # Use generate() and yield a single structured chunk
            result = await self.generate(prompt, provider=provider, model=model)
            # If the result already looks like a dict / structured content, yield as-is
            if isinstance(result, dict):
                yield result
                return
            # Otherwise yield a simple structured dict with text
            yield {"text": result}
            return
        except HTTPException:
            raise
        except Exception as e:
            print("⚠️ stream_structured fallback failed:", e)
            raise HTTPException(status_code=500, detail=f"stream_structured failed: {str(e)}")

