import os
import asyncio
from typing import Any, Optional
from fastapi import HTTPException

# === Optional vendor imports ===
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except Exception:
    AsyncAnthropic = None

# === Internal helpers ===
from enums.llm_provider import LLMProvider
from utils.get_env import (
    get_anthropic_api_key_env,
    get_google_api_key_env,
    get_openai_api_key_env,
)


# ===============================================================
# ✅ UNIVERSAL LLM CLIENT — Gemini 2.0-exp + OpenAI + Anthropic
# ===============================================================
class LLMClient:
    """
    Unified LLM client for Presenton.
    Handles Gemini 2.0-exp by default, plus OpenAI & Anthropic.
    Fully compatible with all message formats and structured streaming.
    """

    def __init__(self):
        # Load API keys
        self.google_api_key = get_google_api_key_env()
        self.openai_api_key = get_openai_api_key_env()
        self.anthropic_api_key = get_anthropic_api_key_env()

        # Configure Gemini once
        if genai and self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
            except Exception as e:
                print("⚠️ Gemini init warning:", e)

        # Default Gemini model
        self.gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-exp")

    # --- Compatibility flag for Presenton ---
    def enable_web_grounding(self) -> bool:
        return False

    # --- Normalize provider (string or enum) ---
    def _normalize_provider(self, provider: Any) -> str:
        if not provider:
            return "google"
        if isinstance(provider, str):
            return provider.lower()
        return str(provider).lower()

    # ===========================================================
    # 🧠 GENERATE — handles Gemini / OpenAI / Anthropic
    # ===========================================================
    async def generate(
        self,
        prompt: str,
        provider: Optional[Any] = "google",
        model: Optional[str] = None,
        **kwargs,
    ):
        provider = self._normalize_provider(provider)
        try:
            # ---- Google Gemini ----
            if provider.startswith("google"):
                if not genai:
                    raise RuntimeError("google-generativeai not installed.")
                model_name = model or self.gemini_model_name
                gem_model = genai.GenerativeModel(model_name)
                response = await asyncio.to_thread(gem_model.generate_content, prompt)
                return getattr(response, "text", str(response))

            # ---- OpenAI ----
            elif provider.startswith("openai"):
                if not AsyncOpenAI:
                    raise RuntimeError("openai not installed.")
                client = AsyncOpenAI(api_key=self.openai_api_key)
                model_name = model or "gpt-4o-mini"
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

            # ---- Anthropic ----
            elif provider.startswith("anthropic"):
                if not AsyncAnthropic:
                    raise RuntimeError("anthropic not installed.")
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                model_name = model or "claude-3-opus-20240229"
                resp = await client.messages.create(
                    model=model_name,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                try:
                    return resp.content[0].text
                except Exception:
                    return str(resp)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"❌ LLMClient.generate({provider}) failed:", e)
            raise HTTPException(status_code=500, detail=f"LLM API error: {e}")

    # ===========================================================
    # ⚡ STREAM_STRUCTURED — FINAL UNIVERSAL FIX
    # ===========================================================
    async def stream_structured(self, *args, **kwargs):
        """
        Handles all input styles:
          - stream_structured(prompt="...")
          - stream_structured("prompt text")
          - stream_structured(messages=[{"role": "user", "content": "..."}])
          - stream_structured(messages=[{"content": [{"text": "..."}]}])
          - stream_structured(messages=[{"content": [{"parts": [{"text": "..."}]}]}])
          - stream_structured(inputs="...") or data="..."
        """
        prompt = None

        # 1️⃣ Direct positional argument
        if len(args) > 0 and isinstance(args[0], str):
            prompt = args[0]

        # 2️⃣ Named keys
        if not prompt:
            prompt = (
                kwargs.get("prompt")
                or kwargs.get("input")
                or kwargs.get("text")
                or kwargs.get("data")
                or kwargs.get("inputs")
            )

        # 3️⃣ Handle OpenAI/Anthropic/Gemini-like nested messages
        if not prompt and "messages" in kwargs:
            try:
                messages = kwargs["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    for m in reversed(messages):
                        # Case A: role=user with text
                        if isinstance(m, dict) and m.get("role") == "user":
                            c = m.get("content")

                            # direct string
                            if isinstance(c, str):
                                prompt = c
                                break

                            # list of dicts
                            if isinstance(c, list):
                                for part in c:
                                    # direct text part
                                    if isinstance(part, dict) and "text" in part:
                                        prompt = part["text"]
                                        break
                                    # nested parts (Gemini 2.0 style)
                                    if (
                                        isinstance(part, dict)
                                        and "parts" in part
                                        and isinstance(part["parts"], list)
                                    ):
                                        for p in part["parts"]:
                                            if isinstance(p, dict) and "text" in p:
                                                prompt = p["text"]
                                                break
                                    if prompt:
                                        break
                            if prompt:
                                break

                        # Case B: No role but content with text/parts
                        if not prompt and isinstance(m, dict) and "content" in m:
                            c = m["content"]
                            if isinstance(c, list):
                                for part in c:
                                    if isinstance(part, dict):
                                        if "text" in part:
                                            prompt = part["text"]
                                            break
                                        if "parts" in part and isinstance(part["parts"], list):
                                            for p in part["parts"]:
                                                if isinstance(p, dict) and "text" in p:
                                                    prompt = p["text"]
                                                    break
                                    if prompt:
                                        break
                            elif isinstance(c, str):
                                prompt = c
                            if prompt:
                                break
            except Exception as e:
                print("⚠️ stream_structured parsing error:", e)

        # 4️⃣ Still missing prompt → fail cleanly
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="LLMClient.stream_structured() missing 'prompt' or valid messages[].content[].parts[].text",
            )

        # 5️⃣ Extract model/provider
        model = kwargs.get("model") or kwargs.get("model_name")
        provider = kwargs.get("provider") or kwargs.get("llm_provider") or "google"

        # 6️⃣ Run generation
        try:
            result = await self.generate(prompt, provider=provider, model=model)
            if isinstance(result, dict):
                yield result
            else:
                yield {"text": result}
        except Exception as e:
            print("⚠️ stream_structured failed:", e)
            raise HTTPException(status_code=500, detail=f"stream_structured failed: {str(e)}")


