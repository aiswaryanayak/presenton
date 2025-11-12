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

        # Default Gemini model (⚡ your version)
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
    # ⚡ STREAM_STRUCTURED — SAFE & DEBUG-FRIENDLY
    # ===========================================================
    async def stream_structured(self, *args, **kwargs):
        print("🔍 DEBUG STREAM INPUT:", kwargs)

        prompt = None

        def extract_text(obj):
            """Recursively search for any 'text' key or string."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and v.strip():
                        return v
                    if k == "text" and isinstance(v, str):
                        return v
                    result = extract_text(v)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = extract_text(item)
                    if result:
                        return result
            elif isinstance(obj, str) and obj.strip():
                return obj
            return None

        if len(args) > 0 and isinstance(args[0], str):
            prompt = args[0]

        if not prompt:
            prompt = (
                kwargs.get("prompt")
                or kwargs.get("input")
                or kwargs.get("text")
                or kwargs.get("data")
                or kwargs.get("inputs")
            )

        if not prompt and "messages" in kwargs:
            prompt = extract_text(kwargs["messages"])

        if not prompt:
            prompt = extract_text(kwargs)

        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="LLMClient.stream_structured() missing 'prompt' or any string content (checked deeply)",
            )

        model = kwargs.get("model") or kwargs.get("model_name")
        provider = kwargs.get("provider") or kwargs.get("llm_provider") or "google"

        try:
            result = await self.generate(prompt, provider=provider, model=model)
            if isinstance(result, dict):
                yield result
            else:
                yield {"text": result}
        except Exception as e:
            print("⚠️ stream_structured failed:", e)
            raise HTTPException(status_code=500, detail=f"stream_structured failed: {str(e)}")

    # ===========================================================
    # 🧩 GENERATE_STRUCTURED — REQUIRED BY PRESENTON
    # ===========================================================
        # ===========================================================
    # 🧩 GENERATE_STRUCTURED — FIXED & VALIDATED
    # ===========================================================
    async def generate_structured(
        self,
        model: str,
        messages: list,
        response_format: dict,
        strict: bool = True,
        provider: Optional[str] = "google",
    ):
        """
        Generate structured JSON responses for slide layouts, outlines, etc.
        Compatible with Gemini 2.0-exp, GPT-4o, and Claude 3 APIs.
        """
        provider = self._normalize_provider(provider)

        try:
            # === GEMINI ===
            if provider.startswith("google"):
                if not genai:
                    raise RuntimeError("google-generativeai not installed.")

                model_name = model or self.gemini_model_name
                gem_model = genai.GenerativeModel(model_name)

                # 🧩 Fix: Normalize message content into a single string
                import json
                prompt_text_parts = []
                for m in messages:
                    if hasattr(m, "content"):
                        if isinstance(m.content, (list, dict)):
                            prompt_text_parts.append(json.dumps(m.content, ensure_ascii=False))
                        else:
                            prompt_text_parts.append(str(m.content))
                    else:
                        prompt_text_parts.append(str(m))
                prompt_text = "\n".join(prompt_text_parts)

                response = await asyncio.to_thread(
                    gem_model.generate_content,
                    f"{prompt_text}\n\nReturn valid JSON strictly matching this schema:\n{json.dumps(response_format, ensure_ascii=False)}",
                )

                import re
                try:
                    return json.loads(response.text)
                except Exception:
                    match = re.search(r"\{.*\}", response.text, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
                    if strict:
                        raise ValueError("Gemini returned invalid JSON.")
                    return {"raw": response.text}

            # === OPENAI ===
            elif provider.startswith("openai"):
                if not AsyncOpenAI:
                    raise RuntimeError("openai not installed.")
                client = AsyncOpenAI(api_key=self.openai_api_key)
                resp = await client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": str(messages)}],
                    response_format={"type": "json_schema", "json_schema": response_format},
                )

                return json.loads(resp.choices[0].message.content)

            # === ANTHROPIC ===
            elif provider.startswith("anthropic"):
                if not AsyncAnthropic:
                    raise RuntimeError("anthropic not installed.")
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                resp = await client.messages.create(
                    model=model or "claude-3-opus-20240229",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": str(messages)}],
                )

                try:
                    return json.loads(resp.content[0].text)
                except Exception:
                    if strict:
                        raise
                    return {"raw": resp.content[0].text}

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"❌ LLMClient.generate_structured() failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"LLMClient.generate_structured() failed: {str(e)}",
            )


            # === OPENAI ===
            elif provider.startswith("openai"):
                if not AsyncOpenAI:
                    raise RuntimeError("openai not installed.")
                client = AsyncOpenAI(api_key=self.openai_api_key)
                resp = await client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": str(messages)}],
                    response_format={"type": "json_schema", "json_schema": response_format},
                )

                import json
                return json.loads(resp.choices[0].message.content)

            # === ANTHROPIC ===
            elif provider.startswith("anthropic"):
                if not AsyncAnthropic:
                    raise RuntimeError("anthropic not installed.")
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                resp = await client.messages.create(
                    model=model or "claude-3-opus-20240229",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": str(messages)}],
                )

                import json
                try:
                    return json.loads(resp.content[0].text)
                except Exception:
                    if strict:
                        raise
                    return {"raw": resp.content[0].text}

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"❌ LLMClient.generate_structured() failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"LLMClient.generate_structured() failed: {str(e)}",
            )



