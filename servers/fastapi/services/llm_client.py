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
from utils.get_env import (
    get_anthropic_api_key_env,
    get_google_api_key_env,
    get_openai_api_key_env,
)


# ===============================================================
# UNIFIED LLM CLIENT — Gemini 2.0 Flash Preview
# ===============================================================
class LLMClient:
    def __init__(self):
        self.google_api_key = get_google_api_key_env()
        self.openai_api_key = get_openai_api_key_env()
        self.anthropic_api_key = get_anthropic_api_key_env()

        if genai and self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
            except Exception as e:
                print("⚠️ Gemini init warning:", e)

        # Correct high quota model (NO preview-image-generation)
        self.gemini_model_name = os.getenv(
            "GEMINI_MODEL_NAME",
            "models/gemini-2.0-flash-preview",
        )

    def enable_web_grounding(self) -> bool:
        return False

    def _normalize_provider(self, provider: Any) -> str:
        if not provider:
            return "google"
        if isinstance(provider, str):
            return provider.lower()
        return str(provider).lower()

    # ===========================================================
    # generate()
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
            # ----------- GEMINI --------------
            if provider.startswith("google"):
                if not genai:
                    raise RuntimeError("google-generativeai not installed")
                model_name = model or self.gemini_model_name
                gem_model = genai.GenerativeModel(model_name)

                if isinstance(prompt, (dict, list)):
                    import json
                    prompt = json.dumps(prompt, ensure_ascii=False)

                response = await asyncio.to_thread(
                    gem_model.generate_content,
                    prompt
                )
                return getattr(response, "text", str(response))

            # ----------- OPENAI --------------
            elif provider.startswith("openai"):
                client = AsyncOpenAI(api_key=self.openai_api_key)
                resp = await client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

            # ----------- ANTHROPIC -----------
            elif provider.startswith("anthropic"):
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                resp = await client.messages.create(
                    model=model or "claude-3-opus-20240229",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text if resp.content else str(resp)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"❌ generate() failed:", e)
            raise HTTPException(500, f"LLM API error: {e}")

    # ===========================================================
    # generate_text() — wrapper
    # ===========================================================
    async def generate_text(self, model, messages, provider="google", **kwargs):
        try:
            prompt = "\n".join(
                (
                    m.content if hasattr(m, "content")
                    else m.get("content") if isinstance(m, dict)
                    else str(m)
                )
                for m in messages
            )
            return await self.generate(prompt, model=model, provider=provider)

        except Exception as e:
            print("⚠️ generate_text failed:", e)
            raise HTTPException(500, f"generate_text failed: {e}")

    # ===========================================================
    # generate_structured() — JSON output
    # ===========================================================
    async def generate_structured(
        self,
        model: str,
        messages: list,
        response_format: dict,
        strict: bool = True,
        provider: Optional[str] = "google"
    ):
        provider = self._normalize_provider(provider)

        try:
            # ----- Gemini -----
            if provider.startswith("google"):
                model_name = model or self.gemini_model_name
                gem_model = genai.GenerativeModel(model_name)

                import json
                prompt = "\n".join(
                    json.dumps(m.content if hasattr(m, "content") else m, ensure_ascii=False)
                    if isinstance(m, (dict, list)) else str(m)
                    for m in messages
                )

                response = await asyncio.to_thread(
                    gem_model.generate_content,
                    f"{prompt}\n\nReturn JSON strictly matching:\n{json.dumps(response_format)}"
                )

                text = response.text
                try:
                    return json.loads(text)
                except:
                    import re
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if m:
                        return json.loads(m.group(0))
                    if strict:
                        raise ValueError("Invalid JSON")
                    return {"raw": text}

            # ----- OpenAI -----
            elif provider.startswith("openai"):
                client = AsyncOpenAI(api_key=self.openai_api_key)
                resp = await client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": str(messages)}],
                    response_format={"type": "json_schema", "json_schema": response_format},
                )
                import json
                return json.loads(resp.choices[0].message.content)

            # ----- Anthropic -----
            elif provider.startswith("anthropic"):
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                resp = await client.messages.create(
                    model=model or "claude-3-opus-20240229",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": str(messages)}],
                )
                import json
                return json.loads(resp.content[0].text)

        except Exception as e:
            print("❌ generate_structured failed:", e)
            raise HTTPException(500, f"generate_structured failed: {e}")

    # ===========================================================
    # stream_structured() — REQUIRED BY outline generator
    # ===========================================================
    async def stream_structured(self, *args, **kwargs):
        """
        Presenton expects this to behave like:
        async for chunk in client.stream_structured(...)

        Your code only needs SINGLE response (not streaming),
        so we just wrap generate_structured.
        """
        model = kwargs.get("model")
        messages = kwargs.get("messages")
        response_format = kwargs.get("schema") or kwargs.get("response_format")
        provider = kwargs.get("provider", "google")
        strict = kwargs.get("strict", True)

        # Yield once — simulated streaming
        result = await self.generate_structured(
            model=model,
            messages=messages,
            response_format=response_format,
            provider=provider,
            strict=strict,
        )

        yield result

