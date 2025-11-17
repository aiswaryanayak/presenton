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
# UNIFIED LLM CLIENT — Using Gemini 2.0 Flash Preview (Correct)
# ===============================================================
class LLMClient:
    def __init__(self):
        self.google_api_key = get_google_api_key_env()
        self.openai_api_key = get_openai_api_key_env()
        self.anthropic_api_key = get_anthropic_api_key_env()

        # Configure Gemini once
        if genai and self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
            except Exception as e:
                print("⚠️ Gemini init warning:", e)

        # THE FIX ➜ Use correct high-quota model
        self.gemini_model_name = os.getenv(
            "GEMINI_MODEL_NAME",
            "models/gemini-2.0-flash-preview"
        )

    def enable_web_grounding(self) -> bool:
        return False

    def _normalize_provider(self, provider: Any) -> str:
        if not provider: return "google"
        if isinstance(provider, str): return provider.lower()
        return str(provider).lower()

    # ===========================================================
    # GENERATE — Gemini / OpenAI / Anthropic
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
            # -------------------- GEMINI --------------------
            if provider.startswith("google"):
                if not genai:
                    raise RuntimeError("google-generativeai not installed.")

                model_name = model or self.gemini_model_name
                gem_model = genai.GenerativeModel(model_name)

                if isinstance(prompt, (list, dict)):
                    import json
                    prompt = json.dumps(prompt, ensure_ascii=False)

                response = await asyncio.to_thread(
                    gem_model.generate_content,
                    prompt
                )
                return getattr(response, "text", str(response))

            # -------------------- OPENAI --------------------
            elif provider.startswith("openai"):
                client = AsyncOpenAI(api_key=self.openai_api_key)
                resp = await client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

            # ------------------- ANTHROPIC ------------------
            elif provider.startswith("anthropic"):
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                resp = await client.messages.create(
                    model=model or "claude-3-opus-20240229",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                try:
                    return resp.content[0].text
                except:
                    return str(resp)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"❌ LLMClient.generate({provider}) failed:", e)
            raise HTTPException(500, f"LLM API error: {e}")

    # ===========================================================
    # COMPAT generate_text()
    # ===========================================================
    async def generate_text(
        self,
        model: str,
        messages: list,
        provider: Optional[str] = "google",
        **kwargs,
    ):
        try:
            prompt_parts = []

            for m in messages:
                if hasattr(m, "content"):
                    prompt_parts.append(str(m.content))
                elif isinstance(m, dict):
                    prompt_parts.append(str(m.get("content", "")))
                else:
                    prompt_parts.append(str(m))

            prompt = "\n".join(prompt_parts)

            return await self.generate(
                prompt=prompt,
                provider=provider,
                model=model,
            )

        except Exception as e:
            print("⚠️ LLMClient.generate_text failed:", e)
            raise HTTPException(500, f"LLMClient.generate_text failed: {e}")

    # ===========================================================
    # Structured JSON Output
    # ===========================================================
    async def generate_structured(
        self,
        model: str,
        messages: list,
        response_format: dict,
        strict: bool = True,
        provider: Optional[str] = "google",
    ):
        provider = self._normalize_provider(provider)

        try:
            # -------- GEMINI --------
            if provider.startswith("google"):
                model_name = model or self.gemini_model_name
                gem_model = genai.GenerativeModel(model_name)

                import json
                prompt_parts = []
                for m in messages:
                    if hasattr(m, "content"):
                        content = m.content
                    else:
                        content = m
                    if isinstance(content, (dict, list)):
                        prompt_parts.append(json.dumps(content, ensure_ascii=False))
                    else:
                        prompt_parts.append(str(content))

                prompt_text = "\n".join(prompt_parts)

                response = await asyncio.to_thread(
                    gem_model.generate_content,
                    f"{prompt_text}\n\nReturn JSON strictly matching:\n{json.dumps(response_format)}"
                )

                import re
                try:
                    return json.loads(response.text)
                except:
                    match = re.search(r"\{.*\}", response.text, re.DOTALL)
                    if match: return json.loads(match.group(0))
                    if strict: raise ValueError("Invalid JSON")
                    return {"raw": response.text}

            # -------- OPENAI --------
            elif provider.startswith("openai"):
                client = AsyncOpenAI(api_key=self.openai_api_key)
                resp = await client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": str(messages)}],
                    response_format={"type": "json_schema", "json_schema": response_format},
                )
                import json
                return json.loads(resp.choices[0].message.content)

            # ------- ANTHROPIC -------
            elif provider.startswith("anthropic"):
                client = AsyncAnthropic(api_key=self.anthropic_api_key)
                resp = await client.messages.create(
                    model=model or "claude-3-opus-20240229",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": str(messages)}],
                )
                import json
                try:
                    return json.loads(resp.content[0].text)
                except:
                    if strict: raise
                    return {"raw": resp.content[0].text}

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print("❌ LLMClient.generate_structured failed:", e)
            raise HTTPException(500, f"generate_structured failed: {e}")

