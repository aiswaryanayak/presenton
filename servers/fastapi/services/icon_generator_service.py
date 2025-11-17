import os
import asyncio
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------
# Utility Functions
# ----------------------------
async def _run_blocking_in_executor(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def _parse_response_text(resp) -> str:
    try:
        if isinstance(resp, str):
            return resp.strip()

        # Gemini 2.0 structure
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()

        if hasattr(resp, "candidates"):
            for cand in resp.candidates or []:
                parts = getattr(cand, "content", None)
                if parts and hasattr(parts, "parts"):
                    for p in parts.parts:
                        if hasattr(p, "text"):
                            return p.text.strip()

        if isinstance(resp, dict):
            return resp.get("text") or ""

    except Exception:
        logger.exception("❌ Failed to parse LLM response")

    return ""


# ----------------------------
# MAIN FUNCTION
# ----------------------------
async def generate_slide_icons(
    slide_text: str,
    n_icons: int = 3,
    style: Optional[str] = None,
) -> List[Dict]:

    slide_text = (slide_text or "").strip()
    if not slide_text:
        return []

    style_part = f"Style: {style}." if style else ""

    prompt = f"""
Suggest {n_icons} simple, flat icon ideas for this slide content:

{slide_text}

Return ONLY comma-separated keywords. Example:
robot, data chart, ai brain

{style_part}
"""

    # ----------------------------
    # Try LLMClient first
    # ----------------------------
    try:
        from servers.fastapi.services.llm_client import LLMClient
        client = LLMClient()

        resp = await client.generate(prompt)
        text = await _parse_response_text(resp)

        if not text:
            raise RuntimeError("Empty response")

    except Exception as e:
        logger.warning(f"⚠️ LLMClient failed, using direct Gemini: {e}")

        # ----------------------------
        # DIRECT GEMINI FALLBACK
        # ----------------------------
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in environment")

        genai.configure(api_key=api_key)

        def _call_gemini():
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            return model.generate_content(prompt)

        resp = await _run_blocking_in_executor(_call_gemini)
        text = await _parse_response_text(resp)

    # ----------------------------
    # Parse keywords into icon dicts
    # ----------------------------
    if not text:
        logger.warning("⚠️ No icons generated.")
        return []

    icons = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]
    icons = icons[:n_icons]

    results = []
    for icon in icons:
        results.append({
            "name": icon.replace(" ", "_").lower(),
            "prompt": f"An icon representing '{icon}' in flat minimal vector style."
        })

    return results


# ----------------------------
# Batch function
# ----------------------------
async def generate_icons_for_slides(
    slides: List[str], n_icons_per_slide: int = 3
) -> List[List[Dict]]:
    tasks = [generate_slide_icons(slide, n_icons=n_icons_per_slide) for slide in slides]
    return await asyncio.gather(*tasks)

