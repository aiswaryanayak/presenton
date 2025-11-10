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
    """Extracts plain text safely from Gemini or custom LLM responses."""
    try:
        if isinstance(resp, str):
            return resp.strip()

        # Handle Gemini 2.0 response objects
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()

        # Handle candidates structure
        if hasattr(resp, "candidates"):
            candidates = resp.candidates or []
            if candidates:
                for cand in candidates:
                    parts = getattr(cand, "content", None)
                    if parts and hasattr(parts, "parts"):
                        for p in parts.parts:
                            if hasattr(p, "text"):
                                return p.text.strip()
        # Dict fallback
        if isinstance(resp, dict):
            return resp.get("text") or str(resp)
    except Exception:
        logger.exception("Failed to parse LLM response")

    return ""


# ----------------------------
# Main Function
# ----------------------------
async def generate_slide_icons(
    slide_text: str,
    n_icons: int = 3,
    style: Optional[str] = None,
) -> List[Dict]:
    """
    Generate a list of icons (by keywords) for a given slide using Gemini 2.0 Exp.
    Returns a list of dicts like:
    [
        {"name": "ai_brain", "prompt": "An icon of a brain with AI circuits"},
        {"name": "data_chart", "prompt": "A simple icon showing analytics bars"},
    ]
    """

    slide_text = (slide_text or "").strip()
    if not slide_text:
        return []

    style_part = f" Style: {style}." if style else ""

    prompt = f"""
You are a presentation design assistant.
Suggest {n_icons} simple, flat icon ideas for this slide content:
---
{slide_text}
---
Each icon should be represented as 1–3 short keywords only, no sentences.
{style_part}
Return them as a comma-separated list (e.g. "robot, brain, innovation").
"""

    # Try to use internal LLMClient first
    try:
        from servers.fastapi.services.llm_client import LLMClient
        client = LLMClient()

        # Attempt to call Gemini 2.0 Exp through your unified LLM client
        resp = await client.generate(prompt)
        text = await _parse_response_text(resp)

        if not text:
            raise ValueError("Empty response from LLMClient")

    except Exception:
        logger.info("⚠️ Falling back to direct Gemini 2.0 API")

        # Fallback to direct Gemini 2.0 Exp model
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in environment")

        genai.configure(api_key=api_key)

        def _call_gemini():
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            return model.generate_content(prompt)

        resp = await _run_blocking_in_executor(_call_gemini)
        text = await _parse_response_text(resp)

    if not text:
        logger.warning("⚠️ No icons generated for slide.")
        return []

    # Parse and clean up response
    icons = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    icons = icons[:n_icons]

    results = []
    for icon in icons:
        results.append(
            {
                "name": icon.replace(" ", "_").lower(),
                "prompt": f"An icon representing '{icon}' in flat, minimal, vector style.",
            }
        )

    return results


# ----------------------------
# Batch Function (Optional)
# ----------------------------
async def generate_icons_for_slides(
    slides: List[str], n_icons_per_slide: int = 3
) -> List[List[Dict]]:
    """Generate icons for multiple slides asynchronously."""
    tasks = [generate_slide_icons(slide, n_icons=n_icons_per_slide) for slide in slides]
    return await asyncio.gather(*tasks)

