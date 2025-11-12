# servers/fastapi/utils/llm_calls/generate_presentation_structure.py
from typing import Optional, Any
import json, re, traceback

from servers.fastapi.models.llm_message import LLMSystemMessage, LLMUserMessage
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import (
    HybridPresentonLayout as PresentationLayoutModel,
)
from servers.fastapi.models.presentation_outline_model import PresentationOutlineModel
from servers.fastapi.models.presentation_structure_model import PresentationStructureModel
from servers.fastapi.services.llm_client import LLMClient
from servers.fastapi.utils.llm_client_error_handler import handle_llm_client_exceptions
from servers.fastapi.utils.llm_provider import get_model
from servers.fastapi.utils.get_dynamic_models import get_presentation_structure_model_with_n_slides


# ==========================================================
# 🧩 Safe Outline → String Converter (Prevents type errors)
# ==========================================================
def _outline_to_string(presentation_outline: PresentationOutlineModel) -> str:
    """
    Converts the presentation outline (which may contain dicts, lists, or objects)
    into a safe, readable string for LLM input.
    """
    slides_data = getattr(presentation_outline, "slides", [])
    text_parts = []

    for i, slide in enumerate(slides_data):
        try:
            # --- handle pydantic models, dicts, or raw strings ---
            if isinstance(slide, dict):
                content = slide.get("content") or slide.get("text") or str(slide)
            elif hasattr(slide, "content"):
                content = getattr(slide, "content")
            else:
                content = str(slide)

            # --- flatten lists or dicts into readable JSON text ---
            if isinstance(content, (list, dict)):
                content = json.dumps(content, ensure_ascii=False)

            text_parts.append(f"## Slide {i+1}\n{content}")

        except Exception as e:
            print(f"⚠️ Failed to serialize slide {i}: {e}")
            traceback.print_exc()
            text_parts.append(f"## Slide {i+1}\n[Unserializable Slide]")

    return "\n\n".join(text_parts)


# ==========================================================
# 🧠 Message Builders
# ==========================================================
def get_messages(
    presentation_layout: PresentationLayoutModel,
    n_slides: int,
    data: str,
    instructions: Optional[str] = None,
):
    """Builds system + user messages for structure generation."""
    user_instructions = f"# User Instruction:\n{instructions}" if instructions else ""

    system_prompt = (
        "You are a professional AI presentation designer generating structured hybrid-style slides.\n\n"
        f"{presentation_layout.to_string()}\n\n"
        "# OUTPUT RULES:\n"
        "- Return valid JSON with key 'slides'.\n"
        "- Each slide must have:\n"
        "  { 'title': str, 'bullets': list[str], 'layout_id': int, "
        "'visuals': list[str]|None, 'chart_type': str|None, 'summary': str|None }\n"
        "- Avoid returning plain integers or strings.\n"
        "- Match layout_id with the type of content.\n"
        f"- Number of slides: {n_slides}\n\n"
        f"{user_instructions}"
    )

    return [
        LLMSystemMessage(content=system_prompt),
        LLMUserMessage(content=data),
    ]


def get_messages_for_slides_markdown(
    presentation_layout: PresentationLayoutModel,
    n_slides: int,
    data: str,
    instructions: Optional[str] = None,
):
    """Same as get_messages() but for prewritten markdown slides."""
    user_instructions = f"# User Instruction:\n{instructions}" if instructions else ""

    system_prompt = (
        "You're an expert creating slide structures for pre-written markdown slides.\n\n"
        f"{presentation_layout.to_string()}\n\n"
        f"{user_instructions}\n\n"
        f"Select an appropriate layout for each of the {n_slides} slides and return valid JSON matching this schema."
    )

    return [
        LLMSystemMessage(content=system_prompt),
        LLMUserMessage(content=data),
    ]


# ==========================================================
# ⚙️ Main Structure Generation Function
# ==========================================================
async def generate_presentation_structure(
    presentation_outline: PresentationOutlineModel,
    presentation_layout: PresentationLayoutModel,
    instructions: Optional[str] = None,
    using_slides_markdown: bool = False,
) -> PresentationStructureModel:
    """Generates slide → layout mapping using hybrid layout."""
    client = LLMClient()
    model = get_model()

    n_slides = len(getattr(presentation_outline, "slides", []))
    response_model = get_presentation_structure_model_with_n_slides(n_slides)

    try:
        # --- Step 1: Convert outline safely into plain text ---
        outline_text = _outline_to_string(presentation_outline)

        # --- Step 2: Choose message builder based on mode ---
        if using_slides_markdown:
            messages = get_messages_for_slides_markdown(
                presentation_layout, n_slides, outline_text, instructions
            )
        else:
            messages = get_messages(
                presentation_layout, n_slides, outline_text, instructions
            )

        # --- Step 3: Generate structured LLM output ---
        response = await client.generate_structured(
            model=model,
            messages=messages,
            response_format=response_model.model_json_schema(),
            strict=True,
        )

        # --- Step 4: Parse + normalize response ---
        parsed = None
        if isinstance(response, dict):
            parsed = response
        elif isinstance(response, str):
            try:
                parsed = json.loads(response)
            except Exception:
                match = re.search(r"(\{.*\})", response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
        else:
            parsed = json.loads(json.dumps(response))

        if not parsed or "slides" not in parsed:
            raise ValueError("LLM returned malformed or empty response")

        slides = parsed["slides"]
        if not isinstance(slides, list):
            slides = [slides]

        # --- Step 5: Ensure structure integrity ---
        normalized_slides = []
        for i, s in enumerate(slides):
            if isinstance(s, dict):
                normalized_slides.append({
                    "title": s.get("title", f"Slide {i+1}"),
                    "bullets": s.get("bullets", []),
                    "layout_id": s.get("layout_id", i + 1),
                    "visuals": s.get("visuals", []),
                    "chart_type": s.get("chart_type"),
                    "summary": s.get("summary", ""),
                })
            elif isinstance(s, str):
                normalized_slides.append({
                    "title": s,
                    "bullets": [],
                    "layout_id": i + 1,
                    "visuals": [],
                    "chart_type": None,
                    "summary": "",
                })
            else:
                normalized_slides.append({
                    "title": f"Slide {i+1}",
                    "bullets": [],
                    "layout_id": i + 1,
                    "visuals": [],
                    "chart_type": None,
                    "summary": "",
                })

        return PresentationStructureModel(slides=normalized_slides)

    except Exception as e:
        print("🔥 Error in generate_presentation_structure:", repr(e))
        traceback.print_exc()
        raise handle_llm_client_exceptions(e)
