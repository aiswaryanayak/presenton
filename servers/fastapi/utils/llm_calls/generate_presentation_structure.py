# servers/fastapi/utils/llm_calls/generate_presentation_structure.py
from typing import Optional, List, Any
import json, re

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


def get_messages(
    presentation_layout: PresentationLayoutModel,
    n_slides: int,
    data: str,
    instructions: Optional[str] = None,
):
    """Build LLM messages for structure generation."""
    user_instructions = f"# User Instruction:\n{instructions}" if instructions else ""
    system_prompt = (
        "You are a professional AI presentation designer generating structured hybrid-style slides.\n\n"
        f"{presentation_layout.to_string()}\n\n"
        "# OUTPUT RULES:\n"
        "- Return valid JSON with key 'slides'.\n"
        "- Each slide must have:\n"
        "  { 'title': str, 'bullets': list[str], 'layout_id': int, 'visuals': list[str]|None, 'chart_type': str|None, 'summary': str|None }\n"
        "- Avoid returning plain integers or strings.\n"
        "- Match layout_id with the type of content.\n"
        "- Number of slides: {n_slides}\n\n"
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
        f"Select an appropriate layout for each of the {n_slides} slides and return valid JSON matching the schema."
    )

    return [
        LLMSystemMessage(content=system_prompt),
        LLMUserMessage(content=data),
    ]


async def generate_presentation_structure(
    presentation_outline: PresentationOutlineModel,
    presentation_layout: PresentationLayoutModel,
    instructions: Optional[str] = None,
    using_slides_markdown: bool = False,
) -> PresentationStructureModel:
    """Generate the slide → layout mapping using hybrid layout and fix malformed responses."""
    client = LLMClient()
    model = get_model()
    response_model = get_presentation_structure_model_with_n_slides(
        len(presentation_outline.slides)
    )

    try:
        messages = (
            get_messages_for_slides_markdown(
                presentation_layout,
                len(presentation_outline.slides),
                presentation_outline.to_string(),
                instructions,
            )
            if using_slides_markdown
            else get_messages(
                presentation_layout,
                len(presentation_outline.slides),
                presentation_outline.to_string(),
                instructions,
            )
        )

        # --- Step 1: call the LLM client safely ---
        try:
            raw_response = await client.stream_structured(
                model=model,
                messages=messages,
            ).__anext__()  # stream returns async generator
        except AttributeError:
            # fallback for older structure (non-stream)
            result = await client.generate(
                prompt=messages[-1].content,
                model=model,
            )
            raw_response = {"text": result}

        text_response = (
            raw_response.get("text")
            if isinstance(raw_response, dict)
            else str(raw_response)
        )

        # --- Step 2: parse possible JSON ---
        parsed = None
        if isinstance(text_response, str):
            match = re.search(r"(\{.*\}|\[.*\])", text_response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None

        # fallback if parsing failed
        if not parsed:
            parsed = {"slides": []}

        # --- Step 3: normalize malformed slide data ---
        def normalize_slide(item: Any, idx: int):
            """Ensures each slide matches SlideContent schema."""
            if isinstance(item, int):
                return {
                    "title": f"Slide {idx+1}",
                    "bullets": [],
                    "layout_id": item,
                    "visuals": [],
                    "chart_type": None,
                    "summary": "",
                }
            if isinstance(item, str):
                return {
                    "title": item.strip() or f"Slide {idx+1}",
                    "bullets": [],
                    "layout_id": idx + 1,
                    "visuals": [],
                    "chart_type": None,
                    "summary": "",
                }
            if isinstance(item, dict):
                return {
                    "title": item.get("title", f"Slide {idx+1}"),
                    "bullets": item.get("bullets", []),
                    "layout_id": item.get("layout_id", idx + 1),
                    "visuals": item.get("visuals", []),
                    "chart_type": item.get("chart_type"),
                    "summary": item.get("summary", ""),
                }
            return {
                "title": f"Slide {idx+1}",
                "bullets": [],
                "layout_id": idx + 1,
                "visuals": [],
                "chart_type": None,
                "summary": "",
            }

        slides = parsed.get("slides", []) if isinstance(parsed, dict) else parsed
        if not isinstance(slides, list):
            slides = [slides]

        normalized_slides = [normalize_slide(s, i) for i, s in enumerate(slides)]

        # fill missing slides if the model gave fewer
        while len(normalized_slides) < len(presentation_outline.slides):
            normalized_slides.append(normalize_slide("", len(normalized_slides)))

        final_data = {"slides": normalized_slides}

        return PresentationStructureModel(**final_data)

    except Exception as e:
        raise handle_llm_client_exceptions(e)

