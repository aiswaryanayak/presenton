# servers/fastapi/utils/llm_calls/generate_presentation_structure.py
from typing import Optional
import json
import re
import traceback

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


def _outline_to_string(presentation_outline: PresentationOutlineModel) -> str:
    """
    Convert PresentationOutlineModel into a stable string for LLM input.
    Uses provided to_string() if available, otherwise builds a safe fallback.
    """
    try:
        if hasattr(presentation_outline, "to_string"):
            s = presentation_outline.to_string()
            if isinstance(s, str) and s.strip():
                return s
    except Exception:
        # continue to fallback
        traceback.print_exc()

    # fallback: try to serialize slides content safely
    pieces = []
    for i, slide in enumerate(getattr(presentation_outline, "slides", []) or []):
        try:
            # slide may be a pydantic model, dict or string
            if hasattr(slide, "content"):
                content = slide.content
            elif isinstance(slide, dict):
                content = slide.get("content", "")
            else:
                content = str(slide)
            if isinstance(content, (list, dict)):
                content = json.dumps(content, ensure_ascii=False)
        except Exception:
            content = str(slide)
        pieces.append(f"## Slide {i+1}:\n{content}")
    return "\n\n".join(pieces) if pieces else ""


def get_messages(
    presentation_layout: PresentationLayoutModel,
    n_slides: int,
    data: str,
    instructions: Optional[str] = None,
):
    """Build LLM messages for structure generation."""
    user_instructions = ""
    if instructions:
        user_instructions = "# User Instruction:\n" + instructions

    system_prompt = (
        "You're a professional presentation designer creating hybrid-style decks "
        "that blend modern visuals and clean general layouts.\n\n"
        f"{presentation_layout.to_string()}\n\n"
        "# DESIGN PRINCIPLES\n"
        "- Combine **visual storytelling** and **data clarity**\n"
        "- Use hybrid styles: gradient hero slides, charts, clean text visuals\n"
        "- Alternate between text, image, and chart slides for rhythm\n"
        "- Never make all slides identical in layout\n\n"
        "# LAYOUT SELECTION RULES\n"
        "1. Match layout to content purpose:\n"
        "   - Title → hero layout\n"
        "   - Problem → image-left-text-right\n"
        "   - Solution → split-modern\n"
        "   - Market → chart or data-heavy layout\n"
        "   - Team → photo grid\n"
        "   - Roadmap → timeline-modern\n"
        "   - CTA → center-cta-gradient\n\n"
        "2. Ensure flow and variety across slides.\n"
        "3. Prioritize clarity, color balance, and audience engagement.\n\n"
        f"{user_instructions}\n\n"
        f"Now, assign a layout index for each of the {n_slides} slides based on purpose and tone.\n\n"
        "Output JSON with a top-level `slides` array, where each item contains:\n"
        "  - title (string)\n"
        "  - layout_id (int)\n"
        "  - bullets (list of strings)\n"
        "  - visuals (optional list of strings)\n"
        "Return valid JSON only."
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
    user_instructions = ""
    if instructions:
        user_instructions = "# User Instruction:\n" + instructions

    system_prompt = (
        "You're a presentation design expert creating a structure for pre-written slides.\n\n"
        f"{presentation_layout.to_string()}\n\n"
        f"{user_instructions}\n\n"
        f"Select the most appropriate layout for each of the {n_slides} slides "
        "from the hybrid presentation layout list.\n\n"
        "Output JSON with a top-level `slides` array where each entry contains:\n"
        "  - title (string)\n"
        "  - layout_id (int)\n"
        "  - bullets (list of strings)\n"
        "Return valid JSON only."
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
    """Generate the slide → layout mapping using the hybrid layout."""
    client = LLMClient()
    model = get_model()

    n_slides = len(getattr(presentation_outline, "slides", []) or [])
    if n_slides <= 0:
        n_slides = 10

    # dynamic expected schema helper (keeps compatibility with existing code)
    response_model = get_presentation_structure_model_with_n_slides(n_slides)
    response_schema = response_model.model_json_schema() if hasattr(response_model, "model_json_schema") else {}

    try:
        # prepare messages
        data_string = _outline_to_string(presentation_outline)
        if using_slides_markdown:
            messages = get_messages_for_slides_markdown(
                presentation_layout, n_slides, data_string, instructions
            )
        else:
            messages = get_messages(presentation_layout, n_slides, data_string, instructions)

        # ask the LLM for structured output
        response = await client.generate_structured(
            model=model,
            messages=messages,
            response_format=response_schema,
            strict=True,
        )

        # Response may already be a dict (parsed), or a JSON string.
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
            except Exception:
                m = re.search(r"(\{.*\})", response, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(1))
                else:
                    raise ValueError("LLM returned non-JSON response")
        elif isinstance(response, dict):
            parsed = response
        else:
            # try convert complex objects
            try:
                parsed = json.loads(json.dumps(response))
            except Exception:
                raise ValueError("Unsupported response type from LLM")

        # Validate essential shape
        if not isinstance(parsed, dict) or "slides" not in parsed or not isinstance(parsed["slides"], list):
            raise ValueError("Parsed LLM response does not contain required 'slides' list")

        # Build and return the Pydantic model
        return PresentationStructureModel(**parsed)

    except Exception as e:
        # Bubble to the centralized handler with enriched context
        print("🔥 DEBUG: generate_presentation_structure failed:", repr(e))
        traceback.print_exc()
        raise handle_llm_client_exceptions(e)


