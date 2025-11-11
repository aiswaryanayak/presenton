# servers/fastapi/utils/llm_calls/generate_presentation_structure.py
from typing import Optional

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
        f"Now, assign a layout index for each of the {n_slides} slides based on purpose and tone."
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
        "from the hybrid presentation layout list."
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
    response_model = get_presentation_structure_model_with_n_slides(
        len(presentation_outline.slides)
    )

    try:
        if using_slides_markdown:
            messages = get_messages_for_slides_markdown(
                presentation_layout,
                len(presentation_outline.slides),
                presentation_outline.to_string(),
                instructions,
            )
        else:
            messages = get_messages(
                presentation_layout,
                len(presentation_outline.slides),
                presentation_outline.to_string(),
                instructions,
            )

        response = await client.generate_structured(
            model=model,
            messages=messages,
            response_format=response_model.model_json_schema(),
            strict=True,
        )

        return PresentationStructureModel(**response)

    except Exception as e:
        raise handle_llm_client_exceptions(e)
