# servers/fastapi/utils/llm_calls/select_slide_type_on_edit.py

from typing import Optional
from servers.fastapi.models.llm_message import LLMSystemMessage, LLMUserMessage
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import (
    HybridPresentonLayout as PresentationLayoutModel,
)
from servers.fastapi.models.presentation_structure_model import SlideLayout
from servers.fastapi.models.sql.slide import SlideModel
from servers.fastapi.services.llm_client import LLMClient
from servers.fastapi.utils.llm_provider import get_model
from servers.fastapi.utils.llm_client_error_handler import handle_llm_client_exceptions


def get_messages_for_layout_selection(
    prompt: str,
    slide: SlideModel,
    layout: PresentationLayoutModel,
):
    """Generate messages for selecting the best slide layout after editing."""
    return [
        LLMSystemMessage(
            content=(
                "You are a professional presentation layout designer.\n"
                "Given the user's edit request and the slide content, "
                "select the most appropriate layout from the provided options.\n"
                "Match layout type, visuals, and theme appropriately."
            )
        ),
        LLMUserMessage(
            content=f"""
                # User Edit Prompt
                {prompt}

                # Current Slide Content
                {slide.content}

                # Available Layouts
                {[l.name for l in layout.slides]}
            """
        ),
    ]


async def get_slide_layout_from_prompt(
    prompt: str,
    layout: PresentationLayoutModel,
    slide: SlideModel,
) -> SlideLayout:
    """Use LLM to decide which layout best fits the edited slide content."""
    client = LLMClient()
    model = get_model()

    try:
        messages = get_messages_for_layout_selection(prompt, slide, layout)
        response = await client.generate_text(model=model, messages=messages)

        # Match layout name from response
        matched = None
        for l in layout.slides:
            if l.name.lower() in response.lower():
                matched = l
                break

        # Default fallback
        return matched or layout.slides[0]

    except Exception as e:
        raise handle_llm_client_exceptions(e)

