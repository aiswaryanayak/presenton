# servers/fastapi/utils/llm_calls/generate_slide_content.py

from typing import Optional

from servers.fastapi.models.llm_message import LLMSystemMessage, LLMUserMessage
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import (
    HybridPresentonLayout as PresentationLayoutModel,
)
from servers.fastapi.models.presentation_structure_model import SlideLayout
from servers.fastapi.services.llm_client import LLMClient
from servers.fastapi.utils.llm_provider import get_model
from servers.fastapi.utils.llm_client_error_handler import handle_llm_client_exceptions


def get_messages(slide_outline: str, slide_type: str, layout: SlideLayout, style: str):
    """Create the LLM message for generating a single slide’s detailed content."""
    return [
        LLMSystemMessage(
            content=(
                f"You are an expert presentation slide generator.\n\n"
                f"Slide Type: {slide_type}\n"
                f"Layout: {layout.type} ({layout.style})\n"
                f"Color Scheme: {layout.color_scheme or 'Default'}\n"
                f"Visual Focus: {layout.visual or 'Mixed'}\n\n"
                "# OBJECTIVE\n"
                "Generate concise, high-impact bullet points for this slide, "
                "keeping tone aligned with professional presentation design.\n\n"
                f"Follow hybrid aesthetics — mix gradient hero styles, clean text visuals, "
                f"and modern contrast."
            ),
        ),
        LLMUserMessage(content=f"Outline: {slide_outline}"),
    ]


async def get_slide_content_from_type_and_outline(
    slide_outline: str,
    slide_type: str,
    layout: Optional[SlideLayout] = None,
    style: Optional[str] = "hybrid-modern",
):
    """Generate full slide content based on layout, type, and outline."""
    client = LLMClient()
    model = get_model()

    if layout is None:
        layout = SlideLayout(
            id=1, type="default", style="modern", color_scheme="blue-gradient"
        )

    try:
        messages = get_messages(slide_outline, slide_type, layout, style)
        response = await client.generate_text(model=model, messages=messages)
        return response.strip()

    except Exception as e:
        raise handle_llm_client_exceptions(e)

