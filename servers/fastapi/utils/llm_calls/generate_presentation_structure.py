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
    """
    Build prompt messages for the LLM to determine layout index per slide.
    """
    return [
        LLMSystemMessage(
            content=f"""
You're a professional presentation designer with creative freedom to design visually engaging, high-impact slides.

{presentation_layout.to_string()}

# DESIGN PRINCIPLES
- Combine **visual storytelling** and **data clarity**
- Use hybrid styles: gradient hero slides, charts, clean text visuals
- Alternate between text, image, and chart slides for rhythm
- Never make all slides identical in layout

# LAYOUT SELECTION RULES
1. **Match layout to content purpose**:
   - Title → hero layout
   - Problem → image-left-text-right
   - Solution → split-modern
   - Market → chart or data-heavy layout
   - Team → photo grid
   - Roadmap → timeline-modern
   - CTA → bold center-cta-gradient

2. **Ensure flow and variety**:
   - Early slides introduce, middle explain, end inspires action
   - Mix visual density naturally across slides

3. **Audience engagement**:
   - Prioritize readability, color balance, and energy
   - Use hybrid visuals (gradient + image + clean data)

{f"# User Instruction:\n{instructions}" if instructions else ""}
                
Now, assign a layout index for each of the {n_slides} slides based on purpose and tone.
"""
        ),
        LLMUserMessage(content=data),
    ]


def get_messages_for_slides_markdown(
    presentation_layout: PresentationLayoutModel,
    n_slides: int,
    data: str,
    instructions: Optional[str] = None,
):
    """
    Same as get_messages() but optimized for user-provided markdown slides.
    """
    return [
        LLMSystemMessage(
            content=f"""
You're a presentation design expert creating a structure for pre-written slides.

{presentation_layout.to_string()}

{f"# User Instruction:\n{instructions}" if instructions else ""}

Select the most appropriate layout for each of the {n_slides} slides from the hybrid presentation layout list.
"""
        ),
        LLMUserMessage(content=data),
    ]


async def generate_presentation_structure(
    presentation_outline: PresentationOutlineModel,
    presentation_layout: PresentationLayoutModel,
    instructions: Optional[str] = None,
    using_slides_markdown: bool = False,
) -> PresentationStructureModel:
    """
    Generate a structured mapping of slides → layout indexes using the Hybrid Layout.
    """
    client = LLMClient()
    model = get_model()
    response_model = get_presentation_structure_model_with_n_slides(
        len(presentation_outline.slides)
    )

    try:
        response = await client.generate_structured(
            model=model,
            messages=(
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
            ),
            response_format=response_model.model_json_schema(),
            strict=True,
        )
        return PresentationStructureModel(**response)
    except Exception as e:
        raise handle_llm_client_exceptions(e)

