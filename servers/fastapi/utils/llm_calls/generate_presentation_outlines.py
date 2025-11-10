from datetime import datetime
from typing import Optional

from models.llm_message import LLMSystemMessage, LLMUserMessage
from models.llm_tools import SearchWebTool
from servers.fastapi.services.llm_client import LLMClient

from utils.get_dynamic_models import get_presentation_outline_model_with_n_slides
from utils.llm_client_error_handler import handle_llm_client_exceptions
from utils.llm_provider import get_model


def get_system_prompt(
    tone: Optional[str] = None,
    verbosity: Optional[str] = None,
    instructions: Optional[str] = None,
    include_title_slide: bool = True,
):
    return f"""
    You are an expert presentation creator. Generate structured presentations based on user requirements and format them according to the specified JSON schema with markdown content.

    Guidelines:
    - Ensure logical flow between slides.
    - Prioritize clarity, visuals, and meaningful structure.
    - Highlight numbers or factual data.
    - Never include any “Table of Contents” slide.
    - If additional context is given, integrate it across slides.
    - Obey user instruction completely, except if it requests a specific number of slides.
    - Generate plain markdown text only (no HTML or images).
    {"- Always make the first slide a title slide." if include_title_slide else "- Do not include a title slide."}

    {"# Tone:" if tone else ""} {tone or ""}
    {"# Verbosity:" if verbosity else ""} {verbosity or ""}
    {"# Custom Instructions:" if instructions else ""} {instructions or ""}

    **You may use SearchWebTool to fetch the latest information if needed.**
    """


def get_user_prompt(
    content: str,
    n_slides: int,
    language: str,
    additional_context: Optional[str] = None,
):
    return f"""
    **Input**
    - Topic: {content or "Untitled Presentation"}
    - Language: {language}
    - Number of Slides: {n_slides}
    - Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    - Extra Context: {additional_context or "N/A"}
    """


def get_messages(
    content: str,
    n_slides: int,
    language: str,
    additional_context: Optional[str] = None,
    tone: Optional[str] = None,
    verbosity: Optional[str] = None,
    instructions: Optional[str] = None,
    include_title_slide: bool = True,
):
    """Builds structured system + user messages for LLM."""
    return [
        LLMSystemMessage(
            content=get_system_prompt(
                tone, verbosity, instructions, include_title_slide
            ),
        ),
        LLMUserMessage(
            content=get_user_prompt(content, n_slides, language, additional_context),
        ),
    ]


async def generate_ppt_outline(
    content: str,
    n_slides: int,
    language: Optional[str] = "English",
    additional_context: Optional[str] = None,
    tone: Optional[str] = None,
    verbosity: Optional[str] = None,
    instructions: Optional[str] = None,
    include_title_slide: bool = True,
    web_search: bool = False,
):
    """
    Generates a presentation outline in structured format using the active LLM model.
    Streams the structured output chunk by chunk.
    """
    model = get_model()
    response_model = get_presentation_outline_model_with_n_slides(n_slides)
    client = LLMClient()

    try:
        async for chunk in client.stream_structured(
            model=model,
            messages=get_messages(
                content,
                n_slides,
                language,
                additional_context,
                tone,
                verbosity,
                instructions,
                include_title_slide,
            ),
            schema=response_model.model_json_schema(),
            strict=True,
            tools=(
                [SearchWebTool]
                if (client.enable_web_grounding() and web_search)
                else None
            ),
        ):
            yield chunk
    except Exception as e:
        yield handle_llm_client_exceptions(e)

