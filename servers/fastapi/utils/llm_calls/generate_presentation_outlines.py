from datetime import datetime
from typing import Optional, AsyncGenerator

from models.llm_message import LLMSystemMessage, LLMUserMessage
from servers.fastapi.services.llm_client import LLMClient
from servers.fastapi.services.icon_generator_service import generate_slide_icons
from servers.fastapi.services.web_scraper_connector import get_content_or_scrape
from utils.get_dynamic_models import get_presentation_outline_model_with_n_slides
from utils.llm_client_error_handler import handle_llm_client_exceptions
from utils.llm_provider import get_model


# -------------------------------------------------
# Prompt Builders
# -------------------------------------------------
def get_system_prompt(
    tone: Optional[str] = None,
    verbosity: Optional[str] = None,
    instructions: Optional[str] = None,
    include_title_slide: bool = True,
):
    return f"""
You are an expert presentation creator. Generate structured presentations
in the required JSON format with slide title + markdown content.

Guidelines:
- Maintain logical flow.
- No “Table of Contents” slide.
- Title slide only if enabled.
- Markdown only (no HTML).
- Use clear, crisp content.

{"- Always generate a title slide." if include_title_slide else "- Do not include a title slide."}

{tone or ""}
{verbosity or ""}
{instructions or ""}
"""


def get_user_prompt(
    content: str,
    n_slides: int,
    language: str,
    additional_context: Optional[str] = None,
):
    return f"""
Topic: {content or "Untitled Presentation"}
Language: {language}
Number of Slides: {n_slides}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Extra Context: {additional_context or "N/A"}
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
    return [
        LLMSystemMessage(
            content=get_system_prompt(
                tone, verbosity, instructions, include_title_slide
            )
        ),
        LLMUserMessage(
            content=get_user_prompt(content, n_slides, language, additional_context)
        ),
    ]


# -------------------------------------------------
# Main outline generator
# -------------------------------------------------
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
) -> AsyncGenerator[dict, None]:

    # Scrape if needed
    cleaned_content = await get_content_or_scrape(content)

    model = get_model()
    response_model = get_presentation_outline_model_with_n_slides(n_slides)

    client = LLMClient()

    try:
        # IMPORTANT FIX: remove invalid tools argument
        async for chunk in client.stream_structured(
            model=model,
            messages=get_messages(
                cleaned_content,
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
        ):

            # Attach icons
            if isinstance(chunk, dict) and "slides" in chunk:
                for slide in chunk["slides"]:
                    text = f"{slide.get('title','')}\n{slide.get('content','')}"
                    slide["icons"] = await generate_slide_icons(text, n_icons=3)

            yield chunk

    except Exception as e:
        yield handle_llm_client_exceptions(e)
