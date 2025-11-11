from datetime import datetime
from typing import Optional
from servers.fastapi.models.llm_message import LLMSystemMessage, LLMUserMessage
from servers.fastapi.models.presentation_layout.hybrid_presenton_layout import (
    HybridPresentonLayout as PresentationLayoutModel,
)
from servers.fastapi.models.presentation_structure_model import SlideLayout
from servers.fastapi.models.sql.slide import SlideModel
from servers.fastapi.services.llm_client import LLMClient
from servers.fastapi.utils.llm_client_error_handler import handle_llm_client_exceptions
from servers.fastapi.utils.llm_provider import get_model
from servers.fastapi.utils.schema_utils import add_field_in_schema, remove_fields_from_schema


def get_system_prompt(
    tone: Optional[str] = None,
    verbosity: Optional[str] = None,
    instructions: Optional[str] = None,
):
    """Generate a system-level instruction prompt for editing a slide."""
    return f"""
    Edit slide data and speaker notes based on the provided prompt. 
    Follow all mentioned steps and maintain the structure and clarity.

    {"# User Instruction:" if instructions else ""}
    {instructions or ""}

    {"# Tone:" if tone else ""}
    {tone or ""}

    {"# Verbosity:" if verbosity else ""}
    {verbosity or ""}

    # Notes
    - Output must be in the same language as the input slide.
    - Modify only requested fields.
    - Keep **Image prompts** and **Icon queries** unchanged unless explicitly asked to modify.
    - Generate **Image prompts** and **Icon queries** only if prompted.
    - Speaker notes must be clear, concise, and non-markdown.
    - Follow all structure and content rules.
    """


def get_user_prompt(prompt: str, slide_data: dict, language: str):
    """Prepare user message with context for the LLM."""
    return f"""
        ## Icon Query And Image Prompt Language
        English

        ## Current Date and Time
        {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        ## Slide Content Language
        {language}

        ## Prompt
        {prompt}

        ## Slide data
        {slide_data}
    """


def get_messages(
    prompt: str,
    slide_data: dict,
    language: str,
    tone: Optional[str] = None,
    verbosity: Optional[str] = None,
    instructions: Optional[str] = None,
):
    """Generate the list of system and user messages for the LLM call."""
    return [
        LLMSystemMessage(content=get_system_prompt(tone, verbosity, instructions)),
        LLMUserMessage(content=get_user_prompt(prompt, slide_data, language)),
    ]


async def get_edited_slide_content(
    prompt: str,
    slide: SlideModel,
    language: str,
    slide_layout: SlideLayout,
    tone: Optional[str] = None,
    verbosity: Optional[str] = None,
    instructions: Optional[str] = None,
):
    """Generate the edited slide content using the LLM."""
    model = get_model()

    # Prepare schema (remove image/icon URLs, add speaker note)
    response_schema = remove_fields_from_schema(
        getattr(slide_layout, "json_schema", {}), ["__image_url__", "__icon_url__"]
    )
    response_schema = add_field_in_schema(
        response_schema,
        {
            "__speaker_note__": {
                "type": "string",
                "minLength": 100,
                "maxLength": 250,
                "description": "Speaker note for the slide",
            }
        },
        True,
    )

    client = LLMClient()

    try:
        response = await client.generate_structured(
            model=model,
            messages=get_messages(
                prompt, slide.content, language, tone, verbosity, instructions
            ),
            response_format=response_schema,
            strict=False,
        )
        return response

    except Exception as e:
        raise handle_llm_client_exceptions(e)

