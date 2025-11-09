from constants.llm import OPENAI_URL  # Keeping for compatibility, unused
from enums.image_provider import ImageProvider
from enums.llm_provider import LLMProvider
from utils.available_models import list_available_google_models
from utils.get_env import (
    get_can_change_keys_env,
    get_google_api_key_env,
    get_google_model_env,
    get_pexels_api_key_env,
    get_pixabay_api_key_env,
)
from utils.image_provider import get_selected_image_provider


async def check_llm_and_image_provider_api_or_model_availability():
    """
    Checks that Gemini and image providers are configured correctly.
    Simplified for Gemini-only backend (no OpenAI, Anthropic, or Ollama).
    """

    can_change_keys = get_can_change_keys_env() != "false"
    if not can_change_keys:
        # ✅ Google Gemini model check
        if get_llm_provider() == LLMProvider.GOOGLE:
            google_api_key = get_google_api_key_env()
            if not google_api_key:
                raise Exception("GOOGLE_API_KEY must be provided")

            google_model = get_google_model_env()
            if google_model:
                available_models = await list_available_google_models(google_api_key)
                if google_model not in available_models:
                    print("-" * 50)
                    print("Available Gemini models: ", available_models)
                    raise Exception(f"Model {google_model} is not available")

        # ✅ Image provider check (optional visuals)
        selected_image_provider = get_selected_image_provider()
        if not selected_image_provider:
            raise Exception("IMAGE_PROVIDER must be provided")

        if selected_image_provider == ImageProvider.PEXELS:
            pexels_api_key = get_pexels_api_key_env()
            if not pexels_api_key:
                raise Exception("PEXELS_API_KEY must be provided")

        elif selected_image_provider == ImageProvider.PIXABAY:
            pixabay_api_key = get_pixabay_api_key_env()
            if not pixabay_api_key:
                raise Exception("PIXABAY_API_KEY must be provided")

        elif selected_image_provider == ImageProvider.GEMINI_FLASH:
            google_api_key = get_google_api_key_env()
            if not google_api_key:
                raise Exception("GOOGLE_API_KEY must be provided")

    print("✅ Gemini + image provider configuration verified successfully.")
