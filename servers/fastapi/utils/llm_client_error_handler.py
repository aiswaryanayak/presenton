from fastapi import HTTPException
import traceback

# ✅ OpenAI error import (always available)
try:
    from openai import APIError as OpenAIAPIError
except ImportError:
    class OpenAIAPIError(Exception):
        pass

# ✅ Anthropic error import
try:
    from anthropic import APIError as AnthropicAPIError
except ImportError:
    class AnthropicAPIError(Exception):
        pass

# ✅ Google Generative AI error import — removed in 0.8.x
try:
    from google.generativeai.types import APIError as GoogleAPIError
except ImportError:
    # Fallback definition for newer versions (>= 0.8.x)
    class GoogleAPIError(Exception):
        """Fallback for google.generativeai.types.APIError (removed upstream)."""
        pass


def handle_llm_client_exceptions(e: Exception) -> HTTPException:
    """
    Convert known LLM provider API errors into consistent HTTP exceptions.
    Ensures FastAPI returns a clear 500 response for all LLM-related issues.
    """
    traceback.print_exc()

    # Handle OpenAI-specific errors
    if isinstance(e, OpenAIAPIError):
        return HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    # Handle Google Gemini-specific errors
    if isinstance(e, GoogleAPIError):
        return HTTPException(status_code=500, detail=f"Google API error: {str(e)}")

    # Handle Anthropic-specific errors
    if isinstance(e, AnthropicAPIError):
        return HTTPException(status_code=500, detail=f"Anthropic API error: {str(e)}")

    # Catch-all fallback
    return HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")
