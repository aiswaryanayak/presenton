import os
import asyncio
import dirtyjson
import json


from typing import AsyncGenerator, List, Optional
from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

# ✅ Stable Gemini import for google-generativeai 0.8.5
import google.generativeai as genai

from google.ai.generativelanguage_v1beta.types import (
    Content as GoogleContent,
    Part as GoogleContentPart,
    Tool as GoogleTool,
    ToolConfig as GoogleToolConfig,
    FunctionCallingConfig as GoogleFunctionCallingConfig,
)

# ✅ In google-generativeai 0.8.5, FunctionCallingConfigMode doesn't exist
# Replace it with a simple Enum-like string fallback
GoogleFunctionCallingConfigMode = type("GoogleFunctionCallingConfigMode", (), {
    "AUTO": "AUTO",
    "ANY": "ANY",
    "NONE": "NONE",
})


# ✅ Correct location for GenerateContentConfig
from google.generativeai.types import GenerateContentConfig




from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage
from anthropic import MessageStreamEvent as AnthropicMessageStreamEvent

from enums.llm_provider import LLMProvider
from models.llm_message import (
    AnthropicAssistantMessage,
    AnthropicUserMessage,
    GoogleAssistantMessage,
    GoogleToolCallMessage,
    OpenAIAssistantMessage,
    LLMMessage,
    LLMSystemMessage,
    LLMUserMessage,
)
from models.llm_tool_call import (
    AnthropicToolCall,
    GoogleToolCall,
    LLMToolCall,
    OpenAIToolCall,
    OpenAIToolCallFunction,
)
from models.llm_tools import LLMDynamicTool, LLMTool
from services.llm_tool_calls_handler import LLMToolCallsHandler
from utils.async_iterator import iterator_to_async
from utils.dummy_functions import do_nothing_async
from utils.get_env import (
    get_anthropic_api_key_env,
    get_custom_llm_api_key_env,
    get_custom_llm_url_env,
    get_disable_thinking_env,
    get_google_api_key_env,
    get_ollama_url_env,
    get_openai_api_key_env,
    get_tool_calls_env,
    get_web_grounding_env,
)
from utils.llm_provider import get_llm_provider, get_model
from utils.parsers import parse_bool_or_none
from utils.schema_utils import (
    ensure_strict_json_schema,
    flatten_json_schema,
    remove_titles_from_schema,
)

