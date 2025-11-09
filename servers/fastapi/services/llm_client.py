import os
import asyncio
import dirtyjson
import json
from typing import AsyncGenerator, List, Optional
from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from google import genai
from google.genai.types import (
    Content as GoogleContent,
    Part as GoogleContentPart,
    GenerateContentConfig,
    GoogleSearch,
    ToolConfig as GoogleToolConfig,
    FunctionCallingConfig as GoogleFunctionCallingConfig,
    FunctionCallingConfigMode as GoogleFunctionCallingConfigMode,
    Tool as GoogleTool,
)
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


class LLMClient:
    def __init__(self):
        self.llm_provider = get_llm_provider()
        self._client = self._get_client()
        self.tool_calls_handler = LLMToolCallsHandler(self)

    # --------------------------------------- #
    # Client setup
    # --------------------------------------- #
    def _get_client(self):
        match self.llm_provider:
            case LLMProvider.OPENAI:
                return self._get_openai_client()
            case LLMProvider.GOOGLE:
                return self._get_google_client()
            case LLMProvider.ANTHROPIC:
                return self._get_anthropic_client()
            case LLMProvider.OLLAMA:
                return self._get_ollama_client()
            case LLMProvider.CUSTOM:
                return self._get_custom_client()
            case _:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid LLM Provider — must be one of: openai, google, anthropic, ollama, custom.",
                )

    def _get_openai_client(self):
        key = get_openai_api_key_env()
        if not key:
            raise HTTPException(status_code=400, detail="OpenAI API Key is missing.")
        return AsyncOpenAI(api_key=key)

    def _get_google_client(self):
        """Initialize Gemini (Google GenAI) client with proper API key."""
        google_api_key = get_google_api_key_env()
        if not google_api_key:
            raise HTTPException(status_code=400, detail="Google API Key is not set.")
        # Explicitly set env so Google SDK finds it even on Render
        os.environ["GOOGLE_API_KEY"] = google_api_key
        return genai.Client(api_key=google_api_key)

    def _get_anthropic_client(self):
        key = get_anthropic_api_key_env()
        if not key:
            raise HTTPException(status_code=400, detail="Anthropic API Key is missing.")
        return AsyncAnthropic(api_key=key)

    def _get_ollama_client(self):
        return AsyncOpenAI(
            base_url=(get_ollama_url_env() or "http://localhost:11434") + "/v1",
            api_key="ollama",
        )

    def _get_custom_client(self):
        url = get_custom_llm_url_env()
        if not url:
            raise HTTPException(status_code=400, detail="Custom LLM URL is not set.")
        return AsyncOpenAI(base_url=url, api_key=get_custom_llm_api_key_env() or "null")

    # --------------------------------------- #
    # Utility toggles
    # --------------------------------------- #
    def use_tool_calls_for_structured_output(self) -> bool:
        if self.llm_provider != LLMProvider.CUSTOM:
            return False
        return parse_bool_or_none(get_tool_calls_env()) or False

    def enable_web_grounding(self) -> bool:
        if self.llm_provider in [LLMProvider.OLLAMA, LLMProvider.CUSTOM]:
            return False
        return parse_bool_or_none(get_web_grounding_env()) or False

    def disable_thinking(self) -> bool:
        return parse_bool_or_none(get_disable_thinking_env()) or False

    # --------------------------------------- #
    # Helpers
    # --------------------------------------- #
    def _get_system_prompt(self, messages: List[LLMMessage]) -> str:
        for m in messages:
            if isinstance(m, LLMSystemMessage):
                return m.content
        return ""

    def _get_google_messages(self, messages: List[LLMMessage]) -> List[GoogleContent]:
        contents = []
        for message in messages:
            if isinstance(message, LLMUserMessage):
                contents.append(
                    GoogleContent(
                        role=message.role,
                        parts=[GoogleContentPart(text=message.content)],
                    )
                )
            elif isinstance(message, GoogleAssistantMessage):
                contents.append(message.content)
            elif isinstance(message, GoogleToolCallMessage):
                contents.append(
                    GoogleContent(
                        role="user",
                        parts=[
                            GoogleContentPart.from_function_response(
                                name=message.name, response=message.response
                            )
                        ],
                    )
                )
        return contents

    def _get_anthropic_messages(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        return [
            msg for msg in messages if not isinstance(msg, LLMSystemMessage)
        ]

    # --------------------------------------- #
    # Generate — Core Functionality
    # --------------------------------------- #
    async def generate(
        self,
        model: str,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None,
    ):
        parsed_tools = self.tool_calls_handler.parse_tools(tools)
        content = None

        match self.llm_provider:
            case LLMProvider.OPENAI:
                content = await self._generate_openai(
                    model, messages, max_tokens, parsed_tools
                )
            case LLMProvider.GOOGLE:
                content = await self._generate_google(
                    model, messages, parsed_tools, max_tokens
                )
            case LLMProvider.ANTHROPIC:
                content = await self._generate_anthropic(
                    model, messages, max_tokens, parsed_tools
                )
            case LLMProvider.OLLAMA:
                content = await self._generate_openai(
                    model, messages, max_tokens, parsed_tools
                )
            case LLMProvider.CUSTOM:
                content = await self._generate_openai(
                    model, messages, max_tokens, parsed_tools
                )

        if not content:
            raise HTTPException(status_code=400, detail="LLM did not return any content.")
        return content

    # --------------------------------------- #
    # Google (Gemini) Generate Function
    # --------------------------------------- #
    async def _generate_google(
        self,
        model: str,
        messages: List[LLMMessage],
        tools: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        depth: int = 0,
    ) -> str | None:
        client: genai.Client = self._client

        google_tools = [GoogleTool(function_declarations=[t]) for t in tools] if tools else None

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=self._get_google_messages(messages),
            config=GenerateContentConfig(
                tools=google_tools,
                system_instruction=self._get_system_prompt(messages),
                response_mime_type="text/plain",
                max_output_tokens=max_tokens,
            ),
        )

        content = response.candidates[0].content if response.candidates else None
        if not content or not content.parts:
            return None

        text_content = None
        for part in content.parts:
            if part.text:
                text_content = part.text
        return text_content or None

    # --------------------------------------- #
    # Web Search (Gemini)
    # --------------------------------------- #
    async def _search_google(self, query: str) -> str:
        client: genai.Client = self._client
        grounding_tool = GoogleTool(google_search=GoogleSearch())
        config = GenerateContentConfig(tools=[grounding_tool])

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=get_model(),
            contents=query,
            config=config,
        )
        return response.text or ""
