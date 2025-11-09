from typing import Any, List, Literal, Optional
from pydantic import BaseModel
import google.generativeai as genai

from models.llm_tool_call import AnthropicToolCall


class LLMMessage(BaseModel):
    """Base class for all LLM message types."""
    pass


class LLMUserMessage(LLMMessage):
    role: Literal["user"] = "user"
    content: str


class LLMSystemMessage(LLMMessage):
    role: Literal["system"] = "system"
    content: str


class OpenAIAssistantMessage(LLMMessage):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None


class GoogleAssistantMessage(LLMMessage):
    """Assistant message format for Google Gemini."""
    role: Literal["assistant"] = "assistant"
    content: genai.types.Content  # ✅ Updated import


class AnthropicAssistantMessage(LLMMessage):
    role: Literal["assistant"] = "assistant"
    content: List[AnthropicToolCall]


class AnthropicToolCallMessage(LLMMessage):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str


class AnthropicUserMessage(LLMMessage):
    role: Literal["user"] = "user"
    content: List[AnthropicToolCallMessage]


class OpenAIToolCallMessage(LLMMessage):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


class GoogleToolCallMessage(LLMMessage):
    role: Literal["tool"] = "tool"
    id: Optional[str] = None
    name: str
    response: dict
