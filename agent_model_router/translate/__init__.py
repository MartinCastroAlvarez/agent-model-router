"""Translation between the Anthropic Messages and OpenAI Chat Completions dialects."""

from agent_model_router.translate.requests import anthropic_to_openai_request
from agent_model_router.translate.requests import openai_to_anthropic_request
from agent_model_router.translate.responses import anthropic_to_openai_response
from agent_model_router.translate.responses import openai_to_anthropic_response
from agent_model_router.translate.streams import make_adapter

__all__ = [
    "anthropic_to_openai_request",
    "anthropic_to_openai_response",
    "make_adapter",
    "openai_to_anthropic_request",
    "openai_to_anthropic_response",
]
