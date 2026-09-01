"""LLM factory for the GWDG academic-cloud OpenAI-compatible backend."""

import os
from typing import List, Any

import litellm
import google.adk.models.lite_llm as adk_lite_llm
from google.adk.models.lite_llm import LiteLlm
from litellm import ChatCompletionAssistantMessage

from machine_learning_engineering.shared_libraries import config, xai_tracker

# Register the per-task usage callback with litellm once at import time. Both the
# ADK LiteLlm agents and the direct complete_text() calls below route through
# litellm, so this single hook captures every LLM call's tokens and latency.
xai_tracker.install_litellm_logger()


# Monkey-patch _ensure_tool_results in google-adk to heal tool-user role sequences.
# OpenAI/GWDG APIs reject sequences where a 'user' message immediately follows a 'tool'
# response. Injecting an intermediate 'assistant' message solves this validation issue.
_original_ensure_tool_results = adk_lite_llm._ensure_tool_results

def _patched_ensure_tool_results(messages: List[Any], model: str) -> List[Any]:
    healed = _original_ensure_tool_results(messages, model)
    if not healed:
        return healed

    expected_tool_role = "tool_responses" if "gemma4" in model.lower() else "tool"
    final_healed = []
    for msg in healed:
        role = msg.get("role")
        if (
            final_healed
            and final_healed[-1].get("role") == expected_tool_role
            and role == "user"
        ):
            final_healed.append(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    content="I have processed the tool execution results.",
                )
            )
        final_healed.append(msg)
    return final_healed

adk_lite_llm._ensure_tool_results = _patched_ensure_tool_results


# Fix Pydantic serialization error for LiteLlm client by excluding llm_client field from serialization.
# This prevents PydanticSerializationError when adk web tries to build/serialize the agent graph.
if "llm_client" in LiteLlm.model_fields:
    LiteLlm.model_fields["llm_client"].exclude = True
    LiteLlm.model_rebuild(force=True)


def _model_name(model_name: str | None = None) -> str:
    name = model_name or config.CONFIG.agent_model
    if not name.startswith("openai/"):
        name = f"openai/{name}"
    return name


def build_llm(model_name: str | None = None) -> LiteLlm:
    """Builds a LiteLlm instance pointing at the configured OpenAI-compatible endpoint."""
    return LiteLlm(
        model=_model_name(model_name),
        api_base=os.environ.get("OPENAI_API_BASE", config.CONFIG.api_base),
        api_key=os.environ.get("OPENAI_API_KEY", config.CONFIG.api_key),
        num_retries=config.CONFIG.max_retry,
        timeout=config.CONFIG.llm_timeout,
    )


def complete_text(
    prompt: str,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> str:
    """One-shot chat completion via LiteLLM (for callbacks outside ADK agents)."""
    response = litellm.completion(
        model=_model_name(model_name),
        api_base=os.environ.get("OPENAI_API_BASE", config.CONFIG.api_base),
        api_key=os.environ.get("OPENAI_API_KEY", config.CONFIG.api_key),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        num_retries=config.CONFIG.max_retry,
        timeout=config.CONFIG.llm_timeout,
    )
    message = response.choices[0].message
    content = getattr(message, "content", None) or ""
    if content:
        return content
    reasoning = getattr(message, "reasoning_content", None)
    return reasoning or ""


