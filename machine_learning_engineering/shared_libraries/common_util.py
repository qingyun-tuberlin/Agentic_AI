"""Common utility functions."""

import ast
import json
import logging
import os
import random
import re
import shutil

import numpy as np
import torch
from google.adk.models import llm_response
from google.genai import types

logger = logging.getLogger(__name__)


def get_text_from_response(
    response: llm_response.LlmResponse,
) -> str:
    """Extracts the final answer text, omitting model reasoning/thinking parts.

    Reasoning models (e.g. Qwen 3.6) return separate content parts with
    ``thought=True`` for chain-of-thought and a final part for the answer.
    Downstream prompts expect only the answer.
    """
    if not response.content or not response.content.parts:
        return ""

    answer_parts: list[str] = []
    all_parts: list[str] = []
    for part in response.content.parts:
        if not hasattr(part, "text") or part.text is None:
            continue
        all_parts.append(part.text)
        if getattr(part, "thought", None) is not True:
            answer_parts.append(part.text)

    if answer_parts:
        return "".join(answer_parts)
    if len(all_parts) > 1:
        return all_parts[-1]
    # Single part marked as thought only — no final answer yet.
    if (
        response.content.parts
        and getattr(response.content.parts[-1], "thought", None) is True
    ):
        return ""
    return "".join(all_parts)


def _function_call_args_dict(function_call: types.FunctionCall) -> dict:
    """Normalizes FunctionCall.args to a dict."""
    args = function_call.args or {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def coerce_model_dicts(
    value,
    required_keys: frozenset[str],
) -> list[dict] | None:
    """Returns a list of model dicts when value is one dict or a list of dicts."""
    if isinstance(value, dict) and all(key in value for key in required_keys):
        return [value]
    if isinstance(value, list):
        if all(
            isinstance(item, dict) and all(key in item for key in required_keys)
            for item in value
        ):
            return value
    return None


def model_list_text_from_function_call(
    function_call: types.FunctionCall,
    required_keys: frozenset[str],
) -> str | None:
    """Serializes a hallucinated tool call's args as a JSON model list string."""
    models = coerce_model_dicts(
        _function_call_args_dict(function_call),
        required_keys,
    )
    if not models:
        return None
    return json.dumps(models)


def repair_unknown_function_calls(
    response: llm_response.LlmResponse,
    *,
    allowed_tool_names: frozenset[str],
    content_keys: frozenset[str] | None = None,
) -> bool:
    """Replaces unknown function_call parts with text so ADK does not execute them.

    Some OpenAI-compatible models emit a function_call named after the agent
    (e.g. ``model_retriever_agent_1``) instead of plain-text JSON. When
    ``content_keys`` is set, matching args are converted to a JSON list in a
    text part for downstream parsers.
    """
    if not response.content or not response.content.parts:
        return False

    new_parts: list[types.Part] = []
    recovered_models: list[dict] = []
    modified = False

    for part in response.content.parts:
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            new_parts.append(part)
            continue

        name = function_call.name or ""
        if name in allowed_tool_names:
            new_parts.append(part)
            continue

        modified = True
        logger.warning(
            "Replacing unknown function_call %r (allowed: %s)",
            name,
            ", ".join(sorted(allowed_tool_names)),
        )
        if content_keys:
            models = coerce_model_dicts(
                _function_call_args_dict(function_call),
                content_keys,
            )
            if models:
                recovered_models.extend(models)
                continue

        args = _function_call_args_dict(function_call)
        if args:
            new_parts.append(types.Part(text=json.dumps(args)))

    if recovered_models:
        new_parts.append(types.Part(text=json.dumps(recovered_models)))

    if modified:
        response.content.parts = new_parts
    return modified


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def extract_code_block(text: str) -> str:
    """Returns the content of the last fenced code block, or the original text.

    Reasoning models (gpt-oss, deepseek-r1, etc.) often prepend their
    chain-of-thought before the final code. Picking the last fenced block keeps
    only the intended answer. If no fence is present, fall back to the legacy
    behavior of stripping bare fence markers.
    """
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        return matches[-1]
    return text.replace("```python", "").replace("```py", "").replace("```", "")


def _parse_json_object(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_json_value(text: str):
    """Parses a JSON or Python-literal value from text."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return None


def _iter_balanced_json_candidates(text: str):
    """Yields balanced {...} substrings in document order."""
    stack: list[int] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(i)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                yield text[start : i + 1]
                start = None


def _iter_balanced_json_arrays(text: str):
    """Yields balanced [...] substrings in document order."""
    stack: list[int] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "[":
            if not stack:
                start = i
            stack.append(i)
        elif ch == "]" and stack:
            stack.pop()
            if not stack and start is not None:
                yield text[start : i + 1]
                start = None


def extract_json_dict(
    text: str,
    required_keys: frozenset[str] | None = None,
) -> dict | None:
    """Returns the last parseable JSON object that contains all required keys.

    Reasoning models often emit chain-of-thought before the final answer. Prefer
    the last valid fenced ```json block, then the last balanced object in text.
    """
    required = required_keys or frozenset()

    def _matches(obj: dict) -> bool:
        return all(key in obj for key in required)

    for block in reversed(_JSON_FENCE_RE.findall(text)):
        obj = _parse_json_object(block.strip())
        if obj and _matches(obj):
            return obj

    for candidate in reversed(list(_iter_balanced_json_candidates(text))):
        obj = _parse_json_object(candidate)
        if obj and _matches(obj):
            return obj

    return None


def extract_json_list(
    text: str,
    required_keys: frozenset[str] | None = None,
) -> list[dict] | None:
    """Returns the last parseable JSON list whose dict items contain required keys.

    Reasoning models often emit chain-of-thought before the final answer. Prefer
    the last valid fenced ```json block, then the last balanced array in text.
    """
    required = required_keys or frozenset()

    def _matches(obj) -> bool:
        if not isinstance(obj, list) or not obj:
            return False
        return all(
            isinstance(item, dict) and all(key in item for key in required)
            for item in obj
        )

    for block in reversed(_JSON_FENCE_RE.findall(text)):
        obj = _parse_json_value(block.strip())
        if _matches(obj):
            return obj

    for candidate in reversed(list(_iter_balanced_json_arrays(text))):
        obj = _parse_json_value(candidate)
        if _matches(obj):
            return obj

    return None


def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_file(source_file_path: str, destination_dir: str) -> None:
    """Copies a file to the specified directory."""
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source_file_path, destination_dir)


def truncate_text_for_llm(text: str, max_chars: int = 150000) -> str:
    """Prevents the agent from swallowing its entire context window with logs."""
    if len(text) > max_chars:
        return f"{text[:max_chars//2]}\n\n[... TRUNCATED BY FRAMEWORK TO PREVENT CONTEXT BLOAT ...]\n\n{text[-max_chars//2:]}"
    return text