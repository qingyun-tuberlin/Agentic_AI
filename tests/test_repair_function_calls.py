"""Tests for repairing hallucinated LLM function calls."""

import json
from unittest.mock import MagicMock

from google.adk.models import llm_response as llm_response_module
from google.genai import types

from machine_learning_engineering.shared_libraries import common_util

_MODEL_KEYS = frozenset({"model_name", "example_code"})


def _response_with_parts(*parts: types.Part) -> llm_response_module.LlmResponse:
    return llm_response_module.LlmResponse(
        content=types.Content(parts=list(parts), role="model"),
    )


def test_repair_converts_hallucinated_agent_tool_to_text():
    resp = _response_with_parts(
        types.Part(
            function_call=types.FunctionCall(
                name="model_retriever_agent_1",
                args={
                    "model_name": "TabPFN",
                    "example_code": "print(1)",
                },
            )
        )
    )
    assert common_util.repair_unknown_function_calls(
        resp,
        allowed_tool_names=frozenset({"web_search"}),
        content_keys=_MODEL_KEYS,
    )
    assert len(resp.content.parts) == 1
    assert resp.content.parts[0].text == json.dumps(
        [{"model_name": "TabPFN", "example_code": "print(1)"}]
    )
    assert common_util.extract_json_list(
        common_util.get_text_from_response(resp),
        required_keys=_MODEL_KEYS,
    ) == [{"model_name": "TabPFN", "example_code": "print(1)"}]


def test_repair_keeps_allowed_web_search_call():
    resp = _response_with_parts(
        types.Part(
            function_call=types.FunctionCall(
                name="web_search",
                args={"query": "tabular models"},
            )
        )
    )
    assert not common_util.repair_unknown_function_calls(
        resp,
        allowed_tool_names=frozenset({"web_search"}),
        content_keys=_MODEL_KEYS,
    )
    assert resp.content.parts[0].function_call.name == "web_search"


def test_repair_merges_multiple_hallucinated_model_calls():
    resp = _response_with_parts(
        types.Part(
            function_call=types.FunctionCall(
                name="model_retriever_agent_1",
                args={"model_name": "A", "example_code": "a"},
            )
        ),
        types.Part(
            function_call=types.FunctionCall(
                name="model_retriever_agent_1",
                args={"model_name": "B", "example_code": "b"},
            )
        ),
    )
    common_util.repair_unknown_function_calls(
        resp,
        allowed_tool_names=frozenset({"web_search"}),
        content_keys=_MODEL_KEYS,
    )
    models = json.loads(resp.content.parts[-1].text)
    assert len(models) == 2
    assert models[0]["model_name"] == "A"
    assert models[1]["model_name"] == "B"


def test_coerce_model_dicts_single_and_list():
    single = {"model_name": "X", "example_code": "x"}
    assert common_util.coerce_model_dicts(single, _MODEL_KEYS) == [single]
    assert common_util.coerce_model_dicts([single], _MODEL_KEYS) == [single]
    assert common_util.coerce_model_dicts({"model_name": "X"}, _MODEL_KEYS) is None

