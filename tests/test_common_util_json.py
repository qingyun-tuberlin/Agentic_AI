"""Tests for JSON extraction from LLM responses."""

from machine_learning_engineering.shared_libraries import common_util


def test_extract_json_dict_fenced():
    text = '```json\n{"verdict": "PASS", "reason": "ok"}\n```'
    assert common_util.extract_json_dict(text, required_keys=frozenset({"verdict"})) == {
        "verdict": "PASS",
        "reason": "ok",
    }


def test_extract_json_dict_prefers_last_fenced_block():
    text = (
        'Draft: ```json\n{"verdict": "FAIL", "reason": "draft"}\n```\n'
        'Final: ```json\n{"verdict": "PASS", "reason": "final"}\n```'
    )
    assert common_util.extract_json_dict(text, required_keys=frozenset({"verdict"})) == {
        "verdict": "PASS",
        "reason": "final",
    }


def test_extract_json_dict_reasoning_before_json():
    text = (
        "Let me think... verdict is pass in my analysis.\n"
        '```json\n{"verdict": "PASS", "reason": "No leakage."}\n```'
    )
    assert common_util.extract_json_dict(text, required_keys=frozenset({"verdict"})) == {
        "verdict": "PASS",
        "reason": "No leakage.",
    }


def test_extract_json_dict_prefers_last_balanced_object():
    text = (
        'Thinking about {"verdict": "FAIL"} as an example.\n'
        '{"verdict": "PASS", "reason": "clean"}'
    )
    assert common_util.extract_json_dict(text, required_keys=frozenset({"verdict"})) == {
        "verdict": "PASS",
        "reason": "clean",
    }


def test_extract_json_list_reasoning_before_final_array():
    text = (
        "Return: list[Model] with draft [{'model_name': 'bad'}].\n"
        '[{"model_name": "CatBoost", "example_code": "print(1)"}]'
    )
    assert common_util.extract_json_list(
        text,
        required_keys=frozenset({"model_name", "example_code"}),
    ) == [{"model_name": "CatBoost", "example_code": "print(1)"}]


def test_extract_json_list_prefers_last_fenced_block():
    text = (
        '```json\n[{"model_name": "Draft", "example_code": "x"}]\n```\n'
        '```json\n[{"model_name": "CatBoost", "example_code": "y"}]\n```'
    )
    assert common_util.extract_json_list(
        text,
        required_keys=frozenset({"model_name", "example_code"}),
    ) == [{"model_name": "CatBoost", "example_code": "y"}]
