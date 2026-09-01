"""Unit tests for common utility functions."""

from unittest.mock import MagicMock
from machine_learning_engineering.shared_libraries import common_util


def test_get_text_from_response_happy_path():
    # Setup mock LlmResponse with standard text parts
    part1 = MagicMock()
    part1.text = "Hello "
    part2 = MagicMock()
    part2.text = "world!"
    
    response = MagicMock()
    response.content.parts = [part1, part2]
    
    assert common_util.get_text_from_response(response) == "Hello world!"


def test_get_text_from_response_with_none_text():
    # Setup mock LlmResponse with a None text part and a non-None text part
    part1 = MagicMock()
    part1.text = "Hello "
    part2 = MagicMock()
    part2.text = None
    part3 = MagicMock()
    part3.text = "world!"
    
    response = MagicMock()
    response.content.parts = [part1, part2, part3]
    
    # It should skip the None part and successfully concatenate the rest
    assert common_util.get_text_from_response(response) == "Hello world!"


def test_get_text_from_response_thought_only_returns_empty():
    thought = MagicMock()
    thought.text = "Still reasoning..."
    thought.thought = True

    response = MagicMock()
    response.content.parts = [thought]

    assert common_util.get_text_from_response(response) == ""


def test_get_text_from_response_skips_thought_parts():
    thought = MagicMock()
    thought.text = "Here's a thinking process:\n1. Analyze..."
    thought.thought = True
    answer = MagicMock()
    answer.text = "Tabular regression task with RMSE."
    answer.thought = False

    response = MagicMock()
    response.content.parts = [thought, answer]

    assert (
        common_util.get_text_from_response(response)
        == "Tabular regression task with RMSE."
    )


def test_get_text_from_response_without_text_attribute():
    # Setup mock LlmResponse with a part that lacks a 'text' attribute (e.g., function call)
    part1 = MagicMock()
    part1.text = "Hello "
    part2 = MagicMock(spec=[])  # Lacks all attributes including 'text'
    part3 = MagicMock()
    part3.text = "world!"
    
    response = MagicMock()
    response.content.parts = [part1, part2, part3]
    
    # It should skip the part without 'text' and successfully concatenate the rest
    assert common_util.get_text_from_response(response) == "Hello world!"


def test_ensure_tool_results_heals_tool_user_transition():
    # Explicitly import llm to apply the _ensure_tool_results monkey-patch
    from machine_learning_engineering.shared_libraries import llm
    from google.adk.models.lite_llm import _ensure_tool_results
    
    # We construct a message sequence containing tool followed by user
    messages = [
        {"role": "user", "content": "Run my code"},
        {"role": "assistant", "content": "Sure", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Success"},
        {"role": "user", "content": ""}
    ]
    
    healed = _ensure_tool_results(messages, "openai/gpt-4o")
    
    # Verify that healed sequence contains 5 messages instead of 4,
    # and has a placeholder assistant message inserted at index 3.
    assert len(healed) == 5
    assert healed[0]["role"] == "user"
    assert healed[1]["role"] == "assistant"
    assert healed[2]["role"] == "tool"
    assert healed[3]["role"] == "assistant"
    assert healed[3]["content"] == "I have processed the tool execution results."
    assert healed[4]["role"] == "user"


def test_extract_performance_from_text():
    from machine_learning_engineering.shared_libraries import code_util
    
    # Test standard format
    assert code_util.extract_performance_from_text("Final Validation Performance: 62724.54") == 62724.54
    assert code_util.extract_performance_from_text("Final Validation Performance: 0.1145") == 0.1145
    
    # Test with prefix (like RMSE =)
    assert code_util.extract_performance_from_text("Final Validation Performance: RMSE = 0.1145") == 0.1145
    assert code_util.extract_performance_from_text("Final Validation Performance: score = -1.23") == -1.23
    
    # Test with scientific notation
    assert code_util.extract_performance_from_text("Final Validation Performance: 1.2e-3") == 0.0012
    
    # Test with suffix
    assert code_util.extract_performance_from_text("Final Validation Performance: 0.85 (higher is better)") == 0.85
    
    # Test non-matching or invalid lines
    assert code_util.extract_performance_from_text("Final Validation Performance: None") is None
    assert code_util.extract_performance_from_text("Some random line") is None


def test_extract_performance_line_from_text():
    from machine_learning_engineering.shared_libraries import code_util

    stdout = (
        "Training complete\n"
        "Final Validation Performance: 0.42\n"
        "Wrote submission.csv\n"
        "Final Validation Performance: 0.99\n"
    )
    assert (
        code_util.extract_performance_line_from_text(stdout)
        == "Final Validation Performance: 0.99"
    )
    assert code_util.extract_performance_line_from_text("no score here") is None
