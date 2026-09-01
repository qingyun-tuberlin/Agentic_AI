"""Unit tests for the XAI Correction Agent utilities and callbacks."""

import pytest
from unittest.mock import MagicMock

from machine_learning_engineering.sub_agents.xai_correction import agent as xai_correction


def test_prepare_xai():
    # Setup mock callback context
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "train_code_0_1": "print('hello')",
    }
    
    xai_correction.prepare_xai(context)
    
    assert context.state["xai_current_code_1"] == "print('hello')"
    assert context.state["xai_loop_count_1"] == 0
    assert context.state["xai_terminate_1"] is False
    assert context.state["xai_action_history_1"] == []
    assert context.state["xai_issue_history_1"] == []


def test_routing_decision_pass():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_audit_result_1": {"verdict": "PASS", "reason": "No leakage found."}
    }
    
    xai_correction.routing_decision(context)
    assert context.state["xai_verdict_1"] == "PASS"


def test_routing_decision_fail():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_audit_result_1": {"verdict": "FAIL", "reason": "Target leakage found."}
    }
    
    xai_correction.routing_decision(context)
    assert context.state["xai_verdict_1"] == "FAIL"


def test_routing_decision_missing():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {}
    
    with pytest.raises(ValueError):
        xai_correction.routing_decision(context)


def test_loop_controller():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_verdict_1": "FAIL",
        "xai_loop_count_1": 0,
        "xai_audit_result_1": {"verdict": "FAIL"},
        "xai_audit_history_1": []
    }
    
    xai_correction.loop_controller(context)
    assert context.state["xai_loop_count_1"] == 1
    assert context.state["xai_terminate_1"] is False
    assert len(context.state["xai_audit_history_1"]) == 1


def test_loop_controller_terminate():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_verdict_1": "FAIL",
        "xai_loop_count_1": 3,
        "xai_audit_result_1": {"verdict": "FAIL"},
        "xai_audit_history_1": []
    }
    
    xai_correction.loop_controller(context)
    assert context.state["xai_loop_count_1"] == 4
    assert context.state["xai_terminate_1"] is True


def test_save_audit_result_valid():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {}
    
    part = MagicMock()
    part.text = '```json\n{"verdict": "PASS", "reason": "Good code"}\n```'
    response = MagicMock()
    response.content.parts = [part]
    
    xai_correction.save_audit_result(context, response)
    
    assert context.state["xai_audit_result_1"] == {"verdict": "PASS", "reason": "Good code"}


def test_save_revised_code():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {}
    
    part = MagicMock()
    part.text = '```python\nprint("revised")\n```'
    response = MagicMock()
    response.content.parts = [part]
    
    xai_correction.save_revised_code(context, response)
    assert context.state["xai_current_code_1"] == 'print("revised")'


from unittest.mock import patch

def test_dynamic_audit_before_model_pass():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_current_code_1": "print('hello')\n# xai_metrics.json",
        "workspace_dir": "/tmp",
        "task_name": "task1",
        "exec_timeout": 10,
    }
    llm_request = MagicMock()

    with patch("machine_learning_engineering.shared_libraries.code_util.run_python_code") as mock_run, \
         patch("machine_learning_engineering.shared_libraries.xai_util.parse_xai_metrics") as mock_parse, \
         patch("machine_learning_engineering.shared_libraries.xai_util.evaluate_leakage_risk") as mock_eval:
        
        mock_run.return_value = {"returncode": 0, "stdout": "done"}
        mock_parse.return_value = {"validation_score": 0.5}
        mock_eval.return_value = (False, "Attribution is clean")

        res = xai_correction.dynamic_audit_before_model(context, llm_request)

        assert res is not None
        assert context.state["xai_audit_result_1"] == {
            "verdict": "PASS",
            "reason": "Attribution is clean",
        }
        assert context.state["xai_audit_mode_1"] == "dynamic"
        assert "PASS" in res.content.parts[0].text

def test_dynamic_audit_before_model_fail_leakage():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_current_code_1": "print('hello')\n# xai_metrics.json",
        "workspace_dir": "/tmp",
        "task_name": "task1",
        "exec_timeout": 10,
    }
    llm_request = MagicMock()

    with patch("machine_learning_engineering.shared_libraries.code_util.run_python_code") as mock_run, \
         patch("machine_learning_engineering.shared_libraries.xai_util.parse_xai_metrics") as mock_parse, \
         patch("machine_learning_engineering.shared_libraries.xai_util.evaluate_leakage_risk") as mock_eval:
        
        mock_run.return_value = {"returncode": 0, "stdout": "done"}
        mock_parse.return_value = {"validation_score": 0.5}
        mock_eval.return_value = (True, "Leakage detected")

        res = xai_correction.dynamic_audit_before_model(context, llm_request)

        assert res is not None
        assert context.state["xai_audit_result_1"] == {
            "verdict": "FAIL",
            "reason": "Leakage detected",
        }
        assert context.state["xai_audit_mode_1"] == "dynamic"
        assert "FAIL" in res.content.parts[0].text

def test_dynamic_audit_before_model_fail_closed():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_current_code_1": "print('hello')\n# xai_metrics.json",
        "workspace_dir": "/tmp",
        "task_name": "task1",
        "exec_timeout": 10,
        "xai_allow_static_fallback": False,
    }
    llm_request = MagicMock()

    # Code runs but never produces valid metrics -> gate returns metrics=None.
    with patch(
        "machine_learning_engineering.shared_libraries.code_util.run_python_code",
        return_value={"returncode": 0, "stdout": "done"},
    ), patch(
        "machine_learning_engineering.shared_libraries.xai_util.parse_xai_metrics",
        return_value={},
    ):
        res = xai_correction.dynamic_audit_before_model(context, llm_request)

    assert res is not None
    assert context.state["xai_audit_mode_1"] == "failed"
    assert context.state["xai_audit_result_1"]["verdict"] == "FAIL"


def test_dynamic_audit_before_model_static_fallback_allowed():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_current_code_1": "print('hello')\n# xai_metrics.json",
        "workspace_dir": "/tmp",
        "task_name": "task1",
        "exec_timeout": 10,
        "xai_allow_static_fallback": True,
    }
    llm_request = MagicMock()

    with patch(
        "machine_learning_engineering.shared_libraries.code_util.run_python_code",
        return_value={"returncode": 0, "stdout": "done"},
    ), patch(
        "machine_learning_engineering.shared_libraries.xai_util.parse_xai_metrics",
        return_value={},
    ):
        res = xai_correction.dynamic_audit_before_model(context, llm_request)

    assert res is None
    assert context.state["xai_audit_mode_1"] == "static_fallback"

def test_save_audit_result_reasoning_with_trailing_json():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {}

    part = MagicMock()
    part.text = (
        "The user wants me to audit the script. Verdict is pass in my draft.\n"
        '```json\n{"verdict": "PASS", "reason": "No target leakage."}\n```'
    )
    response = MagicMock()
    response.content.parts = [part]

    xai_correction.save_audit_result(context, response)

    assert context.state["xai_audit_result_1"] == {
        "verdict": "PASS",
        "reason": "No target leakage.",
    }


def test_save_audit_result_heuristic_pass():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {}

    part = MagicMock()
    part.text = "No further outputs are needed. The audit is complete, and the verdict is PASS with the provided reasoning."
    response = MagicMock()
    response.content.parts = [part]

    xai_correction.save_audit_result(context, response)

    assert context.state["xai_audit_result_1"]["verdict"] == "PASS"
    assert "unstructured model output" in context.state["xai_audit_result_1"]["reason"]


def test_save_audit_result_heuristic_fail():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {}

    part = MagicMock()
    part.text = "The target column is leaked, verdict is FAIL."
    response = MagicMock()
    response.content.parts = [part]

    xai_correction.save_audit_result(context, response)

    assert context.state["xai_audit_result_1"]["verdict"] == "FAIL"
    assert "unstructured model output" in context.state["xai_audit_result_1"]["reason"]


def test_dynamic_audit_instruments_via_llm():
    context = MagicMock()
    context.agent_name = "agent_1"
    context.state = {
        "xai_current_code_1": "print('hello')",
        "workspace_dir": "/tmp",
        "task_name": "task1",
        "exec_timeout": 10,
        "lower": True,
    }
    llm_request = MagicMock()

    instrumented = "print('hello')\n# xai_metrics.json"

    # Code lacks instrumentation -> the gate must call the LLM to add it, then
    # run the instrumented code and parse the resulting metrics.
    with patch(
        "machine_learning_engineering.shared_libraries.llm.complete_text",
        return_value=f"```python\n{instrumented}\n```",
    ) as mock_complete, patch(
        "machine_learning_engineering.shared_libraries.code_util.run_python_code",
        return_value={"returncode": 0, "stdout": "done"},
    ), patch(
        "machine_learning_engineering.shared_libraries.xai_util.parse_xai_metrics",
        return_value={
            "feature_attributions": {"f1": 1.0, "f2": 0.1},
            "validation_score": 1.0,
            "masked_validation_score": 1.0,
            "lower_is_better": True,
        },
    ), patch(
        "machine_learning_engineering.shared_libraries.xai_util.evaluate_leakage_report",
        return_value=(False, "clean"),
    ):
        res = xai_correction.dynamic_audit_before_model(context, llm_request)

    mock_complete.assert_called_once()
    assert res is not None
    assert "xai_metrics.json" in context.state["xai_current_code_1"]
