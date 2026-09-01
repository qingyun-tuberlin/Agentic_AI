"""Test fallback logic for rank_candidate_solutions when no solutions execute successfully."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock
from machine_learning_engineering.sub_agents.initialization.agent import rank_candidate_solutions


def test_rank_candidate_solutions_fallback_ultimate_template():
    # Setup temporary directory for workspace inside the project workspace
    temp_dir = tempfile.mkdtemp(dir=".")
    try:
        workspace_dir = temp_dir
        task_name = "test_task"
        task_id = "123"
        run_cwd = os.path.join(workspace_dir, task_name, task_id)
        os.makedirs(run_cwd, exist_ok=True)

        mock_context = MagicMock()
        mock_context.agent_name = f"rank_agent_{task_id}"
        mock_context.state = {
            "workspace_dir": workspace_dir,
            "task_name": task_name,
            "num_model_candidates": 1,
            "lower": True,
        }

        # Call rank_candidate_solutions. Since performance_results and code are empty, it should hit ultimate template fallback.
        rank_candidate_solutions(mock_context)

        # Assert that fallback populated state keys correctly
        assert f"best_score_{task_id}" in mock_context.state
        assert f"base_solution_{task_id}" in mock_context.state
        assert f"best_idx_{task_id}" in mock_context.state
        assert mock_context.state[f"best_score_{task_id}"] == 0.0

        # Assert train0_0.py was created and has content
        train_file = os.path.join(run_cwd, "train0_0.py")
        assert os.path.exists(train_file)
        with open(train_file, "r") as f:
            content = f.read()
            assert "RandomForestRegressor" in content
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_rank_candidate_solutions_fallback_existing_code():
    # Setup temporary directory for workspace inside the project workspace
    temp_dir = tempfile.mkdtemp(dir=".")
    try:
        workspace_dir = temp_dir
        task_name = "test_task"
        task_id = "456"
        run_cwd = os.path.join(workspace_dir, task_name, task_id)
        os.makedirs(run_cwd, exist_ok=True)

        mock_context = MagicMock()
        mock_context.agent_name = f"rank_agent_{task_id}"
        mock_context.state = {
            "workspace_dir": workspace_dir,
            "task_name": task_name,
            "num_model_candidates": 1,
            "lower": True,
            f"init_code_{task_id}_1": "print('Some custom model candidate code')",
        }

        # Call rank_candidate_solutions. Since performance_results is empty but code exists, it should hit fallback with custom code.
        rank_candidate_solutions(mock_context)

        # Assert that fallback populated state keys correctly
        assert f"best_score_{task_id}" in mock_context.state
        assert f"base_solution_{task_id}" in mock_context.state
        assert f"best_idx_{task_id}" in mock_context.state
        assert mock_context.state[f"best_score_{task_id}"] == 1e9

        # Assert train0_0.py was created and has content
        train_file = os.path.join(run_cwd, "train0_0.py")
        assert os.path.exists(train_file)
        with open(train_file, "r") as f:
            content = f.read()
            assert "Some custom model candidate code" in content
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
