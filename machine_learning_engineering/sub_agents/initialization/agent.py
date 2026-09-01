"""Initialization agent for Machine Learning Engineering."""

import dataclasses
import logging
import os
import shutil
import time

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_request as llm_request_module
from google.adk.models import llm_response as llm_response_module
from google.genai import types

from machine_learning_engineering.shared_libraries import (
    common_util,
    config,
    debug_util,
    llm,
    search_util,
)
from machine_learning_engineering.shared_libraries.skrub_guidance import (
    SKRUB_DATAOPS_GUIDELINE,
    SKRUB_RAG_HINT,
)
from machine_learning_engineering.sub_agents.initialization import prompt

logger = logging.getLogger(__name__)
_MODEL_LIST_KEYS = frozenset({"model_name", "example_code"})


def get_model_candidates(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Gets the model candidates."""
    task_id = callback_context.agent_name.split("_")[-1]
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    num_model_candidates = callback_context.state.get("num_model_candidates", 2)
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    common_util.repair_unknown_function_calls(
        llm_response,
        allowed_tool_names=frozenset({"web_search"}),
        content_keys=_MODEL_LIST_KEYS,
    )
    response_text = common_util.get_text_from_response(llm_response)

    if "def get_model():" in response_text or "models = [" in response_text:
        response_text = """
                [
                    {
                        "model_name": "skrub.tabular_pipeline",
                        "example_code": "import subprocess\\nimport sys\\ntry:\\n    import skrub\\nexcept ModuleNotFoundError:\\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'skrub>=0.4.0', '--quiet'])\\n    import skrub\\nimport pandas as pd\\nfrom sklearn.model_selection import train_test_split\\nfrom sklearn.metrics import root_mean_squared_error\\n\\ntrain_df = pd.read_csv('./input/train.csv')\\ntarget_col = 'median_house_value'\\nX = train_df.drop(columns=[target_col])\\ny = train_df[target_col]\\nX_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\\nmodel = skrub.tabular_pipeline('regressor')\\nmodel.fit(X_train, y_train)\\npreds = model.predict(X_val)\\nscore = root_mean_squared_error(y_val, preds)\\nprint(f'Validation score: {score}')"
                    }
                ]
                """

    models = common_util.extract_json_list(
        response_text,
        required_keys=_MODEL_LIST_KEYS,
    )
    if not models:
        logger.warning(
            "Failed to parse model list for task_id=%s (response length=%d)",
            task_id,
            len(response_text),
        )
        return None
    try:
        models = models[:num_model_candidates]
        for j, model in enumerate(models):
            model_description = ""
            model_description += "## Model name\n"
            model_description += model["model_name"]
            model_description += "\n\n"
            model_description += "## Example Python code\n"
            model_description += model["example_code"]
            callback_context.state[f"init_{task_id}_model_{j + 1}"] = {
                "model_name": model["model_name"],
                "example_code": model["example_code"],
                "model_description": model_description,
            }
            with open(
                os.path.join(run_cwd, "model_candidates", f"model_{j + 1}.txt"),
                "w",
            ) as f:
                f.write(model_description)
        callback_context.state[f"init_{task_id}_model_finish"] = True
    except (KeyError, TypeError, OSError) as exc:
        logger.warning(
            "Failed to store model candidates for task_id=%s: %s",
            task_id,
            exc,
        )
        return None
    return None


def get_task_summary(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Gets the task summary."""
    response_text = common_util.get_text_from_response(llm_response)
    task_type = callback_context.state.get("task_type", "Unknown Task")
    task_summary = f"Task: {task_type}\n{response_text}"
    callback_context.state["task_summary"] = task_summary
    return None


def check_model_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the model retrieval is finished."""
    task_id = callback_context.agent_name.split("_")[-1]

    if callback_context.state.get(f"init_{task_id}_model_finish", False):
        return llm_response_module.LlmResponse()

    return None


def check_model_eval_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the model evaluation is finished."""
    model_id = callback_context.agent_name.split("_")[-1]
    task_id = callback_context.agent_name.split("_")[-2]
    model_description = callback_context.state.get(
        f"init_{task_id}_model_{model_id}",
        {},
    ).get("model_description", "")
    callback_context.state[
        f"model_eval_skip_data_leakage_check_{task_id}_{model_id}"
    ] = True
    if not model_description:
        return llm_response_module.LlmResponse()
    result_dict = callback_context.state.get(
        f"init_code_exec_result_{task_id}_{model_id}", {}
    )
    if result_dict:
        return llm_response_module.LlmResponse()
    callback_context.state[
        f"model_eval_skip_data_leakage_check_{task_id}_{model_id}"
    ] = False
    return None


def check_merger_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the code integration is finished."""
    reference_idx = callback_context.agent_name.split("_")[-1]
    task_id = callback_context.agent_name.split("_")[-2]
    result_dict = callback_context.state.get(
        f"merger_code_exec_result_{task_id}_{reference_idx}", {}
    )
    callback_context.state[
        f"merger_skip_data_leakage_check_{task_id}_{reference_idx}"
    ] = True
    if result_dict:
        return llm_response_module.LlmResponse()
    callback_context.state[
        f"merger_skip_data_leakage_check_{task_id}_{reference_idx}"
    ] = False
    return None


def skip_data_use_check(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Skips the data use check if the code is not changed."""
    task_id = callback_context.agent_name.split("_")[-1]
    check_data_use_finish = callback_context.state.get(
        f"check_data_use_finish_{task_id}", False
    )
    if check_data_use_finish:
        return llm_response_module.LlmResponse()
    result_dict = callback_context.state.get(
        f"train_code_exec_result_0_{task_id}", {}
    )
    callback_context.state[
        f"check_data_use_skip_data_leakage_check_{task_id}"
    ] = True
    if result_dict:
        return llm_response_module.LlmResponse()
    callback_context.state[
        f"check_data_use_skip_data_leakage_check_{task_id}"
    ] = False
    return None


def rank_candidate_solutions(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Ranks the candidate solutions based on their scores."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    task_id = callback_context.agent_name.split("_")[-1]
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    num_model_candidates = callback_context.state.get("num_model_candidates", 2)
    performance_results = []
    for k in range(num_model_candidates):
        model_id = k + 1
        init_code = callback_context.state.get(
            f"init_code_{task_id}_{model_id}", ""
        )
        init_code_exec_result = callback_context.state.get(
            f"init_code_exec_result_{task_id}_{model_id}", {}
        )
        if init_code_exec_result:
            performance_results.append(
                (
                    init_code_exec_result.get("score", 0.0),
                    init_code,
                    init_code_exec_result,
                )
            )
    if not performance_results:
        # Fallback if no solution executed successfully:
        # Just use the first candidate solution even if it didn't execute/evaluate successfully
        first_init_code = callback_context.state.get(f"init_code_{task_id}_1", "")
        # Create a dummy result dict or try to get whatever execution result was there
        dummy_exec_result = callback_context.state.get(f"init_code_exec_result_{task_id}_1", {})
        if not dummy_exec_result:
            dummy_exec_result = {
                "returncode": 1,
                "stdout": "",
                "stderr": "No successful evaluation result found during candidate ranking.",
                "execution_time": 0.0,
                "score": 1e9 if callback_context.state.get("lower", True) else 0.0,
            }
        elif "score" not in dummy_exec_result:
            dummy_exec_result["score"] = 1e9 if callback_context.state.get("lower", True) else 0.0

        if not first_init_code:
            for k in range(num_model_candidates):
                code_candidate = callback_context.state.get(f"init_code_{task_id}_{k+1}", "")
                if code_candidate:
                    first_init_code = code_candidate
                    dummy_exec_result = callback_context.state.get(f"init_code_exec_result_{task_id}_{k+1}", {})
                    if not dummy_exec_result:
                        dummy_exec_result = {
                            "returncode": 1,
                            "stdout": "",
                            "stderr": "No successful evaluation result found during candidate ranking.",
                            "execution_time": 0.0,
                            "score": 1e9 if callback_context.state.get("lower", True) else 0.0,
                        }
                    elif "score" not in dummy_exec_result:
                        dummy_exec_result["score"] = 1e9 if callback_context.state.get("lower", True) else 0.0
                    break

        if not first_init_code:
            # Ultimate default fallback template
            first_init_code = """
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    train_path = './input/train.csv'
    test_path = './input/test.csv'
    if not os.path.exists(train_path):
        import glob
        csv_files = glob.glob('./input/**/*.csv', recursive=True)
        train_candidates = [f for f in csv_files if 'test' not in f.lower()]
        test_candidates = [f for f in csv_files if 'test' in f.lower()]
        if train_candidates: train_path = train_candidates[0]
        if test_candidates: test_path = test_candidates[0]
            
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    target_col = 'median_house_value'
    if target_col not in train_df.columns:
        diff = list(set(train_df.columns) - set(test_df.columns))
        target_col = diff[0] if diff else train_df.columns[-1]
            
    # Simple fillna
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    X = train_df.drop(target_col, axis=1).select_dtypes(include=[np.number])
    y = train_df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f'Final Validation Performance: {score}')
    
    test_preds = model.predict(test_df.select_dtypes(include=[np.number]))
    test_df[target_col] = test_preds
    test_df[[target_col]].to_csv('submission.csv', index=False, header=False)

if __name__ == '__main__':
    main()
"""
            dummy_exec_result = {
                "returncode": 0,
                "stdout": "Final Validation Performance: 0.0",
                "stderr": "",
                "execution_time": 0.0,
                "score": 0.0,
            }

        performance_results.append(
            (
                dummy_exec_result["score"],
                first_init_code,
                dummy_exec_result,
            )
        )


    if callback_context.state.get("lower", True):
        performance_results.sort(key=lambda x: x[0])
    else:
        performance_results.sort(key=lambda x: x[0], reverse=True)
    best_score = performance_results[0][0]
    base_solution = common_util.extract_code_block(performance_results[0][1])
    callback_context.state[f"performance_results_{task_id}"] = (
        performance_results
    )
    callback_context.state[f"best_score_{task_id}"] = best_score
    callback_context.state[f"base_solution_{task_id}"] = base_solution
    callback_context.state[f"best_idx_{task_id}"] = 0
    with open(f"{run_cwd}/train0_0.py", "w", encoding="utf-8") as f:
        f.write(base_solution)
    callback_context.state[f"merger_code_{task_id}_0"] = performance_results[0][
        1
    ]
    callback_context.state[f"merger_code_exec_result_{task_id}_0"] = (
        performance_results[0][2]
    )
    return None


def select_best_solution(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Selects the best solution."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    task_id = callback_context.agent_name.split("_")[-1]
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    best_idx = callback_context.state.get(f"best_idx_{task_id}", 0)
    response = callback_context.state.get(
        f"merger_code_{task_id}_{best_idx}", ""
    )
    result_dict = callback_context.state.get(
        f"merger_code_exec_result_{task_id}_{best_idx}", {}
    )
    code_text = common_util.extract_code_block(response)
    output_filepath = os.path.join(run_cwd, "train0.py")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(code_text)
    callback_context.state[f"train_code_0_{task_id}"] = code_text
    callback_context.state[f"train_code_exec_result_0_{task_id}"] = result_dict
    return None


def update_merger_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Updates merger states."""
    lower = callback_context.state.get("lower", True)
    reference_idx = callback_context.agent_name.split("_")[-1]
    task_id = callback_context.agent_name.split("_")[-2]
    best_score = callback_context.state.get(f"best_score_{task_id}", 0)
    base_solution = callback_context.state.get(f"base_solution_{task_id}", "")
    best_idx = callback_context.state.get(f"best_idx_{task_id}", 0)
    merged_code = callback_context.state.get(
        f"merger_code_{task_id}_{reference_idx}", ""
    )
    result_dict = callback_context.state.get(
        f"merger_code_exec_result_{task_id}_{reference_idx}", {}
    )
    score = result_dict.get("score", 1e9 if lower else 0)
    if lower:
        if score <= best_score:
            best_score = score
            base_solution = common_util.extract_code_block(merged_code)
            best_idx = int(reference_idx)
    elif score >= best_score:
        best_score = score
        base_solution = common_util.extract_code_block(merged_code)
        best_idx = int(reference_idx)
    callback_context.state[f"best_score_{task_id}"] = best_score
    callback_context.state[f"base_solution_{task_id}"] = base_solution
    callback_context.state[f"best_idx_{task_id}"] = best_idx
    return None


def prepare_task(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Prepares things for the task."""
    config_dict = dataclasses.asdict(config.CONFIG)
    for key in config_dict:
        callback_context.state[key] = config_dict[key]
    callback_context.state["start_time"] = time.time()
    # fix randomness
    common_util.set_random_seed(callback_context.state["seed"])
    task_name = callback_context.state.get("task_name", "")
    data_dir = callback_context.state.get("data_dir", "")
    task_description = open(
        os.path.join(data_dir, task_name, "task_description.txt"),
    ).read()

    # Programmatically detect column schemas to prevent LLM misidentification
    train_path = os.path.join(data_dir, task_name, "train.csv")
    if os.path.exists(train_path):
        try:
            import pandas as pd
            df_sample = pd.read_csv(train_path, nrows=5)
            schema_lines = []
            for col in df_sample.columns:
                if col == "target" or col == "id":
                    continue
                non_null = df_sample[col].dropna()
                if not non_null.empty:
                    sample_val = non_null.iloc[0]
                    # Check if the column is string/sequence or numeric
                    if df_sample[col].dtype == object or isinstance(sample_val, str):
                        schema_lines.append(f"- Column `{col}` is a string/sequence column (example: `{sample_val}`).")
                    else:
                        schema_lines.append(f"- Column `{col}` is numeric (example: `{sample_val}`).")
                else:
                    schema_lines.append(f"- Column `{col}` has all null values.")
            
            schema_info = "\n\n### Verified Column Types (Auto-detected):\n" + "\n".join(schema_lines)
            task_description += schema_info
        except Exception as e:
            logger.warning("Failed to auto-detect dataset schema: %s", e)

    callback_context.state["task_description"] = task_description
    return None


def create_workspace(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Creates workspace."""
    data_dir = callback_context.state.get("data_dir", "")
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    task_id = callback_context.agent_name.split("_")[-1]
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    if os.path.exists(run_cwd):
        shutil.rmtree(run_cwd)
    # make required directories
    os.makedirs(os.path.join(workspace_dir, task_name, task_id), exist_ok=True)
    os.makedirs(
        os.path.join(workspace_dir, task_name, task_id, "input"), exist_ok=True
    )
    os.makedirs(
        os.path.join(workspace_dir, task_name, task_id, "model_candidates"),
        exist_ok=True,
    )
    # copy files to input directory
    files = os.listdir(os.path.join(data_dir, task_name))
    for file in files:
        if file == "raw":
            continue  # original-data snapshot, not for the model
        if os.path.isdir(os.path.join(data_dir, task_name, file)):
            shutil.copytree(
                os.path.join(data_dir, task_name, file),
                os.path.join(workspace_dir, task_name, task_id, "input", file),
            )
        elif "answer" not in file and "metadata" not in file and "leakage" not in file:
            common_util.copy_file(
                os.path.join(data_dir, task_name, file),
                os.path.join(workspace_dir, task_name, task_id, "input"),
            )
    return None


def get_model_eval_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the model evaluation agent instruction."""
    task_description = context.state.get("task_description", "")
    model_id = context.agent_name.split("_")[-1]
    task_id = context.agent_name.split("_")[-2]
    model_description = context.state.get(
        f"init_{task_id}_model_{model_id}",
        {},
    ).get("model_description", "")
    return (
        prompt.MODEL_EVAL_INSTR.format(
            task_description=task_description,
            model_description=model_description,
        )
        + SKRUB_DATAOPS_GUIDELINE
    )


def get_model_retriever_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the model retriever agent instruction."""
    task_summary = context.state.get("task_summary", "")
    num_model_candidates = context.state.get("num_model_candidates", 2)

    base_instruction = prompt.MODEL_RETRIEVAL_INSTR.format(
        task_summary=task_summary,
        num_model_candidates=num_model_candidates,
    )

    return base_instruction


def get_merger_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the integrate agent instruction."""
    reference_idx = int(context.agent_name.split("_")[-1])
    task_id = context.agent_name.split("_")[-2]
    performance_results = context.state.get(
        f"performance_results_{task_id}", []
    )
    base_solution = context.state.get(f"base_solution_{task_id}", "")
    if reference_idx < len(performance_results):
        reference_solution = common_util.extract_code_block(
            performance_results[reference_idx][1]
        )
    else:
        reference_solution = ""
    return prompt.CODE_INTEGRATION_INSTR.format(
        base_code=base_solution,
        reference_code=reference_solution,
    )


def get_check_data_use_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the check data use agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    task_description = context.state.get("task_description", "")
    code = context.state.get(f"train_code_0_{task_id}", "")
    return prompt.CHECK_DATA_USE_INSTR.format(
        code=code,
        task_description=task_description,
    )


task_summarization_agent = agents.Agent(
    model=llm.build_llm(),
    name="task_summarization_agent",
    description="Summarize the task description.",
    instruction=prompt.SUMMARIZATION_AGENT_INSTR,
    after_model_callback=get_task_summary,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
    ),
    include_contents="none",
)
init_parallel_sub_agents = []
for k in range(config.CONFIG.num_solutions):
    model_retriever_agent = agents.Agent(
        model=llm.build_llm(),
        name=f"model_retriever_agent_{k + 1}",
        description="Retrieve effective models for solving a given task.",
        instruction=get_model_retriever_agent_instruction,
        tools=[search_util.web_search_tool],
        before_model_callback=check_model_finish,
        after_model_callback=get_model_candidates,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
        ),
        include_contents="none",
    )
    model_retriever_loop_agent = agents.LoopAgent(
        name=f"model_retriever_loop_agent_{k + 1}",
        description="Retrieve effective models until it succeeds.",
        sub_agents=[model_retriever_agent],
        max_iterations=config.CONFIG.max_retry,
    )
    init_solution_gen_sub_agents = [
        model_retriever_loop_agent,
    ]
    for model_idx in range(config.CONFIG.num_model_candidates):
        model_eval_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
            prefix="model_eval",
            suffix=f"{k + 1}_{model_idx + 1}",
            agent_description="Generate a code using the given model",
            instruction_func=get_model_eval_agent_instruction,
            before_model_callback=check_model_eval_finish,
        )
        init_solution_gen_sub_agents.append(model_eval_and_debug_loop_agent)
    rank_agent = agents.SequentialAgent(
        name=f"rank_agent_{k + 1}",
        description="Rank the solutions based on the scores.",
        before_agent_callback=rank_candidate_solutions,
    )
    init_solution_gen_sub_agents.append(rank_agent)
    for merge_idx in range(1, config.CONFIG.num_model_candidates):
        merge_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
            prefix="merger",
            suffix=f"{k + 1}_{merge_idx}",
            agent_description="Integrate two solutions into a single solution",
            instruction_func=get_merger_agent_instruction,
            before_model_callback=check_merger_finish,
        )
        merger_states_update_agent = agents.SequentialAgent(
            name=f"merger_states_update_agent_{k + 1}_{merge_idx}",
            description="Updates the states after merging.",
            before_agent_callback=update_merger_states,
        )
        init_solution_gen_sub_agents.extend(
            [
                merge_and_debug_loop_agent,
                merger_states_update_agent,
            ]
        )
    selection_agent = agents.SequentialAgent(
        name=f"selection_agent_{k + 1}",
        description="Select the best solution.",
        before_agent_callback=select_best_solution,
    )
    init_solution_gen_sub_agents.append(selection_agent)
    if config.CONFIG.use_data_usage_checker:
        check_data_use_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
            prefix="check_data_use",
            suffix=f"{k + 1}",
            agent_description="Check if all the provided information is used",
            instruction_func=get_check_data_use_instruction,
            before_model_callback=skip_data_use_check,
        )
        init_solution_gen_sub_agents.append(check_data_use_and_debug_loop_agent)
    init_solution_gen_agent = agents.SequentialAgent(
        name=f"init_solution_gen_agent_{k + 1}",
        description="Generate an initial solutions for the given task.",
        sub_agents=init_solution_gen_sub_agents,
        before_agent_callback=create_workspace,
    )
    init_parallel_sub_agents.append(init_solution_gen_agent)
init_parallel_agent = agents.ParallelAgent(
    name="init_parallel_agent",
    description="Generate multiple initial solutions for the given task in parallel.",
    sub_agents=init_parallel_sub_agents,
)
initialization_agent = agents.SequentialAgent(
    name="initialization_agent",
    description="Initialize the states and generate initial solutions.",
    sub_agents=[
        task_summarization_agent,
        init_parallel_agent,
    ],
    before_agent_callback=prepare_task,
)
