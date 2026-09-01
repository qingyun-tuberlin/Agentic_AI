"""Refinement agent for Machine Learning Engineering."""

import functools
import json
import os

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_request as llm_request_module
from google.adk.models import llm_response as llm_response_module
from google.genai import types

from machine_learning_engineering.shared_libraries import (
    check_leakage_util,
    common_util,
    config,
    debug_util,
    llm,
    xai_gate,
    xai_instrumentation_guide,
)
from machine_learning_engineering.shared_libraries.skrub_guidance import (
    SKRUB_DATAOPS_GUIDELINE,
    SKRUB_DATAOPS_PLAN_CONSTRAINT,
    SKRUB_RAG_HINT,
)
from machine_learning_engineering.sub_agents.refinement import prompt


def update_inner_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Updates inner loop states."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"inner_iter_{task_id}"] += 1
    return None


def _compute_improvements(
    callback_context: callback_context_module.CallbackContext,
    *,
    step: str | int,
    task_id: str,
    lower: bool,
    inner_loop_round: int,
) -> list[float]:
    """Per-candidate score improvement over the previous step's solution."""
    prev_exec_result = callback_context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    improvements = []
    for inner_iter in range(inner_loop_round):
        exec_result = callback_context.state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}", {}
        )
        prev_score = prev_exec_result.get("score", 1e9 if lower else 0)
        curr_score = exec_result.get("score", 1e9 if lower else 0)
        if lower:
            improvement = prev_score - curr_score
        else:
            improvement = curr_score - prev_score
        improvements.append(improvement)
    return improvements


def _finalize_outer_step(
    callback_context: callback_context_module.CallbackContext,
    *,
    task_id: str,
    step: int,
    run_cwd: str,
    best_idx: int,
    accept: bool,
    prev_solution: str,
    prev_exec_result: dict,
    extra_note: str = "",
) -> None:
    """Write the chosen solution for the next step and roll the loop bookkeeping.

    ``accept`` selects candidate ``best_idx``; otherwise the previous solution is
    kept (rollback). ``extra_note`` is appended to the ablation record so the next
    outer plan sees any leakage findings.
    """
    output_filepath = os.path.join(run_cwd, f"train{step + 1}.py")
    if not accept:
        solution = prev_solution
        exec_result = prev_exec_result
    else:
        solution = callback_context.state.get(
            f"train_code_improve_{best_idx}_{step}_{task_id}", ""
        )
        exec_result = callback_context.state.get(
            f"train_code_improve_exec_result_{best_idx}_{step}_{task_id}", {}
        )
    callback_context.state[f"train_code_{step + 1}_{task_id}"] = solution
    callback_context.state[f"train_code_exec_result_{step + 1}_{task_id}"] = exec_result
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(solution)

    ablation_results = callback_context.state.get(
        f"ablation_summary_{step}_{task_id}", ""
    )
    if extra_note:
        ablation_results = f"{ablation_results}\n\n## XAI leakage findings\n{extra_note}"
    code_block = callback_context.state.get(
        f"refine_code_block_{step}_{task_id}", ""
    )
    callback_context.state[f"prev_ablations_{task_id}"].append(ablation_results)
    callback_context.state[f"prev_code_blocks_{task_id}"].append(code_block)
    callback_context.state[f"refine_step_{task_id}"] += 1


def update_outer_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Updates outer loop states (selects the best candidate by score)."""
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    lower = callback_context.state.get("lower", True)
    inner_loop_round = callback_context.state.get("inner_loop_round", 2)
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    prev_solution = callback_context.state.get(f"train_code_{step}_{task_id}", "")
    prev_exec_result = callback_context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    improvements = _compute_improvements(
        callback_context,
        step=step,
        task_id=task_id,
        lower=lower,
        inner_loop_round=inner_loop_round,
    )
    best_improvement = max(improvements)
    best_idx = improvements.index(best_improvement)
    _finalize_outer_step(
        callback_context,
        task_id=task_id,
        step=step,
        run_cwd=run_cwd,
        best_idx=best_idx,
        accept=best_improvement > 0.0,
        prev_solution=prev_solution,
        prev_exec_result=prev_exec_result,
    )
    return None


def update_outer_loop_states_gated(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Outer-loop selection with a per-step XAI leakage gate.

    Candidates are considered best-score first; each is audited via the shared
    ``xai_gate`` and the first leakage-free candidate with positive improvement is
    accepted. Leaky (or unverifiable) candidates are never selected, so a feature
    that only becomes dominant late in refinement cannot win on its inflated score.
    Findings are recorded so the next outer plan is told to drop the feature.
    """
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    lower = callback_context.state.get("lower", True)
    inner_loop_round = callback_context.state.get("inner_loop_round", 2)
    exec_timeout = callback_context.state.get("exec_timeout", 1800)
    max_conc = callback_context.state.get("xai_max_concentration", 0.80)
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    prev_solution = callback_context.state.get(f"train_code_{step}_{task_id}", "")
    prev_exec_result = callback_context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    improvements = _compute_improvements(
        callback_context,
        step=step,
        task_id=task_id,
        lower=lower,
        inner_loop_round=inner_loop_round,
    )

    order = sorted(
        range(len(improvements)), key=lambda i: improvements[i], reverse=True
    )
    best_idx = -1
    notes = []
    for idx in order:
        if improvements[idx] <= 0.0:
            break  # remaining candidates do not improve on the previous step
        cand_code = callback_context.state.get(
            f"train_code_improve_{idx}_{step}_{task_id}", ""
        )
        if not cand_code:
            continue
        result = xai_gate.audit_code_for_leakage(
            code=cand_code,
            run_cwd=run_cwd,
            lower=lower,
            exec_timeout=exec_timeout,
            py_filepath=f"xai_gate_probe_{step}.py",
            max_concentration=max_conc,
            instrument_fn=lambda c, e: xai_instrumentation_guide.instrument_via_llm(
                c, lower_is_better=lower, error=e
            ),
            metrics_archive_label=f"refinement_step{step}_cand{idx}",
        )
        callback_context.state[f"xai_verdict_{idx}_{step}_{task_id}"] = result.verdict
        if result.metrics is None:
            notes.append(f"candidate {idx}: audit inconclusive — {result.error}")
            continue
        if result.verdict == xai_gate.PASS:
            best_idx = idx
            print(
                f"\n[XAI Refinement step {step}/{task_id}] Accepted candidate {idx} "
                f"(improvement={improvements[idx]:.5f}); leakage gate PASS.\n"
            )
            break
        notes.append(f"candidate {idx}: leakage FAIL — {result.reason}")
        print(
            f"\n[XAI Refinement step {step}/{task_id}] Rejected candidate {idx} "
            f"(improvement={improvements[idx]:.5f}); leakage gate FAIL: {result.reason}\n"
        )

    if best_idx < 0 and notes:
        print(
            f"\n[XAI Refinement step {step}/{task_id}] No leakage-free improvement; "
            f"rolling back to previous solution.\n"
        )
    _finalize_outer_step(
        callback_context,
        task_id=task_id,
        step=step,
        run_cwd=run_cwd,
        best_idx=best_idx,
        accept=best_idx >= 0,
        prev_solution=prev_solution,
        prev_exec_result=prev_exec_result,
        extra_note="\n".join(notes),
    )
    return None


def init_inner_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Initializes inner loop states."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"inner_iter_{task_id}"] = 0
    return None


def init_outer_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Initializes outer loop states."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"refine_step_{task_id}"] = 0
    callback_context.state[f"prev_ablations_{task_id}"] = []
    callback_context.state[f"prev_code_blocks_{task_id}"] = []
    return None


def get_leakage_warning(context: callback_context_module.ReadonlyContext) -> str:
    """Extracts target leakage/XAI warning for this solution from the state."""
    task_id = context.agent_name.split("_")[-1]
    warnings = []
    audit_history = context.state.get(f"xai_audit_history_{task_id}", [])
    if isinstance(audit_history, list):
        for audit in audit_history:
            if isinstance(audit, dict) and audit.get("verdict") == "FAIL":
                reason = audit.get("reason", "")
                if reason and reason not in warnings:
                    warnings.append(reason)
    audit_result = context.state.get(f"xai_audit_result_{task_id}")
    if isinstance(audit_result, dict) and audit_result.get("verdict") == "FAIL":
        reason = audit_result.get("reason", "")
        if reason and reason not in warnings:
            warnings.append(reason)
    if not warnings:
        return ""
    warning_str = "\n".join(f"- {w}" for w in warnings)
    return (
        "\n\n# CRITICAL: TARGET LEAKAGE WARNING\n"
        "The following features or issues were flagged as TARGET LEAKAGE during previous XAI audits and MUST REMAIN REMOVED/DROPPED:\n"
        f"{warning_str}\n"
        "DO NOT re-introduce, recreate, or perform feature engineering on any of these leaked features. "
        "Any code modification or ablation study MUST NOT add these features back under any circumstances.\n"
    )


def get_ablation_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ablation agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    prev_ablations = context.state.get(f"prev_ablations_{task_id}", [])
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    prev_ablations_str = ""
    for i, ablation_result in enumerate(prev_ablations):
        prev_ablations_str += f"## Previous ablation study result {i + 1}\n"
        prev_ablations_str += f"{ablation_result}\n\n"
    if prev_ablations_str:
        instruction = prompt.ABLATION_SEQ_INSTR.format(
            code=code,
            prev_ablations=prev_ablations_str,
        )
    else:
        instruction = prompt.ABLATION_INSTR.format(
            code=code,
        )
    return instruction + get_leakage_warning(context)


def get_ablation_summary_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ablation summary agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"ablation_code_{step}_{task_id}", "")
    result_dict = context.state.get(
        f"ablation_code_exec_result_{step}_{task_id}", {}
    )
    return prompt.SUMMARIZE_ABLATION_INSTR.format(
        code=code,
        result=result_dict["ablation_result"],
    )


def get_init_plan_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the initial plan agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    ablation_results = context.state.get(
        f"ablation_summary_{step}_{task_id}", ""
    )
    prev_code_blocks = context.state.get(f"prev_code_blocks_{task_id}", [])
    if not prev_code_blocks:
        instruction = prompt.EXTRACT_BLOCK_AND_PLAN_INSTR.format(
            code=code,
            ablation_results=ablation_results,
        )
    else:
        # Format list of code blocks into a readable markdown sequence
        # and keep only the last 2 blocks to prevent context window explosion
        blocks_to_show = prev_code_blocks[-2:]
        prev_code_blocks_str = ""
        for i, block in enumerate(blocks_to_show):
            prev_code_blocks_str += f"## Previous code block {i + 1} tried:\n```python\n{block}\n```\n\n"
        
        instruction = prompt.EXTRACT_BLOCK_AND_PLAN_SEQ_INSTR.format(
            code=code,
            ablation_results=ablation_results,
            prev_code_blocks=prev_code_blocks_str,
        )

    return instruction + SKRUB_RAG_HINT + SKRUB_DATAOPS_PLAN_CONSTRAINT + get_leakage_warning(context)


def get_plan_refinement_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets plan refinement instruction."""
    lower = context.state.get("lower", True)
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code_block = context.state.get(f"refine_code_block_{step}_{task_id}", "")
    prev_plans = context.state.get(f"refine_plans_{step}_{task_id}", [])
    prev_exec_result = context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    score_plan_time_list = []
    for inner_iter, curr_plan in enumerate(prev_plans):
        exec_result = context.state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}", {}
        )
        prev_score = prev_exec_result.get("score", 1e9 if lower else 0)
        curr_score = exec_result.get("score", 1e9 if lower else 0)
        if lower:
            improvement = prev_score - curr_score
        else:
            improvement = curr_score - prev_score
        score_plan_time_list.append(
            (improvement, curr_plan, exec_result.get("execution_time", 0.0))
        )
    num_top_plans = context.state.get("num_top_plans", 3)
    score_plan_time_list.sort(key=lambda x: x[0], reverse=True)
    prev_plan_summary = ""
    selected_score_plan_time_list = score_plan_time_list[:num_top_plans]
    for score, curr_plan, execution_time in selected_score_plan_time_list:
        prev_plan_summary += f"## Plan: {curr_plan}\n"
        prev_plan_summary += (
            f"## Execution time after implement: {execution_time}s\n"
        )
        prev_plan_summary += f"## Score: {score:.5f}\n\n"

    base_instruction = prompt.PLAN_REFINEMENT_INSTR.format(
        code_block=code_block,
        prev_plan_summary=prev_plan_summary,
    )
    return base_instruction + SKRUB_DATAOPS_PLAN_CONSTRAINT + get_leakage_warning(context)


def get_plan_implement_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the plan implement agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code_block = context.state.get(f"refine_code_block_{step}_{task_id}", "")
    plan = context.state.get(f"refine_plans_{step}_{task_id}", [""])[-1]
    return (
        prompt.IMPLEMENT_PLAN_INSTR.format(
            code_block=code_block,
            plan=plan,
        )
        + SKRUB_DATAOPS_GUIDELINE
        + get_leakage_warning(context)
    )


def check_ablation_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the ablation study is finished."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"ablation_skip_data_leakage_check_{task_id}"] = True
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    result_dict = callback_context.state.get(
        f"ablation_code_exec_result_{step}_{task_id}", {}
    )
    if result_dict.get("returncode", 1) == 0:
        return llm_response_module.LlmResponse()
    callback_context.state[f"ablation_skip_data_leakage_check_{task_id}"] = (
        False
    )
    return None


def check_init_plan_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the initial plan is finished."""
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    code = callback_context.state.get(f"train_code_{step}_{task_id}", "")
    code_block = callback_context.state.get(
        f"refine_code_block_{step}_{task_id}", ""
    )
    status = code and code_block and (code_block in code)
    if status:
        return llm_response_module.LlmResponse()
    return None


def check_plan_implement_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the plan implement is finished."""
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
    suffix = f"{inner_iter}_{step}_{task_id}"
    result_dict = callback_context.state.get(
        f"train_code_improve_exec_result_{suffix}", {}
    )
    callback_context.state[
        f"plan_implement_skip_data_leakage_check_{suffix}"
    ] = True
    if result_dict:
        return llm_response_module.LlmResponse()
    callback_context.state[
        f"plan_implement_skip_data_leakage_check_{suffix}"
    ] = False
    return None


def get_ablation_summary(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Gets the ablation summary from the response."""
    response_text = common_util.get_text_from_response(llm_response)
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    callback_context.state[f"ablation_summary_{step}_{task_id}"] = response_text
    return None


def get_plan_and_code_block(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Gets the plan and code block from the response."""
    response_text = common_util.get_text_from_response(llm_response)
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    plan = ""
    code_block = ""
    try:
        # Try JSON array format first: [{...}]
        start_idx = response_text.find("[")
        end_idx = response_text.rfind("]") + 1
        if start_idx != -1 and end_idx > start_idx:
            result = json.loads(response_text[start_idx:end_idx])[0]
        else:
            raise ValueError("No JSON array found in response")
        plan = result["plan"]
        code_block = common_util.extract_code_block(result["code_block"])
    except Exception:
        # Fall back to JSON object format: {...}
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                result = json.loads(response_text[start_idx:end_idx])
                plan = result["plan"]
                code_block = common_util.extract_code_block(
                    result["code_block"]
                )
        except Exception:
            plan = ""
            code_block = ""
    callback_context.state[f"refine_plans_{step}_{task_id}"] = [plan]
    callback_context.state[f"refine_code_block_{step}_{task_id}"] = code_block
    return None



def get_refined_plan(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Gets the refined plan from the response."""
    response_text = common_util.get_text_from_response(llm_response)
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    callback_context.state[f"refine_plans_{step}_{task_id}"].append(
        response_text
    )
    return None


def build_refinement_agent(
    *,
    xai_gate_enabled: bool = False,
    name: str = "refinement_agent",
) -> agents.ParallelAgent:
    """Build the refinement ParallelAgent.

    When ``xai_gate_enabled`` is True, the per-step outer-loop selection runs the
    shared XAI leakage gate on each candidate and refuses to accept a leaking one
    (see :func:`update_outer_loop_states_gated`). The agent graph is otherwise
    identical, so the gated and ungated variants cannot drift apart.
    """
    use_data_leakage_checker = config.CONFIG.use_data_leakage_checker
    outer_loop_callback = (
        update_outer_loop_states_gated
        if xai_gate_enabled
        else update_outer_loop_states
    )
    refinement_parallel_sub_agents = []
    for k in range(config.CONFIG.num_solutions):
        ablation_agent = agents.Agent(
            model=llm.build_llm(),
            name=f"ablation_agent_{k + 1}",
            description="Perform ablation studies to improve the solution.",
            instruction=get_ablation_agent_instruction,
            before_model_callback=check_ablation_finish,
            after_model_callback=functools.partial(
                debug_util.get_code_from_response,
                do_eval=not use_data_leakage_checker,
            ),
            generate_content_config=types.GenerateContentConfig(
                temperature=1.0,
            ),
            include_contents="none",
        )
        ablation_sequential_sub_agents = [ablation_agent]
        if use_data_leakage_checker:
            data_leakage_checker_agent = (
                check_leakage_util.get_data_leakage_checker_agent(
                    prefix="ablation",
                    suffix=f"{k + 1}",
                )
            )
            ablation_sequential_sub_agents.append(data_leakage_checker_agent)
            additional_agent_description = (
                " and check if there are data leakage issues"
            )
        else:
            additional_agent_description = ""
        ablation_sequential_agent = agents.SequentialAgent(
            name=f"ablation_sequential_agent_{k + 1}",
            description=f"Perform ablation studies{additional_agent_description}.",
            sub_agents=ablation_sequential_sub_agents,
        )
        debug_inner_loop_agent = debug_util.get_debug_inner_loop_agent(
            prefix="ablation",
            suffix=f"{k + 1}",
        )
        ablation_and_debug_loop_agent = agents.LoopAgent(
            name=f"ablation_and_debug_loop_agent_{k + 1}",
            description="Perform ablation studies and debug the code until it succeeds.",
            sub_agents=[
                ablation_sequential_agent,
                debug_inner_loop_agent,
            ],
            max_iterations=config.CONFIG.max_rollback_round,
        )
        ablation_summary_agent = agents.Agent(
            model=llm.build_llm(),
            name=f"ablation_summary_agent_{k + 1}",
            description="Summarize the ablation study results.",
            instruction=get_ablation_summary_agent_instruction,
            after_model_callback=get_ablation_summary,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.0,
            ),
            include_contents="none",
        )
        init_plan_agent = agents.Agent(
            model=llm.build_llm(),
            name=f"init_plan_agent_{k + 1}",
            description="Generate an initial plan and a code block.",
            instruction=get_init_plan_agent_instruction,
            before_model_callback=check_init_plan_finish,
            after_model_callback=get_plan_and_code_block,
            tools=[],
            generate_content_config=types.GenerateContentConfig(
                temperature=1.0,
            ),
            include_contents="none",
        )
        init_plan_loop_agent = agents.LoopAgent(
            name=f"init_plan_loop_agent_{k + 1}",
            description=(
                "Generate an initial plan and a code block until the code block is valid."
            ),
            sub_agents=[init_plan_agent],
            before_agent_callback=init_inner_loop_states,
            max_iterations=config.CONFIG.max_retry,
        )
        init_plan_implement_agent = debug_util.get_run_and_debug_agent(
            prefix="plan_implement_initial",
            suffix=f"{k + 1}",
            agent_description="Implement the initial plan to generate a solution.",
            instruction_func=get_plan_implement_agent_instruction,
            before_model_callback=check_plan_implement_finish,
        )
        plan_refine_agent = agents.Agent(
            model=llm.build_llm(),
            name=f"plan_refine_agent_{k + 1}",
            description="Refine the plan.",
            instruction=get_plan_refinement_instruction,
            after_model_callback=get_refined_plan,
            tools=[],
            generate_content_config=types.GenerateContentConfig(
                temperature=1.0,
            ),
            include_contents="none",
        )
        plan_implement_agent = debug_util.get_run_and_debug_agent(
            prefix="plan_implement",
            suffix=f"{k + 1}",
            agent_description="Implement the plan to generate a solution.",
            instruction_func=get_plan_implement_agent_instruction,
            before_model_callback=check_plan_implement_finish,
        )
        plan_refine_and_implement_agent = agents.SequentialAgent(
            name=f"plan_refine_and_implement_agent_{k + 1}",
            description="Refine the plan and then implement it.",
            sub_agents=[
                plan_refine_agent,
                plan_implement_agent,
            ],
            after_agent_callback=update_inner_loop_states,
        )
        refine_inner_loop_agent = agents.LoopAgent(
            name=f"refine_inner_loop_agent_{k + 1}",
            description="Refine the given solution.",
            sub_agents=[plan_refine_and_implement_agent],
            before_agent_callback=update_inner_loop_states,
            max_iterations=config.CONFIG.inner_loop_round,
        )
        ablation_and_refine_agent = agents.SequentialAgent(
            name=f"ablation_and_refine_agent_{k + 1}",
            description="Perform ablation study and refine the code.",
            sub_agents=[
                ablation_and_debug_loop_agent,
                ablation_summary_agent,
                init_plan_loop_agent,
                init_plan_implement_agent,
                refine_inner_loop_agent,
            ],
            after_agent_callback=outer_loop_callback,
        )
        ablation_and_refine_loop_agent = agents.LoopAgent(
            name=f"ablation_and_refine_loop_agent_{k + 1}",
            description="Perform ablation study and refine the code for multiple rounds.",
            sub_agents=[ablation_and_refine_agent],
            before_agent_callback=init_outer_loop_states,
            max_iterations=config.CONFIG.outer_loop_round,
        )
        refinement_parallel_sub_agents.append(ablation_and_refine_loop_agent)
    return agents.ParallelAgent(
        name=name,
        description="Refine each solution by performing ablation studies.",
        sub_agents=refinement_parallel_sub_agents,
        before_agent_callback=None,
    )


# Default (ungated) refinement agent, used when the XAI step gate is disabled.
refinement_agent = build_refinement_agent(xai_gate_enabled=False)
