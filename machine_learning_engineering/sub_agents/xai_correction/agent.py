import os
from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_request as llm_request_module
from google.adk.models import llm_response as llm_response_module
from google.genai import types
from machine_learning_engineering.shared_libraries import (
    code_util,
    common_util,
    config,
    llm,
    xai_gate,
    xai_instrumentation_guide,
    xai_tracker,
    xai_util,
)
from machine_learning_engineering.sub_agents.xai_correction import prompt
from machine_learning_engineering.shared_libraries.skrub_guidance import (
    SKRUB_DATAOPS_GUIDELINE,
)


def _task_id(ctx) -> str:
    """Extract the 1-based solution index from the (suffixed) agent name.

    Every agent in the correction pipeline is named ``..._{k+1}`` so that its
    callbacks/instructions can recover which parallel solution they operate on,
    mirroring the convention used by the initialization and refinement stages.
    """
    return ctx.agent_name.split("_")[-1]


def prepare_xai(callback_context):
    # Open the XAI overhead region; pass_to_refinement (always the last sub-agent)
    # closes it. Everything in between -- audits, instrumentation, feature
    # revision, re-execution, termination report -- is attributed to XAI.
    # The depth counter composes across the parallel per-solution regions.
    xai_tracker.enter_xai_phase()
    task_id = _task_id(callback_context)
    code = callback_context.state.get(f"train_code_0_{task_id}", "")  # best solution from last step
    callback_context.state[f"xai_current_code_{task_id}"] = code
    callback_context.state[f"xai_loop_count_{task_id}"] = 0
    callback_context.state[f"xai_terminate_{task_id}"] = False
    callback_context.state[f"xai_action_history_{task_id}"] = []
    callback_context.state[f"xai_issue_history_{task_id}"] = []
    callback_context.state[f"xai_audit_mode_{task_id}"] = ""
    callback_context.state[f"xai_audit_error_{task_id}"] = ""


# Tim's xai calculation and audit part




# 1. Routing decision: read JSON verdict and determine PASS or FAIL
def routing_decision(callback_context):
    task_id = _task_id(callback_context)
    audit_result = callback_context.state.get(f"xai_audit_result_{task_id}", None)
    # No result at all: previous step did not run
    if audit_result is None:
        callback_context.state[f"xai_verdict_{task_id}"] = "ERROR"
        raise ValueError("XAI audit result is None")
    # Empty result: audit ran but returned nothing valid
    if audit_result == {}:
        callback_context.state[f"xai_verdict_{task_id}"] = "EMPTY"
        raise ValueError("XAI audit result is empty")

    verdict = audit_result.get("verdict", None)
    if verdict is None:
        callback_context.state[f"xai_verdict_{task_id}"] = "EMPTY"
        raise ValueError("XAI verdict is missing")

    callback_context.state[f"xai_verdict_{task_id}"] = verdict
    reason = audit_result.get("reason", "")
    print(f"\n==========================================")
    print(f"[XAI Correction Audit {task_id}] Verdict: {verdict}")
    print(f"[XAI Correction Audit {task_id}] Reason: {reason}")
    print(f"==========================================\n")

# 2. Control loop iterations: increment count and decide whether to terminate
def loop_controller(callback_context):
    # Verdict is not FAIL (PASS): no revision needed, so break the LoopAgent
    # immediately instead of burning the remaining max_iterations re-running the
    # audit. Escalate stops the LoopAgent but NOT the outer SequentialAgent, so
    # pass_to_refinement still runs afterward.
    task_id = _task_id(callback_context)
    if callback_context.state.get(f"xai_verdict_{task_id}") != "FAIL":
        callback_context.actions.escalate = True
        return types.Content(parts=[])

    loop_count = callback_context.state.get(f"xai_loop_count_{task_id}", 0)
    loop_count += 1
    callback_context.state[f"xai_loop_count_{task_id}"] = loop_count

    audit_result = callback_context.state.get(f"xai_audit_result_{task_id}", {})
    audit_history = callback_context.state.get(f"xai_audit_history_{task_id}", [])
    audit_history.append(audit_result)
    callback_context.state[f"xai_audit_history_{task_id}"] = audit_history

    if loop_count > 3:
        # Hit the revision budget; stop looping and let the termination report run.
        callback_context.state[f"xai_terminate_{task_id}"] = True
        callback_context.actions.escalate = True
        return

    callback_context.state[f"xai_terminate_{task_id}"] = False


def check_should_revise(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    task_id = _task_id(callback_context)
    if callback_context.state.get(f"xai_verdict_{task_id}") != "FAIL":
        return llm_response_module.LlmResponse()
    if callback_context.state.get(f"xai_terminate_{task_id}", False):
        return llm_response_module.LlmResponse()
    return None

def get_feature_revision_instruction(context):
    task_id = _task_id(context)
    audit_result = context.state.get(f"xai_audit_result_{task_id}", {})
    current_code = context.state.get(f"xai_current_code_{task_id}", "")
    return prompt.FEATURE_REVISION_INSTR.format(
        audit_result = audit_result,
        current_code = current_code,
    ) + SKRUB_DATAOPS_GUIDELINE

def save_revised_code(callback_context, llm_response):
    task_id = _task_id(callback_context)
    response_text = common_util.get_text_from_response(llm_response)
    code = common_util.extract_code_block(response_text)
    if code:
        callback_context.state[f"xai_current_code_{task_id}"] = code
        print(f"\n[XAI Correction {task_id}] Feature revision proposed and applied to the training script.\n")
    return None


def re_execute_revised_code(callback_context):
    task_id = _task_id(callback_context)
    if callback_context.state.get(f"xai_verdict_{task_id}") != "FAIL":
        return

    code = callback_context.state.get(f"xai_current_code_{task_id}", "")
    if not code:
        return

    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    exec_timeout = callback_context.state.get("exec_timeout", 1800)
    run_cwd = os.path.join(workspace_dir, task_name, task_id)

    result_dict = code_util.run_python_code(
        code_text = code,
        run_cwd = run_cwd,
        py_filepath = "train0.py",
        exec_timeout = exec_timeout,
    )

    # Parse validation score to avoid KeyError down the line
    lower = callback_context.state.get("lower", True)
    if result_dict.get("returncode", 1) == 0:
        try:
            score = code_util.extract_performance_from_text(result_dict.get("stdout", ""))
            result_dict["score"] = float(score)
        except Exception:
            result_dict["score"] = 1e9 if lower else 0.0
    else:
        result_dict["score"] = 1e9 if lower else 0.0

    callback_context.state[f"train_code_0_{task_id}"] = code
    callback_context.state[f"train_code_exec_result_0_{task_id}"] = result_dict


def check_should_terminate(callback_context, llm_request):
    task_id = _task_id(callback_context)
    if callback_context.state.get(f"xai_terminate_{task_id}", False):
        return None
    return llm_response_module.LlmResponse()

def get_termination_report_instruction(context):
    task_id = _task_id(context)
    audit_history = context.state.get(f"xai_audit_history_{task_id}", [])
    loop_count = context.state.get(f"xai_loop_count_{task_id}", 0)
    return prompt.TERMINATION_REPORT_INSTR.format(
        loop_count = loop_count,
        audit_history = audit_history
    )

def after_termination_report(callback_context, llm_response):
    callback_context.actions.escalate = True
    return None


def pass_to_refinement(callback_context):
    # Close the XAI overhead region opened in prepare_xai. Do this first so the
    # depth counter is balanced regardless of which branch we take below.
    xai_tracker.exit_xai_phase()
    task_id = _task_id(callback_context)
    if callback_context.state.get(f"xai_verdict_{task_id}") != "PASS":
        return types.Content(parts=[])

    original_code = callback_context.state.get(f"train_code_0_{task_id}", "")
    revised_code = callback_context.state.get(f"xai_current_code_{task_id}", "")
    if original_code != revised_code:
        callback_context.state[f"train_code_0_{task_id}"] = revised_code
        print(f"\n[XAI Correction {task_id}] Successfully revised features and handed off to Refinement phase!\n")
    else:
        print(f"\n[XAI Correction {task_id}] Code passed the initial XAI audit without requiring any changes.\n")


def get_audit_instruction(context):
    task_id = _task_id(context)
    code = context.state.get(f"xai_current_code_{task_id}", "")
    audit_error = context.state.get(f"xai_audit_error_{task_id}", "") or "None"
    metrics = context.state.get(f"xai_audit_metrics_preview_{task_id}", "")
    if not metrics:
        metrics = "None (dynamic metrics unavailable)"
    return prompt.AUDIT_INSTR.format(
        code=code,
        audit_error=audit_error,
        xai_metrics_summary=metrics,
    )


def _normalize_audit_verdict(raw_verdict: object) -> str | None:
    if not isinstance(raw_verdict, str):
        return None
    verdict = raw_verdict.strip().upper()
    if verdict in ("PASS", "FAIL"):
        return verdict
    return None


def save_audit_result(callback_context, llm_response):
    task_id = _task_id(callback_context)
    response_text = common_util.get_text_from_response(llm_response)
    audit_dict = common_util.extract_json_dict(
        response_text,
        required_keys=frozenset({"verdict"}),
    )
    if audit_dict:
        verdict = _normalize_audit_verdict(audit_dict.get("verdict"))
        if verdict:
            callback_context.state[f"xai_audit_result_{task_id}"] = {
                "verdict": verdict,
                "reason": str(audit_dict.get("reason", "")),
            }
            return None

    # Fallback to heuristic text-based parsing if JSON parsing fails
    text_lower = response_text.lower()
    verdict = None
    if '"verdict": "pass"' in text_lower or '"verdict":"pass"' in text_lower:
        verdict = "PASS"
    elif '"verdict": "fail"' in text_lower or '"verdict":"fail"' in text_lower:
        verdict = "FAIL"
    elif "verdict: pass" in text_lower or "verdict is pass" in text_lower:
        verdict = "PASS"
    elif "verdict: fail" in text_lower or "verdict is fail" in text_lower:
        verdict = "FAIL"

    if verdict:
        callback_context.state[f"xai_audit_result_{task_id}"] = {
            "verdict": verdict,
            "reason": (
                "Audit verdict inferred from unstructured model output; "
                "response was not valid JSON."
            ),
        }
    else:
        callback_context.state[f"xai_audit_result_{task_id}"] = {
            "verdict": "FAIL",
            "reason": (
                "Failed to parse audit response as JSON. "
                f"Response preview: {response_text[:500]}"
            ),
        }
    return None


def _allow_static_fallback(callback_context: callback_context_module.CallbackContext) -> bool:
    if "xai_allow_static_fallback" in callback_context.state:
        return bool(callback_context.state["xai_allow_static_fallback"])
    return config.CONFIG.xai_allow_static_fallback


def _audit_llm_response(audit_dict: dict) -> llm_response_module.LlmResponse:
    import json

    response_text = f"```json\n{json.dumps(audit_dict)}\n```"
    part = types.Part(text=response_text)
    return llm_response_module.LlmResponse(
        content=types.Content(parts=[part], role="model")
    )


def _instrument_code_via_llm(
    code: str,
    *,
    lower_is_better: bool,
    error: str = "",
) -> str | None:
    """Ask the LLM to add or fix XAI instrumentation using the shared prompt example."""
    label = "fix" if error else "add"
    print(f"\n[XAI Correction Audit] LLM will {label} XAI instrumentation...\n")
    instrumented = xai_instrumentation_guide.instrument_via_llm(
        code, lower_is_better=lower_is_better, error=error
    )
    if not instrumented:
        print("[XAI Correction Audit] Failed to extract instrumented code block.")
        return None
    return instrumented


def dynamic_audit_before_model(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Run numeric XAI audit; bypass static LLM audit when metrics are valid."""
    task_id = _task_id(callback_context)
    try:
        code = callback_context.state.get(f"xai_current_code_{task_id}", "")
        if not code:
            callback_context.state[f"xai_audit_error_{task_id}"] = "No training code in state"
            if not _allow_static_fallback(callback_context):
                audit_dict = {
                    "verdict": "FAIL",
                    "reason": "Dynamic XAI audit failed: no training code.",
                }
                callback_context.state[f"xai_audit_mode_{task_id}"] = "failed"
                callback_context.state[f"xai_audit_result_{task_id}"] = audit_dict
                return _audit_llm_response(audit_dict)
            callback_context.state[f"xai_audit_mode_{task_id}"] = "static_fallback"
            return None

        lower = callback_context.state.get("lower", True)
        workspace_dir = callback_context.state.get("workspace_dir", "")
        task_name = callback_context.state.get("task_name", "")
        run_cwd = os.path.join(workspace_dir, task_name, task_id)
        os.makedirs(run_cwd, exist_ok=True)

        loop_count = callback_context.state.get(f"xai_loop_count_{task_id}", 0)
        print("\n[XAI Correction Audit] Running instrumented code...\n")
        result = xai_gate.audit_code_for_leakage(
            code=code,
            run_cwd=run_cwd,
            lower=lower,
            exec_timeout=callback_context.state.get("exec_timeout", 1800),
            py_filepath="train0.py",
            max_concentration=callback_context.state.get("xai_max_concentration", 0.80),
            instrument_fn=lambda c, e: _instrument_code_via_llm(
                c, lower_is_better=lower, error=e
            ),
            metrics_archive_label=f"correction_audit_loop{loop_count}",
        )
        # Persist the (possibly instrumented) code the gate actually ran.
        callback_context.state[f"xai_current_code_{task_id}"] = result.code

        if result.metrics is not None:
            callback_context.state[f"train_code_exec_result_0_{task_id}"] = result.exec_result
            audit_dict = {"verdict": result.verdict, "reason": result.reason}
            callback_context.state[f"xai_audit_result_{task_id}"] = audit_dict
            callback_context.state[f"xai_audit_mode_{task_id}"] = "dynamic"
            callback_context.state[f"xai_audit_error_{task_id}"] = ""
            callback_context.state[f"xai_audit_metrics_preview_{task_id}"] = result.metrics_preview
            print("[XAI Correction Audit] Dynamic audit complete!")
            print(f"[XAI Correction Audit] Verdict: {result.verdict}")
            print(f"[XAI Correction Audit] Reason: {result.reason}\n")
            return _audit_llm_response(audit_dict)

        last_error = result.error
        callback_context.state[f"xai_audit_error_{task_id}"] = last_error
        callback_context.state[f"xai_audit_metrics_preview_{task_id}"] = ""
        print(f"[XAI Correction Audit] Dynamic audit failed: {last_error}")

        if not _allow_static_fallback(callback_context):
            audit_dict = {
                "verdict": "FAIL",
                "reason": (
                    "Dynamic XAI audit could not produce valid metrics. "
                    f"{last_error}"
                ),
            }
            callback_context.state[f"xai_audit_result_{task_id}"] = audit_dict
            callback_context.state[f"xai_audit_mode_{task_id}"] = "failed"
            print(
                "[XAI Correction Audit] Fail-closed: verdict FAIL (static fallback disabled).\n"
            )
            return _audit_llm_response(audit_dict)

        callback_context.state[f"xai_audit_mode_{task_id}"] = "static_fallback"
        print("[XAI Correction Audit] Falling back to static LLM audit.\n")
        return None
    except Exception as e:
        err = f"Exception during dynamic audit: {e}"
        callback_context.state[f"xai_audit_error_{task_id}"] = err
        callback_context.state[f"xai_audit_metrics_preview_{task_id}"] = ""
        print(f"[XAI Correction Audit] {err}")
        if not _allow_static_fallback(callback_context):
            audit_dict = {
                "verdict": "FAIL",
                "reason": f"Dynamic XAI audit failed: {e}",
            }
            callback_context.state[f"xai_audit_result_{task_id}"] = audit_dict
            callback_context.state[f"xai_audit_mode_{task_id}"] = "failed"
            return _audit_llm_response(audit_dict)
        callback_context.state[f"xai_audit_mode_{task_id}"] = "static_fallback"
        return None


def build_xai_correction_solution_agent(task_id: int) -> agents.SequentialAgent:
    """Build the full XAI self-correction pipeline for one parallel solution.

    ``task_id`` is the 1-based solution index; every agent is named ``..._{task_id}``
    so its callbacks/instructions can recover which solution they operate on and
    read/write the correct per-solution state keys and workspace directory.
    """
    xai_audit_agent = agents.Agent(
        model=llm.build_llm(),
        name=f"xai_audit_agent_{task_id}",
        description="Audit the current code for target leakage and explainability issues.",
        instruction=get_audit_instruction,
        before_model_callback=dynamic_audit_before_model,
        after_model_callback=save_audit_result,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        include_contents="none",
    )
    routing_agent = agents.SequentialAgent(
        name=f"xai_routing_agent_{task_id}",
        description="Route based on audit verdict.",
        before_agent_callback=routing_decision,
    )
    loop_control_agent = agents.SequentialAgent(
        name=f"xai_loop_control_agent_{task_id}",
        description="Control loop iterations.",
        before_agent_callback=loop_controller,
    )
    feature_revision_agent = agents.Agent(
        model=llm.build_llm(),
        name=f"feature_revision_agent_{task_id}",
        description="Revise features based on XAI audit findings.",
        instruction=get_feature_revision_instruction,
        before_model_callback=check_should_revise,
        after_model_callback=save_revised_code,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        include_contents="none",
    )
    re_execute_agent = agents.SequentialAgent(
        name=f"xai_re_execute_agent_{task_id}",
        description="Re-execute revised code after XAI feature revision.",
        before_agent_callback=re_execute_revised_code,
    )
    termination_report_agent = agents.Agent(
        model=llm.build_llm(),
        name=f"xai_termination_report_agent_{task_id}",
        description="Generate termination report when max loop count is reached.",
        instruction=get_termination_report_instruction,
        before_model_callback=check_should_terminate,
        after_model_callback=after_termination_report,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        include_contents="none",
    )
    pass_to_refinement_agent = agents.SequentialAgent(
        name=f"xai_pass_to_refinement_agent_{task_id}",
        description="Update train code with XAI-revised code before entering refinement.",
        before_agent_callback=pass_to_refinement,
    )

    xai_inner_loop = agents.LoopAgent(
        name=f"xai_inner_loop_{task_id}",
        description="XAI audit + revision loop, max 3 iterations.",
        sub_agents=[
            xai_audit_agent,
            routing_agent,
            loop_control_agent,
            feature_revision_agent,
            re_execute_agent,
        ],
        max_iterations=3,
    )

    return agents.SequentialAgent(
        name=f"xai_correction_agent_{task_id}",
        description="XAI self-correction loop.",
        sub_agents=[
            xai_inner_loop,
            termination_report_agent,
            pass_to_refinement_agent,
        ],
        before_agent_callback=prepare_xai,
    )


# One correction pipeline per solution, run in parallel -- mirrors the
# initialization and refinement stages so that *every* candidate solution
# (train_code_0_1, train_code_0_2, ...) is audited and corrected, not just the
# first one.
xai_correction_agent = agents.ParallelAgent(
    name="xai_correction_agent",
    description="Audit and correct every parallel solution for target leakage.",
    sub_agents=[
        build_xai_correction_solution_agent(k + 1)
        for k in range(config.CONFIG.num_solutions)
    ],
)
