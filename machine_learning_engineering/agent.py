# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demonstration of Machine Learning Engineering Agent using Agent Development Kit."""

import json
import os

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.genai import types

from machine_learning_engineering import prompt
from machine_learning_engineering.shared_libraries import code_util, config, llm, xai_tracker
from machine_learning_engineering.sub_agents.ensemble import (
    agent as ensemble_agent_module,
)
from machine_learning_engineering.sub_agents.initialization import (
    agent as initialization_agent_module,
)
from machine_learning_engineering.sub_agents.xai_correction import (
    agent as xai_correction_agent_module,
)
from machine_learning_engineering.sub_agents.refinement import (
    agent as refinement_agent_module,
)
from machine_learning_engineering.sub_agents.submission import (
    agent as submission_agent_module,
)


def reset_overhead_tracker(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Clear the per-task XAI overhead counters at the start of the run."""
    xai_tracker.reset(callback_context.state.get("task_name", ""))
    return None


def _persist_final_validation_performance(
    callback_context: callback_context_module.CallbackContext,
) -> None:
    """Promote final_solution.py validation output to top-level state keys."""
    submission_res = callback_context.state.get("submission_code_exec_result")
    if not isinstance(submission_res, dict):
        return
    stdout = submission_res.get("stdout") or ""
    perf_line = code_util.extract_performance_line_from_text(stdout)
    if perf_line is None:
        return
    callback_context.state["final_validation_performance_print"] = perf_line
    score = submission_res.get("score")
    if score is None:
        score = code_util.extract_performance_from_text(stdout)
    if score is not None:
        callback_context.state["final_validation_performance"] = score


def save_state(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Prints the current state of the callback context."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name)
    _persist_final_validation_performance(callback_context)
    with open(os.path.join(run_cwd, "final_state.json"), "w") as f:
        json.dump(callback_context.state.to_dict(), f, indent=2)
    # Emit the per-task XAI overhead report (extra API calls, tokens, runtime).
    try:
        report = xai_tracker.write_report(
            run_cwd,
            xai_correction_enabled=bool(
                callback_context.state.get(
                    "use_xai_correction", config.CONFIG.use_xai_correction
                )
            ),
            xai_refinement_enabled=bool(
                callback_context.state.get(
                    "use_xai_refinement", config.CONFIG.use_xai_refinement
                )
            ),
        )
        print(
            f"[xai_tracker] Wrote xai_overhead.json: "
            f"{report['xai_api_calls']} XAI calls, "
            f"{report['xai_tokens']} XAI tokens, "
            f"{report['xai_total_overhead_s']:.1f}s XAI runtime."
        )
    except Exception as exc:  # never fail the run on telemetry
        print(f"[xai_tracker] Failed to write overhead report: {exc}")
    return None


pipeline_sub_agents = [initialization_agent_module.initialization_agent]

if config.CONFIG.use_xai_correction:
    pipeline_sub_agents.append(xai_correction_agent_module.xai_correction_agent)

# `use_xai_refinement` now selects the standard refinement pipeline with the
# per-step XAI leakage gate enabled (built from the same code as the ungated one,
# so the two cannot drift). The legacy standalone `xai_refinement` agent is retired.
if config.CONFIG.use_xai_refinement:
    pipeline_sub_agents.append(
        refinement_agent_module.build_refinement_agent(
            xai_gate_enabled=True,
            name="xai_refinement_agent",
        )
    )
else:
    pipeline_sub_agents.append(refinement_agent_module.refinement_agent)

pipeline_sub_agents.extend([
    ensemble_agent_module.ensemble_agent,
    submission_agent_module.submission_agent,
])


mle_pipeline_agent = agents.SequentialAgent(
    name="mle_pipeline_agent",
    sub_agents=pipeline_sub_agents,
    description="Executes a sequence of sub-agents for solving the MLE task.",
    before_agent_callback=reset_overhead_tracker,
    after_agent_callback=save_state,
)

# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = agents.Agent(
    model=llm.build_llm(),
    name="mle_frontdoor_agent",
    instruction=prompt.FRONTDOOR_INSTRUCTION,
    global_instruction=prompt.SYSTEM_INSTRUCTION,
    sub_agents=[mle_pipeline_agent],
    tools=[],
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
