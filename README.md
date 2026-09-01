# Data Engineering for AI & ML (Group 1, Theme 1a: Improve MLE-Star)

A fork of Google's [MLE-STAR ADK sample](https://github.com/google/adk-samples/tree/main/python/agents/machine-learning-engineering/machine_learning_engineering).

See also: [EXPERIMENTS.md](EXPERIMENTS.md) (how to run the experiments), [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md) (results and discussion).

---

## TL;DR: Execute Full Pipeline

To set up, download the datasets, inject data leakage, run the entire matrix, and generate final evaluation results:

```bash
# 1. Install dependencies & configure env (add keys to .env)
uv sync --dev
cp .env.example .env

# 2. Download tasks and inject proxy leakage
python machine_learning_engineering/download_tasks.py
python scripts/prepare_leakage_tasks.py

# 3. Run the full experimental matrix (XAI-off vs XAI-on, Clean vs Corrupted)
python machine_learning_engineering/auto_run_all_tasks.py --matrix --corrupted && \
python machine_learning_engineering/auto_run_all_tasks.py --matrix --clean

# 4. Generate evaluation metrics and figures
python eval2.py
python make_figures.py
```

---

## Task Description

Improve MLE-STAR

* Start from the MLE-Star implementation available at
https://github.com/google/adk-samples/tree/main/python/agents/machine-learning-engineering/machine_learning_engineering
[1]
* Make it produce skrub DataOps pipelines
* Pick one component and improve it with your own ideas
* EXPERIMENT WITH AT LEAST 10 TASKS from
MLE-Bench Lite
https://github.com/openai/mle-bench/blob/main/experiments/splits/low.txt
[2] or Kaggle
https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions
[3]

We fulfil the requirements as follows:

1. **skrub DataOps pipelines** - every code-generating stage is steered (via layered prompt injection) to express its solution as a skrub DataOps computation graph. See [Contribution 1](#contribution-1-skrub-dataops-pipelines).
2. **Improved component** - we extend MLE-STAR's data-leakage checker. The baseline has a *static* checker: an LLM reads the generated script and guesses whether the model trains on validation or test data. We add a *dynamic, XAI-based* leakage checker: the generated pipeline is executed under a probe suite (SHAP/permutation attribution, masking, proxy-power analysis), and a bounded correction loop lets the LLM fix what the probes flag. See [Contribution 2](#contribution-2-xai-data-leakage-checker).
3. **≥10 tasks** - 10 tabular Kaggle tasks (see [Datasets](#datasets)), each run in a full A/B matrix: {clean, leak-injected} × {XAI gate off, XAI gate on} = 40 pipeline runs. See [Evaluation methodology](#evaluation-methodology-the-xai-ab-experiment).

---

## Contribution 1: skrub DataOps pipelines

Every code-generating agent is steered to produce [skrub](https://skrub-data.org/) DataOps pipelines instead of plain sklearn/pandas code. This works entirely through prompt injection. The LLM writes the DataOps code itself, and the debug loop repairs it within the same abstraction.

### What the guideline enforces

[skrub_guidance.py](machine_learning_engineering/shared_libraries/skrub_guidance.py) is the single source of truth. Its main string, [SKRUB_DATAOPS_GUIDELINE](machine_learning_engineering/shared_libraries/skrub_guidance.py#L45), mandates the **graph shape**:

1. **Load** inputs through `skrub.var(...)` (so the same graph replays on unseen test data),
2. **mark** target/features via `.skb.mark_as_y()` / `.skb.mark_as_X()` immediately after load,
3. **feature-engineer** only on tracked nodes (`.skb.apply_func`, `.assign`),
4. **model** via `.skb.apply(estimator, y=y)`,
5. **export** a fit/predict object with `.skb.make_learner()` and predict on test data by re-binding the environment (`learner.predict({"data": test_df})`).

It explicitly forbids the failure modes we saw most often: raw `model.fit(X, y)` / `sklearn.pipeline.Pipeline` escapes, the legacy `skrub.tabular_pipeline` API, hallucinated `skrub.preprocessing.*` modules, `train_test_split` on raw arrays (which silently breaks fit/predict duality), emitting the placeholder column name `"target"` literally, and doing math on string columns. It also documents how to combine several models' predictions as DataOps nodes rather than falling back to numpy arrays, which is what keeps the ensemble stage inside the abstraction.

### Tiered injection policy

Each agent type only receives the part of the guideline it needs, to keep prompts short:

| Constant | Injected into | Purpose |
| --- | --- | --- |
| [SKRUB_DATAOPS_GUIDELINE](machine_learning_engineering/shared_libraries/skrub_guidance.py#L45) | Code-writing agents ([prompt.py](machine_learning_engineering/sub_agents/initialization/prompt.py), [prompt.py](machine_learning_engineering/sub_agents/refinement/prompt.py), the XAI revision prompt) | The full guide with do's/don'ts and patterns. |
| [SKRUB_DATAOPS_DEBUG_GUIDELINE](machine_learning_engineering/shared_libraries/skrub_guidance.py#L442) | Debug/repair agents ([debug_prompt.py](machine_learning_engineering/shared_libraries/debug_prompt.py)) | A condensed "stay inside DataOps while fixing this traceback" reminder. |
| [SKRUB_DATAOPS_PLAN_CONSTRAINT](machine_learning_engineering/shared_libraries/skrub_guidance.py#L542) | Plan-only agents ([agent.py](machine_learning_engineering/sub_agents/refinement/agent.py) appends it to the planning instruction) | Keeps ablation/refinement *plans* inside the DataOps graph so a plan can't propose a plain-sklearn rewrite that the implementer would then follow. |
| [SKRUB_SUBMISSION_CONSTRAINT](machine_learning_engineering/shared_libraries/skrub_guidance.py#L552) | Submission agent | How to produce `submission.csv` from the exported learner. |
| [SKRUB_RAG_HINT](machine_learning_engineering/shared_libraries/skrub_guidance.py#L619) | Agents with RAG tool access only | Points at the optional `query_skrub_documentation` tool. |

In the initial phase of the project we built a RAG system to produce skrub DataOps, however due to poor performance it was abandoned for the prompt-injection approach.
When RAG is disabled (the default, `MLE_USE_RAG=0`), [strip_rag_tool_mentions()](machine_learning_engineering/shared_libraries/skrub_guidance.py#L36) removes every reference to the tool from the prompts at import time, so the model is never told about a tool it cannot call.

## Contribution 2: XAI data-leakage checker

### Motivation

Upstream MLE-STAR's leakage checker ([check_leakage_util.py](machine_learning_engineering/shared_libraries/check_leakage_util.py), flag [use_data_leakage_checker](machine_learning_engineering/shared_libraries/config.py#L54)) is **static**: an LLM reads the generated script and judges whether it leaks. That works for syntactic leaks (fitting a scaler on train+test) but not for leaks that live in the data itself, e.g. a column that is a near-perfect proxy of the target under an innocent name. Reading the code cannot reveal that `MerchantRiskIndex` correlates 0.99 with the label; you have to run the model and look at what it relies on.

Our gate is *dynamic*: it executes the generated pipeline, computes feature attributions and targeted probe statistics on the fitted model, and turns them into a PASS/FAIL verdict. On FAIL, the LLM is asked to revise its own feature engineering and the audit runs again, in a bounded loop.

### How one audit works

1. **Instrumentation.** The LLM is shown [xai_instrumentation_guide.py](machine_learning_engineering/shared_libraries/xai_instrumentation_guide.py), which instructs it to add exactly one call to its script - [run_leakage_suite()](machine_learning_engineering/shared_libraries/xai_probes.py#L656) - wiring in its own learner/data variable names. It is explicitly told *not* to compute SHAP or write any JSON itself.
2. **Shipped probe library.** [xai_probes.py](machine_learning_engineering/shared_libraries/xai_probes.py) (copied into each run's workspace) does all the actual measurement. We ship it as a fixed library instead of letting the LLM generate the instrumentation, for two reasons: a multi-probe suite is too complex to regenerate reliably every run, and our earlier LLM-generated SHAP code silently mis-mapped one-input→many-output encoders (skrub's `GapEncoder` names outputs `"col: ..."`, which a `col_`-prefix rule misses). That fragmented the attribution and deflated the concentration signal, producing false negatives in the detector. The library aggregates SHAP values (with a permutation-importance fallback for non-tree models) from *preprocessed* feature space back onto *original input columns*, so attribution is comparable across whatever encoding the generated pipeline chose.
3. **Probes.** Each leak type has a dedicated, independently-guarded probe (one probe crashing never aborts the suite):

   | Probe | Question it answers |
   | --- | --- |
   | `direct` | If the top-attributed feature is masked (train-mean/mode), does validation performance collapse? A large robustness drop on a highly concentrated feature indicates the model leans on a leak. |
   | `proxy_power` | Can a shallow decision stump/tree on a *single* feature nearly match the full model's lift over a majority/mean baseline? If one column alone explains the target, it's a proxy for it. |
   | `semantic_candidates` | Exports the top-attributed features with stats so the *LLM* can judge semantics (is this a post-outcome variable / identifier?). The probe library itself has no LLM access. |


4. **Verdict.** The suite writes a structured `xai_metrics.json` (per-probe suspects, severity, evidence). [xai_util.py](machine_learning_engineering/shared_libraries/xai_util.py) parses it and [evaluate_leakage_report()](machine_learning_engineering/shared_libraries/xai_util.py#L224) combines the signals (attribution concentration above `xai_max_concentration`, default 0.80, on a suspicious feature; probe severities; robustness collapse) into PASS/FAIL with a recommended action.
5. **Single audit entry point.** [xai_gate.py](machine_learning_engineering/shared_libraries/xai_gate.py) (specifically [audit_code_for_leakage()](machine_learning_engineering/shared_libraries/xai_gate.py#L101)) wraps steps 1–4 ("run this training code, return a `GateResult`") and has no ADK context coupling, so both integration points below share one audit implementation and cannot drift apart.

### Where the gate plugs into the pipeline

* **[xai_correction/](machine_learning_engineering/sub_agents/xai_correction/) - the correction stage** (flag `use_xai_correction`, **on by default**). Runs between `initialization` and `refinement`, once per parallel solution branch. An ADK `LoopAgent` audits the current best baseline (`train0.py`); on FAIL, a routing callback hands the probe evidence (flagged features, robustness numbers, semantic candidates) to a revision agent that rewrites the feature engineering, and the loop re-audits. At most 3 revision rounds are allowed, and the full action/issue history is kept in state so later rounds see what was already tried. On PASS (or exhaustion) the surviving code is handed to `refinement`. If the dynamic audit itself cannot produce metrics (script crash, missing JSON), `xai_allow_static_fallback` decides whether to fall back to an LLM code review or treat it as a hard FAIL.
* **Gated refinement** (flag `use_xai_refinement`, off by default). Alternatively, the same gate is applied *inside* the refinement loop: every improving candidate is re-audited before being accepted ([update_outer_loop_states_gated()](machine_learning_engineering/sub_agents/refinement/agent.py#L150) vs. the ungated [update_outer_loop_states()](machine_learning_engineering/sub_agents/refinement/agent.py#L113), both in [agent.py](machine_learning_engineering/sub_agents/refinement/agent.py)). This catches leaks that refinement itself introduces, at the cost of one audit per accepted candidate.

### Overhead accounting

[xai_tracker.py](machine_learning_engineering/shared_libraries/xai_tracker.py) measures what the gate costs in LLM API calls, tokens, and wall-clock seconds. All LLM traffic funnels through a single litellm callback and all code executions through [code_util.py](machine_learning_engineering/shared_libraries/code_util.py) (specifically `run_python_code`), so two hooks capture everything; a depth counter marks "currently inside the XAI module", attributing nested regions (correction loop wrapping the shared gate) correctly. Each run writes `xai_overhead.json` next to its `final_state.json`, which the evaluation reads (columns `xai_api_calls`, `xai_tokens`, `xai_total_overhead_s` in the metrics table).

### Config knobs

In [config.py](machine_learning_engineering/shared_libraries/config.py) (all overridable via environment variables): `use_xai_correction` (env `USE_XAI_CORRECTION`, default **True**), `use_xai_refinement` (env `USE_XAI_REFINEMENT`, default False), `xai_max_concentration` (default 0.80), `xai_allow_static_fallback` (env `XAI_ALLOW_STATIC_FALLBACK`, default False). The pre-existing static checker remains available as `use_data_leakage_checker`; the XAI gate is additive, not a replacement of that code path.

---

## Evaluation methodology: the XAI A/B experiment

The evaluation answers three questions: does the gate catch injected leaks, how many legitimate features does it falsely remove on clean data, and how much compute does it add?

### 1. Controlled leak injection

[prepare_leakage_tasks.py](scripts/prepare_leakage_tasks.py) copies each clean task to a corrupted variant with exactly **one** injected leak (one leak per target feature, so detection is attributable):

* **`<task>_proxy`** (default): a plausibly-named feature that is a ~0.99 Pearson correlate of the target but contains no syntactic reference to it (e.g. `MerchantRiskIndex`, `palate balance index`). A code-reading static checker cannot detect this by construction; it is the case that motivates the dynamic gate.

The leaked column is neutralized on the test split (set to train-mean/mode), simulating a leak that is unavailable in production. A model that relies on it therefore gets a near-perfect validation score and a much worse test score, which is what the generalization-gap metric measures. Injection is seeded (`LEAK_SEED`) and recorded in each variant's `leakage_metadata.json` (ground-truth feature name, used by the eval to score detection).

### 2. The 40-run A/B matrix

`python machine_learning_engineering/auto_run_all_tasks.py --matrix --corrupted && python machine_learning_engineering/auto_run_all_tasks.py --matrix --clean` runs each selected task through both variants, on clean and corrupted data:

| Variant | Env overrides | Workspace |
| --- | --- | --- |
| `baseline` ("vanilla") | `USE_XAI_CORRECTION=False` | [ws_baseline/](ws_baseline/) |
| `baseline_xai` ("ours") | `USE_XAI_CORRECTION=True` | [ws_baseline_xai/](ws_baseline_xai/) |

10 tasks × {clean, `_proxy`} × 2 variants = 40 runs, each a full MLE-STAR pipeline. Finished cells (existing `final_state.json`) are skipped, so the run can be resumed; this matters because the GWDG API budget spreads it over multiple days. [scan_tasks_status.py](scripts/scan_tasks_status.py) prints the completion grid with the exact command to fill each missing cell. For running the matrix on a SLURM cluster instead of locally, see [HPC batch runs](#hpc-batch-runs).

### 3. Metrics and aggregation

[eval2.py](eval2.py) walks both workspaces, re-scores every run's `submission.csv` against held-out ground truth, mines the final generated code and XAI audit artifacts, and writes one row per run to [metrics_table2.csv](metrics_table2.csv). It prints five headline metrics:

1. **DR_feat** - feature-level detection rate: on corrupted runs of *ours*, fraction where the injected feature was flagged **and actually removed** from the final code.
2. **FPR_feat** - false-positive rate: fraction of legitimate features falsely dropped by the gate.
3. **ΔPerf_clean** - per-task performance delta ours-vs-vanilla on clean data (what the gate costs when there is nothing to catch).
4. **Generalization-gap reduction** - |validation − test| on corrupted tasks, ours vs. vanilla. Vanilla pipelines exploit the leak and reach near-perfect validation scores that do not hold up on test; a working gate closes that gap.
5. **Agent overhead** - extra API calls / tokens / wall-clock attributed to the XAI module (from [xai_tracker.py](machine_learning_engineering/shared_libraries/xai_tracker.py)).

[make_figures.py](make_figures.py) renders these into [figs/](figs/) (`detection_matrix.png`, `gap_reduction.png`, `clean_delta.png`, `overhead.png`). [benchmark_xai_probe.py](benchmark_xai_probe.py) additionally micro-benchmarks the probe suite's own runtime on a real generated pipeline, isolating probe cost from LLM cost.

---

## Project Structure & Code Guide

```
machine_learning_engineering/
├── agent.py                    # Root agent + pipeline wiring
├── prompt.py                   # Root agent instruction strings
├── datasets.yaml               # Kaggle dataset/competition registry (10 tasks)
├── download_datasets.py        # Downloads datasets.yaml entries from Kaggle (LEGACY)
├── download_tasks.py           # Packs/fetches tasks/ as a tarball on Hugging Face
├── auto_run_all_tasks.py       # Batch/matrix runner over tasks × variants (see Evaluation)
├── sub_agents/                 # The 5 pipeline stages (see below)
├── shared_libraries/           # Utilities shared across sub-agents (see below)
├── tasks/                      # Per-task data (gitignored, see "Data & tasks")
└── workspace/                  # Default per-run outputs (gitignored)

ws_baseline/ ws_baseline_xai/   # A/B experiment workspaces (vanilla vs. ours), one final_state.json per run
eval2.py                        # A/B aggregation -> metrics_table2.csv + 5 headline metrics
make_figures.py                 # metrics_table2.csv -> figs/*.png
benchmark_xai_probe.py          # Micro-benchmark of the probe suite's runtime
make_eval_splits.py             # Builds stratified train/test splits per task from raw/ snapshots
dump_latest_session.py          # Dumps the latest ADK session (debugging helper)
scripts/                        # Experiment tooling (leak injector, status grid, overhead reports; see EXPERIMENTS.md)
hpc/                            # SLURM + Singularity batch (see hpc/README.md)
figs/                           # Generated result figures
eval/                           # ADK evaluation-framework fixtures (full_eval/, simple_eval/); unrelated to eval*.py
tests/                          # Unit/integration tests (pytest)
deployment/                     # Vertex AI Agent Engine deployment script
```

### Entry point: [agent.py](machine_learning_engineering/agent.py)

`root_agent` ([mle_frontdoor_agent](machine_learning_engineering/agent.py#L139)) is the chat-facing agent: it answers questions directly or hands off to [mle_pipeline_agent](machine_learning_engineering/agent.py#L130), a `SequentialAgent` assembled from [sub_agents/](machine_learning_engineering/sub_agents/) based on flags in [config.py](machine_learning_engineering/shared_libraries/config.py):

```
initialization  →  [xai_correction]  →  refinement (or xai-gated refinement)  →  ensemble  →  submission
                    only if                only if
                    use_xai_correction      use_xai_refinement
```

After the pipeline finishes, [save_state()](machine_learning_engineering/agent.py#L70) (registered as `after_agent_callback`) dumps the full session state to `<workspace>/<task_name>/final_state.json`, the machine-readable record that everything downstream ([eval2.py](eval2.py), overhead reports) is built from.

### [sub_agents/](machine_learning_engineering/sub_agents/): the 5 pipeline stages

| Stage | Responsibility |
| --- | --- |
| **[initialization](machine_learning_engineering/sub_agents/initialization/)** | For each of `num_solutions` parallel branches: retrieves candidate model approaches via web search, generates/runs/debugs code for each candidate (steered toward skrub, see Contribution 1), ranks them by score, and merges the best ones into a working baseline (`train0.py`). |
| **[xai_correction](machine_learning_engineering/sub_agents/xai_correction/)** | *New in this fork.* The bounded audit-and-revise loop; see Contribution 2. |
| **[refinement](machine_learning_engineering/sub_agents/refinement/)** | For each solution, loops (`outer_loop_round` × `inner_loop_round`) over ablation studies → plan generation → plan implementation, keeping only improving candidates. Optionally XAI-gated (Contribution 2). |
| **[ensemble](machine_learning_engineering/sub_agents/ensemble/)** | Generates an initial plan to combine all refined solutions inside the DataOps graph, implements it, then refines that ensembling plan for `ensemble_loop_round` iterations, keeping the best-scoring version. |
| **[submission](machine_learning_engineering/sub_agents/submission/)** | Takes whichever solution (refined or ensembled) scored best and generates the code that writes the final `submission.csv`, with its own debug/repair loop. |

Every stage's `agent.py` (e.g., in [refinement](machine_learning_engineering/sub_agents/refinement/agent.py)) composes generic building blocks from [debug_util.py](machine_learning_engineering/shared_libraries/debug_util.py) (the "generate → run → debug on failure" loop used everywhere code gets executed) rather than reimplementing execution/retry logic per stage.

### [shared_libraries/](machine_learning_engineering/shared_libraries/): shared utilities

| File | Purpose |
| --- | --- |
| [config.py](machine_learning_engineering/shared_libraries/config.py) | `DefaultConfig` dataclass + [get_config()](machine_learning_engineering/shared_libraries/config.py#L77) (selects by `MLE_PROFILE` env var, default `google`); exposes the `CONFIG` singleton every module reads. |
| [llm.py](machine_learning_engineering/shared_libraries/llm.py) | [build_llm()](machine_learning_engineering/shared_libraries/llm.py#L64): LiteLLM client wired to the GWDG Academic Cloud OpenAI-compatible endpoint. |
| [code_util.py](machine_learning_engineering/shared_libraries/code_util.py) | Sandboxed subprocess execution of generated scripts (`run_python_code`, `evaluate_code`) and score extraction. |
| [debug_util.py](machine_learning_engineering/shared_libraries/debug_util.py) | Generic "run code → summarize error → retry" `LoopAgent` factories reused by every code-generating stage. |
| [common_util.py](machine_learning_engineering/shared_libraries/common_util.py) | Extracting text/JSON/code blocks from LLM output, repairing malformed function calls, seeding, file copy helpers. |
| [check_leakage_util.py](machine_learning_engineering/shared_libraries/check_leakage_util.py) | The original static, prompt-driven (LLM-judged) data-leakage checker (`use_data_leakage_checker`). |
| [xai_gate.py](machine_learning_engineering/shared_libraries/xai_gate.py), [xai_probes.py](machine_learning_engineering/shared_libraries/xai_probes.py), [xai_util.py](machine_learning_engineering/shared_libraries/xai_util.py), [xai_instrumentation_guide.py](machine_learning_engineering/shared_libraries/xai_instrumentation_guide.py), [xai_tracker.py](machine_learning_engineering/shared_libraries/xai_tracker.py) | *New in this fork.* The dynamic XAI leakage gate + overhead accounting, see Contribution 2. |
| [skrub_guidance.py](machine_learning_engineering/shared_libraries/skrub_guidance.py) | *New in this fork.* Prompt-injection strings steering generated code toward the skrub DataOps API, see Contribution 1. |
| [search_util.py](machine_learning_engineering/shared_libraries/search_util.py) | [web_search()](machine_learning_engineering/shared_libraries/search_util.py#L15) tool using DuckDuckGo (`ddgs`), used in place of the Gemini-only Google Search tool. |
| [data_leakage_prompt.py](machine_learning_engineering/shared_libraries/data_leakage_prompt.py), [debug_prompt.py](machine_learning_engineering/shared_libraries/debug_prompt.py) | Prompt templates for the static leakage checker and the debug/bug-summary loop. |

Key [config.py](machine_learning_engineering/shared_libraries/config.py) knobs: `task_name`/`task_type`/`data_dir`/`workspace_dir`, `agent_model`/`api_base`/`api_key`, `num_solutions`, `num_model_candidates`, `inner_loop_round`/`outer_loop_round`/`ensemble_loop_round`, `num_top_plans`, `max_retry`/`max_debug_round`/`max_rollback_round`, `use_data_leakage_checker`, `use_data_usage_checker`, `use_xai_correction`, `use_xai_refinement`, `xai_max_concentration`, `xai_allow_static_fallback`. Most are also settable per-run via env vars (`MLE_TASK_NAME`, `MLE_TASK_TYPE`, `MLE_LOWER`, `MLE_WORKSPACE_DIR`, `ROOT_AGENT_MODEL`, `USE_XAI_CORRECTION`, …); this is how the matrix runner and the HPC jobs configure each subprocess. [get_config()](machine_learning_engineering/shared_libraries/config.py#L77) starts from `DefaultConfig` (a fast single-solution profile) and, under the default `MLE_PROFILE=google`, scales up for real runs (`num_solutions=2`, `num_model_candidates=2`, `exec_timeout=600`, `max_retry=5`, `max_debug_round=3`). Any other profile value (e.g. `MLE_PROFILE=test`) uses the `DefaultConfig` values unchanged.

### [scripts/](scripts/), [eval/](eval/), [tests/](tests/), [deployment/](deployment/)

* [scripts/](scripts/): experiment tooling for the XAI contribution: [prepare_leakage_tasks.py](scripts/prepare_leakage_tasks.py) (leak injector), [scan_tasks_status.py](scripts/scan_tasks_status.py) (matrix completion grid + repro commands), plus [smoke_tool_calling.py](scripts/smoke_tool_calling.py) (sanity-checks the configured model's tool-calling behavior). See [EXPERIMENTS.md](EXPERIMENTS.md) for usage.
* [full_eval/](eval/full_eval/) and [simple_eval/](eval/simple_eval/): ADK evaluation-framework fixtures (`test_config.json` scoring weights + `*.test.json` cases), run via `pytest eval`.
* [tests/](tests/): unit/integration tests (`pytest tests`): agent wiring ([test_agents.py](tests/test_agents.py)), LLM-output parsing ([test_common_util.py](tests/test_common_util.py) / [test_common_util_json.py](tests/test_common_util_json.py)), initialization fallback behavior, function-call repair, and the XAI gate/correction logic ([test_xai_agent.py](tests/test_xai_agent.py), [test_xai_correction.py](tests/test_xai_correction.py)).
* [deploy.py](deployment/deploy.py): creates/lists/deletes a Vertex AI Agent Engine deployment of `root_agent` (`--create` / `--list` / `--delete`).

### Data & tasks ([tasks/](machine_learning_engineering/tasks/), workspaces), gitignored

Task data and run outputs are excluded from git; data is distributed separately via [download_tasks.py](machine_learning_engineering/download_tasks.py) and outputs are regenerated by running the pipeline.

* **[tasks/](machine_learning_engineering/tasks/)`<task-name>/`**: `raw/` (original downloaded snapshot), `task_description.txt` (read by the pipeline), `train.csv`/`test.csv` (stratified split produced by [make_eval_splits.py](make_eval_splits.py)). `<task-name>_proxy` / `<task-name>_c` variants (from [prepare_leakage_tasks.py](scripts/prepare_leakage_tasks.py)) additionally carry `leakage_metadata.json` with the injected-leak ground truth.
* **`<workspace>/<task-name>/<solution-id>/`**: one directory per parallel solution branch: progressive code versions (`init_code_1.py`, `train0.py`, `train0_improve*.py`), `ablation_0.py`, `model_candidates/`, a copy of `input/` and of `xai_probes.py`, plus `xai_metrics.json` (and archived audit rounds) when the gate ran.
* **`<workspace>/<task-name>/ensemble/`**: the ensembling plans/code and `final_solution.py`.
* **`<workspace>/<task-name>/final_state.json`** and **`xai_overhead.json`**: the session-state dump and XAI cost report for that run; these are what [eval2.py](eval2.py) reads.
* `<workspace>` is [workspace/](machine_learning_engineering/workspace/) for interactive runs and [ws_baseline/](ws_baseline/) / [ws_baseline_xai/](ws_baseline_xai/) (repo root) for A/B matrix runs (set via `MLE_WORKSPACE_DIR`).

### Datasets

Tasks are defined in [datasets.yaml](machine_learning_engineering/datasets.yaml) (Kaggle `dataset`/`competition` entries with `task_type`/`target`/`lower`). [download_tasks.py](machine_learning_engineering/download_tasks.py) reads that file and pulls the pre-packaged task data (including stratified `train.csv`/`test.csv` splits and `raw/` snapshots) from Hugging Face into [tasks/](machine_learning_engineering/tasks/)`<task-name>/`. See [EXPERIMENTS.md](EXPERIMENTS.md) for the step-by-step execution workflow.

The 10 configured tasks: `california-housing-prices`, `credit-card-fraud`, `wine-quality-red`, `adult-income`, `pima-diabetes`, `medical-insurance`, `employee-attrition` (datasets) and `tabular-playground-series-may-2022`, `nomad2018-predict-transparent-conductors`, `titanic` (competitions). Note that the `task_type`/`lower` flags in [datasets.yaml](machine_learning_engineering/datasets.yaml) feed both the agent and the sign logic of the XAI probes; a mislabeled metric direction silently inverts leak detection (found and fixed for `wine-quality-red`, see [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md)).

---

## Infrastructure: Porting from Gemini to OpenAI-Compatible Backends (LiteLLM)

### Motivation

The original upstream MLE-STAR repository was built to run exclusively on Google Vertex AI using native Gemini APIs and Google Cloud services. To run experiments on open/academic endpoints (such as the GWDG Academic Cloud or any standard OpenAI-format API), the LLM client, tool interfaces, and config infrastructure had to be ported to a model-agnostic abstraction.

### Key Modifications

We replaced Gemini-specific assumptions with a generalized `litellm` layer. The port consists of three main components:

| Component / File | Adaptation | Purpose |
| --- | --- | --- |
| **LLM Client Factory**<br>[llm.py](machine_learning_engineering/shared_libraries/llm.py) | Wrapped the backend client in a `litellm`-compatible class (`LiteLlm`) pointing at an arbitrary OpenAI-compatible endpoint. | Decouples the agent from Google Vertex AI, enabling it to run on any OpenAI-compatible provider. |
| **Tool-Call Sequence Patches**<br>[llm.py](machine_learning_engineering/shared_libraries/llm.py) | Monkey-patched ADK's [_ensure_tool_results()](machine_learning_engineering/shared_libraries/llm.py#L24) to insert intermediate `assistant` messages ("I have processed the tool execution results.") between sequential tool responses and user queries. | Prevents API validation crashes on GWDG/OpenAI endpoints, which reject message histories where a `user` role immediately follows a `tool` role. |
| **Pydantic Client Serialization**<br>[llm.py](machine_learning_engineering/shared_libraries/llm.py) | Excluded the `llm_client` field from model serialization via Pydantic metadata. | Bypasses `PydanticSerializationError` when building/serializing the agent graph for the `adk web` UI. |
| **DuckDuckGo Web Search**<br>[search_util.py](machine_learning_engineering/shared_libraries/search_util.py) | Implemented a custom [web_search_tool](machine_learning_engineering/shared_libraries/search_util.py#L54) powered by the `ddgs` library to query DuckDuckGo. | Restores web search capability; with Gemini gone, the built-in Gemini Google Search tool (`google_search`) was also gone. |
| **Agnostic Configuration**<br>[config.py](machine_learning_engineering/shared_libraries/config.py) | Defined standard config keys (overridable via `ROOT_AGENT_MODEL`, `OPENAI_API_BASE`, `OPENAI_API_KEY` env vars). | Allows runtime switching of the target LLM and API endpoints without modifying code. |

---

## GWDG Academic Cloud rate limits

This fork uses the GWDG Academic Cloud OpenAI-compatible endpoint as the LLM backend. The default (free, no paid contract) rate limits are:

| Window | Limit (requests) |
| --- | --- |
| per minute | 10 * |
| per hour | 200 |
| per day | 400 |
| per month | 3,000 |

\* The per-minute limit cannot be increased without a paid contract. The hour/day/month limits can be raised on request; email GWDG with an estimate of expected request volume.

A full MLE-STAR pipeline run fires dozens of completions (initialization × `num_solutions` × `num_model_candidates`, plus refinement, ensemble, submission; 30–150 API calls per run in our matrix, see `metrics_table2.csv`). Watch your usage when iterating; one careless `adk run` can chew a substantial fraction of the daily budget. To check remaining quota:

```bash
curl -s -D - -o /dev/null \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://chat-ai.academiccloud.de/v1/models \
  | grep -i ratelimit-remaining
```

The `x-ratelimit-remaining-*` lines tell you which bucket is closest to empty. See https://docs.hpc.gwdg.de/services/ai-services/saia/index.html#api-limits for full details.

## Setup and Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Git
- **System libraries for common ML models** (needed for LightGBM/XGBoost/CatBoost OpenMP support on Linux):
  ```bash
  sudo apt-get update && sudo apt-get install -y \
     libgomp1 \
     libomp-dev \
     build-essential
  ```

### Install dependencies

```bash
uv sync --dev
```

### Configure environment

Copy [.env.example](.env.example) to [.env](.env) and fill in the values (GWDG API key/model, and `KAGGLE_API_KEY` if you need to (re)download task data from Kaggle using [download_datasets.py](machine_learning_engineering/download_datasets.py); see [.env.example](.env.example)).

### Get the task data

Either fetch the pre-packaged archive:

```bash
python machine_learning_engineering/download_tasks.py
```

## Usage

Run a single task interactively (whatever `task_name` is currently set to via `MLE_TASK_NAME` / [config.py](machine_learning_engineering/shared_libraries/config.py)):

```bash
uv run adk run machine_learning_engineering
```

Or with the web UI (select `machine_learning_engineering` from the dropdown):

```bash
uv run adk web
```

Batch/matrix runs (the experiment driver; see [Evaluation methodology](#evaluation-methodology-the-xai-ab-experiment) and [EXPERIMENTS.md](EXPERIMENTS.md)):

```bash
# running everything
python machine_learning_engineering/auto_run_all_tasks.py --matrix --corrupted && python machine_learning_engineering/auto_run_all_tasks.py --matrix --clean

# what's done / what's missing across the whole matrix
python scripts/scan_tasks_status.py

# aggregate results + figures
python eval2.py
python make_figures.py --pdf
```

### Development

```bash
uv sync --dev
uv run pytest tests
uv run pytest eval
```
