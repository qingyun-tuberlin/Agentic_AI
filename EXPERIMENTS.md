# EXPERIMENTS.md -- A file pointing to the scripts to execute the experiments and their log files

This document details how the experiments for **MLE-STAR** (both clean baseline and XAI-corrected pipelines) were conducted, provides pointers to the execution scripts, explains the data leakage injection process, and outlines where the resulting run artifacts and log files are located in the repository.

See [README.md](README.md) for setup and details on the codebase structure, and [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md) for the evaluation of these experiments.

---

## TL;DR: Execute Full Pipeline

```bash
# 1. Install dependencies & configure env
# Ensure you copy .env.example to .env and fill in your API keys (e.g., OPENAI_API_KEY)
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

### Generated Run Artifacts

Running the steps above generates the following directory structure and artifacts:

```text
.
├── machine_learning_engineering/
│   └── tasks/                          # Task datasets (clean and proxy variants) (Step 2)
│       └── <task_name>[_proxy]/
│           ├── train.csv               # Training split (leakage injected in _proxy variant)
│           ├── test.csv                # Test split (leakage neutralized in _proxy variant)
│           └── leakage_metadata.json   # Injection configuration metadata (only in _proxy variant)
├── ws_baseline/                        # Vanilla baseline (XAI-off) runs (Step 3)
│   └── <task_name>[_proxy]/
│       ├── final_state.json            # Run scores, verdicts, and submission code
│       └── ensemble/
│           ├── final_solution.py
│           └── final/submission.csv
├── ws_baseline_xai/                    # Ours (XAI-on) runs (Step 3)
│   └── <task_name>[_proxy]/
│       ├── final_state.json            # Run scores, verdicts, and submission code
│       ├── xai_overhead.json           # API calls, token counts, and latency logs
│       └── ensemble/
│           ├── final_solution.py
│           └── final/submission.csv
├── figs/                               # Diagnostic plots (Step 4)
│   ├── clean_delta.png                 # Performance delta (Ours vs Vanilla) on clean tasks
│   ├── detection_matrix.png            # Status matrix of flagged/dropped leakage features
│   ├── gap_reduction.png               # Relative generalization gap comparison
│   └── overhead.png                    # API calls, tokens, and execution time overhead comparison
└── metrics_table2.csv                  # Compiled evaluation metrics from eval2.py (Step 4)
```

For easy navigation, you can access these directories directly:
* [machine_learning_engineering/tasks/](machine_learning_engineering/tasks/) - Contains all clean and proxy variant datasets.
* [ws_baseline/](ws_baseline/) - Stores Vanilla (XAI-off) execution outputs.
* [ws_baseline_xai/](ws_baseline_xai/) - Stores Ours (XAI-on) execution outputs.
* [figs/](figs/) - Stores output evaluation plots and charts.

---

## 1. How the Experiments Were Conducted

The experiments were conducted using the **`gpt-5.4-mini-2026-03-17`** model configured via the `ROOT_AGENT_MODEL` environment variable. The evaluation framework is structured as a complete grid of **40 experimental runs in total** ($10 \text{ tasks} \times 2 \text{ variants} \times 2 \text{ configurations}$). This design allows paired comparisons across every task and condition.

### A. The 40-Run Evaluation Matrix
For each of the 10 base Kaggle tasks, the following 4 runs were executed:
1. **Vanilla (XAI-off) on Clean data**: Baseline agent model performance with no leakage present and leakage correction disabled. Used to establish clean baseline performance.
2. **Vanilla (XAI-off) on Corrupted data**: Baseline agent performance when target proxy leakage is present but the correction gate is disabled. This measures the vulnerability to leakage (i.e., model overfits to the proxy leak, collapsing test performance).
3. **Ours (XAI-on) on Clean data**: Agent performance with the correction gate enabled on clean data. Used to measure the feature-level false positive rate (FPR) and check for any negative impact (clean baseline performance delta) of the audit gate.
4. **Ours (XAI-on) on Corrupted data**: Agent performance under the active correction gate on corrupted data. Used to measure the leakage detection rate (DR) and check if self-correction successfully drops the leak to restore generalization.

### B. Clean Baseline Runs (Skrub Pipeline)
* **Goal**: Measure the baseline performance of the agent-generated Skrub models when training on valid features.
* **Skrub Steering**: The agent is guided via prompt injection defined in [skrub_guidance.py](machine_learning_engineering/shared_libraries/skrub_guidance.py) to output skrub `.skb.apply(...)` operations rather than standard pandas/sklearn dataframes.
* **Outputs**: Run details are stored in the task's workspace subdirectory in `final_state.json`.

### C. Data Leakage Injection Methodology
To evaluate the dynamic leakage gate, clean datasets are corrupted using the automation script [prepare_leakage_tasks.py](scripts/prepare_leakage_tasks.py) to inject a "proxy leak". The injection process executes according to the following systematic stages:

1. **Target Identification**: The target column is identified by querying the metadata in [datasets.yaml](machine_learning_engineering/datasets.yaml). If absent, a heuristic checks for columns present in `train.csv` but absent in `test.csv`.
2. **Proxy Target Leakage Generation (`_proxy` variants)**:
   * **Encoding & Standardization**: For classification tasks, the target classes are ordinally encoded. The target signal $y$ (or its encoded representation) is standardized to a $z$-score:
     $$z = \frac{y - \mu_y}{\sigma_y}$$
   * **Noise Addition**: Independent Gaussian noise is added to the standardized signal:
     $$z_{\text{leak}} = z + \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$$
   * **Correlation Tuning**: To target a Pearson correlation coefficient of $R = 0.99$ (99% correlation), the noise standard deviation is scaled as:
     $$\sigma_{\text{noise}} = \sqrt{\frac{1}{R^2} - 1} \approx 0.1425$$
   * **Shifting & Scaling**: The raw proxy signal is shifted and scaled to standard feature values:
     $$\text{Proxy Feature} = 50.0 + 15.0 \times z_{\text{leak}}$$
3. **Style-Matched Naming**: Column names are mapped to match each task dataset's unique naming style (e.g. PascalCase `GlucoseRiskIndex` for `pima-diabetes` or snake_case `block_median_price` for `california-housing-prices`). This hides the injected column from naive regex/keyword-based code checkers.
4. **Neutralization**: The leaked feature is neutralized on the test set (`test.csv`) by overwriting all test rows with the training set's feature mean (regression) or mode (classification). This simulates production inference where the leak is unavailable, causing models reliant on the leak to fail.
5. **Mid-Schema Reordering**: To prevent order-detection bias (e.g. a model ignoring the first or last column), the injected feature is inserted in the middle of the schema (mid-schema) in both `train.csv` and `test.csv`.
6. **Task Description Update**: The script parses task description templates (e.g., [task_description.txt](machine_learning_engineering/tasks/adult-income/task_description.txt)) and regenerates the feature listing and CSV preview blocks, showing the leaked column as an ordinary pre-existing feature.
7. **Metadata Tracking**: Injection stats (correlation, naming, leakage type) are written to `leakage_metadata.json` in the task folder.

### D. XAI Data-Leakage Audit & Self-Correction
* **Goal**: Dynamic checking and auto-elimination of target leakage features.
* **Audit Phase**: During execution, the XAI probe suite in [xai_probes.py](machine_learning_engineering/shared_libraries/xai_probes.py) computes feature attribution (via SHAP and permutation importance) of the generated pipelines and evaluates proxy power by fitting shallow decision trees on individual features.
* **Decision Gate**: The gate in [xai_gate.py](machine_learning_engineering/shared_libraries/xai_gate.py) matches attributions against leakage thresholds. If a feature is flagged, the gate issues a `FAIL` verdict with details.
* **Self-Correction Loop**: The LLM agent receives the audit trail showing the flagged columns, rewrites its data-preparation code to drop the flagged features, and retries training. Once the audit returns `PASS`, the final model is compiled and submitted.

### E. Overhead & Cost Profiling
* **Goal**: Measure the resource overhead (tokens, API calls, execution latency) introduced by running the dynamic XAI gate.
* **Methodology**: Each pipeline run generates an `xai_overhead.json` file (tracked by [xai_tracker.py](machine_learning_engineering/shared_libraries/xai_tracker.py)) tracking these metrics to allow comparing run costs.

---

## 2. Evaluated Kaggle Tasks & Dataset Links

The table below lists the 10 Kaggle tasks used throughout the experiments:

| Task Name | Task Type | Kaggle URL / ID | Target Column(s) | Metric Direction |
| --- | --- | --- | --- | --- |
| [`california-housing-prices`](machine_learning_engineering/tasks/california-housing-prices) | Tabular Regression | [camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) | `median_house_value` | Lower is better (RMSE) |
| [`credit-card-fraud`](machine_learning_engineering/tasks/credit-card-fraud) | Tabular Classification | [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `Class` | Higher is better (AUPRC) |
| [`wine-quality-red`](machine_learning_engineering/tasks/wine-quality-red) | Tabular Regression | [uciml/red-wine-quality-cortez-et-al-2009](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) | `quality` | Lower is better (RMSE) |
| [`adult-income`](machine_learning_engineering/tasks/adult-income) | Tabular Classification | [uciml/adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income) | `income` | Higher is better (Accuracy) |
| [`pima-diabetes`](machine_learning_engineering/tasks/pima-diabetes) | Tabular Classification | [uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | `Outcome` | Higher is better (Accuracy) |
| [`medical-insurance`](machine_learning_engineering/tasks/medical-insurance) | Tabular Regression | [mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) | `charges` | Lower is better (RMSE) |
| [`employee-attrition`](machine_learning_engineering/tasks/employee-attrition) | Tabular Classification | [pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) | `Attrition` | Higher is better (Accuracy) |
| [`tabular-playground-series-may-2022`](machine_learning_engineering/tasks/tabular-playground-series-may-2022) | Tabular Classification | [tabular-playground-series-may-2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022) | `target` | Higher is better (ROC AUC) |
| [`nomad2018-predict-transparent-conductors`](machine_learning_engineering/tasks/nomad2018-predict-transparent-conductors) | Tabular Regression | [nomad2018-predict-transparent-conductors](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors) | `formation_energy_ev_natom`, `bandgap_energy_ev` | Lower is better (RMSLE) |
| [`titanic`](machine_learning_engineering/tasks/titanic) | Tabular Classification | [titanic](https://www.kaggle.com/competitions/titanic) | `Survived` | Higher is better (Accuracy) |

---

## 3. Scripts and Commands

### A. Dataset Setup & Leakage Injection

We host custom-prepared versions of these datasets on Hugging Face at [tschuster/mle-star-tasks](https://huggingface.co/datasets/tschuster/mle-star-tasks). Because standard Kaggle competitions do not provide labels in `test.csv` (which are required for automated evaluation), we generated our own splits by slicing the labeled training data to create a new, labeled `test.csv` and discarding the original unlabeled Kaggle test split. All of this splitting and setup is bundled into our download script so the end-user does not have to perform any manual steps. For classification tasks, particularly `credit-card-fraud` (which features an extreme class imbalance of ~0.17% fraud transactions), we performed stratified splitting to ensure the test set remains representative and to avoid evaluation issues.

To download these datasets using [download_tasks.py](machine_learning_engineering/download_tasks.py) and inject the data leakage using [prepare_leakage_tasks.py](scripts/prepare_leakage_tasks.py) (generating proxy variants under the [tasks/](machine_learning_engineering/tasks/) directory), run the following commands:
```bash
# Setup datasets
python machine_learning_engineering/download_tasks.py

# Inject proxy leakage (default)
python scripts/prepare_leakage_tasks.py
```

### B. Running Experiments

There are three ways to execute the experimental runs depending on your objective:

#### Option 1: Run the Full Evaluation Matrix (Recommended)
This runs the full experimental matrix (both XAI-off baseline and XAI-on treatments, on both clean and corrupted versions of the tasks) using the [auto_run_all_tasks.py](machine_learning_engineering/auto_run_all_tasks.py) script, saving results into the distinct workspace directories [ws_baseline/](ws_baseline/) and [ws_baseline_xai/](ws_baseline_xai/) expected by [eval2.py](eval2.py) and [make_figures.py](make_figures.py).
```bash
# 1. Run the grid of tasks with data leakage (proxy variants)
python machine_learning_engineering/auto_run_all_tasks.py --matrix --corrupted

# 2. Run the grid of clean tasks
python machine_learning_engineering/auto_run_all_tasks.py --matrix --clean
```

#### Option 2: Run a Specific Pipeline Variant (Single Pass)
To run a specific variant on clean or corrupted datasets using [auto_run_all_tasks.py](machine_learning_engineering/auto_run_all_tasks.py):
```bash
# Run baseline (XAI-off) on clean datasets (writes to ./ws_baseline/)
python machine_learning_engineering/auto_run_all_tasks.py --clean --variant baseline

# Run baseline (XAI-off) on corrupted datasets (writes to ./ws_baseline/)
python machine_learning_engineering/auto_run_all_tasks.py --corrupted --variant baseline

# Run Ours (XAI-on) on corrupted datasets (writes to ./ws_baseline_xai/)
python machine_learning_engineering/auto_run_all_tasks.py --corrupted --variant baseline_xai
```

You can target a specific single task using the `--task` argument (which accepts either the base name, e.g. `titanic`, or the full name, e.g. `titanic_proxy`):
```bash
# Run baseline on a single corrupted task (titanic_proxy)
python machine_learning_engineering/auto_run_all_tasks.py --task titanic --corrupted --variant baseline
```

> [!NOTE]
> The batch execution script is **idempotent**. If a task's `final_state.json` file is already present in the target workspace directory, the script will skip execution for that task.

### C. Evaluation & Overhead Metrics
To aggregate all `final_state.json` states from the vanilla and ours workspaces and calculate the 5 headline metrics using [eval2.py](eval2.py):
```bash
# Computes metrics and writes metrics_table2.csv
python eval2.py
```

### D. Generating Figures and Visualization
To generate the diagnostic plots from [metrics_table2.csv](metrics_table2.csv) and save them to [figs/](figs/) using [make_figures.py](make_figures.py):
```bash
# Reads metrics_table2.csv and saves plots under figs/
python make_figures.py
```


---

## 4. Log Files and Run Artifacts

The logs and structured run files are committed to git or generated locally. Below are pointers to these files:

### A. Workspace Directories & Run Outputs
The experiment directories are situated in the repository root:
* **[ws_baseline/](ws_baseline/)**: Subdirectories for each task run in the Vanilla baseline (XAI-off).
* **[ws_baseline_xai/](ws_baseline_xai/)**: Subdirectories for each task run under the XAI-enabled correction gate (Ours).

We provide all generated code during all runs from `ws_baseline` and `ws_baseline_xai` at https://tubcloud.tu-berlin.de/s/ziSKb7WRi4S5eMp.

Each task run subdirectory (e.g. `ws_baseline_xai/adult-income_proxy/`) contains the following structured artifacts:
* `final_state.json`: The complete state dictionary containing final scores, XAI verdicts, flagged suspects, and the generated submission code.
* `xai_overhead.json`: Structured resource log logging API calls, token counts, and compute latency specifically consumed by the XAI gate.
* `ensemble/final_solution.py`: The final Python pipeline script generated by the agent.
* `ensemble/final/submission.csv`: Predictions generated on the neutralized test split.

### B. Evaluation Metrics and Figures (Committed to Git)
The following evaluation outputs and figures are tracked and committed to Git:
* **[metrics_table2.csv](metrics_table2.csv)**: Compiled evaluation metrics from [eval2.py](eval2.py).
* **[figs/](figs/)**: Diagnostic plots generated by [make_figures.py](make_figures.py) representing:
  * `clean_delta.png`: Performance delta (Ours vs Vanilla) on clean tasks.
  * `detection_matrix.png`: Status matrix of flagged/dropped leakage features.
  * `gap_reduction.png`: Relative generalization gap comparison.
  * `overhead.png`: API calls, tokens, and execution time overhead comparison.
