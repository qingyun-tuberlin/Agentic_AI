"""Shared XAI instrumentation instructions for LLM prompts.

The dynamic leakage analysis lives in the shipped `xai_probes` module (copied
next to every generated script by `code_util.run_python_code`). The model must
NOT hand-write SHAP / permutation / masking / feature-name remapping anymore —
it only wires the script's `learner`, split `data`, and final `pred` node into
`xai_probes.run_leakage_suite(...)`, which writes `xai_metrics.json` itself.

This keeps the fragile attribution logic in version-controlled, tested code
instead of regenerating it per run, and shrinks the prompt surface to a handful
of values the model is reliable at filling in.
"""

# Reference wiring block: the model adapts the metric, task_type, target column,
# and the learner/data/pred variable names to the script it is instrumenting.
XAI_INSTRUMENTATION_EXAMPLE = '''
# --- Step 1: name the two graph steps so the suite can run TreeSHAP ---
# Add .skb.set_name("xai_features") to the FINAL preprocessing node (the model's
# input) and .skb.set_name("xai_model") to the model .skb.apply(...) step, e.g.:
#     X_features = X.skb.apply(TableVectorizer()).skb.set_name("xai_features")
#     pred = X_features.skb.apply(model, y=y).skb.set_name("xai_model")
# This changes ONLY the names; the pipeline behaviour is identical. If you omit
# them the suite still works (it falls back to permutation importance).

# --- Step 2: after fit, call the shipped probe suite ---
# `xai_probes` is already available next to this script — a bare import works.
# Do NOT write SHAP, permutation importance, masking, feature-name remapping,
# or the xai_metrics.json file yourself; run_leakage_suite does all of that.
import xai_probes
from sklearn.metrics import accuracy_score  # <- use THIS script's metric

# `data` is the dict returned by `pred.skb.train_test_split(...)`.
# `learner` is the fitted learner; `pred` is the final prediction DataOp node.
task_meta = xai_probes.TaskMeta(
    lower_is_better=__LOWER_IS_BETTER__,
    metric_fn=accuracy_score,             # SAME metric as 'Final Validation Performance'
    task_type="Tabular Classification",   # or "Tabular Regression"
    target_column="target",               # the ACTUAL target column name in the data
)
xai_probes.run_leakage_suite(
    learner,
    data,
    task_meta,
    train_df=pd.read_csv("./input/train.csv"),
    learner_factory=lambda: pred.skb.make_learner(),
)
'''.strip()

XAI_INSTRUMENTATION_RULES = """
## XAI instrumentation rules (mandatory)
The dynamic leakage check is performed by the shipped `xai_probes` module, which
is importable with a bare `import xai_probes` (it is staged next to your script).
You MUST call it instead of writing your own explainability code.

1. Do NOT change feature engineering, preprocessing, or the model. Do NOT wrap
   the model in a custom estimator (no `TreeModelWithShap`). Do NOT compute SHAP,
   permutation importance, masking, or feature-name remapping, and do NOT write
   `xai_metrics.json` yourself — `xai_probes.run_leakage_suite(...)` does all of it.
   The ONLY edits to the existing graph you may make are adding two
   `.skb.set_name(...)` tags (step 1 below): `"xai_features"` on the final
   preprocessing node and `"xai_model"` on the model `.skb.apply(...)` step. These
   let the suite extract the fitted model + preprocessed matrix for TreeSHAP;
   they do not change behaviour.
2. After `learner.fit(...)` runs and the split `data = pred.skb.train_test_split(...)`
   exists, append the wiring block below, filling in values from THIS script:
   - `metric_fn`: the exact metric used to print 'Final Validation Performance'.
   - `lower_is_better`: __LOWER_IS_BETTER__.
   - `task_type`: "Tabular Classification" or "Tabular Regression".
   - `target_column`: the real target column name in the dataset.
   - `learner`, `data`, and `pred` must be the variables already defined above.
3. The metric must score `data["y_test"]` against predictions. If the graph maps
   string labels to integers at `mark_as_y` (recommended), the split already
   yields numeric `y` and the metric works directly. Otherwise pass a `metric_fn`
   that maps the labels before scoring. Never call `.astuple()` on the target
   Series and never cast raw string labels with `.astype(int)`.
4. `learner_factory=lambda: pred.skb.make_learner()` supplies a fresh learner for
   the resampling probes; keep it as shown. For a non-skrub estimator, pass the
   fitted estimator as `learner` and omit `learner_factory`/`train_df`.
5. Keep `import xai_probes` at module top level with the other imports.
6. Do not add `pip install` guards for `xai_probes`; it is a local module, not a
   PyPI package.

## Wiring block (adapt names/metric to this script)
```python
__EXAMPLE__
```
""".strip()


def build_instrumentation_instructions(*, lower_is_better: bool) -> str:
    """Build prompt text with the wiring block and rules for a given task."""
    lower_repr = repr(bool(lower_is_better))
    example = XAI_INSTRUMENTATION_EXAMPLE.replace("__LOWER_IS_BETTER__", lower_repr)
    return XAI_INSTRUMENTATION_RULES.replace("__LOWER_IS_BETTER__", lower_repr).replace(
        "__EXAMPLE__", example
    )


# Wrapper prompts that ask the model to add / repair the instrumentation block
# above. Shared by every leakage gate so the wording can't drift between them.
_INSTRUMENT_INSTR = """
You are a machine learning engineer.
Instrument the training script with explainability (SHAP / feature importance) and robustness evaluation.
Do NOT change training logic — only add instrumentation wired to this script's model and validation data.

{instrumentation_guide}

Here is the current training script:
```python
{code}
```

Return only the complete instrumented Python script in a single ```python markdown block.
"""

_INSTRUMENT_FIX_INSTR = """
You are a machine learning engineer.
The script below failed to run or XAI metrics were missing/invalid. If the execution failed due to a syntax/runtime error in the training code or pipeline, fix the root cause. Otherwise, fix ONLY the XAI instrumentation block. Keep all other logic unchanged where possible.

- If the error contains "AttributeError: 'Series' object has no attribute 'astuple'", it means you should map the pandas Series of target labels using `.map(...)` instead of calling `.astuple()`. Do not try to convert target string labels directly using `.astype(int)`, as that will raise a ValueError.
- If the error contains "AttributeError: 'numpy.dtypes.Int64DType' object has no attribute 'na_value'" or "Evaluation of '.na_value' failed", it means you passed `stratify=y` (or a `DataOp` derivative) to `train_test_split`. In Skrub, `stratify` is NOT supported on lazy `DataOp` nodes. Remove the `stratify` keyword argument from `.skb.train_test_split(...)`.
- If the error contains "TypeError: This object is a DataOp ... it is not possible to eagerly iterate over it now", it means you tried to iterate over `df.columns` or `X.columns` eagerly (e.g. in a list comprehension `[c for c in X.columns]`). Use `.drop(columns=[...])` or `.skb.select()` instead.
- If the error contains "AttributeError: 'dict' object has no attribute 'to_pandas'", it is because you called `.to_pandas()` on `X_train` or `data["train"]`, which are environment dictionaries, not DataFrames. Do not call `.to_pandas()` on them.

## Error from the last run
{error}

{instrumentation_guide}

Here is the current training script:
```python
{code}
```

Return only the complete fixed Python script in a single ```python markdown block.
"""


def is_gpt_model() -> bool:
    """Check if the configured agent model is in the GPT-3/4/5 family."""
    from machine_learning_engineering.shared_libraries.config import CONFIG
    model = CONFIG.agent_model.lower()
    return any(x in model for x in ("gpt-3", "gpt-4", "gpt-5"))

def strip_gpt_mentions(text: str) -> str:
    """Remove GPT-specific guidelines when not using a GPT model."""
    if is_gpt_model():
        return text
    
    gpt_bullets = [
        '''- If the error contains "AttributeError: 'numpy.dtypes.Int64DType' object has no attribute 'na_value'" or "Evaluation of '.na_value' failed", it means you passed `stratify=y` (or a `DataOp` derivative) to `train_test_split`. In Skrub, `stratify` is NOT supported on lazy `DataOp` nodes. Remove the `stratify` keyword argument from `.skb.train_test_split(...)`.\n''',
        '''- If the error contains "TypeError: This object is a DataOp ... it is not possible to eagerly iterate over it now", it means you tried to iterate over `df.columns` or `X.columns` eagerly (e.g. in a list comprehension `[c for c in X.columns]`). Use `.drop(columns=[...])` or `.skb.select()` instead.\n''',
        '''- If the error contains "AttributeError: 'dict' object has no attribute 'to_pandas'", it is because you called `.to_pandas()` on `X_train` or `data["train"]`, which are environment dictionaries, not DataFrames. Do not call `.to_pandas()` on them.\n'''
    ]
    for bullet in gpt_bullets:
        text = text.replace(bullet, "")
    return text



def instrument_via_llm(code: str, *, lower_is_better: bool, error: str = "") -> str | None:
    """Ask the LLM to add (or repair, when ``error`` is set) XAI instrumentation.

    Returns the instrumented script, or ``None`` if no code block was produced.
    """
    from machine_learning_engineering.shared_libraries import common_util, llm

    guide = build_instrumentation_instructions(lower_is_better=lower_is_better)
    template = _INSTRUMENT_FIX_INSTR if error else _INSTRUMENT_INSTR
    prompt_text = (
        template.replace("{instrumentation_guide}", guide)
        .replace("{code}", code)
        .replace("{error}", error)
    )
    prompt_text = strip_gpt_mentions(prompt_text)
    response_text = llm.complete_text(prompt_text, temperature=0.0)

    return common_util.extract_code_block(response_text) or None
