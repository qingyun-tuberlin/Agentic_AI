"""Defines the prompts for debugging."""

from machine_learning_engineering.shared_libraries.skrub_guidance import (
    SKRUB_DATAOPS_DEBUG_GUIDELINE,
)

BUG_SUMMARY_INSTR = """# Error report
{bug}

# Your task
- Remove all unnecessary parts of the above error report.
- We are now running {filename}.py. Do not remove where the error occurred.
- If the error is a TIMEOUT (e.g. "timed out", "TimeoutExpired"), this is NOT a
  code bug — the pipeline is too slow. Your summary MUST identify which parts of
  the code are most expensive (e.g. high iteration counts, deep ensembles,
  repeated .assign()/.fillna() loops in a Skrub graph) and suggest concrete
  simplifications (halve iterations, drop ensemble layers, consolidate feature
  engineering into fewer steps).
- If the code uses Skrub DataOps (`skrub.var`, `.skb.*`), the fix MUST stay inside
  the DataOps graph. NEVER suggest replacing a skrub call with a pandas/sklearn
  equivalent (e.g. do NOT suggest `pd.concat` in place of `.skb.concat`, or
  `model.fit`/`sklearn.Pipeline`/`train_test_split` in place of `.skb.apply` /
  `.skb.make_learner` / `.skb.train_test_split`). An error mentioning a skrub
  function means it was used incorrectly, NOT that skrub should be abandoned.
- A `skrub.concat(...)` AttributeError means concat is a METHOD on a DataOp node:
  use `first_node.skb.concat([other_node], axis=1)`, not the pandas function.
- Do not invent a fix that imports a symbol or private module you are unsure
  exists; prefer pointing at the misused public API."""

TIMEOUT_GUIDANCE = """

# CRITICAL: EXECUTION TIMEOUT — YOU MUST SIMPLIFY THE CODE
The previous code exceeded the execution time limit of {exec_timeout} seconds.
This is NOT a code bug — the pipeline is too computationally expensive.
Re-emitting the same or similar code WILL time out again. You MUST make it
significantly faster by applying SEVERAL of the following changes:
- Reduce model iterations/n_estimators by at least 50% (e.g. 700 → 200, 500 → 150)
- Use a single lightweight model instead of ensembles or stacking
- Reduce tree depth (e.g. depth=6 → depth=4)
- Remove redundant feature engineering steps
- Consolidate repeated .assign()/.fillna() calls in loops into a single .assign()
  call to avoid rebuilding the Skrub DataOps graph on every iteration
- If using TableVectorizer with many high-cardinality columns, consider dropping
  some columns or using a simpler encoding strategy
- Prefer LGBMClassifier/LGBMRegressor (fast) over CatBoost (slower) when speed
  is critical
DO NOT simply re-emit the same code with cosmetic changes — the identical
pipeline will time out again and waste another {exec_timeout} seconds."""

BUG_REFINE_INSTR = (
    """# Task description
{task_description}

# Code with an error:
{code}

# Error:
{bug}

# Your task
- Please revise the code to fix the error.
- The standard ML stack is ALREADY installed and importable: scikit-learn, skrub, pandas, numpy, xgboost, lightgbm, catboost, shap. Do NOT add `pip install` / subprocess guards for these — just import them normally. Adding install boilerplate for already-available packages is noise and is discouraged.
- For categorical encoding, use skrub's native encoders (`TableVectorizer`, `StringEncoder`, `GapEncoder`, `MinHashEncoder`, `SimilarityEncoder`) or `sklearn.preprocessing.TargetEncoder`. Do NOT pull in `category_encoders` (e.g. `CatBoostEncoder`) — skrub/sklearn cover these natively and `category_encoders` is not installed.
- ONLY if a `ModuleNotFoundError` is for a package OUTSIDE that list, add a single compact guard for that one package at the top of the script:
  ```python
  import subprocess, sys
  try:
      import <module>
  except ModuleNotFoundError:
      subprocess.check_call([sys.executable, "-m", "pip", "install", "<package-name>"])
      import <module>
  ```
- Do not remove subsampling if exists.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- All the provided input data is stored in \"./input\" directory.
- Remember to print a line in the code with 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code.
- MANDATORY OUTPUT FORMAT: Your reply MUST be exactly one ```python ... ``` fenced code block containing the full revised script. Even if you believe the previous code was already correct or that "no changes are needed", you MUST re-emit the complete script inside a fenced code block. Prose-only replies, status reports, summaries, or "the implementation looks correct" responses are forbidden and will be discarded."""
    + SKRUB_DATAOPS_DEBUG_GUIDELINE
)
