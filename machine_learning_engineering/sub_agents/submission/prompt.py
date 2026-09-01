"""Defines the prompts for the submission agent."""

from machine_learning_engineering.shared_libraries.config import CONFIG
from machine_learning_engineering.shared_libraries.skrub_guidance import (
    SKRUB_SUBMISSION_CONSTRAINT,
)

ADD_TEST_FINAL_INSTR = (
    """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to come up with an excellent solution in Python.
- We will now provide a task description and a Python solution.
- What you have to do on the solution is just loading test samples and create a submission file.

# Task description
{task_description}

# Python solution
```python
{code}
```

# Your task
- Load the test samples and create a submission file.
- The ONLY things you add to the provided solution are: (1) obtaining predictions for the test set through the EXISTING trained learner, and (2) writing `./final/submission.csv`. Do not change anything else.
- Get test predictions EXCLUSIVELY through the existing learner, e.g. `learner.predict({{"file_path": "./input/test.csv"}})` (or the env dict matching how the solution declared its data var). The learner already contains ALL preprocessing and feature engineering.
- NEVER call `pd.read_csv("./input/test.csv")` and then re-apply binning, encoding, scaling, or any feature engineering by hand, and NEVER rebuild the preprocessing pipeline for the test set. Re-deriving preprocessing on test is the most common cause of failure here and is forbidden.
- The test set has NO target column. Never reference the target column on the test data.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Test data is available in the `./input` directory.
- Save the test predictions in a `submission.csv` file. Put the `submission.csv` into `./final` directory.
- You should not drop any test samples. Predict the target value for all test samples.
- If the task has multiple target columns, produce each submission column from a model trained on THAT target: split a multi-output learner's prediction array by column (`preds[:, 0]`, `preds[:, 1]`), or predict each column with its own per-target learner. Never duplicate one target's predictions across columns, and never rely on a single leftover `learner` variable for all targets.
- If you choose to train the model on the full training set, remember that you must never pass symbolic Skrub DataOp objects (like X_features, y, df) inside the environment dictionary passed to learner.fit(). Passing an empty dictionary `{{}}` will fail. Instead, you must load the training data and pass the concrete pandas objects (e.g., passing train DataFrame under its variable name like `{{"data": train_df}}` or `{{"file_path": "./input/train.csv"}}`, or passing concrete placeholders `{{"_skrub_X": train_X_df, "_skrub_y": train_y_series}}`).

# Required
- Do not modify the given Python solution code too much. Try to integarte test submission with minimal changes.
- There should be no additional headings or text in your response.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not forget the ./final/submission.csv file.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
- You MUST keep the existing `print(f"Final Validation Performance: {{...}}")` line (and the validation split and score computation it depends on) exactly as in the provided solution. This line is parsed by the grader; if it is missing the run is scored as a failure. Emit it even when you retrain on the full training set.
- Remove any XAI instrumentation code (SHAP, permutation importance, xai_metrics.json writing) from the solution. It is not needed for submission. This cleanup applies ONLY to XAI/explainability code — never remove the `Final Validation Performance:` print, the validation split, or the metric computation, which are not XAI instrumentation."""
    + SKRUB_SUBMISSION_CONSTRAINT
)

if not CONFIG.use_xai_correction and not CONFIG.use_xai_refinement:
    ADD_TEST_FINAL_INSTR = ADD_TEST_FINAL_INSTR.replace(
        "\n- Remove any XAI instrumentation code (SHAP, permutation importance, xai_metrics.json writing) from the solution. It is not needed for submission. This cleanup applies ONLY to XAI/explainability code — never remove the `Final Validation Performance:` print, the validation split, or the metric computation, which are not XAI instrumentation.",
        ""
    )
