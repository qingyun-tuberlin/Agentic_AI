"""Defines the prompts for the initialization agent."""

from machine_learning_engineering.shared_libraries.skrub_guidance import (
    SKRUB_DATAOPS_GUIDELINE,
    strip_rag_tool_mentions,
)

SUMMARIZATION_AGENT_INSTR = """# Task description
{task_description}

# Your task
- Summarize this task description.
- Your summary will be used for searching recent effective models for {task_type}.

# Requirement
- We will directly use your response, so be simple and concise.
"""

MODEL_RETRIEVAL_INSTR = """# Competition
{task_summary}

# Your task
- List {num_model_candidates} recent effective models and their example codes to win the above competition.
- Do NOT select or suggest TabPFN under any circumstance, as it requires interactive license acceptance or an online token to download model weights and is unsuitable for headless execution.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.
Use this JSON schema:
Model = {{'model_name': str, 'example_code': str}}
Return: list[Model]

# Output format
- Reply with plain text only: a Python-style JSON list of Model objects.
- Use web_search only when you need current information.
- Do not return the model list as a function or tool call."""

MODEL_EVAL_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a task description and a model description.
- You need to implement your Python solution using the provided model.

# Task description
{task_description}

# Model description
{model_description}

# Your task
- Implement the solution in Python using the Skrub DataOps graph API.
- Only use the provided train data in the `./input` directory.
- Start from all train.csv columns except the target; you may drop columns during feature engineering when superseded by derived features.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
"""

CODE_INTEGRATION_INSTR = (
    """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a base solution and an additional reference solution.
- You need to implement your Python solution by integrating reference solution to the base solution.

# Base solution
```python
{base_code}
```

# Reference solution
```python
{reference_code}
```

# Your task
- Implement the solution in Python.
- You have to integrate the reference solution to the base solution.
- Your code base should be the base solution.
- Try to train additional model of the reference solution.
- When integrating, try to keep code with similar functionality in the same place (e.g., all preprocessing should be done and then all training).
- When integrating, ensemble the models inside the Skrub DataOps graph by implementing a custom Scikit-learn estimator class and passing it to `.skb.apply(ensemble_model, y=y)`. IMPORTANT: Match the mixin and predictions to the task type: for classification tasks, inherit ClassifierMixin, implement predict_proba and predict, and output class/probabilities; for regression tasks, inherit RegressorMixin, implement predict, and output continuous values. Do NOT introduce a plain sklearn block or use `@skrub.deferred` / `.skb.apply_func` to combine model prediction nodes directly.
- Keep model training silent: always set `verbose=False` or `verbose=0` for all models (XGBoost, LightGBM, CatBoost, RandomForest, etc.). Verbose training prints will accumulate in the history and exceed the context window limit.
- The solution design should be relatively simple.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.
- Start from all train.csv columns except the target; you may drop columns during feature engineering when superseded by derived features.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""
    + SKRUB_DATAOPS_GUIDELINE
)

MODEL_RETRIEVAL_INSTR = strip_rag_tool_mentions(MODEL_RETRIEVAL_INSTR)
MODEL_EVAL_INSTR = strip_rag_tool_mentions(MODEL_EVAL_INSTR)

CHECK_DATA_USE_INSTR = (
    """I have provided Python code for a machine learning task (attached below):
# Solution Code
```python
{code}
```

# Task description
{task_description}

# Your task
If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.
Columns listed in the train.csv schema should reach the model unless they were intentionally dropped during feature engineering (e.g. raw lat/lon replaced by a derived distance column).
DO NOT USE TRY and EXCEPT; just occur error so we can debug it!
See the task description carefully, to know how to extract unused information effectively.
When improving the solution code by incorporating unused information, DO NOT FORGET to print out 'Final Validation Performance: {{final_validation_score}}' as in original solution code.
Add any newly-used features as additional nodes in the existing Skrub DataOps graph (e.g. via `.assign` / `.skb.apply` / `.skb.apply_func`). Do NOT introduce a parallel plain-sklearn pipeline that bypasses Skrub.

Response format:
Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. There should be no additional headings or text in your response.
Option 2: If the code used all the provided information, simply state that "All the provided information is used."
"""
    + SKRUB_DATAOPS_GUIDELINE
)
