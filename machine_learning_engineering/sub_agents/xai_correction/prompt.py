

FEATURE_REVISION_INSTR = """
You are a machine learning engineer.
The XAI auditor has reviewed the current model and produced the following findings:

## XAI Audit Result
{audit_result}

## Current Code
{current_code}

Based on the audit findings, revise the feature engineering in the code.
Make sure the rest of the code remains consistent with your changes.
Preserve the XAI instrumentation that writes `xai_metrics.json` (see instrumentation rules: _to_json_float, atomic write, required schema).
Return only the complete modified Python code, nothing else.
"""

# NOTE: the XAI instrumentation prompts (add / fix) now live in
# `shared_libraries/xai_instrumentation_guide.instrument_via_llm`, shared by every
# leakage gate so their wording cannot drift between agents.


TERMINATION_REPORT_INSTR = """
You are a machine learning engineer.
The XAI self-correction loop has reached the maximum number of iterations ({loop_count}), but the model still failed the XAI audit.

## Loop Count
{loop_count}

## Audit History
{audit_history}

Please generate a report for the user with the following sections:
- Report title
- Summary of what was audited and attempted in each iteration
- Remaining issues that could not be resolved
- Recommendations for the user to manually address the remaining issues
"""


AUDIT_INSTR = """
You are an expert Machine Learning Explainability (XAI) and Data Leakage Auditor.
Dynamic XAI metrics could not be computed. Use this static review only as a secondary check.

## Dynamic audit error (if any)
{audit_error}

## Numeric XAI metrics (if any were produced)
{xai_metrics_summary}

Specifically, check if:
1. The code uses features derived from the target or future information (target leakage).
2. Train/validation split or preprocessing causes leakage (e.g. scaling before split).

If dynamic metrics are missing or invalid, prefer verdict FAIL unless the code is clearly safe.

Produce a JSON response:
```json
{{
  "verdict": "PASS" | "FAIL",
  "reason": "Detailed explanation."
}}
```

Output rules (strict):
- Return ONLY the JSON object above.
- Put the full explanation inside the JSON "reason" field only.

Here is the current training code to audit:
{code}
"""
