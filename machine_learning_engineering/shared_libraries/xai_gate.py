"""Reusable dynamic XAI leakage gate.

Single source of truth for "run this training code and return a leakage
verdict". Both the pre-refinement gate (``xai_correction``) and the in-loop
gate (gated ``refinement``) call :func:`audit_code_for_leakage`, so the audit
logic can never drift between them.

The function is intentionally free of any ADK ``callback_context`` coupling:
callers own all state reads/writes. Instrumentation (adding the SHAP /
robustness block that writes ``xai_metrics.json``) is delegated to an optional
``instrument_fn`` so this module does not depend on any prompt or sub-agent.
"""

import dataclasses
from typing import Callable, Optional

from machine_learning_engineering.shared_libraries import code_util, xai_tracker, xai_util

PASS = "PASS"
FAIL = "FAIL"

# instrument_fn(code, error) -> instrumented_code | None
InstrumentFn = Callable[[str, str], Optional[str]]


@dataclasses.dataclass
class GateResult:
    """Outcome of a single leakage audit.

    ``metrics is None`` means the audit could not produce valid dynamic metrics
    (execution failed or ``xai_metrics.json`` was missing/invalid); the caller
    decides the fallback policy in that case. When metrics are present,
    ``verdict`` is the authoritative PASS/FAIL from ``evaluate_leakage_report``.
    """

    verdict: str
    reason: str
    metrics: Optional[dict]
    metrics_preview: str
    code: str
    exec_result: dict
    error: str


def metrics_preview(xai_metrics: dict) -> str:
    """One-line human summary of an XAI metrics dict, for prompts/logs."""
    attrs = xai_metrics.get("feature_attributions", {})
    if not attrs:
        return "No attributions"
    top = max(attrs, key=lambda k: abs(attrs[k]))
    concentration = xai_util.calculate_attribution_concentration(attrs)
    preview = (
        f"top_feature={top!r}, concentration={concentration:.2f}, "
        f"validation_score={xai_metrics.get('validation_score')}, "
        f"masked_validation_score={xai_metrics.get('masked_validation_score')}, "
        f"lower_is_better={xai_metrics.get('lower_is_better')}"
    )
    if xai_metrics.get("probes") is not None:
        preview += (
            f", suite_verdict={xai_metrics.get('verdict')}, "
            f"probes=[{xai_util.summarize_probes(xai_metrics)}]"
        )
    return preview


def _run_and_parse(
    code: str,
    run_cwd: str,
    *,
    lower: bool,
    exec_timeout: int,
    py_filepath: str,
) -> tuple[Optional[dict], dict, str]:
    """Execute ``code`` and parse ``xai_metrics.json``; always score the run."""
    result = code_util.run_python_code(
        code_text=code,
        run_cwd=run_cwd,
        py_filepath=py_filepath,
        exec_timeout=exec_timeout,
    )
    if result.get("returncode", 1) == 0:
        try:
            result["score"] = float(
                code_util.extract_performance_from_text(result.get("stdout", ""))
            )
        except Exception:
            result["score"] = 1e9 if lower else 0.0
    else:
        result["score"] = 1e9 if lower else 0.0
        stderr = result.get("stderr") or ""
        if len(stderr) > 4000:
            stderr = stderr[:1000] + "\n... [TRUNCATED] ...\n" + stderr[-3000:]
        return None, result, f"Execution failed: {stderr}"

    metrics = xai_util.parse_xai_metrics(run_cwd)
    if not metrics:
        return None, result, "xai_metrics.json missing or invalid after execution"
    return metrics, result, ""


def audit_code_for_leakage(
    code: str,
    run_cwd: str,
    *,
    lower: bool,
    exec_timeout: int,
    py_filepath: str = "train0.py",
    max_concentration: float = 0.80,
    instrument_fn: Optional[InstrumentFn] = None,
    max_attempts: int = 2,
    metrics_archive_label: Optional[str] = None,
) -> GateResult:
    """Run (optionally instrumenting) training code and return a leakage verdict.

    If ``instrument_fn`` is provided, it is invoked when the code lacks XAI
    instrumentation or to repair a failed attempt. With no ``instrument_fn`` the
    code is expected to already write ``xai_metrics.json`` itself.
    """
    # Bracket the whole audit as an XAI region so its instrumented code run(s)
    # and any instrument_fn LLM call(s) are attributed to XAI overhead, no matter
    # which caller (pre-refinement correction or in-loop gate) invoked us.
    xai_tracker.enter_xai_phase()
    try:
        last_error = ""
        last_result: dict = {}
        for attempt in range(max_attempts):
            needs_instrument = "xai_metrics.json" not in code or (attempt > 0 and last_error)
            if needs_instrument and instrument_fn is not None:
                instrumented = instrument_fn(code, last_error if attempt > 0 else "")
                if not instrumented:
                    last_error = (
                        last_error or "instrumentation failed: no code block extracted"
                    )
                    continue
                code = instrumented

            metrics, last_result, err = _run_and_parse(
                code,
                run_cwd,
                lower=lower,
                exec_timeout=exec_timeout,
                py_filepath=py_filepath,
            )
            if metrics is not None:
                if metrics_archive_label:
                    xai_util.archive_xai_metrics(run_cwd, metrics_archive_label)
                is_leaky, reason = xai_util.evaluate_leakage_report(
                    xai_metrics=metrics,
                    max_concentration_threshold=max_concentration,
                )
                return GateResult(
                    verdict=FAIL if is_leaky else PASS,
                    reason=reason,
                    metrics=metrics,
                    metrics_preview=metrics_preview(metrics),
                    code=code,
                    exec_result=last_result,
                    error="",
                )
            last_error = err

        return GateResult(
            verdict=FAIL,
            reason=f"Dynamic XAI audit could not produce valid metrics. {last_error}",
            metrics=None,
            metrics_preview="",
            code=code,
            exec_result=last_result,
            error=last_error or "Unknown dynamic audit error",
        )
    finally:
        xai_tracker.exit_xai_phase()
