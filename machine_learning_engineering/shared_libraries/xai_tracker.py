"""Per-task XAI overhead tracker.

Single source of truth for *how much extra* the XAI module costs a task, in
three currencies: LLM API calls, tokens, and wall-clock seconds.

Design
------
There are exactly two cost primitives in the pipeline, and both funnel through a
single chokepoint, so we only need two hooks plus one attribution flag:

* **LLM calls** -- every call (ADK ``LiteLlm`` agents *and* the direct
  ``llm.complete_text`` instrumentation calls) goes through ``litellm``. A
  single ``litellm`` callback (:func:`install_litellm_logger`) records tokens +
  latency for *all* of them, so nothing is missed -- including the instrumentation
  call that bypasses ADK events.
* **Code executions** -- every generated script runs through
  ``code_util.run_python_code``, which already measures ``execution_time``. That
  function calls :func:`record_code_run`.

Attribution is a depth counter (:func:`enter_xai_phase` / :func:`exit_xai_phase`)
rather than a boolean, so nested XAI regions (the correction loop wrapping the
shared leakage gate) compose correctly. Anything recorded while depth > 0 is
counted as XAI overhead; everything is also counted toward the run total, so the
report can express overhead as an absolute number *and* as a share of the run.

The accumulator is process-global and reset per task via :func:`reset`. This is
safe because tasks never run concurrently in one process (the batch runner uses
one subprocess per task; the eval runner awaits tasks one at a time).
"""

import json
import os
import threading
from typing import Any, Dict

_lock = threading.Lock()

# Depth of nested XAI regions. > 0 means "we are currently inside the XAI
# module" (correction loop and/or the shared leakage gate).
_xai_depth = 0


def enter_xai_phase() -> None:
    """Mark the start of an XAI region. Safe to nest."""
    global _xai_depth
    with _lock:
        _xai_depth += 1


def exit_xai_phase() -> None:
    """Mark the end of an XAI region. Never drops below zero."""
    global _xai_depth
    with _lock:
        if _xai_depth > 0:
            _xai_depth -= 1


def in_xai_phase() -> bool:
    """True while execution is inside an XAI region (lock-free read)."""
    return _xai_depth > 0


def _new_bucket() -> Dict[str, Any]:
    return {
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_latency_s": 0.0,
        "code_runs": 0,
        "compute_s": 0.0,
    }


_total = _new_bucket()
_xai = _new_bucket()
_task_name = ""


def reset(task_name: str = "") -> None:
    """Clear all counters for a fresh task run."""
    global _total, _xai, _task_name, _xai_depth
    with _lock:
        _total = _new_bucket()
        _xai = _new_bucket()
        _task_name = task_name
        _xai_depth = 0


def record_llm_call(prompt_tokens: int, completion_tokens: int, latency_s: float) -> None:
    """Record one completed LLM call. Attributed to XAI iff inside a phase."""
    xai = in_xai_phase()
    pt, ct, lat = int(prompt_tokens or 0), int(completion_tokens or 0), float(latency_s or 0.0)
    with _lock:
        for bucket in (_total, _xai) if xai else (_total,):
            bucket["api_calls"] += 1
            bucket["prompt_tokens"] += pt
            bucket["completion_tokens"] += ct
            bucket["llm_latency_s"] += lat


def record_code_run(execution_time: float) -> None:
    """Record one code execution. Attributed to XAI iff inside a phase."""
    xai = in_xai_phase()
    secs = float(execution_time or 0.0)
    with _lock:
        for bucket in (_total, _xai) if xai else (_total,):
            bucket["code_runs"] += 1
            bucket["compute_s"] += secs


def snapshot() -> Dict[str, Any]:
    """Return a copy of the current counters."""
    with _lock:
        return {"task": _task_name, "total": dict(_total), "xai": dict(_xai)}


def build_report(
    *, xai_correction_enabled: bool, xai_refinement_enabled: bool
) -> Dict[str, Any]:
    """Flatten the current snapshot into the per-task overhead report schema."""
    snap = snapshot()
    t, x = snap["total"], snap["xai"]
    return {
        "task": snap["task"],
        "xai_correction_enabled": bool(xai_correction_enabled),
        "xai_refinement_enabled": bool(xai_refinement_enabled),
        # --- XAI-attributed overhead ---
        "xai_api_calls": x["api_calls"],
        "xai_prompt_tokens": x["prompt_tokens"],
        "xai_completion_tokens": x["completion_tokens"],
        "xai_tokens": x["prompt_tokens"] + x["completion_tokens"],
        "xai_llm_latency_s": round(x["llm_latency_s"], 2),
        "xai_code_runs": x["code_runs"],
        "xai_compute_s": round(x["compute_s"], 2),
        "xai_total_overhead_s": round(x["llm_latency_s"] + x["compute_s"], 2),
        # --- whole-run totals (for share-of-run framing) ---
        "total_api_calls": t["api_calls"],
        "total_prompt_tokens": t["prompt_tokens"],
        "total_completion_tokens": t["completion_tokens"],
        "total_tokens": t["prompt_tokens"] + t["completion_tokens"],
        "total_llm_latency_s": round(t["llm_latency_s"], 2),
        "total_code_runs": t["code_runs"],
        "total_compute_s": round(t["compute_s"], 2),
    }


def write_report(
    run_cwd: str, *, xai_correction_enabled: bool, xai_refinement_enabled: bool
) -> Dict[str, Any]:
    """Write ``xai_overhead.json`` into ``run_cwd`` and return the report dict."""
    report = build_report(
        xai_correction_enabled=xai_correction_enabled,
        xai_refinement_enabled=xai_refinement_enabled,
    )
    os.makedirs(run_cwd, exist_ok=True)
    with open(os.path.join(run_cwd, "xai_overhead.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------
# litellm integration: one global callback captures every LLM call's tokens and
# latency, regardless of whether it came from an ADK agent or a direct
# litellm.completion (instrumentation) call.
# ---------------------------------------------------------------------------

try:  # litellm is a hard dependency of llm.py, but guard so import never fails.
    from litellm.integrations.custom_logger import CustomLogger as _LiteLLMCustomLogger
except Exception:  # pragma: no cover - defensive
    _LiteLLMCustomLogger = object  # type: ignore[assignment, misc]


def _extract_usage(response_obj: Any) -> tuple[int, int]:
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return 0, 0
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    if pt is None and isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
    return int(pt or 0), int(ct or 0)


def _latency_seconds(start_time: Any, end_time: Any) -> float:
    try:
        return (end_time - start_time).total_seconds()
    except Exception:
        return 0.0


class _XAIUsageLogger(_LiteLLMCustomLogger):
    """litellm CustomLogger that forwards every successful call to the tracker."""

    def log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D102
        pt, ct = _extract_usage(response_obj)
        record_llm_call(pt, ct, _latency_seconds(start_time, end_time))

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D102
        pt, ct = _extract_usage(response_obj)
        record_llm_call(pt, ct, _latency_seconds(start_time, end_time))


_logger_singleton: Any = None


def install_litellm_logger() -> None:
    """Register the usage callback with litellm exactly once per process."""
    global _logger_singleton
    if _logger_singleton is not None:
        return
    try:
        import litellm

        _logger_singleton = _XAIUsageLogger()
        callbacks = list(getattr(litellm, "callbacks", []) or [])
        callbacks.append(_logger_singleton)
        litellm.callbacks = callbacks
    except Exception as exc:  # pragma: no cover - never block startup on telemetry
        print(f"[xai_tracker] Could not install litellm usage logger: {exc}")
