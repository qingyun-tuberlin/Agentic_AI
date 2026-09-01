"""Utility functions for XAI analysis and target leakage detection."""

import json
import logging
import os
import shutil
from typing import Any

logger = logging.getLogger(__name__)

XAI_METRICS_FILENAME = "xai_metrics.json"
XAI_METRICS_ARCHIVE_DIR = "xai_metrics_archive"

# Suite verdicts / probe ids (mirror xai_probes).
SUITE_PASS = "PASS"
SUITE_WARN = "WARN"
SUITE_FAIL = "FAIL"
SEMANTIC_PROBE = "semantic_candidates"
REQUIRED_XAI_METRIC_KEYS = frozenset(
    {
        "feature_attributions",
        "validation_score",
        "masked_validation_score",
        "lower_is_better",
    }
)


def to_json_float(value: Any) -> float:
    """Coerce numpy/scalar values to a JSON-serializable float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def validate_xai_metrics(raw: Any) -> tuple[bool, str]:
    """Return (ok, error_message)."""
    if not isinstance(raw, dict) or not raw:
        return False, "metrics is empty or not a dict"

    missing = REQUIRED_XAI_METRIC_KEYS - raw.keys()
    if missing:
        return False, f"missing keys: {sorted(missing)}"

    attrs = raw["feature_attributions"]
    if not isinstance(attrs, dict) or not attrs:
        return False, "feature_attributions must be a non-empty dict"

    for name, value in attrs.items():
        if not isinstance(name, str):
            return False, "feature_attributions keys must be strings"
        try:
            to_json_float(value)
        except (TypeError, ValueError):
            return False, f"invalid attribution for {name!r}"

    try:
        to_json_float(raw["validation_score"])
        to_json_float(raw["masked_validation_score"])
    except (TypeError, ValueError):
        return False, "validation scores must be numeric"

    if not isinstance(raw["lower_is_better"], bool):
        return False, "lower_is_better must be a boolean"

    return True, ""


def normalize_xai_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe metrics dict; assumes validate_xai_metrics passed."""
    expanded = dict(raw)
    if isinstance(expanded.get("feature_attributions"), list):
        expanded["feature_attributions"] = _normalize_feature_attributions(expanded)

    ok, err = validate_xai_metrics(expanded)
    if not ok:
        raise ValueError(err)

    feature_attributions = expanded["feature_attributions"]
    return {
        "feature_attributions": {
            str(k): to_json_float(v) for k, v in feature_attributions.items()
        },
        "validation_score": to_json_float(expanded["validation_score"]),
        "masked_validation_score": to_json_float(expanded["masked_validation_score"]),
        "lower_is_better": bool(expanded["lower_is_better"]),
    }


def _normalize_feature_attributions(raw: dict[str, Any]) -> dict[str, float]:
    feature_attributions = raw.get("feature_attributions", {})
    feature_names = raw.get("feature_names", [])

    if isinstance(feature_attributions, dict):
        return {str(k): to_json_float(v) for k, v in feature_attributions.items()}

    if not isinstance(feature_attributions, list):
        return {}

    if feature_attributions and isinstance(feature_attributions[0], list):
        num_features = len(feature_attributions[0])
        sums = [0.0] * num_features
        count = len(feature_attributions)
        for row in feature_attributions:
            for idx, val in enumerate(row):
                if idx < num_features:
                    try:
                        sums[idx] += abs(to_json_float(val))
                    except (TypeError, ValueError):
                        pass
        mean_attrs = [s / count for s in sums]
        if feature_names and len(feature_names) == num_features:
            return {str(n): v for n, v in zip(feature_names, mean_attrs)}
        return {f"feature_{i}": v for i, v in enumerate(mean_attrs)}

    if feature_attributions and isinstance(feature_attributions[0], (int, float)):
        if feature_names and len(feature_names) == len(feature_attributions):
            return {
                str(n): to_json_float(v) for n, v in zip(feature_names, feature_attributions)
            }
        return {
            f"feature_{i}": to_json_float(v) for i, v in enumerate(feature_attributions)
        }

    return {}


def write_xai_metrics(
    filepath: str,
    metrics: dict[str, Any],
) -> None:
    """Atomically write validated XAI metrics to disk."""
    ok, err = validate_xai_metrics(metrics)
    if not ok:
        raise ValueError(f"Invalid XAI metrics: {err}")
    payload = normalize_xai_metrics(metrics)
    directory = os.path.dirname(filepath) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, filepath)


def archive_xai_metrics(run_cwd: str, label: str) -> str | None:
    """Copy the current ``xai_metrics.json`` to ``xai_metrics_archive/<label>.json``."""
    src = os.path.join(run_cwd, XAI_METRICS_FILENAME)
    if not os.path.exists(src):
        logger.warning("Cannot archive XAI metrics: %s not found", src)
        return None
    archive_dir = os.path.join(run_cwd, XAI_METRICS_ARCHIVE_DIR)
    os.makedirs(archive_dir, exist_ok=True)
    dest = os.path.join(archive_dir, f"{label}.json")
    shutil.copy2(src, dest)
    return dest


def parse_xai_metrics(
    run_cwd: str,
    filename: str = XAI_METRICS_FILENAME,
) -> dict[str, Any]:
    """Parse and validate XAI metrics from the run directory."""
    filepath = os.path.join(run_cwd, filename)
    if not os.path.exists(filepath):
        logger.warning("XAI metrics file not found at %s", filepath)
        return {}

    try:
        with open(filepath, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in XAI metrics %s: %s", filepath, e)
        return {}
    except OSError as e:
        logger.error("Error reading XAI metrics from %s: %s", filepath, e)
        return {}

    expanded = _expand_metrics_input(raw)
    ok, err = validate_xai_metrics(expanded)
    if not ok:
        logger.error("Invalid XAI metrics schema in %s: %s", filepath, err)
        return {}

    metrics = normalize_xai_metrics(raw)
    # Preserve the structured leakage-suite block (xai_probes) alongside the
    # legacy flat keys so downstream routing can use per-probe verdicts.
    for key in ("probes", "verdict", "recommended_action"):
        if key in raw:
            metrics[key] = raw[key]
    return metrics


def calculate_attribution_concentration(feature_attributions: dict[str, float]) -> float:
    """Max absolute attribution / sum of absolute attributions."""
    if not feature_attributions:
        return 0.0

    abs_values = [abs(val) for val in feature_attributions.values()]
    total_abs = sum(abs_values)

    if total_abs == 0:
        return 0.0

    return max(abs_values) / total_abs


def summarize_probes(xai_metrics: dict[str, Any]) -> str:
    """One-line human summary of the probe suite findings, for prompts/logs."""
    probes = xai_metrics.get("probes") or []
    parts = []
    for p in probes:
        if p.get("status") != "ran" or p.get("type") == SEMANTIC_PROBE:
            continue
        suspects = p.get("suspects") or []
        if not suspects or float(p.get("severity", 0.0)) <= 0.0:
            continue
        parts.append(
            f"{p.get('type')}(sev={float(p.get('severity', 0.0)):.2f}, "
            f"suspects={suspects[:3]})"
        )
    return "; ".join(parts) if parts else "no probes flagged a leak"


def evaluate_leakage_report(
    xai_metrics: dict[str, Any],
    max_concentration_threshold: float = 0.80,
    max_robustness_drop_pct: float = 50.0,
) -> tuple[bool, str]:
    """Route off the structured xai_probes block when present; else legacy heuristic.

    Returns (is_leaky, reason). A FAIL from the suite means a quantitative probe
    (direct/proxy/temporal/group) fired strongly; the advisory ``semantic_candidates``
    probe never drives this verdict on its own (it is routed to the LLM review).
    """
    if not xai_metrics:
        return False, "No XAI metrics available for analysis."

    verdict = xai_metrics.get("verdict")
    probes = xai_metrics.get("probes")
    if verdict in (SUITE_PASS, SUITE_WARN, SUITE_FAIL) and probes is not None:
        summary = summarize_probes(xai_metrics)
        action = xai_metrics.get("recommended_action", "review")
        if verdict == SUITE_FAIL:
            fired = [
                p for p in probes
                if p.get("status") == "ran" and p.get("suspects")
                and p.get("type") != SEMANTIC_PROBE
                and float(p.get("severity", 0.0)) > 0.0
            ]
            top = max(fired, key=lambda p: float(p.get("severity", 0.0)), default=None)
            label = top.get("type") if top else "unknown"
            suspects = (top.get("suspects") if top else []) or []
            return True, (
                f"SUSPECTED LEAKAGE [{label}] suspects={suspects[:3]} "
                f"action={action}. Probes: {summary}"
            )
        if verdict == SUITE_WARN:
            return False, (
                f"No blocking leakage; moderate signal flagged for review. "
                f"Probes: {summary}"
            )
        return False, f"No leakage detected by probes. Probes: {summary}"

    # Legacy flat-metrics fallback (no structured block in the file).
    return evaluate_leakage_risk(
        xai_metrics, max_concentration_threshold, max_robustness_drop_pct
    )


def _expand_metrics_input(raw: dict[str, Any]) -> dict[str, Any]:
    expanded = dict(raw)
    if isinstance(expanded.get("feature_attributions"), list):
        expanded["feature_attributions"] = _normalize_feature_attributions(expanded)
    return expanded


def evaluate_leakage_risk(
    xai_metrics: dict[str, Any],
    max_concentration_threshold: float = 0.80,
    max_robustness_drop_pct: float = 50.0,
) -> tuple[bool, str]:
    """Flag leakage when concentration and masked-score fragility are both high."""
    if not xai_metrics:
        return False, "No XAI metrics available for analysis."

    expanded = _expand_metrics_input(xai_metrics)
    ok, err = validate_xai_metrics(expanded)
    if not ok:
        return False, f"Invalid XAI metrics: {err}"

    metrics = normalize_xai_metrics(xai_metrics)
    feature_attributions = metrics["feature_attributions"]

    concentration = calculate_attribution_concentration(feature_attributions)
    top_feature = max(feature_attributions, key=lambda k: abs(feature_attributions[k]))

    val_score = metrics["validation_score"]
    masked_score = metrics["masked_validation_score"]
    lower = metrics["lower_is_better"]

    reason = f"Top feature: '{top_feature}' with concentration: {concentration:.2f}."

    is_fragile = False
    drop_info = ""
    if val_score != 0:
        if lower:
            drop_pct = ((masked_score - val_score) / val_score) * 100.0
            is_fragile = drop_pct >= max_robustness_drop_pct
            drop_info = f" Masked score increased by {drop_pct:.1f}% (lower is better)."
        else:
            drop_pct = ((val_score - masked_score) / val_score) * 100.0
            is_fragile = drop_pct >= max_robustness_drop_pct
            drop_info = f" Masked score dropped by {drop_pct:.1f}% (higher is better)."

    if concentration >= max_concentration_threshold:
        if is_fragile:
            return (
                True,
                f"SUSPECTED TARGET LEAKAGE! Feature '{top_feature}' has "
                f"{concentration * 100:.1f}% of model attribution.{drop_info}",
            )
        return (
            False,
            reason
            + f" Highly dominant but robustness drop did not exceed threshold.{drop_info}",
        )

    return False, reason + " Feature attribution is distributed reasonably."
