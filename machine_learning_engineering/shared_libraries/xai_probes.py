"""Leakage-detection probe suite — shipped, NOT LLM-generated.

This module is the single source of truth for the dynamic (post-fit) XAI
leakage checks. Generated solution scripts import it and call exactly one entry
point::

    import xai_probes
    xai_probes.run_leakage_suite(
        learner, data, task_meta,
        train_df=train_df, learner_factory=lambda: pred.skb.make_learner(),
    )

It then writes a structured ``xai_metrics.json`` that ``xai_util.py`` parses and
that the ``xai_correction`` agent routes corrections from.

Why a shipped library instead of LLM-generated instrumentation:
- A multi-probe suite is too complex to regenerate reliably each run.
- The old LLM-generated SHAP + prefix-remap code silently mis-maps one-input →
  many-output encoders (e.g. skrub ``GapEncoder`` names columns ``"col: ..."``
  which a ``col_`` prefix rule misses), fragmenting attribution and DEFLATING
  the concentration metric — i.e. false negatives in the leakage detector.

Design notes:
- Each leak type has a dedicated probe (see the leak-type × method matrix in the
  project docs). Probes are independent and individually guarded: one probe
  raising must never abort the suite.
- This code runs inside the sandboxed solution process and has NO LLM access.
  The *semantic* judgement (is a high-attribution feature a post-outcome
  variable / identifier?) is done by the ``xai_correction`` agent, which reads
  the ``semantic_candidates`` block this module exports.
- Probes that need to refit under a different split (temporal, group) receive a
  ``learner_factory`` (fresh unfitted learner) and ``train_df`` (full training
  frame). Keep cost bounded with ``subsample``.

Status: P0 skeleton. ``direct`` and ``proxy_power`` are implemented;
``temporal`` and ``group`` are wired stubs (interface fixed, logic TODO).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from typing import Any, Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Probe identifiers. Keep in sync with the leak-type × method matrix.
DIRECT = "direct"
PROXY_POWER = "proxy_power"
SEMANTIC = "semantic_candidates"
TEMPORAL = "temporal"
GROUP = "group"

DEFAULT_PROBES: tuple[str, ...] = (DIRECT, PROXY_POWER, SEMANTIC, TEMPORAL, GROUP)

# Verdict labels (consumed by xai_util / xai_correction routing).
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TaskMeta:
    """Everything the probes need to know about the task, supplied by the script."""

    lower_is_better: bool
    metric_fn: Callable[[Any, Any], float]
    metric_name: str = "score"
    task_type: str = ""  # e.g. "Tabular Classification" / "Tabular Regression"
    target_column: str = ""
    # Hints for the resampling probes; empty => auto-infer in the probe.
    datetime_columns: Sequence[str] = ()
    group_columns: Sequence[str] = ()

    @property
    def is_classification(self) -> bool:
        return "classif" in self.task_type.lower()


@dataclasses.dataclass
class ProbeResult:
    """One probe's structured finding."""

    type: str
    status: str  # "ran" | "skipped" | "not_implemented" | "error"
    suspects: list[str] = dataclasses.field(default_factory=list)
    severity: float = 0.0  # 0..1, comparable across probes
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["severity"] = _json_float(self.severity)
        d["evidence"] = _json_safe(self.evidence)
        return d


@dataclasses.dataclass
class SuiteReport:
    probes: list[ProbeResult]
    verdict: str
    recommended_action: str
    legacy: dict[str, Any]  # back-compat flat schema for current xai_util

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.legacy)  # keep top-level legacy keys for back-compat
        out["probes"] = [p.to_dict() for p in self.probes]
        out["verdict"] = self.verdict
        out["recommended_action"] = self.recommended_action
        return out


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _json_float(x: Any) -> float:
    return float(x.item()) if hasattr(x, "item") else float(x)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return _json_float(obj)
    return obj


def _is_skrub_learner(learner: Any) -> bool:
    return hasattr(learner, "data_op")


def _predict(learner: Any, X: Any, is_skrub: bool) -> Any:
    """Uniform predict for skrub learners (env dict) vs sklearn estimators."""
    if is_skrub:
        return learner.predict({"_skrub_X": X})
    return learner.predict(X)


def _score(learner: Any, X: Any, y: Any, meta: TaskMeta, is_skrub: bool) -> float:
    return _json_float(meta.metric_fn(y, _predict(learner, X, is_skrub)))


def _split_parent(out_col: str, raw_cols_by_len: list[str]) -> Optional[str]:
    """Map a preprocessed output column back to its originating input column.

    Handles the separators skrub/sklearn encoders actually emit:
      OneHot/TableVectorizer -> "col_value"
      MinHash/StringEncoder  -> "col_0", "col_1"
      GapEncoder             -> "col: topic words"   (the case the old code missed)
      DatetimeEncoder        -> "col_year", "col_month"
    Longest raw name first so "age_group" wins over "age".
    """
    for raw in raw_cols_by_len:
        if out_col == raw:
            return raw
        for sep in ("_", " ", ": ", ":"):
            if out_col.startswith(raw + sep):
                return raw
    return None


# --------------------------------------------------------------------------- #
# Attribution at ORIGINAL-column granularity (fixes the one-to-many problem)
# --------------------------------------------------------------------------- #
def original_column_attributions(
    learner: Any,
    X_val: Any,
    y_val: Any,
    meta: TaskMeta,
    is_skrub: bool,
) -> dict[str, float]:
    """Mean |attribution| per ORIGINAL input column.

    Strategy:
      1. Try SHAP on the preprocessed matrix and fold derived columns back onto
         their parent input column via `_split_parent` (separator-aware).
      2. If anything is uncertain (no column names on the preprocessed matrix,
         or many outputs fail to map), fall back to permutation importance over
         the RAW input columns — which is inherently one-to-one and needs no
         remap, so it cannot fragment.
    """
    raw_cols = list(X_val.columns)
    raw_by_len = sorted(raw_cols, key=len, reverse=True)

    try:
        out_cols, mean_abs = _shap_on_preprocessed(learner, X_val, is_skrub)
        if out_cols is not None:
            agg: dict[str, float] = {}
            unmapped = 0
            for col, val in zip(out_cols, mean_abs):
                parent = _split_parent(str(col), raw_by_len)
                if parent is None:
                    unmapped += 1
                    parent = str(col)
                agg[parent] = agg.get(parent, 0.0) + float(val)
            # If too many outputs couldn't be attributed to a real input column,
            # the remap is untrustworthy -> use the robust raw-column fallback.
            if unmapped <= 0.25 * max(1, len(out_cols)):
                return {k: _json_float(v) for k, v in agg.items()}
            logger.warning(
                "SHAP remap unreliable (%d/%d outputs unmapped); using permutation.",
                unmapped, len(out_cols),
            )
    except Exception as exc:  # noqa: BLE001 - probe must degrade, not crash
        logger.warning("SHAP attribution failed (%s); using permutation.", exc)

    return _permutation_on_raw(learner, X_val, y_val, meta, is_skrub, raw_cols)


_TREE_MODEL_NAMES = frozenset({
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
})


def _is_tree_model(model: Any) -> bool:
    return (
        hasattr(model, "tree_")
        or hasattr(model, "estimators_")
        or type(model).__name__ in _TREE_MODEL_NAMES
    )


# Step names the generated pipeline assigns via `.skb.set_name(...)` so the
# suite can reach the fitted model and the preprocessed matrix for TreeSHAP.
SKRUB_FEATURES_STEP = "xai_features"  # final preprocessing node (model input)
SKRUB_MODEL_STEP = "xai_model"  # the .skb.apply(model, y=y) step

_SKRUB_WRAPPERS = frozenset({"ApplyToCols", "ApplyToFrame", "ApplyToSubFrame"})


def _unwrap_skrub_estimator(est: Any) -> Any:
    """Unwrap skrub's column-wise wrappers; leave sklearn estimators untouched.

    Note: do NOT blindly read `.estimator_` — on a fitted sklearn ensemble (e.g.
    RandomForest) that is the unfitted base template, not the model we want.
    """
    if type(est).__name__ in _SKRUB_WRAPPERS:
        for attr in ("estimator_", "transformer_"):
            if hasattr(est, attr):
                return getattr(est, attr)
    return est


def _shap_on_preprocessed(
    learner: Any, X_val: Any, is_skrub: bool
) -> tuple[Optional[list[str]], Optional[np.ndarray]]:
    """Fast TreeSHAP path. Returns (column_names, mean_abs_shap) or (None, None).

    skrub: requires the pipeline to have named its steps with `.skb.set_name`:
      - ``SKRUB_MODEL_STEP`` on the model apply -> fitted model via
        ``learner.find_fitted_estimator(...)``;
      - ``SKRUB_FEATURES_STEP`` on the final preprocessing node -> the exact
        preprocessed matrix (with real column names) via
        ``learner.truncated_after(...).transform({"_skrub_X": X_val})``.
      The caller then folds the preprocessed columns back onto their original
      input columns. If the steps are unnamed/unreachable, returns (None, None)
      and the robust raw-column permutation path is used instead.
    non-skrub: a tree estimator whose direct input IS X_val (no remap needed).
    """
    try:
        import shap  # noqa: F401
    except Exception:  # noqa: BLE001
        return None, None

    if is_skrub:
        try:
            model = _unwrap_skrub_estimator(
                learner.find_fitted_estimator(SKRUB_MODEL_STEP)
            )
            if not _is_tree_model(model):
                return None, None
            X_pre = learner.truncated_after(SKRUB_FEATURES_STEP).transform(
                {"_skrub_X": X_val}
            )
        except Exception:  # noqa: BLE001 - steps unnamed/unreachable -> permutation
            return None, None
    else:
        model, X_pre = learner, X_val
        if not _is_tree_model(model):
            return None, None

    try:
        sv = shap.TreeExplainer(model).shap_values(X_pre)
        if isinstance(sv, list):  # one array per class -> average |contrib|
            arr = np.mean([np.abs(s) for s in sv], axis=0)
        else:
            arr = np.abs(sv)
        if arr.ndim == 3:  # (n_samples, n_features, n_classes)
            arr = arr.mean(axis=2)
        cols = (
            list(X_pre.columns)
            if hasattr(X_pre, "columns")
            else [f"feature_{i}" for i in range(np.asarray(X_pre).shape[1])]
        )
        return cols, np.asarray(arr, dtype=float).mean(axis=0)
    except Exception:  # noqa: BLE001 - any SHAP failure -> permutation fallback
        return None, None


def _permutation_on_raw(
    learner: Any,
    X_val: Any,
    y_val: Any,
    meta: TaskMeta,
    is_skrub: bool,
    raw_cols: list[str],
) -> dict[str, float]:
    """Permutation importance over raw input columns (one-to-one, no remap)."""
    rng = np.random.default_rng(0)
    base = _score(learner, X_val, y_val, meta, is_skrub)
    sign = 1.0 if meta.lower_is_better else -1.0  # positive => importance
    importances: dict[str, float] = {}
    for col in raw_cols:
        Xp = X_val.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        permuted = _score(learner, Xp, y_val, meta, is_skrub)
        importances[col] = _json_float(abs(sign * (permuted - base)))
    return importances


def attribution_concentration(attrs: dict[str, float]) -> float:
    """max(|a|) / sum(|a|); mirrors xai_util.calculate_attribution_concentration."""
    if not attrs:
        return 0.0
    abs_vals = [abs(v) for v in attrs.values()]
    total = sum(abs_vals)
    return (max(abs_vals) / total) if total else 0.0


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def probe_direct(ctx: "ProbeContext") -> ProbeResult:
    """Direct leak: one original column dominates attribution AND masking it
    collapses validation performance."""
    attrs = ctx.attributions
    if not attrs:
        return ProbeResult(DIRECT, "skipped", message="no attributions available")

    concentration = attribution_concentration(attrs)
    top = max(attrs, key=lambda k: abs(attrs[k]))

    # Mask the top feature and re-score on the existing val split (no refit).
    X_masked = ctx.X_val.copy()
    col = X_masked[top]
    X_masked[top] = col.median() if col.dtype.kind in "biufc" else col.mode().iloc[0]
    masked_score = _score(ctx.learner, X_masked, ctx.y_val, ctx.meta, ctx.is_skrub)

    drop_pct = _robustness_drop_pct(ctx.val_score, masked_score, ctx.meta.lower_is_better)
    fragile = drop_pct >= ctx.max_robustness_drop_pct
    dominant = concentration >= ctx.max_concentration

    severity = min(1.0, concentration) * (1.0 if fragile else 0.4)
    status_msg = (
        f"top='{top}' concentration={concentration:.2f} "
        f"masked_drop={drop_pct:.1f}%"
    )
    return ProbeResult(
        type=DIRECT,
        status="ran",
        suspects=[top] if (dominant and fragile) else [],
        severity=severity if (dominant and fragile) else min(0.4, severity),
        evidence={
            "concentration": concentration,
            "validation_score": ctx.val_score,
            "masked_validation_score": masked_score,
            "robustness_drop_pct": drop_pct,
            "dominant": dominant,
            "fragile": fragile,
        },
        message=status_msg,
    )


def probe_proxy_power(ctx: "ProbeContext") -> ProbeResult:
    """Proxy/direct leak: a single feature alone recovers nearly all of the
    model's *skill over the base rate* — a hallmark of a leaked/proxy column.

    The discriminator is lift over a naive baseline, NOT raw performance. A
    strong-but-legitimate predictor recovers a large fraction of the model's
    accuracy simply because the base rate is already high; only a column that is
    individually almost *sufficient* (its lift over baseline approaches the full
    model's) is suspicious. Each candidate is fit on a held-out split (no
    in-sample scoring) so a shallow tree cannot memorise its way to a flag.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(PROXY_POWER, "skipped", message=f"sklearn unavailable: {exc}")

    attrs = ctx.attributions or {}
    ranked = sorted(attrs, key=lambda k: abs(attrs[k]), reverse=True)
    candidates = ranked[: ctx.top_k] if ranked else list(ctx.X_val.columns)[: ctx.top_k]

    y_arr = np.asarray(ctx.y_val)
    idx = np.arange(len(y_arr))
    strat = y_arr if ctx.meta.is_classification else None
    try:
        tr_idx, va_idx = train_test_split(
            idx, test_size=0.3, random_state=0, stratify=strat
        )
    except Exception:  # noqa: BLE001 - tiny/imbalanced classes -> unstratified
        tr_idx, va_idx = train_test_split(idx, test_size=0.3, random_state=0)
    y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

    base_score = _baseline_score(y_tr, y_va, ctx.meta)
    full = ctx.val_score

    # If the full model barely beats the naive baseline there is no skill to
    # attribute to any single feature, so "sufficiency" is meaningless — stay
    # silent rather than risk flagging (and dropping) the model's only signal.
    margin = (base_score - full) if ctx.meta.lower_is_better else (full - base_score)
    if margin <= 0.02 * max(abs(full), abs(base_score), 1e-9):
        return ProbeResult(
            type=PROXY_POWER,
            status="ran",
            suspects=[],
            severity=0.0,
            evidence={"baseline_score": base_score, "full_model_score": full},
            message=(
                f"full model has negligible lift over baseline "
                f"(baseline={base_score:.3f}, full={full:.3f}); probe inactive"
            ),
        )

    suspects: list[str] = []
    per_feature: dict[str, float] = {}
    detail: dict[str, dict[str, float]] = {}
    for col in candidates:
        try:
            xcol = ctx.X_val[[col]]
            if xcol[col].dtype.kind not in "biufc":
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                enc.fit(xcol.iloc[tr_idx].astype(str))
                xvals = enc.transform(xcol.astype(str))
            else:
                xvals = xcol.to_numpy()
            est = (
                DecisionTreeClassifier(max_depth=3, random_state=0)
                if ctx.meta.is_classification
                else DecisionTreeRegressor(max_depth=3, random_state=0)
            )
            est.fit(xvals[tr_idx], y_tr)
            single_score = _json_float(ctx.meta.metric_fn(y_va, est.predict(xvals[va_idx])))
            # Fraction of the model's lift-over-baseline recovered by one feature.
            lift = _lift_fraction(single_score, full, base_score, ctx.meta.lower_is_better)
            per_feature[col] = lift
            detail[col] = {
                "single_score": single_score,
                "lift_fraction": lift,
            }
            if lift >= ctx.single_feature_ratio:
                suspects.append(col)
        except Exception as exc:  # noqa: BLE001 - per-feature guard
            logger.debug("proxy_power: feature %s skipped (%s)", col, exc)

    severity = min(1.0, max(per_feature.values(), default=0.0))
    return ProbeResult(
        type=PROXY_POWER,
        status="ran",
        suspects=suspects,
        severity=severity,
        evidence={
            "baseline_score": base_score,
            "full_model_score": full,
            "lift_fraction": per_feature,
            "per_feature": detail,
        },
        message=(
            f"{len(suspects)} feature(s) recover >= "
            f"{ctx.single_feature_ratio:.0%} of model lift over baseline "
            f"(baseline={base_score:.3f}, full={full:.3f})"
        ),
    )


def probe_semantic_candidates(ctx: "ProbeContext") -> ProbeResult:
    """No LLM here. Export the top-attribution features so the xai_correction
    agent can ask the LLM whether any is semantically a post-outcome variable
    or identifier (the headline 'semantic leakage' novelty)."""
    attrs = ctx.attributions or {}
    ranked = sorted(attrs, key=lambda k: abs(attrs[k]), reverse=True)[: ctx.top_k]
    return ProbeResult(
        type=SEMANTIC,
        status="ran",
        suspects=ranked,
        severity=0.0,  # severity assigned by the agent after LLM judgement
        evidence={"top_features": {k: attrs[k] for k in ranked}},
        message="exported top features for LLM semantic review",
    )


def probe_temporal(ctx: "ProbeContext") -> ProbeResult:
    """Temporal leak: random-split score >> time-ordered-split score.

    TODO(P3): using ctx.learner_factory + ctx.train_df, refit on a time-ordered
    split (sort by an inferred/declared datetime column, hold out the tail) and
    compare to the random-split score. A large favourable gap under random
    splitting indicates temporal leakage. Needs learner_factory + train_df.
    """
    if ctx.learner_factory is None or ctx.train_df is None:
        return ProbeResult(
            TEMPORAL, "skipped",
            message="needs learner_factory + train_df (not provided)",
        )
    return ProbeResult(TEMPORAL, "not_implemented", message="P3")


def probe_group(ctx: "ProbeContext") -> ProbeResult:
    """Group/entity leak: random-split score >> grouped-split score.

    TODO(P3): infer candidate id/high-cardinality group columns, refit with a
    GroupKFold/grouped holdout via ctx.learner_factory, and compare to the
    random-split score. Needs learner_factory + train_df.
    """
    if ctx.learner_factory is None or ctx.train_df is None:
        return ProbeResult(
            GROUP, "skipped",
            message="needs learner_factory + train_df (not provided)",
        )
    return ProbeResult(GROUP, "not_implemented", message="P3")


_PROBE_FNS: dict[str, Callable[["ProbeContext"], ProbeResult]] = {
    DIRECT: probe_direct,
    PROXY_POWER: probe_proxy_power,
    SEMANTIC: probe_semantic_candidates,
    TEMPORAL: probe_temporal,
    GROUP: probe_group,
}


def _robustness_drop_pct(val: float, masked: float, lower_is_better: bool) -> float:
    if val == 0.0:
        if lower_is_better:
            return 100.0 if masked > 0.0 else 0.0
        else:
            return 100.0 if masked < 0.0 else 0.0
    return ((masked - val) / val * 100.0) if lower_is_better else ((val - masked) / val * 100.0)


def _baseline_score(y_train: Any, y_val: Any, meta: TaskMeta) -> float:
    """Score of a naive constant predictor (majority class / training mean).

    This is the floor the full model must beat. Comparing single-feature
    performance against this floor — rather than against raw accuracy — is what
    separates a genuine leak from a feature that merely rides a high base rate.
    """
    from sklearn.dummy import DummyClassifier, DummyRegressor

    dummy = (
        DummyClassifier(strategy="most_frequent")
        if meta.is_classification
        else DummyRegressor(strategy="mean")
    )
    x_tr = np.zeros((len(y_train), 1))
    x_va = np.zeros((len(y_val), 1))
    dummy.fit(x_tr, y_train)
    return _json_float(meta.metric_fn(y_val, dummy.predict(x_va)))


def _lift_fraction(
    single: float, full: float, base: float, lower_is_better: bool
) -> float:
    """Fraction of the full model's lift-over-baseline recovered by one feature.

    ~1.0 (or higher) means the single feature is essentially sufficient — the
    rest of the features add nothing — which is the signature of a leak/proxy.
    Values well below 1.0 indicate a strong but non-sufficient predictor. Returns
    0.0 when the full model itself barely beats the baseline (lift undefined).
    """
    if lower_is_better:
        denom, num = base - full, base - single
    else:
        denom, num = full - base, single - base
    if denom <= 1e-12:
        return 0.0
    return max(0.0, num / denom)


# --------------------------------------------------------------------------- #
# Probe context + suite runner
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ProbeContext:
    learner: Any
    X_val: Any
    y_val: Any
    meta: TaskMeta
    is_skrub: bool
    val_score: float
    attributions: dict[str, float]
    train_df: Any = None
    learner_factory: Optional[Callable[[], Any]] = None
    # Thresholds (mirror config.xai_*; can be overridden by run_leakage_suite).
    max_concentration: float = 0.80
    max_robustness_drop_pct: float = 50.0
    # Min fraction of the model's lift-over-baseline a single feature must
    # recover to be flagged as a likely leak/proxy (see probe_proxy_power).
    single_feature_ratio: float = 0.95
    top_k: int = 10


def combine_results(results: list[ProbeResult]) -> tuple[str, str]:
    """Aggregate probe results into (verdict, recommended_action).

    `semantic_candidates` is advisory (severity 0; it only exports features for a
    separate LLM review) and never drives the verdict, nor do zero-severity
    results. Only quantitative probes that actually fired count.
    """
    fired = [
        r for r in results
        if r.status == "ran"
        and r.suspects
        and r.type != SEMANTIC
        and r.severity > 0.0
    ]
    if not fired:
        return PASS, "none"

    worst = max(fired, key=lambda r: r.severity)
    action = {
        DIRECT: "drop_feature",
        PROXY_POWER: "drop_feature",
        SEMANTIC: "llm_semantic_review",
        TEMPORAL: "use_time_ordered_split",
        GROUP: "use_grouped_cv",
    }.get(worst.type, "review")

    # Strong, corroborated signal => FAIL; otherwise WARN for agent review.
    verdict = FAIL if worst.severity >= 0.8 else WARN
    return verdict, action


def run_leakage_suite(
    learner: Any,
    data: dict[str, Any],
    task_meta: TaskMeta,
    *,
    train_df: Any = None,
    learner_factory: Optional[Callable[[], Any]] = None,
    probes: Sequence[str] = DEFAULT_PROBES,
    out_path: str = "xai_metrics.json",
    subsample: Optional[int] = 5000,
    max_concentration: float = 0.80,
    max_robustness_drop_pct: float = 50.0,
) -> dict[str, Any]:
    """Run the leakage probe suite and atomically write a structured report.

    Args:
        learner: fitted learner (skrub learner or sklearn estimator).
        data: split dict from `pred.skb.train_test_split(...)`; uses
            ``X_test``/``y_test`` as the validation set.
        task_meta: TaskMeta describing metric/target/task type.
        train_df: full training frame (required by temporal/group probes).
        learner_factory: callable returning a fresh unfitted learner
            (required by temporal/group probes).
        probes: which probes to run.
        out_path: where to write the JSON report.
        subsample: cap validation rows used by probes (None = all).
    Returns:
        The report dict that was written.
    """
    is_skrub = _is_skrub_learner(learner)
    X_val, y_val = data["X_test"], data["y_test"]
    if subsample and hasattr(X_val, "shape") and X_val.shape[0] > subsample:
        idx = np.random.default_rng(0).choice(X_val.shape[0], subsample, replace=False)
        X_val, y_val = X_val.iloc[idx], y_val.iloc[idx]

    val_score = _score(learner, X_val, y_val, task_meta, is_skrub)
    attrs = original_column_attributions(learner, X_val, y_val, task_meta, is_skrub)

    ctx = ProbeContext(
        learner=learner, X_val=X_val, y_val=y_val, meta=task_meta,
        is_skrub=is_skrub, val_score=val_score, attributions=attrs,
        train_df=train_df, learner_factory=learner_factory,
        max_concentration=max_concentration,
        max_robustness_drop_pct=max_robustness_drop_pct,
    )

    results: list[ProbeResult] = []
    for name in probes:
        fn = _PROBE_FNS.get(name)
        if fn is None:
            logger.warning("Unknown probe %r; skipping.", name)
            continue
        try:
            results.append(fn(ctx))
        except Exception as exc:  # noqa: BLE001 - one probe must not abort the suite
            logger.exception("Probe %s failed", name)
            results.append(ProbeResult(name, "error", message=str(exc)))

    verdict, action = combine_results(results)

    # Legacy flat schema so the current xai_util keeps working during migration.
    top_feature = max(attrs, key=lambda k: abs(attrs[k])) if attrs else None
    masked = next(
        (r.evidence.get("masked_validation_score") for r in results if r.type == DIRECT),
        val_score,
    )
    legacy = {
        "feature_attributions": {str(k): _json_float(v) for k, v in attrs.items()},
        "validation_score": _json_float(val_score),
        "masked_validation_score": _json_float(masked),
        "lower_is_better": bool(task_meta.lower_is_better),
    }

    report = SuiteReport(results, verdict, action, legacy).to_dict()
    _atomic_write_json(out_path, report)
    return report


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
    os.replace(tmp, path)
