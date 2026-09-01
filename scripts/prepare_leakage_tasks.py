#!/usr/bin/env python3
"""
prepare_leakage_tasks.py
Automates copying clean ML tasks under `machine_learning_engineering/tasks/`
to corrupted variants, injecting data leakage, updating task_description.txt
schema previews with actual data values, and tracking the injection details in
`leakage_metadata.json`.

Two leak types are available, each as its own variant directory so that detection
can be attributed to a single leak type (one leak per variant):

  * `<task>_proxy` -- PROXY leakage (default): a plausibly-named feature that is
                      a near-perfect correlate of the target (~0.99 Pearson on
                      the standardized signal). It contains no syntactic reference
                      to the target, so a code-reading static checker is blind to
                      it -- this is the case that motivates the dynamic XAI
                      attribution probe.
  * `<task>_c`     -- DIRECT target leakage: a near-copy of the target (small
                      noise / occasional class flips). The leaked column name is
                      task-specific and matches that dataset's column naming style.

By default only the PROXY variants are emitted; pass `--direct` to also emit the
DIRECT variants.

In both variants the leaked feature is NEUTRALIZED on the test set (set to the
train-mean / mode), simulating "the leak is unavailable in production". This is
what lets the evaluation measure the val-vs-test generalization gap.

Injection is seeded (LEAK_SEED env var, default 0) so corrupted datasets are
reproducible across runs.
"""

import os
import shutil
import json
import argparse
import numpy as np
import pandas as pd
import yaml

MLE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "machine_learning_engineering",
    )
)
TASKS_DIR = os.path.join(MLE_DIR, "tasks")
DATASETS_YAML = os.path.join(MLE_DIR, "datasets.yaml")


def load_target_map():
    """Map task_name -> declared target column from datasets.yaml.

    The yaml has two top-level sections (`datasets`, `competitions`); both hold
    entries with a `task_name` and a `target` field.
    """
    if not os.path.exists(DATASETS_YAML):
        print(f"[Warning] {DATASETS_YAML} not found; falling back to heuristic target detection.")
        return {}

    with open(DATASETS_YAML, "r") as f:
        config = yaml.safe_load(f) or {}

    target_map = {}
    for section in ("datasets", "competitions"):
        for entry in (config.get(section) or {}).values():
            name = entry.get("task_name")
            target = entry.get("target")
            if name and target:
                target_map[name] = target
    return target_map


# Populated in main(); keyed by task_name.
TARGET_MAP = {}

# Target Pearson correlation between the proxy and the (standardized) target
# signal. A signal z mixed with independent noise of std-dev `ratio` correlates
# with z at 1 / sqrt(1 + ratio^2); invert that to get the noise std-dev that
# hits the target correlation. 0.99 yields ~99% single-feature predictive power
# on classification tasks while still not being a literal copy of the target.
PROXY_TARGET_CORR = 0.99
PROXY_NOISE_RATIO = float(np.sqrt(1.0 / PROXY_TARGET_CORR**2 - 1.0))

# Per-task leaked column names, matched to each dataset's native naming style.
# Direct (`<task>_c`): near-copy of the target — name should sound like a strong
# predictor agents would keep, not disposable metadata.
# Proxy (`<task>_proxy`): imperfect correlate — same style, different name.
DIRECT_LEAK_FEATURE_NAMES: dict[str, str | list[str]] = {
    # snake_case economic / housing features
    "california-housing-prices": "block_median_price",
    # PascalCase anomaly measurements
    "credit-card-fraud": "TransactionAnomalyScore",
    # space-separated wine chemistry descriptors
    "wine-quality-red": "expert quality rating",
    # dot-separated census fields
    "adult-income": "household.income.estimate",
    # PascalCase clinical measurements
    "pima-diabetes": "InsulinResponseIndex",
    # snake_case insurance fields
    "medical-insurance": "annual_cost_estimate",
    # PascalCase HR analytics fields
    "employee-attrition": "RetentionRiskScore",
    # anonymous feature numbering (f_00 … f_30)
    "tabular-playground-series-may-2022": "f_32",
    # scientific snake_case with unit suffix
    "nomad2018-predict-transparent-conductors": [
        "dft_formation_energy_estimate_ev",
        "dft_bandgap_estimate_ev",
    ],
    # PascalCase passenger attributes
    "titanic": "PassengerPriorityScore",
}

PROXY_LEAK_FEATURE_NAMES: dict[str, str | list[str]] = {
    "california-housing-prices": "tract_price_index",
    "credit-card-fraud": "MerchantRiskIndex",
    "wine-quality-red": "palate balance index",
    "adult-income": "census.wage.index",
    "pima-diabetes": "GlucoseRiskIndex",
    "medical-insurance": "risk_cost_index",
    "employee-attrition": "EngagementIndex",
    "tabular-playground-series-may-2022": "f_33",
    "nomad2018-predict-transparent-conductors": [
        "computed_formation_energy_ev",
        "computed_bandgap_energy_ev",
    ],
    "titanic": "TicketPriorityScore",
}


def get_direct_leak_feature_name(task_name: str) -> str | list[str]:
    """Return the direct-leak column name(s) for ``task_name``."""
    try:
        return DIRECT_LEAK_FEATURE_NAMES[task_name]
    except KeyError as exc:
        raise KeyError(
            f"No direct-leak feature name configured for task '{task_name}'. "
            f"Add an entry to DIRECT_LEAK_FEATURE_NAMES in prepare_leakage_tasks.py."
        ) from exc


def get_proxy_leak_feature_name(task_name: str) -> str | list[str]:
    """Return the proxy-leak column name(s) for ``task_name``."""
    try:
        return PROXY_LEAK_FEATURE_NAMES[task_name]
    except KeyError as exc:
        raise KeyError(
            f"No proxy-leak feature name configured for task '{task_name}'. "
            f"Add an entry to PROXY_LEAK_FEATURE_NAMES in prepare_leakage_tasks.py."
        ) from exc


def get_target_column(train_df, test_df, task_name):
    """Identifies the target column, preferring the value declared in datasets.yaml."""
    declared = TARGET_MAP.get(task_name)
    if declared:
        if isinstance(declared, list):
            if all(c in train_df.columns for c in declared):
                return declared
        elif declared in train_df.columns:
            return declared
        print(
            f"  [Warning] Declared target '{declared}' for '{task_name}' not in train.csv; "
            "falling back to heuristic detection."
        )

    # Find columns in train but not in test
    diff = list(set(train_df.columns) - set(test_df.columns))
    if len(diff) == 1:
        return diff[0]
    elif len(diff) > 1:
        # If multiple, prefer columns that look like a target (contains price, value, label, target, etc.)
        for col in diff:
            col_lower = col.lower()
            if any(
                keyword in col_lower
                for keyword in ["value", "price", "target", "label", "class", "count", "y"]
            ):
                return col
        return diff[0]

    # Fallback to the last column if no difference found
    return train_df.columns[-1]


def is_classification_target(target_series):
    """Heuristic shared by both injectors: object dtype or few unique values."""
    return target_series.dtype == object or len(target_series.unique()) < 15


def reorder_leak_columns(
    df: pd.DataFrame, target_col: str | list[str], leak_col: str | list[str]
) -> pd.DataFrame:
    """Place the leak(s) among features (mid-schema) and keep the target last when present."""
    cols = list(df.columns)
    leak_cols = leak_col if isinstance(leak_col, list) else [leak_col]
    if not any(lc in cols for lc in leak_cols):
        return df

    target_cols = target_col if isinstance(target_col, list) else [target_col]
    features = [c for c in cols if c not in target_cols and c not in leak_cols]
    mid = len(features) // 2
    ordered = features[:mid] + leak_cols + features[mid:]
    for t_col in target_cols:
        if t_col in cols:
            ordered.append(t_col)
    return df[ordered]


def _preamble_before_dataset(clean_desc: str) -> str:
    """Return the static part of a task description (everything before `# Dataset`)."""
    marker = "# Dataset"
    idx = clean_desc.find(marker)
    if idx == -1:
        return clean_desc.rstrip()
    return clean_desc[:idx].rstrip()


def _csv_preview_block(df: pd.DataFrame, n_rows: int = 3) -> list[str]:
    """Header + first ``n_rows`` data rows for a task_description.csv fenced block."""
    lines = [",".join(map(str, df.columns))]
    for _, row in df.head(n_rows).iterrows():
        lines.append(",".join(str(v) for v in row))
    lines.append("etc.")
    return lines


def build_task_description(
    clean_task_dir: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | list[str],
) -> str:
    """Compute task_description.txt for a corrupted variant from the clean template."""
    clean_desc_path = os.path.join(clean_task_dir, "task_description.txt")
    preamble = ""
    if os.path.exists(clean_desc_path):
        with open(clean_desc_path, encoding="utf-8") as f:
            preamble = _preamble_before_dataset(f.read())

    target_cols = target_col if isinstance(target_col, list) else [target_col]
    feature_cols = [c for c in train_df.columns if c not in target_cols]
    target_col_str = ", ".join(f"`{c}`" for c in target_cols)
    parts = [
        preamble,
        "",
        "# Dataset",
        "",
        "train.csv",
        "```",
        *_csv_preview_block(train_df),
        "```",
        "",
        "test.csv",
        "```",
        *_csv_preview_block(test_df),
        "```",
        "",
        "# Feature columns",
        "",
        f"Feature columns in train.csv (exclude only {target_col_str} as the target; "
        "others may be dropped during feature engineering if superseded by derived features):",
        ", ".join(feature_cols),
        "",
    ]
    return "\n".join(parts)


def write_task_description(
    variant_dir: str,
    clean_task_dir: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> None:
    desc_path = os.path.join(variant_dir, "task_description.txt")
    body = build_task_description(clean_task_dir, train_df, test_df, target_col)
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(body)
    print("  Wrote task_description.txt (computed schema; leak column before target).")


def prepare_variant(task_name, suffix):
    """Copy a clean task dir to a fresh `<task><suffix>` variant directory.

    Returns (variant_dir, train_df, test_df, target_col) or None when the task
    has no train/test split to corrupt.
    """
    clean_dir = os.path.join(TASKS_DIR, task_name)
    variant_dir = os.path.join(TASKS_DIR, f"{task_name}{suffix}")

    print(f"  Clean path:     {clean_dir}")
    print(f"  Variant path:   {variant_dir}")

    # Remove existing variant directory if it exists to ensure a fresh start
    if os.path.exists(variant_dir):
        shutil.rmtree(variant_dir)
    os.makedirs(variant_dir, exist_ok=True)

    # Copy files
    for item in os.listdir(clean_dir):
        s = os.path.join(clean_dir, item)
        d = os.path.join(variant_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    train_path = os.path.join(variant_dir, "train.csv")
    test_path = os.path.join(variant_dir, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"  [Warning] Missing train.csv or test.csv in {task_name}. Skipping.")
        shutil.rmtree(variant_dir)
        return None

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target_col = get_target_column(train_df, test_df, task_name)
    print(f"  Identified target column: '{target_col}'")
    return variant_dir, train_df, test_df, target_col


def finalize_variant(
    variant_dir,
    clean_task_dir,
    train_df,
    test_df,
    injected_feature,
    metadata,
):
    """Persist the corrupted CSVs, refresh the description preview, write metadata."""
    target_col = metadata["target_column"]
    train_df = reorder_leak_columns(train_df, target_col, injected_feature)
    test_df = reorder_leak_columns(test_df, target_col, injected_feature)

    train_df.to_csv(os.path.join(variant_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(variant_dir, "test.csv"), index=False)
    print("  Successfully updated train.csv and test.csv with data leakage column.")

    write_task_description(
        variant_dir, clean_task_dir, train_df, test_df, target_col
    )

    metadata_path = os.path.join(variant_dir, "leakage_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote leakage metadata to: {metadata_path}")


def inject_direct_leakage(task_name):
    """DIRECT target leakage: inject a near-copy of the target column(s)."""
    print(f"Processing task (direct): {task_name}...")
    prepared = prepare_variant(task_name, "_c")
    if prepared is None:
        return
    variant_dir, train_df, test_df, target_col = prepared

    injected_features = get_direct_leak_feature_name(task_name)
    if not isinstance(injected_features, list):
        injected_features = [injected_features]

    target_cols = target_col if isinstance(target_col, list) else [target_col]
    assert len(injected_features) == len(target_cols), f"Mismatch between targets and leak columns for {task_name}"

    leakage_type = "target_leakage"
    correlations = []

    for injected_feature, t_col in zip(injected_features, target_cols):
        target_series = train_df[t_col]
        is_classification = is_classification_target(target_series)

        if is_classification:
            # Categorical Target Leakage: Copy target with 2% random flips
            unique_classes = list(target_series.dropna().unique())
            if not unique_classes:
                unique_classes = [0, 1]

            leak_vals = []
            for val in target_series:
                if pd.isna(val):
                    leak_vals.append(np.random.choice(unique_classes))
                elif np.random.rand() < 0.98:
                    leak_vals.append(val)
                else:
                    leak_vals.append(np.random.choice(unique_classes))

            train_df[injected_feature] = leak_vals
            # Test feature gets the mode class (or random if missing)
            test_df[injected_feature] = target_series.mode()[0] if not target_series.mode().empty else unique_classes[0]

            # Calculate exact matching rate
            matching = (train_df[injected_feature] == target_series).mean()
            correlation_val = float(matching)
            print(f"  [Direct] Injected categorical leak '{injected_feature}' with matching rate: {correlation_val:.4f}")
        else:
            # Numeric/Regression Target Leakage: Target plus small Gaussian noise
            std_val = target_series.std()
            if pd.isna(std_val) or std_val == 0:
                std_val = 1.0

            # Inject noise with std dev being 1% of the target's std dev
            noise = np.random.normal(0, 0.01 * std_val, size=len(train_df))
            train_df[injected_feature] = target_series + noise
            # Test feature gets the mean value
            test_df[injected_feature] = target_series.mean()

            # Calculate Pearson correlation coefficient
            correlation_val = float(train_df[injected_feature].corr(target_series))
            print(f"  [Direct] Injected numeric leak '{injected_feature}' with Pearson correlation: {correlation_val:.4f}")
        
        correlations.append(correlation_val)

    metadata = {
        "original_task": task_name,
        "corrupted_task": f"{task_name}_c",
        "target_column": target_col,
        "injected_feature_name": injected_features if len(injected_features) > 1 else injected_features[0],
        "leakage_type": leakage_type,
        "correlation_metric": correlations if len(correlations) > 1 else correlations[0],
        "is_classification": False,
    }
    finalize_variant(
        variant_dir,
        os.path.join(TASKS_DIR, task_name),
        train_df,
        test_df,
        injected_features,
        metadata,
    )


def inject_proxy_leakage(task_name):
    """PROXY leakage: a strong-but-imperfect, plausibly-named correlate of the target."""
    print(f"Processing task (proxy): {task_name}...")
    prepared = prepare_variant(task_name, "_proxy")
    if prepared is None:
        return
    variant_dir, train_df, test_df, target_col = prepared

    injected_features = get_proxy_leak_feature_name(task_name)
    if not isinstance(injected_features, list):
        injected_features = [injected_features]

    target_cols = target_col if isinstance(target_col, list) else [target_col]
    assert len(injected_features) == len(target_cols), f"Mismatch between targets and leak columns for {task_name}"

    leakage_type = "proxy_leakage"
    n = len(train_df)
    correlations = []

    for injected_feature, t_col in zip(injected_features, target_cols):
        target_series = train_df[t_col]
        is_classification = is_classification_target(target_series)

        if is_classification:
            # Ordinal-encode classes; the encoded index is the signal we standardize.
            classes = sorted(target_series.dropna().unique(), key=lambda x: str(x))
            class_to_idx = {c: i for i, c in enumerate(classes)}
            signal = target_series.map(class_to_idx).fillna(0).astype(float)
        else:
            signal = target_series

        std_val = signal.std()
        if pd.isna(std_val) or std_val == 0:
            std_val = 1.0
        z = (signal - signal.mean()) / std_val
        noise = np.random.normal(0, PROXY_NOISE_RATIO, size=n)
        proxy = 50.0 + 15.0 * (z + noise)
        train_df[injected_feature] = proxy
        correlation_val = float(train_df[injected_feature].corr(signal))
        
        # Neutralize the proxy on the test set: production-simulated "leak unavailable".
        test_df[injected_feature] = float(train_df[injected_feature].mean())
        
        kind = "classification" if is_classification else "regression"
        print(f"  [Proxy] Injected {kind} proxy '{injected_feature}'; Pearson correlation: {correlation_val:.4f}")
        correlations.append(correlation_val)

    metadata = {
        "original_task": task_name,
        "corrupted_task": f"{task_name}_proxy",
        "target_column": target_col,
        "injected_feature_name": injected_features if len(injected_features) > 1 else injected_features[0],
        "leakage_type": leakage_type,
        "correlation_metric": correlations if len(correlations) > 1 else correlations[0],
        "is_classification": False,
        "noise_ratio": PROXY_NOISE_RATIO,
        "target_correlation": PROXY_TARGET_CORR,
    }
    finalize_variant(
        variant_dir,
        os.path.join(TASKS_DIR, task_name),
        train_df,
        test_df,
        injected_features,
        metadata,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Also emit the DIRECT leak variants (`<task>_c`). "
        "By default only the PROXY leak variants (`<task>_proxy`) are created.",
    )
    args = parser.parse_args()

    global TARGET_MAP
    TARGET_MAP = load_target_map()

    seed = int(os.environ.get("LEAK_SEED", "0"))
    np.random.seed(seed)
    print(f"Scanning tasks directory: {TASKS_DIR} (LEAK_SEED={seed})")
    if not os.path.exists(TASKS_DIR):
        print(f"Tasks directory {TASKS_DIR} does not exist.")
        return

    clean_tasks = []
    for item in os.listdir(TASKS_DIR):
        item_path = os.path.join(TASKS_DIR, item)
        # Skip already-generated variants (direct `_c`, proxy `_proxy`).
        if os.path.isdir(item_path) and not item.endswith("_c") and not item.endswith("_proxy"):
            clean_tasks.append(item)

    clean_tasks.sort()
    print(f"Found clean tasks: {clean_tasks}")

    for task in clean_tasks:
        inject_proxy_leakage(task)
        if args.direct:
            inject_direct_leakage(task)

    print("\nData leakage injection complete!")


if __name__ == "__main__":
    main()
