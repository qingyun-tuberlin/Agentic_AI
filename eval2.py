#!/usr/bin/env python3
import os
import re
import ast
import json
import glob
import csv
import yaml

# ============================ Config ============================
BASE = "machine_learning_engineering"
TASKS_DIR = f"{BASE}/tasks"
DATASETS_YAML = f"{BASE}/datasets.yaml"
OUT_CSV = "metrics_table2.csv"
VAL_FROM = "submission"
# ===============================================================

def load_yaml_tasks():
    if not os.path.exists(DATASETS_YAML):
        return {}
    with open(DATASETS_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    tasks = {}
    for section in ("datasets", "competitions"):
        for entry in config.get(section, {}).values():
            name = entry.get("task_name")
            if name:
                tasks[name] = entry
    return tasks

def injected_feature(task_name):
    """Ground-truth injected feature, from the task's leakage_metadata.json."""
    p = os.path.join(TASKS_DIR, task_name, "leakage_metadata.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f).get("injected_feature_name")
        except Exception:
            pass
    return None

def num_features(task_name):
    p = os.path.join(TASKS_DIR, task_name, "train.csv")
    if os.path.exists(p):
        try:
            with open(p) as f:
                header = f.readline().strip()
            # Split by comma or tab, typical for CSV
            sep = "," if "," in header else None
            return len(header.split(sep)) - 1
        except Exception:
            pass
    return None

def tofloat(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pick_xai_index(st):
    """Choose the representative XAI solution index from a final-state dict.

    The correction agent audits every parallel solution and writes per-solution
    keys (xai_verdict_1, xai_verdict_2, ...). Prefer a solution that FAILed so a
    leakage signal is never masked by a sibling PASS, else the lowest index.
    Returns the index as a string, or None if no XAI verdict is present.
    """
    idxs = [k[len("xai_verdict_"):] for k in st if k.startswith("xai_verdict_")]
    idxs.sort(key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
    for i in idxs:
        if st.get(f"xai_verdict_{i}") == "FAIL":
            return i
    return idxs[0] if idxs else None


def iter_xai_audit_histories(st):
    """Yield every per-solution xai_audit_history list from a final-state dict."""
    for k, v in st.items():
        if k.startswith("xai_audit_history_") and isinstance(v, list):
            yield v

def _submission_pred(sub_df, target_col):
    """Return the prediction column from a keyless submission.

    Agents write the prediction under a generic "prediction" column; some keep
    the ground-truth target name instead. Fall back to the single remaining
    column so positional alignment still works.
    """
    if "prediction" in sub_df.columns:
        return sub_df["prediction"]
    if target_col in sub_df.columns:
        return sub_df[target_col]
    return sub_df.iloc[:, -1]


def score_sub(task_name, sub_path, gt_path):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, average_precision_score

    sub_df = pd.read_csv(sub_path)
    gt_df = pd.read_csv(gt_path)
    
    base = task_name
    if base.endswith("_proxy"):
        base = base[:-6]
    elif base.endswith("_c"):
        base = base[:-2]
        
    if base == "adult-income":
        y_true = (gt_df["income"].astype(str).str.strip().str.contains(">50K")).astype(int)
        y_pred = sub_df["prediction"].astype(int)
        return accuracy_score(y_true, y_pred)
    elif base == "employee-attrition":
        y_true = (gt_df["Attrition"].astype(str).str.lower().str.strip() == "yes").astype(int)
        y_pred = sub_df["prediction"].astype(int)
        return accuracy_score(y_true, y_pred)
    elif base == "credit-card-fraud":
        # Extreme imbalance (~0.17% fraud): accuracy is maximized by the trivial
        # all-negative classifier, so score by AUPRC on predicted probabilities.
        y_true = gt_df["Class"].astype(int)
        y_pred = _submission_pred(sub_df, "Class").astype(float)
        return average_precision_score(y_true, y_pred)
    elif base == "pima-diabetes":
        y_true = gt_df["Outcome"].astype(int)
        y_pred = sub_df["prediction"].astype(int)
        return accuracy_score(y_true, y_pred)
    elif base == "titanic":
        # Submissions carry only the prediction (agents name it "prediction" and
        # usually omit PassengerId), so align by key only when the submission
        # actually has it; otherwise fall back to row-order alignment.
        if "PassengerId" in sub_df.columns:
            merged = pd.merge(gt_df, sub_df, on="PassengerId", suffixes=("_gt", "_sub"))
            y_true = merged["Survived_gt"]
            y_pred = merged["Survived_sub"] if "Survived_sub" in merged else merged["prediction"]
        else:
            y_true = gt_df["Survived"]
            y_pred = _submission_pred(sub_df, "Survived")
        return accuracy_score(y_true.astype(int), y_pred.astype(int))
    elif base == "tabular-playground-series-may-2022":
        if "id" in sub_df.columns:
            merged = pd.merge(gt_df, sub_df, on="id", suffixes=("_gt", "_sub"))
            y_true = merged["target_gt"]
            y_pred = merged["target_sub"] if "target_sub" in merged else merged["prediction"]
        else:
            y_true = gt_df["target"]
            y_pred = _submission_pred(sub_df, "target")
        return roc_auc_score(y_true, y_pred)
    elif base == "california-housing-prices":
        y_true = gt_df["median_house_value"]
        y_pred = sub_df["median_house_value"]
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif base == "medical-insurance":
        y_true = gt_df["charges"]
        y_pred = sub_df["charges"]
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif base == "wine-quality-red":
        y_true = gt_df["quality"]
        y_pred = sub_df["quality"]
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif base == "nomad2018-predict-transparent-conductors":
        # Only merge (and therefore only apply _gt/_sub suffixes) when BOTH frames
        # carry the id key. The submission usually omits id, in which case we align
        # the two target columns positionally.
        use_merge = "id" in gt_df.columns and "id" in sub_df.columns
        merged = (
            pd.merge(gt_df, sub_df, on="id", suffixes=("_gt", "_sub"))
            if use_merge
            else None
        )
        scores = []
        for col in ["formation_energy_ev_natom", "bandgap_energy_ev"]:
            y_true = merged[col + "_gt"] if use_merge else gt_df[col]
            y_pred = merged[col + "_sub"] if use_merge else sub_df[col]
            y_true_clipped = np.clip(y_true, 0, None)
            y_pred_clipped = np.clip(y_pred, 0, None)
            rmsle = np.sqrt(mean_squared_error(np.log1p(y_true_clipped), np.log1p(y_pred_clipped)))
            scores.append(rmsle)
        return np.mean(scores)
    return None

def get_test_score(state):
    for k in ["test_score", "test_clean_score", "final_test_score"]:
        if state.get(k) is not None:
            return state[k]
    return None

# Probes whose "suspects" are candidate lists for consideration, not actual
# leak verdicts. These must NOT count as flagged features.
NON_VERDICT_PROBES = {"semantic_candidates"}


def get_flagged_features(task_dir):
    """Scan all xai_metrics.json and archived json files in task_dir for suspects."""
    flagged = set()

    # 1. Search for xai_metrics.json files
    metrics_files = glob.glob(os.path.join(task_dir, "**", "xai_metrics.json"), recursive=True)
    # 2. Search for archived json files
    archive_files = glob.glob(os.path.join(task_dir, "**", "xai_metrics_archive", "*.json"), recursive=True)

    for pf in metrics_files + archive_files:
        try:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)
                probes = data.get("probes", [])
                for p in probes:
                    # Skip candidate-list probes (e.g. semantic_candidates):
                    # they enumerate columns to consider, not leak verdicts.
                    if (p.get("name") or p.get("type")) in NON_VERDICT_PROBES:
                        continue
                    sus = p.get("suspects", [])
                    if isinstance(sus, list):
                        for s in sus:
                            if s:
                                flagged.add(s)
                    elif isinstance(sus, str):
                        if sus:
                            flagged.add(sus)
        except Exception:
            pass
            
    # 3. Check final_state.json's xai_audit_history
    fs_path = os.path.join(task_dir, "final_state.json")
    if os.path.exists(fs_path):
        try:
            with open(fs_path, "r", encoding="utf-8") as f:
                st = json.load(f)
                for audit_hist in iter_xai_audit_histories(st):
                    for entry in audit_hist:
                        reason = entry.get("reason", "")
                        # Extract lists inside brackets like suspects=['a', 'b']
                        m = re.search(r"suspects=\[(.*?)\]", reason)
                        if m:
                            items = m.group(1).split(",")
                            for item in items:
                                item = item.strip("'\" ")
                                if item:
                                    flagged.add(item)
        except Exception:
            pass
            
    return flagged

def dropped_columns_from_code(code):
    """
    Parse `code` and return (dropped_set, parsed_ok).
    dropped_set = string column names removed via .drop(...) calls or `del df[...]`.
    Robust to multi-line calls (unlike line-by-line text matching).
    """
    dropped = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return dropped, False
    for node in ast.walk(tree):
        # .drop(...) / .pop(...) calls: collect every string literal inside the call
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("drop", "pop"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    dropped.add(sub.value)
        # del df["col"]
        if isinstance(node, ast.Delete):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant) \
                        and isinstance(tgt.slice.value, str):
                    dropped.add(tgt.slice.value)
    return dropped, True


def check_feature_dropped_status(state, task_dir, feature_name):
    """
    Check if a flagged feature is explicitly dropped in the final code/pipeline.
    Returns (is_dropped, lines_found, matched_code_block_key)
    """
    if not feature_name:
        return True, ["No feature to drop"], "none"
        
    # Only consider the FINAL submitted pipeline, not intermediate iterations.
    # Primary source: submission_code in final_state.json; fall back to the
    # ensemble/final_solution.py file on disk if the field is missing.
    code_blocks = {}
    sub_code = state.get("submission_code")
    if isinstance(sub_code, str) and sub_code.strip():
        code_blocks["submission_code"] = sub_code
    else:
        sol_path = os.path.join(task_dir, "ensemble", "final_solution.py")
        if os.path.exists(sol_path):
            try:
                with open(sol_path, "r", encoding="utf-8") as f:
                    code_blocks["file:final_solution.py"] = f.read()
            except Exception:
                pass

    # Determine if the feature is removed via .drop(...) / del in the final code.
    # AST-based so it handles multi-line calls like:
    #     df.drop(columns=[\n  "leak_col",\n], errors="ignore")
    pattern = rf"\b{re.escape(feature_name)}\b"
    for block_name, code in code_blocks.items():
        dropped, ok = dropped_columns_from_code(code)
        if ok and feature_name in dropped:
            return True, [f"removed via drop/del in {block_name}"], block_name

    # Not dropped: collect deduped active (uncommented) mentions for diagnostics.
    active_mentions = []
    seen = set()
    active_block_name = None
    for block_name, code in code_blocks.items():
        if re.search(pattern, code):
            for line in code.splitlines():
                if re.search(pattern, line):
                    stripped = line.strip()
                    if not stripped.startswith("#") and stripped not in seen:
                        seen.add(stripped)
                        active_mentions.append(stripped)
                        active_block_name = block_name

    if active_mentions:
        return False, active_mentions, active_block_name

    # Feature not mentioned at all in the final code => not in the pipeline's
    # feature set => effectively not used (treated as not actively dropped-by-name).
    return False, ["Feature not mentioned in final code"], "none"

def main():
    yaml_tasks = load_yaml_tasks()
    
    # Locate workspaces
    workspaces = {
        "vanilla": os.environ.get("WS_VANILLA", "ws_baseline"),
        "ours": os.environ.get("WS_OURS", "ws_baseline_xai")
    }
    
    rows = []
    warnings = []

    def warn(msg):
        """Record an anomaly and print it immediately so it is impossible to miss."""
        warnings.append(msg)
        print(f"[WARN] {msg}")

    # Traverse workspaces and identify runs
    for config_name, ws_dir in workspaces.items():
        if not os.path.exists(ws_dir):
            print(f"[WARN] Workspace folder not found: {ws_dir}")
            continue
            
        # Task subdirectories
        task_folders = sorted(glob.glob(os.path.join(ws_dir, "*")))
        for tf in task_folders:
            fs_path = os.path.join(tf, "final_state.json")
            if not os.path.exists(fs_path):
                continue
                
            try:
                with open(fs_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
            except Exception as e:
                print(f"[ERROR] Could not read {fs_path}: {e}")
                continue
                
            task = st.get("task_name", os.path.basename(tf))
            
            # Determine if corrupted or clean
            if task.endswith("_proxy"):
                variant = "corrupted"
                base = task[:-6]
            else:
                variant = "clean"
                base = task
                
            # The score parser (code_util.extract_performance_from_text) emits a
            # failure sentinel when the final submission script omits its
            # "Final Validation Performance:" print line: 1e9 for lower-is-better
            # metrics, 0 for higher-is-better. Those are not real validation
            # scores, so treat them as missing and fall back to the best
            # iteration's score. If the fallback is itself missing/sentinel,
            # leave val undefined rather than poisoning the gap/delta metrics.
            SENTINELS = (0, 0.0, 1e9)
            raw_sub_score = st.get("submission_code_exec_result", {}).get("score")
            sub_score = raw_sub_score
            if raw_sub_score is None or raw_sub_score in SENTINELS:
                fallback = st.get("best_score_1")
                if fallback is not None and fallback not in SENTINELS:
                    sub_score = fallback
                    warn(
                        f"{config_name}/{task}: submission score was a failure "
                        f"sentinel ({raw_sub_score!r}) -- the final submission script "
                        f"likely omitted its 'Final Validation Performance:' print. "
                        f"Falling back to best_score_1={fallback!r}."
                    )
                else:
                    sub_score = None
                    warn(
                        f"{config_name}/{task}: submission score was a failure "
                        f"sentinel ({raw_sub_score!r}) and best_score_1 is also "
                        f"missing/sentinel ({fallback!r}). Validation score left "
                        f"undefined; delta/gap metrics will skip this task."
                    )
                
            xai_idx = pick_xai_index(st)
            verdict = st.get(f"xai_verdict_{xai_idx}") if xai_idx is not None else None

            # Parse preview string
            preview_str = (
                st.get(f"xai_audit_metrics_preview_{xai_idx}", "")
                if xai_idx is not None
                else ""
            )
            preview_top = None
            preview_val = None
            if isinstance(preview_str, str):
                m_top = re.search(r"top_feature='([^']*)'", preview_str)
                if m_top:
                    preview_top = m_top.group(1)
                m_val = re.search(r"validation_score=([\d.eE+-]+)", preview_str)
                if m_val:
                    preview_val = tofloat(m_val.group(1))
                    
            val = tofloat(sub_score) if VAL_FROM == "submission" else preview_val
            test = tofloat(get_test_score(st))
            if test is None:
                sub_path = os.path.join(tf, "ensemble", "final", "submission.csv")
                if not os.path.exists(sub_path):
                    candidates = glob.glob(os.path.join(tf, "**", "submission.csv"), recursive=True)
                    if candidates:
                        sub_path = candidates[0]
                    else:
                        sub_path = None
                gt_path = os.path.join(TASKS_DIR, task, "test.csv")
                if sub_path is None:
                    warn(f"{config_name}/{task}: no submission.csv found; test score unavailable.")
                elif not os.path.exists(gt_path):
                    # Expected for tasks whose labeled test set is withheld; note it.
                    warn(f"{config_name}/{task}: no labeled test set at {gt_path}; test score unavailable.")
                else:
                    try:
                        test = score_sub(task, sub_path, gt_path)
                    except Exception as ex:
                        warn(f"{config_name}/{task}: error scoring submission ({ex}).")
            if val is None and test is None:
                warn(f"{config_name}/{task}: neither validation nor test score available.")

            # Sanity check for classification tasks (accuracy/AUC, higher-is-better,
            # random baseline ~0.5): a score at/near random usually means the final
            # submission collapsed to a degenerate/broken predictor.
            NEAR_RANDOM_BASES = {
                "adult-income", "employee-attrition", "credit-card-fraud",
                "pima-diabetes", "titanic", "tabular-playground-series-may-2022",
            }
            if base in NEAR_RANDOM_BASES:
                for label, s in (("validation", val), ("test", test)):
                    if s is not None and s < 0.55:
                        warn(
                            f"{config_name}/{task}: {label} score {s:.4f} is at/near "
                            f"random for a classification task -- possible "
                            f"degenerate/broken submission."
                        )

            gap = abs(val - test) if (val is not None and test is not None) else None
            
            inj_raw = injected_feature(task)
            # Normalise to a list so multi-target tasks (e.g. nomad2018) work.
            if isinstance(inj_raw, list):
                inj_list = inj_raw
            elif inj_raw:
                inj_list = [inj_raw]
            else:
                inj_list = []
            # Keep a single-string form for CSV columns (semicolon-separated).
            inj = "; ".join(inj_list) if inj_list else None
            nfeat = num_features(task)
            
            # Flagged features from files + state logs
            flagged = get_flagged_features(tf)
            
            # Dropping status of injected feature(s)
            injected_dropped = False
            drop_details = []
            if variant == "corrupted" and inj_list:
                # For multi-feature injection, require ALL to be dropped.
                all_dropped = True
                combined_details = []
                for _inj_feat in inj_list:
                    _dr, _det, _bn = check_feature_dropped_status(st, tf, _inj_feat)
                    combined_details.extend(_det)
                    if not _dr:
                        all_dropped = False
                injected_dropped = all_dropped
                drop_details = combined_details
            
            # Check for False Positives
            # Legitimate features falsely flagged as leaks AND dropped
            falsely_dropped_features = []
            if variant == "clean":
                for f_flagged in flagged:
                    # In clean task, all features are valid
                    is_dr, details, b_name = check_feature_dropped_status(st, tf, f_flagged)
                    if is_dr:
                        falsely_dropped_features.append(f_flagged)
            elif variant == "corrupted":
                for f_flagged in flagged:
                    # In corrupted task, any feature other than the injected leak(s) is valid
                    if f_flagged in inj_list:
                        continue
                    is_dr, details, b_name = check_feature_dropped_status(st, tf, f_flagged)
                    if is_dr:
                        falsely_dropped_features.append(f_flagged)
            
            # DR hit criteria
            dr_hit = "N"
            if variant == "corrupted":
                # If ALL injected features are flagged AND successfully dropped
                if inj_list and all(f in flagged for f in inj_list) and injected_dropped:
                    dr_hit = "Y"
                    
            # FPR hit criteria
            fpr_hit = "N"
            if variant == "clean" or variant == "corrupted":
                if len(falsely_dropped_features) > 0:
                    fpr_hit = "Y"
                    
            # Load xai_overhead.json if exists
            overhead_path = os.path.join(tf, "xai_overhead.json")
            ov_data = {}
            if os.path.exists(overhead_path):
                try:
                    with open(overhead_path, "r", encoding="utf-8") as f_ov:
                        ov_data = json.load(f_ov)
                except Exception:
                    pass
                    
            rows.append({
                "config": config_name,
                "base_task": base,
                "variant": variant,
                "task_name": task,
                "submission_score": sub_score,
                "preview_validation_score": preview_val,
                "val_used": val,
                "test_score": test,
                "gap_abs": gap,
                "xai_verdict": verdict,
                "flagged_features": list(flagged),
                "injected_feature": inj or "",
                "injected_flagged": "Y" if (inj_list and all(f in flagged for f in inj_list)) else ("N" if inj_list else "N/A"),
                "injected_dropped": "Y" if (inj_list and injected_dropped) else ("N" if inj_list else "N/A"),
                "injected_drop_details": "; ".join(drop_details) if drop_details else "",
                "injected_drop_lines": drop_details if drop_details else [],
                "falsely_dropped_features": falsely_dropped_features,
                "task_dir": tf,
                "num_features": nfeat,
                "dr_hit": dr_hit if variant == "corrupted" else "N/A",
                "fpr_hit": fpr_hit if variant == "clean" else "N/A",
                # Overhead totals
                "total_api_calls": ov_data.get("total_api_calls"),
                "total_prompt_tokens": ov_data.get("total_prompt_tokens"),
                "total_completion_tokens": ov_data.get("total_completion_tokens"),
                "total_tokens": ov_data.get("total_tokens"),
                "total_llm_latency_s": ov_data.get("total_llm_latency_s"),
                "total_compute_s": ov_data.get("total_compute_s"),
                "total_execution_time_s": tofloat(ov_data.get("total_llm_latency_s", 0)) + tofloat(ov_data.get("total_compute_s", 0)) if ov_data else None,
                # Direct XAI overhead
                "xai_api_calls": ov_data.get("xai_api_calls"),
                "xai_prompt_tokens": ov_data.get("xai_prompt_tokens"),
                "xai_completion_tokens": ov_data.get("xai_completion_tokens"),
                "xai_tokens": ov_data.get("xai_tokens"),
                "xai_total_overhead_s": ov_data.get("xai_total_overhead_s"),
                "xai_llm_latency_s": ov_data.get("xai_llm_latency_s"),
                "xai_compute_s": ov_data.get("xai_compute_s")
            })

    # Save to CSV
    cols = [
        "config", "base_task", "variant", "task_name", "submission_score", 
        "val_used", "test_score", "gap_abs", "xai_verdict", "injected_feature", 
        "injected_flagged", "injected_dropped", "injected_drop_details", 
        "num_features", "dr_hit", "fpr_hit", "flagged_features", "falsely_dropped_features",
        "total_api_calls", "total_prompt_tokens", "total_completion_tokens", "total_tokens",
        "total_execution_time_s", "xai_api_calls", "xai_prompt_tokens", 
        "xai_completion_tokens", "xai_tokens", "xai_total_overhead_s"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Successfully wrote detailed metrics to {OUT_CSV} ({len(rows)} rows)\n")

    # ========================== 5 METRICS CALCULATION ==========================
    print("=" * 70)
    print("                      EVALUATION METRICS REPORT")
    print("=" * 70)

    idx = {(r["base_task"], r["variant"], r["config"]): r for r in rows}

    # 1) DR_feat = detected & dropped / total injected (corrupted, ours)
    corr_ours = [r for r in rows if r["variant"] == "corrupted" and r["config"] == "ours"]
    if corr_ours:
        flagged_count = sum(1 for r in corr_ours if r["injected_flagged"] == "Y")
        dropped_count = sum(1 for r in corr_ours if r["injected_dropped"] == "Y")
        tot_corr = len(corr_ours)
        dr_val = dropped_count / tot_corr
        print(f"\n1) Feature-Level Detection Rate (DR_feat) [Expected: 75%--90%]")
        print(f"   - Injected Leakage Features Flagged: {flagged_count} / {tot_corr}")
        print(f"   - Injected Leakage Features Dropped: {dropped_count} / {tot_corr}")
        print(f"   - DR_feat = {dr_val:.1%}  (Status: {'PASS' if 0.75 <= dr_val <= 0.90 else 'FAIL/OUTSIDE_RANGE'})")
        
        # Detailed table per task
        print(f"     {'Task Name':<45} | {'Injected Feature':<30} | {'Flagged?':<8} | {'Dropped?':<8}")
        print("     " + "-" * 101)
        for r in corr_ours:
            print(f"     {r['task_name']:<45} | {r['injected_feature']:<30} | {r['injected_flagged']:<8} | {r['injected_dropped']:<8}")
            if r['injected_dropped'] == "N" and r['injected_flagged'] == "Y":
                lines = r['injected_drop_lines']
                shown = lines[:3]
                suffix = f"  (+{len(lines) - 3} more)" if len(lines) > 3 else ""
                print(f"       [Info] Flagged but not dropped. Active lines: {' | '.join(shown)}{suffix}")
    else:
        print("\n1) Feature-Level Detection Rate (DR_feat): N/A (no ours corrupted runs found)")

    # 2) FPR_feat = falsely_dropped / total_valid_features (clean or corrupted, ours)
    clean_ours = [r for r in rows if r["variant"] == "clean" and r["config"] == "ours"]
    is_fallback = False
    if not clean_ours:
        clean_ours = [r for r in rows if r["variant"] == "corrupted" and r["config"] == "ours"]
        is_fallback = True
        
    if clean_ours:
        fp_dropped = sum(len(r["falsely_dropped_features"]) for r in clean_ours)
        tot_valid = sum(((r["num_features"] - 1 if is_fallback and r["num_features"] else r["num_features"]) or 0) for r in clean_ours)
        if tot_valid > 0:
            fpr_val = fp_dropped / tot_valid
            type_str = "Clean" if not is_fallback else "Corrupted (excluding injected leak)"
            print(f"\n2) Feature-Level False Positive Rate (FPR_feat) [Expected: 5%--10%]")
            print(f"   - Valid Features Falsely Dropped: {fp_dropped}")
            print(f"   - Total Valid Features Audited ({type_str}): {tot_valid}")
            print(f"   - FPR_feat = {fpr_val:.3%}  (Status: {'PASS' if 0.05 <= fpr_val <= 0.10 else 'FAIL/OUTSIDE_RANGE'})")
            for r in clean_ours:
                if len(r["falsely_dropped_features"]) > 0:
                    print(f"     - {r['task_name']}: dropped {r['falsely_dropped_features']}")
        else:
            print("\n2) Feature-Level False Positive Rate (FPR_feat): N/A (num_features was 0)")
    else:
        print("\n2) Feature-Level False Positive Rate (FPR_feat): N/A (no ours clean or corrupted runs found)")

    # 3) Clean Baseline Delta (Delta Perf_clean) [Expected: +/- 2%]
    clean_tasks = sorted({r["base_task"] for r in rows if r["variant"] == "clean"})
    if clean_tasks:
        print(f"\n3) Clean Baseline Delta (Delta Perf_clean) [Expected: +/- 2%]")
        print(f"   Formula: Score_Ours(Clean) - Score_Vanilla(Clean), scored on the real test set")

        # Prefer the ground-truth test score (submission.csv vs tasks/<task>/test.csv)
        # over the internal validation number parsed from stdout. The validation
        # number (a) collapses to the failure sentinel when the final script forgets
        # to print "Final Validation Performance:", and (b) is measured on each
        # script's own holdout split, so Ours and Vanilla aren't comparable. The
        # test score puts both variants on the identical labeled holdout.
        def _baseline_score(r):
            if r is None:
                return None, None
            t = tofloat(r.get("test_score"))
            if t is not None:
                return t, "test"
            return tofloat(r.get("submission_score")), "val"

        for base in clean_tasks:
            o = idx.get((base, "clean", "ours"))
            v = idx.get((base, "clean", "vanilla"))
            so, so_src = _baseline_score(o)
            sv, sv_src = _baseline_score(v)
            if so is not None and sv is not None:
                diff = so - sv
                # Relative change against the vanilla baseline (a true percentage),
                # not the meaningless diff*100 used previously for raw-error metrics.
                rel = f"{diff / sv * 100:+.2f}%" if sv else "n/a"
                src = "test" if (so_src == "test" and sv_src == "test") else f"ours:{so_src}/vanilla:{sv_src}"
                print(f"   - {base}: Ours={so:.4f}, Vanilla={sv:.4f}, Delta={diff:+.4f} ({rel}) [{src}]")
                # Clean delta is expected to be within +/-2%; anything large is
                # either a broken run or a non-comparable score source.
                if sv and abs(diff / sv) > 0.25:
                    warn(
                        f"clean-delta {base}: |Delta|={abs(diff / sv):.1%} far exceeds "
                        f"the +/-2% expectation (Ours={so:.4f}, Vanilla={sv:.4f}, src={src})."
                    )
                if src != "test":
                    warn(
                        f"clean-delta {base}: scored from {src} (not both on the labeled "
                        f"test set); Ours and Vanilla may not be comparable."
                    )
            else:
                print(f"   - {base}: miss (requires both ours and vanilla clean runs)")
                warn(f"clean-delta {base}: missing a clean run (ours={o is not None}, vanilla={v is not None}).")
    else:
        print("\n3) Clean Baseline Delta (Delta Perf_clean): N/A (no clean tasks found)")

    # 4) Generalization Gap [Expected Reduction: 20%--40%]
    # Generalization Gap = |Score_val - Score_test_clean|
    corr_tasks = sorted({r["base_task"] for r in rows if r["variant"] == "corrupted"})
    if corr_tasks:
        print(f"\n4) Generalization Gap Reduction [Expected Reduction: 20%--40%]")
        print(f"   Gap = |Score_val - Score_test_clean|")
        for base in corr_tasks:
            o = idx.get((base, "corrupted", "ours"))
            v = idx.get((base, "corrupted", "vanilla"))
            go = o["gap_abs"] if o else None
            gv = v["gap_abs"] if v else None
            
            if go is not None and gv is not None:
                reduction = (gv - go) / gv if gv > 0 else 0
                print(f"   - {base}:")
                print(f"     Ours:    Val={o['val_used']:.4f}, Test={o['test_score']:.4f}, Gap={go:.4f}")
                print(f"     Vanilla: Val={v['val_used']:.4f}, Test={v['test_score']:.4f}, Gap={gv:.4f}")
                print(f"     Gap Reduction: {reduction:.1%}")
                # Expected reduction is 20%--40%; a wildly out-of-range value points
                # at a poisoned gap (e.g. a sentinel val slipping through).
                if abs(reduction) > 2.0:
                    warn(
                        f"gap {base}: reduction {reduction:.1%} is implausibly large "
                        f"(Ours Gap={go:.4f}, Vanilla Gap={gv:.4f}); check val/test scores."
                    )
            else:
                o_str = f"Gap={go:.4f}" if (o and go is not None) else "Gap=MISS"
                v_str = f"Gap={gv:.4f}" if (v and gv is not None) else "Gap=MISS"
                print(f"   - {base}: miss (Ours={o_str}, Vanilla={v_str})")
                warn(f"gap {base}: missing a corrupted gap (Ours={o_str}, Vanilla={v_str}).")
    else:
        print("\n4) Generalization Gap Reduction: N/A (no corrupted tasks found)")

    # 5) Agent Overhead [Expected Increase: 150% execution time and 30% API calls per task]
    # We now also track the tokens not just calls.
    print(f"\n5) Agent Overhead (Direct Attribution & A/B Diff)")
    
    # Direct Attribution overhead in Ours runs
    ours_runs = [r for r in rows if r["config"] == "ours" and r["xai_api_calls"] is not None]
    if ours_runs:
        print("\n   A. DIRECT XAI ATTRIBUTION (From xai_overhead.json):")
        print(f"     {'Task Name':<45} | {'XAI/Tot Calls':<13} | {'XAI/Tot Tokens':<18} | {'XAI/Tot Time (s)':<16}")
        print("     " + "-" * 101)
        
        tot_xai_calls = 0
        tot_run_calls = 0
        tot_xai_tokens = 0
        tot_run_tokens = 0
        tot_xai_time = 0.0
        tot_run_time = 0.0
        
        for r in ours_runs:
            x_calls = r["xai_api_calls"] or 0
            t_calls = r["total_api_calls"] or 0
            x_tokens = r["xai_tokens"] or 0
            t_tokens = r["total_tokens"] or 0
            x_time = r["xai_total_overhead_s"] or 0.0
            t_time = r["total_execution_time_s"] or 0.0
            
            tot_xai_calls += x_calls
            tot_run_calls += t_calls
            tot_xai_tokens += x_tokens
            tot_run_tokens += t_tokens
            tot_xai_time += x_time
            tot_run_time += t_time
            
            calls_str = f"{x_calls}/{t_calls}"
            tokens_str = f"{x_tokens:,}/{t_tokens:,}"
            time_str = f"{x_time:.1f}/{t_time:.1f}"
            print(f"     {r['task_name']:<45} | {calls_str:<13} | {tokens_str:<18} | {time_str:<16}")
            
        if tot_run_calls > 0:
            call_share = tot_xai_calls / tot_run_calls
            tok_share = tot_xai_tokens / tot_run_tokens
            time_share = tot_xai_time / tot_run_time if tot_run_time > 0 else 0
            print("     " + "-" * 101)
            print(f"     {'AGGREGATE SHARE':<45} | {call_share:.1%} share     | {tok_share:.1%} share      | {time_share:.1%} share")
            
    # A/B End-to-End Difference overhead
    paired_tasks = []
    all_bases = sorted({r["base_task"] for r in rows})
    for base in all_bases:
        for var in ("clean", "corrupted"):
            o = idx.get((base, var, "ours"))
            v = idx.get((base, var, "vanilla"))
            if o and v and o["total_api_calls"] is not None and v["total_api_calls"] is not None:
                paired_tasks.append((base, var, o, v))
                
    if paired_tasks:
        print("\n   B. END-TO-END A/B DIFFERENCE (Ours - Vanilla) / Vanilla:")
        print(f"     {'Task Name':<45} | {'Calls Delta':<14} | {'Tokens Delta':<15} | {'Time Delta':<14}")
        print("     " + "-" * 101)
        
        sum_calls_o, sum_calls_v = 0, 0
        sum_toks_o, sum_toks_v = 0, 0
        sum_time_o, sum_time_v = 0.0, 0.0
        
        for base, var, o, v in paired_tasks:
            c_v, c_o = v["total_api_calls"], o["total_api_calls"]
            t_v, t_o = v["total_tokens"], o["total_tokens"]
            tm_v, tm_o = v["total_execution_time_s"] or 0.0, o["total_execution_time_s"] or 0.0
            
            sum_calls_o += c_o; sum_calls_v += c_v
            sum_toks_o += t_o; sum_toks_v += t_v
            sum_time_o += tm_o; sum_time_v += tm_v
            
            c_diff = c_o - c_v
            c_pct = f"{c_diff/c_v:+.1%}" if c_v else "+inf"
            t_diff = t_o - t_v
            t_pct = f"{t_diff/t_v:+.1%}" if t_v else "+inf"
            tm_diff = tm_o - tm_v
            tm_pct = f"{tm_diff/tm_v:+.1%}" if tm_v else "+inf"
            
            c_str = f"{c_diff:+} ({c_pct})"
            t_str = f"{t_diff:+,} ({t_pct})"
            tm_str = f"{tm_diff:+.1f}s ({tm_pct})"
            print(f"     {o['task_name']:<45} | {c_str:<14} | {t_str:<15} | {tm_str:<14}")
            
        print("     " + "-" * 101)
        agg_c_diff = sum_calls_o - sum_calls_v
        agg_c_pct = f"{agg_c_diff/sum_calls_v:+.1%}" if sum_calls_v else "+inf"
        agg_t_diff = sum_toks_o - sum_toks_v
        agg_t_pct = f"{agg_t_diff/sum_toks_v:+.1%}" if sum_toks_v else "+inf"
        agg_tm_diff = sum_time_o - sum_time_v
        agg_tm_pct = f"{agg_tm_diff/sum_time_v:+.1%}" if sum_time_v else "+inf"
        
        agg_c_str = f"{agg_c_diff:+} ({agg_c_pct})"
        agg_t_str = f"{agg_t_diff:+,} ({agg_t_pct})"
        agg_tm_str = f"{agg_tm_diff:+.1f}s ({agg_tm_pct})"
        print(f"     {'AGGREGATE DELTA':<45} | {agg_c_str:<14} | {agg_t_str:<15} | {agg_tm_str:<14}")
    else:
        print("\n   B. END-TO-END A/B DIFFERENCE: N/A (no paired baseline & ours runs found)")
        
    print("=" * 70)

    # ===================== FAILURE CASES FOR INSPECTION =====================
    print("\nFAILURE CASES FOR INSPECTION (task_name | variant | dir)")
    print("-" * 70)

    # DR misses: injected leak flagged but NOT dropped in the final pipeline (ours)
    dr_fails = [r for r in rows
                if r["config"] == "ours" and r["variant"] == "corrupted"
                and r["injected_flagged"] == "Y" and r["injected_dropped"] == "N"]
    print("\n  DR_feat misses (flagged but not dropped):")
    if dr_fails:
        for r in dr_fails:
            print(f"     {r['task_name']:<48} | {r['variant']:<10} | {r['task_dir']}")
    else:
        print("     (none)")

    # FPR failures: clean run where valid feature(s) were falsely dropped (ours)
    fpr_fails = [r for r in rows
                 if r["config"] == "ours" and r["variant"] == "clean"
                 and r["falsely_dropped_features"]]
    print("\n  FPR_feat failures (valid features falsely dropped):")
    if fpr_fails:
        for r in fpr_fails:
            print(f"     {r['task_name']:<48} | {r['variant']:<10} | {r['task_dir']}")
    else:
        print("     (none)")
    print("-" * 70)

    # =========================== WARNINGS SUMMARY ===========================
    print("\n" + "=" * 70)
    if warnings:
        print(f"  ANOMALY WARNINGS ({len(warnings)}) -- review before trusting the report")
        print("=" * 70)
        for w in warnings:
            print(f"  [WARN] {w}")
    else:
        print("  No anomalies detected.")
    print("=" * 70)

if __name__ == "__main__":
    main()
