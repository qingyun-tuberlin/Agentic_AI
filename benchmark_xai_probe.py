#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import argparse
import numpy as np

# Literal copy of ws_baseline_xai/adult-income_proxy/1/train0.py
TRAIN_CODE = """import numpy as np
import pandas as pd
import skrub
import xai_probes
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

# Load data through Skrub DataOps
df = skrub.var("data", pd.read_csv("./input/train.csv"))

# Target and features
y = df["income"].skb.apply_func(lambda s: s.astype(str).str.contains(">50K").astype(int))
y = y.skb.mark_as_y()
X = df.drop(columns=["income"], errors="ignore")
X = X.skb.mark_as_X()

# Revised feature engineering: remove suspected leakage proxy feature and avoid
# creating overly direct wage-related proxies.
def add_features(X_op):
    X_fe = X_op.assign(
        age_x_hours=(X_op["age"] * X_op["hours.per.week"]),
        gain_loss_sum=(X_op["capital.gain"] + X_op["capital.loss"]),
        gain_loss_diff=(X_op["capital.gain"] - X_op["capital.loss"]),
        education_per_age=(X_op["education.num"] / (X_op["age"] + 1)),
    )
    # Drop the suspected leakage proxy feature identified by XAI audit.
    X_fe = X_fe.drop(columns=["census.wage.index"], errors="ignore")
    return X_fe

X_features = add_features(X).skb.set_name("xai_features")

# Custom CatBoost classifier to keep everything inside the graph
class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        random_seed=42,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed

    def fit(self, X, y):
        self.cat_cols_ = X.select_dtypes(include=["object"]).columns.tolist()
        X_fit = X.copy()
        for c in self.cat_cols_:
            X_fit[c] = X_fit[c].astype(str).fillna("NA")
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="Logloss",
            eval_metric="Accuracy",
            verbose=False,
            random_seed=self.random_seed,
        )
        self.model_.fit(X_fit, y, cat_features=self.cat_cols_)
        return self

    def predict(self, X):
        X_pred = X.copy()
        for c in getattr(self, "cat_cols_", []):
            if c in X_pred.columns:
                X_pred[c] = X_pred[c].astype(str).fillna("NA")
        preds = self.model_.predict(X_pred)
        return np.asarray(preds).astype(int).ravel()

model = CatBoostWrapper()

pred = X_features.skb.apply(model, y=y).skb.set_name("xai_model")
learner = pred.skb.make_learner()

# Validation split on final graph node
data = pred.skb.train_test_split(test_size=0.2, random_state=0)
learner.fit(data["train"])
y_pred = learner.predict(data["test"])
y_true = data["y_test"]

final_validation_score = accuracy_score(y_true, y_pred)
print(f"Final Validation Performance: {final_validation_score}")

task_meta = xai_probes.TaskMeta(
    lower_is_better=False,
    metric_fn=accuracy_score,
    task_type="Tabular Classification",
    target_column="income",
)
xai_probes.run_leakage_suite(
    learner,
    data,
    task_meta,
    train_df=pd.read_csv("./input/train.csv"),
    learner_factory=lambda: pred.skb.make_learner(),
)
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark XAI probe overhead with a self-contained training script.")
    parser.add_argument("--runs", type=int, default=15, help="Number of repetitions for each variant (default: 15)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of CatBoost iterations for faster model training (default: 100)")
    return parser.parse_args()

def generate_bench_variants(iterations, with_path, without_path):
    # Adjust iterations for speed
    code = TRAIN_CODE.replace("iterations=2000,", f"iterations={iterations},")

    # 1. With probe version
    with open(with_path, "w", encoding="utf-8") as f:
        f.write(code)

    # 2. Without probe version
    idx = code.find("xai_probes.run_leakage_suite")
    if idx == -1:
        print("[ERROR] Could not locate 'xai_probes.run_leakage_suite' in embedded code.")
        return False

    header = code[:idx]
    probe_call = code[idx:]
    commented_call = "\n".join(f"# {line}" for line in probe_call.splitlines())
    
    without_code = header + "\n" + commented_call
    with open(without_path, "w", encoding="utf-8") as f:
        f.write(without_code)

    return True

def run_script(script_name, cwd):
    start = time.perf_counter()
    res = subprocess.run(
        ["python", script_name],
        cwd=cwd,
        capture_output=True,
        text=True
    )
    end = time.perf_counter()
    if res.returncode != 0:
        print(f"\n[WARN] Script {script_name} failed with exit code {res.returncode}")
        print(f"Stderr: {res.stderr[:500]}")
        return None
    return end - start

def main():
    args = parse_args()
    
    # Run directory needs to be where dataset and support files are located
    run_dir = os.path.join("ws_baseline_xai", "adult-income_proxy", "1")
    if not os.path.exists(run_dir):
        print(f"[ERROR] Run directory {run_dir} does not exist. Run the pipeline first.")
        sys.exit(1)
        
    with_script_name = "bench_with_probe.py"
    without_script_name = "bench_without_probe.py"
    
    with_script_path = os.path.join(run_dir, with_script_name)
    without_script_path = os.path.join(run_dir, without_script_name)
    
    print(f"=== Self-Contained XAI Probe Benchmark ===")
    print(f"Model training iterations: {args.iterations}")
    print(f"Repetitions:               {args.runs} runs per variant")
    print("Generating benchmark files...")
    
    if not generate_bench_variants(args.iterations, with_script_path, without_script_path):
        sys.exit(1)
        
    with_times = []
    without_times = []
    
    try:
        for i in range(args.runs):
            print(f"\nIteration {i+1}/{args.runs}:")
            
            # Interleave to prevent hardware temperature/throttling bias
            # 1. Run without probe
            print("  Running without XAI probe...", end="", flush=True)
            t_without = run_script(without_script_name, run_dir)
            if t_without is not None:
                without_times.append(t_without)
                print(f" {t_without:.3f}s")
            else:
                print(" Failed!")
                
            # 2. Run with probe
            print("  Running with XAI probe...", end="", flush=True)
            t_with = run_script(with_script_name, run_dir)
            if t_with is not None:
                with_times.append(t_with)
                print(f" {t_with:.3f}s")
            else:
                print(" Failed!")
                
    finally:
        # Clean up temp files
        for p in (with_script_path, without_script_path):
            if os.path.exists(p):
                os.remove(p)
                
    if len(with_times) == 0 or len(without_times) == 0:
        print("[ERROR] No successful benchmark runs to report.")
        sys.exit(1)
        
    # Stats calculation
    mean_with = np.mean(with_times)
    std_with = np.std(with_times)
    mean_without = np.mean(without_times)
    std_without = np.std(without_times)
    
    diff = mean_with - mean_without
    pct_increase = (diff / mean_without) * 100 if mean_without > 0 else 0.0
    
    print("\n" + "="*60)
    print("                      BENCHMARK RESULTS")
    print("="*60)
    print(f"Without XAI Probe: Mean = {mean_without:6.3f}s (StdDev = {std_without:.3f}s)")
    print(f"With XAI Probe:    Mean = {mean_with:6.3f}s (StdDev = {std_with:.3f}s)")
    print("-"*60)
    print(f"Absolute Probe Overhead:   {diff:+.3f}s")
    print(f"Relative Probe Increase:   {pct_increase:+.2f}%")
    print("="*60)
    print("\nRaw Times (Without Probe): " + ", ".join(f"{t:.2f}s" for t in without_times))
    print("Raw Times (With Probe):    " + ", ".join(f"{t:.2f}s" for t in with_times))

if __name__ == "__main__":
    main()
