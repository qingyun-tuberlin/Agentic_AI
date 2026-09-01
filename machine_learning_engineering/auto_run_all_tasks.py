"""Automated batch execution utility for the leaked MLE-bench-lite tasks.

Runs the ADK pipeline over tasks discovered from `datasets.yaml`. Per-task
configuration (name, problem type, metric direction) is read from `datasets.yaml`
and injected into each `adk run` subprocess via environment variables
(`MLE_TASK_NAME`, `MLE_TASK_TYPE`, `MLE_LOWER`), which `shared_libraries/config.py`
consumes. The shared config file is never mutated on disk, so heterogeneous tasks
(regression vs classification) are not mislabeled by a single static config.

Selection flags:
  --task NAME       Run one task (base name or variant name, e.g. titanic / titanic_proxy)
  --clean           Use clean task dirs (no suffix)
  --corrupted       Use corrupted (_proxy) dirs (default)
  --variant NAME    Single-pass pipeline variant: baseline or baseline_xai

Matrix mode (`--matrix`) runs every selected task once per variant (see VARIANTS),
each into its own `./ws_<variant>` workspace tree, so the runs can be diffed
end-to-end with `scripts/measure_xai_overhead.py --ab-workspace ws_<a> ws_<b>`.
"""

import argparse
import os
import subprocess
import sys
import time

import yaml
from dotenv import load_dotenv

# Automatically load environment variables from local .env if present
load_dotenv()

# Path Configuration: Resolves paths relative to this script's position
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_YAML = os.path.join(PROJECT_ROOT, "datasets.yaml")
TASKS_DIR = os.path.join(PROJECT_ROOT, "tasks")

# Framework Execution Root: Navigates one level up to the 'mle-star' workspace root
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

# Suffix identifying the leaked task variant to run. The PROXY-leak variants
# (`<task>_proxy`) are produced by scripts/prepare_leakage_tasks.py.
LEAK_SUFFIX = "_proxy"

# Matrix mode variants: each maps to the env-var overrides applied to the
# `adk run` subprocess. MLE_WORKSPACE_DIR is set automatically to ./ws_<variant>.
VARIANTS = {
    "baseline": {"USE_XAI_CORRECTION": "False", "USE_XAI_REFINEMENT": "False"},
    "baseline_xai": {"USE_XAI_CORRECTION": "True", "USE_XAI_REFINEMENT": "False"},
}


def load_task_configs(suffix: str = LEAK_SUFFIX, only: str | None = None) -> list[dict]:
    """Build a per-task config list from datasets.yaml.

    Args:
        suffix: Variant suffix to append to each base task name. Use LEAK_SUFFIX
            (`_proxy`) for the PROXY-leak variants, or "" for the clean tasks.
        only: If given, keep only the task whose base name OR resolved variant
            name matches this string (lets `--task foo` select `foo` or `foo_proxy`).

    Returns a list of dicts with the resolved `task_name`, its `task_type`, and
    the `lower` flag (True if a lower metric value is better). Variants whose
    directory is missing on disk are skipped with a warning.
    """
    if not os.path.exists(DATASETS_YAML):
        raise FileNotFoundError(f"datasets.yaml missing at resolved path: {DATASETS_YAML}")

    with open(DATASETS_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    tasks: list[dict] = []
    for section in ("datasets", "competitions"):
        for entry in (config.get(section) or {}).values():
            base_name = entry.get("task_name")
            if not base_name:
                continue
            variant_name = f"{base_name}{suffix}"
            if only is not None and only not in (base_name, variant_name):
                continue
            variant_dir = os.path.join(TASKS_DIR, variant_name)
            if not os.path.isdir(variant_dir):
                print(
                    f"[WARN] Task directory missing, skipping: {variant_dir}",
                    file=sys.stderr,
                )
                continue
            tasks.append(
                {
                    "task_name": variant_name,
                    "task_type": entry.get("task_type", "Tabular Regression"),
                    "lower": bool(entry.get("lower", True)),
                }
            )
    return tasks


def run_one_task(task: dict, base_env: dict | None = None) -> bool:
    """Run a single task through one `adk run` subprocess.

    Args:
        task: Per-task config dict (task_name / task_type / lower).
        base_env: Environment to layer the per-task vars on top of. When None,
            this process's environment is used (single-pass behavior). In matrix
            mode this carries the variant flags + dedicated MLE_WORKSPACE_DIR.

    Returns:
        True on success, False if the run failed (caller continues to next run).
    """
    name = task["task_name"]
    # Check if final_state.json already exists in the target workspace directory
    workspace_dir = (base_env or {}).get("MLE_WORKSPACE_DIR", ".")
    final_state_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, workspace_dir, name, "final_state.json"))
    if os.path.exists(final_state_path):
        print(f"[SKIP] Task {name} already has final_state.json at {final_state_path} (idempotent)")
        return True

    # Per-task configuration is injected through the environment so the shared
    # config file is never mutated. config.py reads these at import.
    run_env = dict(base_env if base_env is not None else os.environ)
    run_env["MLE_TASK_NAME"] = name
    run_env["MLE_TASK_TYPE"] = task["task_type"]
    run_env["MLE_LOWER"] = "True" if task["lower"] else "False"

    try:
        # Short buffer ensuring OS file system synchronization.
        time.sleep(1)

        # Continuous inputs to bypass the interactive prompt mode.
        automated_input = "Please start the machine learning engineering pipeline.\nexit\n"

        print("[EXEC] Calling CLI from workspace: adk run machine_learning_engineering")
        subprocess.run(
            ["adk", "run", "machine_learning_engineering"],
            input=automated_input,
            text=True,
            check=True,
            cwd=WORKSPACE_ROOT,  # Forces resolution relative to 'mle-star' root directory
            env=run_env,
        )
        print(f"[SUCCESS] Finished processing task: {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] CLI runtime execution failed for task {name} (Exit code: {e.returncode})", file=sys.stderr)
        print("Proceeding to next run...", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Script exception encountered during task {name}: {str(e)}", file=sys.stderr)
        print("Proceeding to next run...", file=sys.stderr)
        return False


def run_single_pass(tasks: list[dict], variant: str | None = None, suffix: str = LEAK_SUFFIX) -> None:
    """Run each selected task once.

    When *variant* is set, applies that entry from VARIANTS and writes into
    ./ws_<variant>. Otherwise inherits the ambient config/env.
    """
    dataset_label = "corrupted" if suffix == LEAK_SUFFIX else "clean"
    print("==================================================================")
    print(f"Initializing ADK Benchmarking Batch Run for {len(tasks)} {dataset_label} tasks.")
    print(f"Resolved Workspace Root: {WORKSPACE_ROOT}")
    print(f"Resolved Project Root:   {PROJECT_ROOT}")
    print(f"Task dataset suffix:     {suffix!r} ({dataset_label})")
    if variant is not None:
        print(f"Pipeline variant:        {variant}  ->  ./ws_{variant}")
        print(f"  flags: {VARIANTS[variant]}")
    else:
        print("Pipeline variant:        (ambient environment)")
    print("==================================================================")

    base_env = os.environ
    if variant is not None:
        base_env = {
            **os.environ,
            **VARIANTS[variant],
            "MLE_WORKSPACE_DIR": f"./ws_{variant}",
        }

    for idx, task in enumerate(tasks, 1):
        print(f"\n[{idx}/{len(tasks)}] STARTING TASK: {task['task_name']}")
        print(f"    task_type={task['task_type']!r}  lower={task['lower']}")
        print("---------------------------------------------------------")
        run_one_task(task, base_env=base_env)

    print("\n=========================================================")
    print("ADK BATCH PIPELINE RUN COMPLETE.")
    print("=========================================================")


def run_matrix(tasks: list[dict], variant_names: list[str], suffix: str = LEAK_SUFFIX) -> None:
    """Run every task once per variant, each into its own ./ws_<variant> tree.

    The resulting workspaces can be diffed end-to-end with, e.g.:
        scripts/measure_xai_overhead.py --ab-workspace ws_baseline ws_baseline_xai
    """
    dataset_label = "corrupted" if suffix == LEAK_SUFFIX else "clean"
    n_runs = len(variant_names) * len(tasks)
    print("==================================================================")
    print(f"Initializing ADK MATRIX Run: {len(variant_names)} variants x "
          f"{len(tasks)} {dataset_label} tasks = {n_runs} runs.")
    print(f"Variants: {', '.join(variant_names)}")
    print(f"Task dataset suffix:     {suffix!r} ({dataset_label})")
    print(f"Resolved Workspace Root: {WORKSPACE_ROOT}")
    print("==================================================================")

    run_idx = 0
    for variant in variant_names:
        workspace = f"./ws_{variant}"
        flags = VARIANTS[variant]
        # Base env for every run of this variant: inherit the parent env, then
        # force the variant flags and a dedicated workspace so no two variants
        # ever write into the same tree. Per-task vars are layered in run_one_task.
        variant_env = {**os.environ, **flags, "MLE_WORKSPACE_DIR": workspace}

        print(f"\n##################################################################")
        print(f"# VARIANT: {variant}  ->  workspace {workspace}")
        print(f"#   flags: {flags}")
        print(f"##################################################################")

        for task in tasks:
            run_idx += 1
            print(f"\n[{run_idx}/{n_runs}] VARIANT={variant} TASK={task['task_name']}")
            print(f"    task_type={task['task_type']!r}  lower={task['lower']}")
            print("---------------------------------------------------------")
            run_one_task(task, base_env=variant_env)

    print("\n=========================================================")
    print("ADK MATRIX PIPELINE RUN COMPLETE.")
    print("Diff variants with:")
    if len(variant_names) >= 2:
        a, b = variant_names[0], variant_names[1]
        print(f"  python scripts/measure_xai_overhead.py "
              f"--ab-workspace ws_{a} ws_{b}")
    print("=========================================================")


def main() -> None:
    """Parse args and dispatch to single-pass or matrix mode."""
    parser = argparse.ArgumentParser(
        description="Batch-run MLE-bench tasks, optionally as a variant matrix.",
        epilog=(
            "Examples:\n"
            "  %(prog)s --task titanic --corrupted --variant baseline\n"
            "  %(prog)s --task credit-card-fraud --clean --variant baseline_xai\n"
            "  %(prog)s --matrix --task titanic --variants baseline baseline_xai\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        metavar="NAME",
        help="Run only this task. Accepts the base name (e.g. titanic) or the "
             "resolved variant name (e.g. titanic_proxy).",
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--clean",
        action="store_true",
        help="Use clean task directories (no suffix). Default: corrupted.",
    )
    dataset_group.add_argument(
        "--corrupted",
        action="store_true",
        help="Use corrupted (_proxy) task directories. This is the default.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run each task once per variant (see VARIANTS), each into its own "
             "./ws_<variant> workspace, for end-to-end A/B comparison.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        help="Pipeline variant for single-pass mode: baseline (no XAI) or "
             "baseline_xai (XAI correction). Sets USE_XAI_* flags and "
             "./ws_<variant> workspace.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        help=f"Subset of variants to run in --matrix mode. "
             f"Default: all ({', '.join(sorted(VARIANTS))}).",
    )
    args = parser.parse_args()

    if args.variants and not args.matrix:
        parser.error("--variants requires --matrix")
    if args.variant and args.matrix:
        parser.error("--variant is for single-pass mode; use --variants with --matrix")

    suffix = "" if args.clean else LEAK_SUFFIX
    tasks = load_task_configs(suffix=suffix, only=args.task)
    if args.task and not tasks:
        parser.error(
            f"No task matching {args.task!r} found with suffix {suffix!r}. "
            "Check datasets.yaml and the tasks/ directory."
        )

    if args.matrix:
        run_matrix(tasks, args.variants or sorted(VARIANTS), suffix=suffix)
    else:
        run_single_pass(tasks, variant=args.variant, suffix=suffix)


if __name__ == "__main__":
    main()
