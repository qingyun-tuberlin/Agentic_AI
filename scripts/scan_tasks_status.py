#!/usr/bin/env python3
"""scan_tasks_status.py

Scans the two evaluation workspaces (ws_baseline and ws_baseline_xai) for the ten tasks
defined in datasets.yaml. Checks for the presence of final_state.json to determine
which runs (baseline/xai, clean/corrupted) are completed or missing, and outputs
reproduction/execution commands.
"""

import os
import sys
import yaml

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_YAML = os.path.join(PROJECT_ROOT, "machine_learning_engineering", "datasets.yaml")
WS_BASELINE = os.path.abspath(os.environ.get("WS_VANILLA", os.path.join(PROJECT_ROOT, "ws_baseline")))
WS_BASELINE_XAI = os.path.abspath(os.environ.get("WS_OURS", os.path.join(PROJECT_ROOT, "ws_baseline_xai")))


def load_task_names() -> list[str]:
    """Load task names from datasets.yaml."""
    if not os.path.exists(DATASETS_YAML):
        print(f"Error: datasets.yaml not found at {DATASETS_YAML}", file=sys.stderr)
        sys.exit(1)

    with open(DATASETS_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    task_names = []
    for section in ("datasets", "competitions"):
        for entry in (config.get(section) or {}).values():
            base_name = entry.get("task_name")
            if base_name:
                task_names.append(base_name)
    return sorted(task_names)


def check_completed(workspace_dir: str, task_folder: str) -> bool:
    """Check if the final_state.json exists in the given workspace task directory."""
    final_state_path = os.path.join(workspace_dir, task_folder, "final_state.json")
    return os.path.exists(final_state_path)


def main():
    tasks = load_task_names()
    
    # Grid of results
    # Each task maps to a dict of {config_name: completed_bool}
    results = {}
    
    # 4 Configurations:
    # 1. baseline_clean: ws_baseline/task_name
    # 2. baseline_corrupted: ws_baseline/task_name_proxy
    # 3. xai_clean: ws_baseline_xai/task_name
    # 4. xai_corrupted: ws_baseline_xai/task_name_proxy
    
    configs = [
        {"id": "bl_clean", "label": "BL-Clean", "ws": WS_BASELINE, "suffix": "", "variant": "baseline", "clean_flag": "--clean"},
        {"id": "bl_corr", "label": "BL-Corrupt", "ws": WS_BASELINE, "suffix": "_proxy", "variant": "baseline", "clean_flag": "--corrupted"},
        {"id": "xai_clean", "label": "XAI-Clean", "ws": WS_BASELINE_XAI, "suffix": "", "variant": "baseline_xai", "clean_flag": "--clean"},
        {"id": "xai_corr", "label": "XAI-Corrupt", "ws": WS_BASELINE_XAI, "suffix": "_proxy", "variant": "baseline_xai", "clean_flag": "--corrupted"},
    ]
    
    for task in tasks:
        results[task] = {}
        for cfg in configs:
            folder_name = f"{task}{cfg['suffix']}"
            results[task][cfg["id"]] = check_completed(cfg["ws"], folder_name)

    # Print a summary table
    header = f"{'Task Name':<45} | {'BL-Clean':^10} | {'BL-Corrupt':^10} | {'XAI-Clean':^10} | {'XAI-Corrupt':^11}"
    print("=" * len(header))
    print("TASK COMPLETION STATUS GRID (✓ = Completed, ✗ = Missing)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    totals = {cfg["id"]: 0 for cfg in configs}
    
    for task in tasks:
        status_strs = []
        for cfg in configs:
            completed = results[task][cfg["id"]]
            status_strs.append("   ✓      " if completed else "   ✗      ")
            if completed:
                totals[cfg["id"]] += 1
        print(f"{task:<45} | {status_strs[0]}| {status_strs[1]}| {status_strs[2]}| {status_strs[3]}")
        
    print("-" * len(header))
    total_str = f"{'TOTAL COMPLETED':<45} | {totals['bl_clean']:^10} | {totals['bl_corr']:^10} | {totals['xai_clean']:^10} | {totals['xai_corr']:^11}"
    print(total_str)
    print("=" * len(header))
    print()

    # Now show missing runs, prioritizing XAI-Corrupt as requested
    print("MISSING RUNS & EXECUTION COMMANDS")
    print("=================================")
    
    # Priority 1: XAI-Corrupt (baseline+xai on corrupted)
    xai_corr_missing = [task for task in tasks if not results[task]["xai_corr"]]
    print(f"\n1. Baseline+XAI on Corrupted ({len(xai_corr_missing)}/{len(tasks)} missing):")
    if not xai_corr_missing:
        print("   ✓ All tasks completed!")
    else:
        for task in xai_corr_missing:
            print(f"   ✗ {task}")
            print(f"     Run command: python machine_learning_engineering/auto_run_all_tasks.py --task {task} --corrupted --variant baseline_xai")
            
    # Priority 2: XAI-Clean (baseline+xai on clean)
    xai_clean_missing = [task for task in tasks if not results[task]["xai_clean"]]
    print(f"\n2. Baseline+XAI on Clean ({len(xai_clean_missing)}/{len(tasks)} missing):")
    if not xai_clean_missing:
        print("   ✓ All tasks completed!")
    else:
        for task in xai_clean_missing:
            print(f"   ✗ {task}")
            print(f"     Run command: python machine_learning_engineering/auto_run_all_tasks.py --task {task} --clean --variant baseline_xai")

    # Priority 3: Baseline on Corrupted ({len(bl_corr_missing)} missing)
    bl_corr_missing = [task for task in tasks if not results[task]["bl_corr"]]
    print(f"\n3. Baseline on Corrupted ({len(bl_corr_missing)}/{len(tasks)} missing):")
    if not bl_corr_missing:
        print("   ✓ All tasks completed!")
    else:
        for task in bl_corr_missing:
            print(f"   ✗ {task}")
            print(f"     Run command: python machine_learning_engineering/auto_run_all_tasks.py --task {task} --corrupted --variant baseline")

    # Priority 4: Baseline on Clean ({len(bl_clean_missing)} missing)
    bl_clean_missing = [task for task in tasks if not results[task]["bl_clean"]]
    print(f"\n4. Baseline on Clean ({len(bl_clean_missing)}/{len(tasks)} missing):")
    if not bl_clean_missing:
        print("   ✓ All tasks completed!")
    else:
        for task in bl_clean_missing:
            print(f"   ✗ {task}")
            print(f"     Run command: python machine_learning_engineering/auto_run_all_tasks.py --task {task} --clean --variant baseline")

    print("\n-----------------------------------------------------------------")
    print("To run the full suite for any missing tasks or configuration, you can use:")
    print("  python machine_learning_engineering/auto_run_all_tasks.py --matrix")
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()
