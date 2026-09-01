#!/usr/bin/env python3
"""Emit the ordered run manifest for the SLURM array.

One line per array element, three whitespace-separated fields:

    <task_name> <dataset> <pipeline>

  dataset  = clean | corrupted        (tasks/<name>  vs  tasks/<name>_proxy)
  pipeline = baseline | baseline_xai  (XAI correction off vs on)

Order is task-major, then dataset, then pipeline, so each task's
`clean baseline` / `clean baseline_xai` A/B pair is adjacent (they run together
under the array's %CONCURRENCY throttle). Only tasks that have BOTH a clean and
a corrupted directory on disk are included, so the array size always matches
what can actually run. The array index equals the 0-based line number.
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MLE = os.path.join(REPO, "machine_learning_engineering")
DATASETS = os.path.join(MLE, "datasets.yaml")
TASKS = os.path.join(MLE, "tasks")

PIPELINES = ("baseline", "baseline_xai")


def main() -> int:
    with open(DATASETS, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    lines = []
    for section in ("datasets", "competitions"):
        for entry in (cfg.get(section) or {}).values():
            name = entry.get("task_name")
            if not name:
                continue
            has_clean = os.path.isdir(os.path.join(TASKS, name))
            has_corr = os.path.isdir(os.path.join(TASKS, f"{name}_proxy"))
            if not (has_clean and has_corr):
                print(
                    f"[gen_run_manifest] skipping {name}: "
                    f"clean={has_clean} corrupted={has_corr}",
                    file=sys.stderr,
                )
                continue
            for dataset in ("clean", "corrupted"):
                for pipeline in PIPELINES:
                    lines.append(f"{name} {dataset} {pipeline}")

    if not lines:
        print("[gen_run_manifest] no runnable tasks found", file=sys.stderr)
        return 1

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
