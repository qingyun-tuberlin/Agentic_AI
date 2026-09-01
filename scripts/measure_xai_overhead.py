#!/usr/bin/env python3
"""measure_xai_overhead.py

Aggregate the per-task ``xai_overhead.json`` reports emitted by the pipeline
(see ``shared_libraries/xai_tracker.py``) into a single comparison table.

Each run writes ``workspace/<task>/xai_overhead.json`` with directly-attributed
XAI cost -- extra API calls, tokens, and runtime -- alongside the run totals, so
overhead can be read as an absolute number and as a share of the whole run.

Usage
-----
  # Table across every task in the workspace
  python scripts/measure_xai_overhead.py

  # Point at a different workspace dir
  python scripts/measure_xai_overhead.py --workspace path/to/workspace

  # A/B end-to-end diff for one task: two xai_overhead.json files (XAI off vs on)
  python scripts/measure_xai_overhead.py --ab baseline.json treatment.json

  # A/B across a whole task set: two workspaces (run each variant into its own
  #   MLE_WORKSPACE_DIR so nothing is overwritten), matched by task name
  python scripts/measure_xai_overhead.py \
      --ab-workspace workspace_xai_off workspace_xai_on
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional


def load_report(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  [warn] could not read {path}: {exc}", file=sys.stderr)
        return None


def find_reports(workspace: str) -> List[Dict[str, Any]]:
    reports = []
    for path in sorted(glob.glob(os.path.join(workspace, "*", "xai_overhead.json"))):
        report = load_report(path)
        if report:
            report["_path"] = path
            reports.append(report)
    return reports


def _pct(part: float, whole: float) -> str:
    return f"{(part / whole * 100):.1f}%" if whole else "  -  "


def print_table(reports: List[Dict[str, Any]]) -> None:
    if not reports:
        print("No xai_overhead.json reports found. Run the pipeline first.")
        return

    header = (
        f"{'Task':<34} {'XAI':<4} {'Calls':>11} {'Tokens':>17} "
        f"{'LLM s':>9} {'Compute s':>11} {'Overhead s':>11}"
    )
    print("=" * len(header))
    print("PER-TASK XAI OVERHEAD  (xai / total, and share of run)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    agg = {k: 0 for k in ("xc", "tc", "xt", "tt", "xl", "tl", "xcp", "tcp", "xov")}
    agg = {k: 0.0 for k in agg}

    for r in reports:
        task = r.get("task") or os.path.basename(os.path.dirname(r["_path"]))
        on = "on" if r.get("xai_correction_enabled") or r.get("xai_refinement_enabled") else "off"

        calls = f"{r['xai_api_calls']}/{r['total_api_calls']}"
        tokens = f"{r['xai_tokens']:,}/{r['total_tokens']:,}"
        calls_pct = _pct(r["xai_api_calls"], r["total_api_calls"])
        tok_pct = _pct(r["xai_tokens"], r["total_tokens"])

        print(
            f"{task[:34]:<34} {on:<4} {calls:>11} {tokens:>17} "
            f"{r['xai_llm_latency_s']:>9.1f} {r['xai_compute_s']:>11.1f} "
            f"{r['xai_total_overhead_s']:>11.1f}"
        )
        print(
            f"{'':<34} {'':<4} {calls_pct:>11} {tok_pct:>17} "
            f"{_pct(r['xai_llm_latency_s'], r['total_llm_latency_s']):>9} "
            f"{_pct(r['xai_compute_s'], r['total_compute_s']):>11} "
            f"{_pct(r['xai_total_overhead_s'], r['total_llm_latency_s'] + r['total_compute_s']):>11}"
        )

        agg["xc"] += r["xai_api_calls"]; agg["tc"] += r["total_api_calls"]
        agg["xt"] += r["xai_tokens"]; agg["tt"] += r["total_tokens"]
        agg["xl"] += r["xai_llm_latency_s"]; agg["tl"] += r["total_llm_latency_s"]
        agg["xcp"] += r["xai_compute_s"]; agg["tcp"] += r["total_compute_s"]
        agg["xov"] += r["xai_total_overhead_s"]

    print("-" * len(header))
    print(
        f"{'TOTAL (' + str(len(reports)) + ' tasks)':<34} {'':<4} "
        f"{str(int(agg['xc'])) + '/' + str(int(agg['tc'])):>11} "
        f"{f'{int(agg['xt']):,}/{int(agg['tt']):,}':>17} "
        f"{agg['xl']:>9.1f} {agg['xcp']:>11.1f} {agg['xov']:>11.1f}"
    )
    print("=" * len(header))
    print(
        "  XAI share of run -> "
        f"calls: {_pct(agg['xc'], agg['tc'])}, "
        f"tokens: {_pct(agg['xt'], agg['tt'])}, "
        f"runtime: {_pct(agg['xov'], agg['tl'] + agg['tcp'])}"
    )
    print(
        "  Note: this is *directly-attributed* cost (audit/instrumentation/revision\n"
        "  LLM calls + instrumented code runs). For true end-to-end delta, use --ab."
    )


def print_ab_diff(baseline_path: str, treatment_path: str) -> None:
    base = load_report(baseline_path)
    treat = load_report(treatment_path)
    if not base or not treat:
        print("Could not load both reports for A/B diff.")
        return

    print("=" * 70)
    print("A/B END-TO-END OVERHEAD  (treatment - baseline)")
    print(f"  baseline:  {baseline_path}  (task={base.get('task')})")
    print(f"  treatment: {treatment_path}  (task={treat.get('task')})")
    print("=" * 70)

    print(f"{'Metric':<18} {'Baseline':>14} {'Treatment':>14} {'Delta':>14} {'%':>8}")
    print("-" * 70)
    for label, key in _AB_METRICS:
        b, t = base.get(key, 0), treat.get(key, 0)
        delta = t - b
        pct = f"{(delta / b * 100):+.1f}%" if b else "   -  "
        print(f"{label:<18} {b:>14,.1f} {t:>14,.1f} {delta:>+14,.1f} {pct:>8}")
    print("=" * 70)


_AB_METRICS = [
    ("API calls", "total_api_calls"),
    ("Total tokens", "total_tokens"),
    ("LLM latency s", "total_llm_latency_s"),
    ("Compute s", "total_compute_s"),
]


def print_ab_workspace(off_dir: str, on_dir: str) -> None:
    off = {r["task"] or os.path.basename(os.path.dirname(r["_path"])): r
           for r in find_reports(off_dir)}
    on = {r["task"] or os.path.basename(os.path.dirname(r["_path"])): r
          for r in find_reports(on_dir)}
    shared = sorted(set(off) & set(on))

    print("=" * 78)
    print("A/B END-TO-END OVERHEAD ACROSS WORKSPACES  (on - off, whole-run totals)")
    print(f"  off: {off_dir}")
    print(f"  on:  {on_dir}")
    print("=" * 78)
    only = (set(off) ^ set(on))
    if only:
        print(f"  [warn] tasks present in only one workspace, skipped: {sorted(only)}")
    if not shared:
        print("  No tasks present in both workspaces.")
        return

    sums = {key: [0.0, 0.0] for _, key in _AB_METRICS}
    for task in shared:
        print(f"\n{task}")
        print(f"  {'Metric':<16} {'Off':>14} {'On':>14} {'Delta':>14} {'%':>8}")
        for label, key in _AB_METRICS:
            b, t = off[task].get(key, 0), on[task].get(key, 0)
            sums[key][0] += b
            sums[key][1] += t
            delta = t - b
            pct = f"{(delta / b * 100):+.1f}%" if b else "   -  "
            print(f"  {label:<16} {b:>14,.1f} {t:>14,.1f} {delta:>+14,.1f} {pct:>8}")

    print("\n" + "=" * 78)
    print(f"AGGREGATE OVER {len(shared)} SHARED TASKS")
    print(f"  {'Metric':<16} {'Off':>14} {'On':>14} {'Delta':>14} {'%':>8}")
    for label, key in _AB_METRICS:
        b, t = sums[key]
        delta = t - b
        pct = f"{(delta / b * 100):+.1f}%" if b else "   -  "
        print(f"  {label:<16} {b:>14,.1f} {t:>14,.1f} {delta:>+14,.1f} {pct:>8}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-task XAI overhead reports.")
    parser.add_argument(
        "--workspace",
        default="machine_learning_engineering/workspace",
        help="Workspace dir containing <task>/xai_overhead.json files.",
    )
    parser.add_argument(
        "--ab",
        nargs=2,
        metavar=("BASELINE_JSON", "TREATMENT_JSON"),
        help="Two xai_overhead.json files to diff end-to-end (XAI off vs on).",
    )
    parser.add_argument(
        "--ab-workspace",
        nargs=2,
        metavar=("OFF_WORKSPACE", "ON_WORKSPACE"),
        help="Two workspace dirs to diff end-to-end, matched per task name.",
    )
    args = parser.parse_args()

    if args.ab:
        print_ab_diff(args.ab[0], args.ab[1])
        return 0

    if args.ab_workspace:
        print_ab_workspace(args.ab_workspace[0], args.ab_workspace[1])
        return 0

    reports = find_reports(args.workspace)
    print_table(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
