#!/usr/bin/env python3
"""Generate report figures from the eval2.py metrics table.

Reproducible: reads metrics_table2.csv (produced by eval2.py) and writes four
PNGs to figs/. No randomness, no network, no hidden state -- rerunning on the
same CSV yields byte-identical inputs to matplotlib.

    python make_figures.py                      # uses ./metrics_table2.csv -> ./figs
    python make_figures.py --csv other.csv --outdir /tmp/figs --dpi 300 --pdf

Figures:
  1. gap_reduction   -- relative generalization gap |val-test|/|test| on the
                        corrupted (leak-injected) tasks, vanilla vs ours.
  2. detection_matrix-- per corrupted task: was the injected leak flagged and
                        dropped? vanilla vs ours.
  3. clean_delta     -- do-no-harm: oriented performance delta of ours vs
                        vanilla on the clean tasks (positive = ours better).
  4. overhead        -- XAI direct share of the run, and end-to-end A/B cost
                        increase (calls / tokens / wall-time).
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Task metadata. Metric direction is fixed by eval2.py's score_sub():
#   higher-is-better -> accuracy / AUC ; else RMSE / RMSLE (lower-is-better).
# ----------------------------------------------------------------------------
HIGHER_IS_BETTER = {
    "adult-income": True,
    "employee-attrition": True,
    "credit-card-fraud": True,
    "pima-diabetes": True,
    "titanic": True,
    "tabular-playground-series-may-2022": True,
    "california-housing-prices": False,
    "medical-insurance": False,
    "wine-quality-red": False,
    "nomad2018-predict-transparent-conductors": False,
}

# Colour-blind-safe, consistent across all figures.
C_VANILLA = "#B0762A"   # ochre
C_OURS = "#2A72B0"      # blue
C_GOOD = "#3C8C5A"      # green  (leak caught / ours better)
C_BAD = "#C24A3F"       # red    (leak missed / ours worse)
C_NEUTRAL = "#9199A3"   # grey
C_GRID = "#D8DCE0"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.7,
    "svg.fonttype": "none",
})


def short(name: str) -> str:
    """Compact task label for axes."""
    return {
        "tabular-playground-series-may-2022": "tabular-playground-2022",
        "nomad2018-predict-transparent-conductors": "nomad2018",
        "california-housing-prices": "california-housing",
    }.get(name, name)


def load(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        sys.exit(f"[ERROR] metrics CSV not found: {csv_path}\n"
                 f"        Run eval2.py first to produce it.")
    df = pd.read_csv(csv_path)
    for col in ("val_used", "test_score", "gap_abs", "submission_score",
                "total_api_calls", "total_tokens", "total_execution_time_s",
                "xai_api_calls", "xai_tokens", "xai_total_overhead_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def lut(df: pd.DataFrame) -> dict:
    """(base_task, variant, config) -> row (as dict)."""
    return {(r.base_task, r.variant, r.config): r._asdict()
            for r in df.itertuples(index=False)}


# ----------------------------------------------------------------------------
# Figure 1: relative generalization gap on corrupted tasks
# ----------------------------------------------------------------------------
def fig_gap(df, idx, outpath):
    bases = sorted({r["base_task"] for k, r in idx.items() if k[1] == "corrupted"})

    def rel_gap(row):
        if row is None:
            return None
        val, test = row["val_used"], row["test_score"]
        if pd.isna(val) or pd.isna(test) or test == 0:
            return None
        return abs(val - test) / abs(test)

    data = []
    for b in bases:
        gv = rel_gap(idx.get((b, "corrupted", "vanilla")))
        go = rel_gap(idx.get((b, "corrupted", "ours")))
        if gv is None or go is None:
            continue
        caught = idx.get((b, "corrupted", "ours"), {}).get("injected_dropped") == "Y"
        data.append((b, gv, go, caught))

    data.sort(key=lambda t: t[1], reverse=True)  # worst vanilla gap on top
    labels = [short(b) for b, *_ in data]
    gv = [d[1] for d in data]
    go = [d[2] for d in data]
    caught = [d[3] for d in data]

    y = range(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(labels) + 1.6))
    ax.barh([i + h / 2 for i in y], gv, height=h, color=C_VANILLA,
            label="vanilla", zorder=3)
    ax.barh([i - h / 2 for i in y], go, height=h, color=C_OURS,
            label="ours", zorder=3)

    xmax = max(gv + go)
    for i, (v, o, c) in enumerate(zip(gv, go, caught)):
        ax.text(v + xmax * 0.01, i + h / 2, f"{v:.2f}", va="center",
                fontsize=8, color=C_VANILLA)
        ax.text(o + xmax * 0.01, i - h / 2, f"{o:.2f}", va="center",
                fontsize=8, color=C_OURS,
                fontweight="bold" if c else "normal")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("relative generalization gap   |val − test| / |test|")
    ax.set_title("Generalization gap on leak-injected (corrupted) tasks\n"
                 "lower is better — a large gap means the model overfit the injected leak",
                 loc="left")
    ax.set_xlim(0, xmax * 1.12)
    ax.grid(axis="y", visible=False)
    handles = [Patch(color=C_VANILLA, label="vanilla"),
               Patch(color=C_OURS, label="ours (bold label = leak dropped)")]
    ax.legend(handles=handles, loc="lower right", framealpha=0.95)
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ----------------------------------------------------------------------------
# Figure 2: detection matrix (corrupted tasks)
# ----------------------------------------------------------------------------
def fig_detection(df, idx, outpath):
    bases = sorted({r["base_task"] for k, r in idx.items() if k[1] == "corrupted"})
    cols = [("vanilla", "injected_flagged", "Vanilla\nflagged"),
            ("vanilla", "injected_dropped", "Vanilla\ndropped"),
            ("ours", "injected_flagged", "Ours\nflagged"),
            ("ours", "injected_dropped", "Ours\ndropped")]

    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(bases) + 1.8))
    for ci, (cfg, field, _) in enumerate(cols):
        for ri, b in enumerate(bases):
            row = idx.get((b, "corrupted", cfg), {})
            v = str(row.get(field, "N/A"))
            color = {"Y": C_GOOD, "N": C_BAD}.get(v, C_NEUTRAL)
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, facecolor=color,
                                       edgecolor="white", linewidth=2))
            ax.text(ci + 0.5, ri + 0.5, v, ha="center", va="center",
                    color="white", fontweight="bold")

    ax.axvline(2, color="#444444", linewidth=1.4)  # vanilla | ours divider
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(bases))
    ax.set_xticks([c + 0.5 for c in range(len(cols))])
    ax.set_xticklabels([c[2] for c in cols])
    ax.set_yticks([r + 0.5 for r in range(len(bases))])
    ax.set_yticklabels([short(b) for b in bases])
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.set_title("Injected-leak detection & removal per task", loc="left", pad=28)
    handles = [Patch(color=C_GOOD, label="Y — caught"),
               Patch(color=C_BAD, label="N — missed")]
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(0, -0.02), ncol=2, frameon=False)
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ----------------------------------------------------------------------------
# Figure 3: clean do-no-harm delta
# ----------------------------------------------------------------------------
def fig_clean_delta(df, idx, outpath):
    bases = sorted({r["base_task"] for k, r in idx.items() if k[1] == "clean"})

    rows = []
    for b in bases:
        o = idx.get((b, "clean", "ours"))
        v = idx.get((b, "clean", "vanilla"))
        if not o or not v:
            continue
        # Prefer the labeled-test score for both; if either is missing, fall
        # back to the internal validation score for both so they stay comparable.
        used_val = False
        so, sv = o["test_score"], v["test_score"]
        if pd.isna(so) or pd.isna(sv):
            so, sv = o["val_used"], v["val_used"]
            used_val = True
        if pd.isna(so) or pd.isna(sv) or sv == 0:
            continue
        # Orient so that positive always means "ours is better".
        if HIGHER_IS_BETTER[b]:
            imp = (so - sv) / abs(sv)
        else:
            imp = (sv - so) / abs(sv)
        rows.append((b, imp * 100.0, used_val))

    rows.sort(key=lambda t: t[1])
    labels = [short(b) for b, _, u in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(rows) + 1.8))
    colors = [C_GOOD if v >= 0 else C_BAD for v in vals]
    y = range(len(rows))
    ax.barh(list(y), vals, color=colors, zorder=3)
    ax.axvspan(-2, 2, color=C_NEUTRAL, alpha=0.18, zorder=0,
               label="±2% do-no-harm band")
    ax.axvline(0, color="#444444", linewidth=1)

    span = max(4.0, max(abs(v) for v in vals) * 1.25)
    for i, v in enumerate(vals):
        off = span * 0.015
        ax.text(v + (off if v >= 0 else -off), i, f"{v:+.1f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(-span, span)
    ax.set_xlabel("performance delta of ours vs vanilla   (positive = ours better)")
    ax.set_title("Do-no-harm check on clean (un-corrupted) tasks\n"
                 "oriented by metric direction",
                 loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ----------------------------------------------------------------------------
# Figure 4: overhead
# ----------------------------------------------------------------------------
def fig_overhead(df, idx, outpath):
    import numpy as np
    ours = df[(df["config"] == "ours") & df["xai_api_calls"].notna()]
    ours_clean = ours[ours["variant"] == "clean"]
    ours_corr = ours[ours["variant"] == "corrupted"]

    # Panel A: direct XAI share of the ours run
    a_calls_clean = (ours_clean["xai_api_calls"] / ours_clean["total_api_calls"] * 100.0).fillna(0).tolist()
    a_calls_corr = (ours_corr["xai_api_calls"] / ours_corr["total_api_calls"] * 100.0).fillna(0).tolist()
    a_tokens_clean = (ours_clean["xai_tokens"] / ours_clean["total_tokens"] * 100.0).fillna(0).tolist()
    a_tokens_corr = (ours_corr["xai_tokens"] / ours_corr["total_tokens"] * 100.0).fillna(0).tolist()
    a_time_clean = (ours_clean["xai_total_overhead_s"] / ours_clean["total_execution_time_s"] * 100.0).fillna(0).tolist()
    a_time_corr = (ours_corr["xai_total_overhead_s"] / ours_corr["total_execution_time_s"] * 100.0).fillna(0).tolist()

    a_data = [
        a_calls_clean, a_calls_corr,
        a_tokens_clean, a_tokens_corr,
        a_time_clean, a_time_corr
    ]
    a_labels = ["API calls", "tokens", "wall-time"]

    # Panel B: end-to-end A/B increase, aggregated over paired tasks
    bases = sorted({r["base_task"] for r in idx.values()})
    b_calls_clean = []
    b_calls_corr = []
    b_tokens_clean = []
    b_tokens_corr = []
    b_time_clean = []
    b_time_corr = []

    for b in bases:
        for var in ("clean", "corrupted"):
            o = idx.get((b, var, "ours"))
            v = idx.get((b, var, "vanilla"))
            if not o or not v:
                continue
            
            calls_list = b_calls_clean if var == "clean" else b_calls_corr
            tokens_list = b_tokens_clean if var == "clean" else b_tokens_corr
            time_list = b_time_clean if var == "clean" else b_time_corr

            if pd.notna(o["total_api_calls"]) and pd.notna(v["total_api_calls"]) and v["total_api_calls"] > 0:
                calls_list.append((o["total_api_calls"] - v["total_api_calls"]) / v["total_api_calls"] * 100.0)
            if pd.notna(o["total_tokens"]) and pd.notna(v["total_tokens"]) and v["total_tokens"] > 0:
                tokens_list.append((o["total_tokens"] - v["total_tokens"]) / v["total_tokens"] * 100.0)
            if pd.notna(o["total_execution_time_s"]) and pd.notna(v["total_execution_time_s"]) and v["total_execution_time_s"] > 0:
                time_list.append((o["total_execution_time_s"] - v["total_execution_time_s"]) / v["total_execution_time_s"] * 100.0)

    b_data = [
        b_calls_clean, b_calls_corr,
        b_tokens_clean, b_tokens_corr,
        b_time_clean, b_time_corr
    ]
    b_labels = ["API calls", "tokens", "wall-time"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))

    positions = [1, 2, 4, 5, 7, 8]
    width = 0.6

    # Style Boxplots for Panel A
    bpA = axA.boxplot(a_data, positions=positions, patch_artist=True, widths=width, showfliers=False, zorder=3)
    axA.set_xticks([1.5, 4.5, 7.5])
    axA.set_xticklabels(a_labels)

    for idx_box, patch in enumerate(bpA['boxes']):
        if idx_box % 2 == 0:
            patch.set_facecolor('#D5F5E3')  # soft light green for clean
            patch.set_edgecolor('#3C8C5A')  # C_GOOD
        else:
            patch.set_facecolor('#FADBD8')  # soft light red for corrupted
            patch.set_edgecolor('#C24A3F')  # C_BAD
        patch.set_linewidth(1.5)

    for idx_whisker, whisker in enumerate(bpA['whiskers']):
        box_idx = idx_whisker // 2
        color = '#3C8C5A' if box_idx % 2 == 0 else '#C24A3F'
        whisker.set_color(color)
        whisker.set_linewidth(1.2)

    for idx_cap, cap in enumerate(bpA['caps']):
        box_idx = idx_cap // 2
        color = '#3C8C5A' if box_idx % 2 == 0 else '#C24A3F'
        cap.set_color(color)
        cap.set_linewidth(1.2)

    for idx_median, median in enumerate(bpA['medians']):
        box_idx = idx_median
        color = '#1E8449' if box_idx % 2 == 0 else '#78281F'
        median.set_color(color)
        median.set_linewidth(2)

    # Overlay data points with jitter for Panel A
    np.random.seed(42)
    for i, y in enumerate(a_data):
        x = np.random.normal(positions[i], 0.04, size=len(y))
        color = '#1E8449' if i % 2 == 0 else '#78281F'
        axA.scatter(x, y, alpha=0.6, color=color, edgecolors='none', s=20, zorder=4)

    axA.set_ylabel("XAI share of total per task (%)")
    axA.set_ylim(-2, 55)  # Max time is ~47%, so 55% is a good limit
    axA.set_title("A. Direct XAI attribution\n(XAI share per task run; dots show individual tasks, n=10)", loc="left")
    axA.grid(axis="x", visible=False)

    # Style Boxplots for Panel B
    bpB = axB.boxplot(b_data, positions=positions, patch_artist=True, widths=width, showfliers=False, zorder=3)
    axB.set_xticks([1.5, 4.5, 7.5])
    axB.set_xticklabels(b_labels)

    for idx_box, patch in enumerate(bpB['boxes']):
        if idx_box % 2 == 0:
            patch.set_facecolor('#D5F5E3')  # soft light green
            patch.set_edgecolor('#3C8C5A')
        else:
            patch.set_facecolor('#FADBD8')  # soft light red
            patch.set_edgecolor('#C24A3F')
        patch.set_linewidth(1.5)

    for idx_whisker, whisker in enumerate(bpB['whiskers']):
        box_idx = idx_whisker // 2
        color = '#3C8C5A' if box_idx % 2 == 0 else '#C24A3F'
        whisker.set_color(color)
        whisker.set_linewidth(1.2)

    for idx_cap, cap in enumerate(bpB['caps']):
        box_idx = idx_cap // 2
        color = '#3C8C5A' if box_idx % 2 == 0 else '#C24A3F'
        cap.set_color(color)
        cap.set_linewidth(1.2)

    for idx_median, median in enumerate(bpB['medians']):
        box_idx = idx_median
        color = '#1E8449' if box_idx % 2 == 0 else '#78281F'
        median.set_color(color)
        median.set_linewidth(2)

    # Overlay data points with jitter for Panel B
    for i, y in enumerate(b_data):
        x = np.random.normal(positions[i], 0.04, size=len(y))
        color = '#1E8449' if i % 2 == 0 else '#78281F'
        axB.scatter(x, y, alpha=0.6, color=color, edgecolors='none', s=20, zorder=4)

    axB.axhline(0, color="#444444", linewidth=1, zorder=2)
    axB.set_ylabel("end-to-end change: (ours − vanilla) / vanilla (%)")
    
    all_b_vals = b_calls_clean + b_calls_corr + b_tokens_clean + b_tokens_corr + b_time_clean + b_time_corr
    max_val = max(all_b_vals) if all_b_vals else 100
    min_val = min(all_b_vals) if all_b_vals else -100
    axB.set_ylim(min_val - 15, max_val + 15)
    axB.set_title("B. End-to-end A/B cost\n(run cost changes; dots show individual tasks, n=10)", loc="left")
    axB.grid(axis="x", visible=False)

    # Add legend
    handles = [
        Patch(facecolor='#D5F5E3', edgecolor='#3C8C5A', linewidth=1.5, label="Clean tasks (n=10)"),
        Patch(facecolor='#FADBD8', edgecolor='#C24A3F', linewidth=1.5, label="Corrupted tasks (n=10)")
    ]
    axA.legend(handles=handles, loc="upper left", framealpha=0.95)
    axB.legend(handles=handles, loc="upper left", framealpha=0.95)

    fig.suptitle("Agent overhead of the XAI leakage checker",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="metrics_table2.csv")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pdf", action="store_true",
                    help="also write a vector .pdf next to each .png")
    args = ap.parse_args()

    plt.rcParams["savefig.dpi"] = args.dpi
    os.makedirs(args.outdir, exist_ok=True)

    df = load(args.csv)
    idx = lut(df)

    figs = {
        "gap_reduction": fig_gap,
        "detection_matrix": fig_detection,
        "clean_delta": fig_clean_delta,
        "overhead": fig_overhead,
    }
    exts = ["png"] + (["pdf"] if args.pdf else [])
    for name, fn in figs.items():
        for ext in exts:
            out = os.path.join(args.outdir, f"{name}.{ext}")
            fn(df, idx, out)
            print(f"  wrote {out}")
    print(f"Done. {len(figs)} figure(s) x {len(exts)} format(s) in {args.outdir}/")


if __name__ == "__main__":
    main()
