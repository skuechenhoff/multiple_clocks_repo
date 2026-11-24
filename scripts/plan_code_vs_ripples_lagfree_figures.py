#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for plan_code_vs_ripples_lagfree.py (plots result.json and the saved
per-session betas as is).

    a  ripple beta per session (mean, 95% CI; sign-flip p), per code variant,
       first 3 vs all execution loops
    b  the same per session: first 3 vs all, paired (future mean, best lag)
    c  pooled partial correlation over blocks (descriptive: blocks are nested
       in sessions, so its p is not the test; the test is a)

    python scripts/plan_code_vs_ripples_lagfree_figures.py <folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CM = 1 / 2.54
VARIANTS = ("lag 60", "future mean", "best lag", "max lag")
V_L = {"lag 60": "60°\n(B1 primary)", "future mean": "mean of\nfuture lags",
       "best lag": "each cell's\nbest lag (CV)", "max lag": "max over\nlags"}
W_C = {"first3": ("first 3 execution loops", "#2E5E45"),
       "all": ("all execution loops", "#9CC5A8")}


def ptxt(p):
    return "p<.001" if p < .001 else f"p={p:.3f}".replace("0.", ".")


def draw(d):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    R = json.load(open(os.path.join(d, "result.json")))
    S = pd.read_csv(os.path.join(d, "session_betas.csv"))
    fig = plt.figure(figsize=(19 * CM, 12 * CM))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], hspace=0.95,
                          wspace=0.55)
    # a
    ax = fig.add_subplot(gs[0, :])
    for i, v in enumerate(VARIANTS):
        for k, (w, (lab, col)) in enumerate(W_C.items()):
            r = R[f"{v} | {w} | beta"]
            x = i + (k - 0.5) * 0.36
            sig = r["p_perm"] < .05
            ax.bar(x, r["mean"], 0.33, color=col if sig else "white",
                   edgecolor=col, lw=1.1, zorder=2)
            h = stats.t.ppf(.975, r["n"] - 1) * r["sem"]
            ax.plot([x, x], [r["mean"] - h, r["mean"] + h], color="0.15",
                    lw=0.9, zorder=3)
            ax.text(x, max(r["mean"] + h, 0) + 0.02, ptxt(r["p_perm"]),
                    rotation=90, ha="center", va="bottom", fontsize=7,
                    color="0.1" if sig else "0.5")
        q = R[f"{v} | first3 - all | beta"]
        ax.text(i, -0.42, f"first 3 − all: {ptxt(q['p_perm'])}",
                ha="center", fontsize=7, color="0.3")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_ylim(-0.45, 0.95)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([V_L[v] for v in VARIANTS])
    ax.set_ylabel("ripple β per session\n(standardised)")
    ax.set_title("a  more learning ripples → stronger mPFC code in "
                 "execution? (25 sessions)", loc="left")
    ax.legend(handles=[Patch(fc=c, ec=c, label=l) for l, c in W_C.values()]
              + [Patch(fc="white", ec="0.4", label="open: p ≥ .05")],
              frameon=False, fontsize=7.5, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.28))
    # b: paired
    for j, v in enumerate(("future mean", "best lag")):
        ax = fig.add_subplot(gs[1, j])
        P = S[S.variant == v].pivot(index="session", columns="which",
                                    values="beta").dropna()
        for _, row in P.iterrows():
            ax.plot([0, 1], [row["first3"], row["all"]], color="0.8", lw=0.7,
                    zorder=1)
        for k, w in enumerate(("first3", "all")):
            m = P[w].mean()
            h = stats.t.ppf(.975, len(P) - 1) * P[w].sem()
            ax.errorbar(k, m, h, color=W_C[w][1], marker="o", ms=6, lw=1.4,
                        capsize=3, zorder=3)
        ax.axhline(0, color="0.2", lw=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["first 3", "all"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("ripple β per session" if j == 0 else "")
        ax.set_title(f"{'bc'[j]}  {V_L[v].replace(chr(10), ' ')}", loc="left")
    # d: pooled partial r
    ax = fig.add_subplot(gs[1, 2])
    for i, v in enumerate(VARIANTS):
        for k, (w, (lab, col)) in enumerate(W_C.items()):
            r = R[f"{v} | {w} | partial r (pooled blocks)"]
            ax.bar(i + (k - 0.5) * 0.36, r["r"], 0.33, color=col, lw=0)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(["60°", "mean", "best", "max"], rotation=40,
                       fontsize=7.5)
    ax.set_ylabel("partial r, 617 blocks")
    ax.set_title("d  pooled partial r\n(descriptive)", loc="left")
    cap = ("B1's per-cell block codes (F2 pooled estimator: this block's "
           "execution-loop lag map vs the execution maps of all OTHER "
           "configurations, dwell-weighted r, Fisher z, mean over the "
           "session's mPFC cells) and B1's model: per session, OLS of the "
           "z-scored code on z-scored log(1 + hippocampal ripples while "
           "learning) + log learning duration + n learning loops + block "
           "order + run index; sign-flip over sessions. Variants: 60° = B1 "
           "primary; mean of the 11 future lags (as the learning-ripples "
           "figure); each cell's best future lag chosen on its OTHER blocks; "
           "max over the 11 future lags per block. a: mean β, 95% CI; filled "
           "= p < .05. b, c: lines = sessions. d: partial correlation of the "
           "pooled within-session residuals (blocks are not independent; "
           "descriptive only). Exploratory; uncorrected.")
    fig.text(0.01, -0.03, textwrap.fill(cap, 165), fontsize=7.5, va="top",
             color="0.2")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"plan_code_vs_ripples_lagfree.{e}"),
                    dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(d, "plan_code_vs_ripples_lagfree.png"))


if __name__ == "__main__":
    draw(sys.argv[1])
