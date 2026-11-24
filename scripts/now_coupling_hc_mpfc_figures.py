#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure for `now_coupling_hc_mpfc.py`. Plots `result.json` as is: session
mean with 95% CI (t-based, from the stored SEM); p = two-sided t-test vs 0
(sign-flip p in result.json agrees).

    a  mPFC content per event: CURRENT and NEXT square (count-matched z)
    b  coupling HC current x mPFC current (r - same-square shuffle null)
    c  coupling HC current x mPFC next

    python scripts/now_coupling_hc_mpfc_figures.py <folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE

CM = 1 / 2.54
MPFC = SHOWGIRL2_DISCRETE[1]
GREY = "#9a9a9a"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "pdf.fonttype": 42})


def bar(ax, x, g, col):
    ci = stats.t.ppf(0.975, g["n"] - 1) * g["sem"]
    ax.bar(x, g["mean"], 0.36, color=col, edgecolor="white")
    ax.errorbar(x, g["mean"], yerr=ci, color="k", lw=0.8, capsize=2)
    y = g["mean"] + ci if g["mean"] >= 0 else g["mean"] - ci
    up = g["mean"] >= 0
    lab = "p < .001" if g["p"] < 0.001 else f"p = {g['p']:.3f}".replace("0.", ".")
    ax.annotate(lab, (x, y),
                xytext=(0, 3 if up else -3), textcoords="offset points",
                ha="center", va="bottom" if up else "top", rotation=90,
                fontsize=7)


def panel(ax, get, title, ylab):
    for i, ph in enumerate(("encode", "execute")):
        for j, (kind, col) in enumerate((("ripple", MPFC), ("ordinary", GREY))):
            bar(ax, i + (j - 0.5) * 0.4, get(kind, ph), col)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["encoding\nloops", "correct\nloops"])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylab)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.5 * (hi - lo), hi + 0.45 * (hi - lo))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main(folder):
    R = json.load(open(os.path.join(folder, "result.json")))
    n_c = R["n_sessions_coupling"]
    fig, axes = plt.subplots(1, 4, figsize=(20 * CM, 7.5 * CM))
    for ax, stat, lab in ((axes[0], "CURRENT", "current square"),
                          (axes[1], "fut1", "next square")):
        g = lambda kind, ph: R["level"][
            f"mPFC | {stat} | zm_{'rip' if kind == 'ripple' else 'ctl'}_{ph}"
            f" | all sessions"]
        n = g("ripple", "encode")["n"]
        panel(ax, g, f"mPFC {lab}\n({n} sessions)",
              "evidence z (session mean, 95% CI)")
    for ax, pair, lab in ((axes[2], "HC now x mPFC now", "HC now × mPFC now"),
                          (axes[3], "HC now x mPFC next", "HC now × mPFC next")):
        panel(ax, lambda kind, ph: R["coupling"][f"{pair} | {kind} | {ph}"],
              f"{lab}\n({n_c} sessions)",
              "coupling r − shuffle r\n(session mean, 95% CI)")
    for ax, l in zip(axes, "abcd"):
        ax.text(-0.35, 1.12, l, transform=ax.transAxes, fontsize=11,
                fontweight="bold")
    h = [plt.Rectangle((0, 0), 1, 1, color=MPFC),
         plt.Rectangle((0, 0), 1, 1, color=GREY)]
    fig.legend(h, ["ripples (window = ripple duration)",
                   "ordinary moments (no ripple within 0.5 s)"],
               loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"now_coupling_hc_mpfc.{ext}"),
                    dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
