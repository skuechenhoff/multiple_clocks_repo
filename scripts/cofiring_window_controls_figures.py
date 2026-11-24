#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure for `cofiring_window_controls.py`. Plots `result.json` as is
(session mean, 95% CI from the stored SEM, two-sided t-test p).

    a  HC-mPFC population co-firing r: ripple windows of their own length,
       the same with duration partialled out, fixed-width ripple windows,
       ordinary moments
    b  content coupling HC now x mPFC now, per window variant, with and
       without the per-event population gain removed
    c  the same for HC now x mPFC next

    python scripts/cofiring_window_controls_figures.py <folder>

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
LIGHT = "#a9cdb9"
GREY = "#9a9a9a"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "pdf.fonttype": 42})
PH = {"encode": "encoding\nloops", "execute": "correct\nloops"}


def bars(ax, items, title, ylab):
    """items: list of (label, colour, {phase: group-dict})."""
    w = 0.8 / len(items)
    for j, (lab, col, g) in enumerate(items):
        for i, ph in enumerate(PH):
            d = g[ph]
            x = i + (j - (len(items) - 1) / 2) * w
            ci = stats.t.ppf(0.975, d["n"] - 1) * d["sem"]
            ax.bar(x, d["mean"], w * 0.9, color=col, edgecolor="white",
                   label=lab if i == 0 else None)
            ax.errorbar(x, d["mean"], yerr=ci, color="k", lw=0.7, capsize=1.5)
            up = d["mean"] >= 0
            p = "<.001" if d["p"] < .001 else f"{d['p']:.3f}".replace("0.", ".")
            ax.annotate(p, (x, d["mean"] + (ci if up else -ci)),
                        xytext=(0, 2 if up else -2), textcoords="offset points",
                        ha="center", va="bottom" if up else "top", rotation=90,
                        fontsize=6.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(PH)))
    ax.set_xticklabels(list(PH.values()))
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel(ylab)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.35 * (hi - lo), hi + 0.35 * (hi - lo))
    ax.legend(fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main(folder):
    R = json.load(open(os.path.join(folder, "result.json")))["tests"]
    g = lambda key: {ph: R[f"{key} | {ph}"] for ph in PH}
    fig, axes = plt.subplots(1, 3, figsize=(19 * CM, 9.5 * CM))
    bars(axes[0], [
        ("ripples, own length", MPFC, g("r_pop | own")),
        ("ripples, own length, duration partialled", LIGHT, g("r_pop_dur_partial | own")),
        ("ripples, fixed width", "#2b5d47", g("r_pop | fixed")),
        ("ordinary moments (fixed width)", GREY, g("r_pop | ordinary"))],
        "all-cell co-firing\nHC × mPFC", "r across events (session mean, 95% CI)")
    for ax, pair in ((axes[1], "HC now x mPFC now"), (axes[2], "HC now x mPFC next")):
        bars(ax, [
            ("ripples, own length", MPFC, g(f"{pair} | own")),
            ("ripples, own length, gain removed", LIGHT, g(f"{pair} | gainfree | own")),
            ("ripples, fixed width, gain removed", "#2b5d47", g(f"{pair} | gainfree | fixed")),
            ("ordinary moments", GREY, g(f"{pair} | ordinary"))],
            pair.replace(" x ", " × ").replace("HC", "content coupling\nHC"),
            "r − same-square shuffle r\n(session mean, 95% CI)")
    for ax, l in zip(axes, "abc"):
        ax.text(-0.4, 1.15, l, transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"cofiring_window_controls.{ext}"),
                    dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
