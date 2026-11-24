#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for scripts/position_index_by_phase.py (plots result.json as is).

    a  index per ROI: learning / execution / execution with matched loop count
    b  paired change, execution - learning, per ROI (same cells)

Bars start at zero and end at the Hodges-Lehmann location; whiskers are its
distribution-free 95% interval. Filled = q < .05 (BH across ROIs), pale = not.

    python scripts/position_index_by_phase_figures.py <run folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import textwrap

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import era_brewer

CM = 1 / 2.54
_p = era_brewer.era_brew("Showgirl2", n=7)
ROI_COLOURS = {"EC": _p[0], "mOFC": _p[4], "mPFC": _p[1],
               "HC_anterior": "#23677E", "HC_mid": _p[2], "PCC": _p[3]}
ROI_LABEL = {"HC_anterior": "HC ant", "HC_mid": "HC mid"}
SETS = (("learning", "learning", ""),
        ("execution", "execution", None),
        ("execution_matched", "execution,\nmatched n loops", "////"))


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False})


def ptxt(p):
    return "p<.001" if p < .001 else f"p={p:.3f}".replace("0.", ".")


def bar(ax, x, st, colour, width, hatch, sig, label_p, light=False):
    face = colour if (sig and not light) else "white"
    edge = colour
    if light:
        face = matplotlib.colors.to_rgba(colour, 0.35 if sig else 0.12)
    ax.bar(x, st["hl"], width, color=face, edgecolor=edge, linewidth=1.0,
           hatch=hatch, zorder=2)
    ax.plot([x, x], [st["lo"], st["hi"]], color="0.15", lw=1.0, zorder=3)
    top = max(st["hi"], 0) + 0.0012
    ax.text(x, top, label_p, ha="center", va="bottom", fontsize=7.5,
            rotation=90, color="0.1" if sig else "0.45")


def draw(d):
    _rc()
    R = json.load(open(os.path.join(d, "result.json")))
    rois = R["settings"]["rois"]
    fig, axes = plt.subplots(1, 2, figsize=(19.5 * CM, 7.5 * CM),
                             gridspec_kw=dict(width_ratios=[1.6, 1.4],
                                              wspace=0.45))
    ax = axes[0]
    w = 0.26
    for i, r in enumerate(rois):
        for j, (v, _, hatch) in enumerate(SETS):
            st = R["levels"][v][r]
            bar(ax, i + (j - 1) * w, st, ROI_COLOURS[r], w * 0.92,
                hatch if hatch else None, st["q"] < .05, ptxt(st["q"]),
                light=(v == "learning"))
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlim(-0.55, len(rois) - 0.45)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([ROI_LABEL.get(r, r).replace(" ", "\n")
                        for r in rois])
    ax.set_ylabel("held-out position r\n(Hodges-Lehmann, 95% CI)")
    ax.set_title("a  position index, learning vs execution", loc="left")
    ax.set_ylim(-0.02, 0.06)
    handles = [Patch(facecolor=(0.4, 0.4, 0.4, 0.35), edgecolor="0.4",
                     label="learning"),
               Patch(facecolor="0.4", edgecolor="0.4", label="execution"),
               Patch(facecolor="white", edgecolor="0.4", hatch="////",
                     label="execution, matched n loops")]
    ax.legend(handles=handles, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.12), handlelength=1.2,
              columnspacing=0.8)

    ax = axes[1]
    w = 0.36
    for i, r in enumerate(rois):
        for j, (c, hatch) in enumerate((("execution - learning", None),
                                        ("execution_matched - learning",
                                         "////"))):
            st = R["paired"][c][r]
            bar(ax, i + (j - 0.5) * w, st, ROI_COLOURS[r], w * 0.92, hatch,
                st["p"] < .05, ptxt(st["p"]))
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlim(-0.55, len(rois) - 0.45)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([ROI_LABEL.get(r, r).replace(" ", "\n")
                        for r in rois])
    ax.set_ylabel("execution - learning\n(paired over cells)")
    mo = R["mOFC_vs_HC"]
    ax.set_title("b  change with learning", loc="left")
    ax.text(0.02, 0.02, "HC vs mOFC change: "
            + ptxt(mo["execution - learning"]["p_mannwhitney"])
            + " (all), " + ptxt(mo["execution_matched - learning"]
                                ["p_mannwhitney"]) + " (matched)",
            transform=ax.transAxes, fontsize=8, va="bottom")
    ax.set_ylim(-0.05, 0.065)
    handles = [Patch(facecolor="0.4", edgecolor="0.4", label="all loops"),
               Patch(facecolor="white", edgecolor="0.4", hatch="////",
                     label="matched n loops")]
    ax.legend(handles=handles, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.22), handlelength=1.2)

    cap = ("Published chapter-5 single-cell index (goal progress and per-loop "
           "ramp removed; 4 states + state x goal-progress, leave-one-"
           "configuration-out; held-out r of the state block), computed on "
           "the loops BEFORE a block's first correct loop (learning) and on "
           "correct loops (execution;\nreproduces the published values). "
           "a: one-sided Wilcoxon of per-cell r > 0, BH across ROIs (q "
           "printed); filled = q < .05. b: Wilcoxon signed-rank of the "
           "per-cell change (uncorrected p printed). Caution: PCC, the "
           "execution negative control, is\npositive in learning, so the "
           "learning measure carries a state-locked signal that is not a "
           "position code (e.g. uneven search/error presses across states).")
    fig.text(0.01, -0.16, textwrap.fill(cap.replace("\n", " "), 150),
             fontsize=7.5, va="top")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"position_index_by_phase.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(d, "position_index_by_phase.png"))


if __name__ == "__main__":
    draw(sys.argv[1])
