#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure for `per_lag_learning_stage.py` (count-matched version). Plots
`result.json` as is.

    a  lag profile per learning stage, session mean +- SEM
    b  lag 0 and the ROI's Fig 3c window per stage: session mean Fisher z
       with 95% CI; stars = vs 0 (one-sided); trend p = per-session slope
       over the four stages (two-sided)

    python scripts/per_lag_learning_stage_figures.py <folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE

CM = 1 / 2.54
X = np.arange(0, 361, 30)
STAGES = ("encode", "exe 1-2", "exe 3-5", "exe 6+")
STAGE_LABEL = {"encode": "encoding loops", "exe 1-2": "correct loops 1-2",
               "exe 3-5": "correct loops 3-5", "exe 6+": "correct loops 6+"}
ROIS = {"HC_mid": ("mid HC", SHOWGIRL2_DISCRETE[2], "(0, 330)", "0°+330°"),
        "mPFC": ("mPFC", SHOWGIRL2_DISCRETE[1], "(30, 60)", "30°+60°")}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "pdf.fonttype": 42})


def ramp(col, i, n=4):
    """Light (encode) to dark (late) version of the ROI colour."""
    c = np.array(to_rgb(col))
    k = i / (n - 1)
    return tuple((1 - k) * (0.6 * np.ones(3) + 0.4 * c) + k * 0.75 * c)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def main(folder):
    R = json.load(open(os.path.join(folder, "result.json")))["matched"]
    fig, axes = plt.subplots(2, 2, figsize=(17 * CM, 13.5 * CM),
                             gridspec_kw=dict(width_ratios=[1.5, 1]))
    for i, (roi, (name, col, win, win_lab)) in enumerate(ROIS.items()):
        ax = axes[i, 0]
        for j, s in enumerate(STAGES):
            c = R["curves"][f"{s} | {roi} | subject"]
            m = np.append(c["mean"], c["mean"][0])
            se = np.append(c["sem"], c["sem"][0])
            ax.fill_between(X, m - se, m + se, color=ramp(col, j), alpha=0.15,
                            lw=0)
            ax.plot(X, m, "-o", ms=2.5, lw=1.3, color=ramp(col, j),
                    label=STAGE_LABEL[s])
        n = R["n"][f"encode | {roi}"]
        ax.set_title(f"{name}: lag profile by learning stage\n"
                     f"({n['sessions']} sessions, {n['cells']} cells; "
                     f"loop counts matched)", fontsize=10)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.set_xticklabels(["0°", "60°", "120°", "180°", "240°", "300°", "0°"])
        ax.set_ylabel("CV r (session mean ± SEM)")
        if i == 1:
            ax.legend(frameon=False, fontsize=8, loc="upper center",
                      bbox_to_anchor=(0.5, -0.3), ncol=4, columnspacing=1.0,
                      handlelength=1.4)

        ax = axes[i, 1]
        w = 0.38
        trends = []
        for k, (key, lab, hatch) in enumerate(
                ((f"{roi} | lag 0 (0,) | subject", "0°", None),
                 (f"{roi} | window {win} | subject", win_lab, "///"))):
            F = R["focused"][key]
            for j, row in enumerate(F["stages"]):
                xpos = j + (k - 0.5) * w
                ax.bar(xpos, row["mean_z"], w * 0.92, yerr=row["ci95"],
                       color=ramp(col, j), hatch=hatch, edgecolor="white",
                       error_kw=dict(lw=0.8, capsize=1.5),
                       label=lab if j == 3 else None)
                top = row["mean_z"] + row["ci95"]
                ax.text(xpos, top + 0.004, stars(row["p_one_sided"]),
                        ha="center", va="bottom", fontsize=8)
            trends.append(f"{lab} p = {F['trend']['p_two_sided']:.3f}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["enc.", "1-2", "3-5", "6+"])
        ax.set_xlabel("learning stage (correct loops)\ntrend over stages:\n"
                      + ",  ".join(trends), fontsize=8.5)
        ax.set_ylabel("Fisher z (session mean, 95% CI)")
        ax.set_title(f"{name}: now vs Fig 3c window", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    for ax in axes.ravel():
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1, 0].set_xlabel("Lag into the future")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"learning_stage_lag_profiles.{ext}"),
                    dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
