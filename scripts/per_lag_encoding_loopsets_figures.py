#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure for `per_lag_encoding_loopsets.py`: lag profiles of mid HC and mPFC
for the three loop sets and the two estimators. Plots `result.json` as is
(curves = session-balanced mean +- SEM; markers = that set's lag-wise test
vs 0 at the session unit, one-sided, uncorrected; filled = p < .05).

    python scripts/per_lag_encoding_loopsets_figures.py <folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE

CM = 1 / 2.54
LAGS = list(range(0, 360, 30))
X = np.array(LAGS + [360])
ROIS = {"HC_mid": ("mid HC", SHOWGIRL2_DISCRETE[2]),
        "mPFC": ("mPFC", SHOWGIRL2_DISCRETE[1])}
SET_STYLE = {"execute": ("correct loops (execute; Fig 3c)", "-", "o", 1.0),
             "all": ("all loops", "--", "s", 0.55),
             "encode": ("encoding loops only", ":", "^", 0.3)}
EST_STYLE = {"per_repeat": ("per-repeat locations", "-", "o"),
             "mode": ("per-bin mode location (old)", "--", "D")}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "pdf.fonttype": 42})


def wrap(v):
    v = np.asarray(v, float)
    return np.append(v, v[0])


def shade(col, k):
    """Mix a colour with black (k < 1 darkens) for the loop-set lines."""
    c = np.array(plt.matplotlib.colors.to_rgb(col))
    return tuple(c * (0.45 + 0.55 * k))


def pvals(res, key, roi):
    rows = [r for r in res["vs0"][key] if r["roi"] == roi
            and r["analysis_unit"] == "subject"]
    rows = sorted(rows, key=lambda r: r["lag_deg"])
    return np.array([r["p_one_sided"] for r in rows])


def draw(ax, res, est, s, roi, label, ls, mk, col):
    c = res["curves"][f"{est} | {s} | {roi} | subject"]
    m, se = wrap(c["mean"]), wrap(c["sem"])
    p = wrap(pvals(res, f"{est} | {s}", roi))
    ax.fill_between(X, m - se, m + se, color=col, alpha=0.15, lw=0)
    ax.plot(X, m, ls=ls, color=col, lw=1.4, label=label)
    sig = p < 0.05
    ax.scatter(X[sig], m[sig], marker=mk, s=22, color=col, zorder=3)
    ax.scatter(X[~sig], m[~sig], marker=mk, s=16, facecolor="white",
               edgecolor=col, lw=0.8, zorder=3)


def main(folder):
    res = json.load(open(os.path.join(folder, "result.json")))
    fig, axes = plt.subplots(2, 2, figsize=(17 * CM, 12.5 * CM), sharex=True)
    for i, (roi, (name, col)) in enumerate(ROIS.items()):
        n = res["n"][f"per_repeat | execute | {roi}"]
        ax = axes[i, 0]
        for s, (lab, ls, mk, k) in SET_STYLE.items():
            draw(ax, res, "per_repeat", s, roi, lab, ls, mk, shade(col, k))
        ax.set_title(f"{name}: which loops go in\n(per-repeat estimator; "
                     f"{n['sessions']} sessions, {n['cells']} cells)",
                     fontsize=10)
        ax = axes[i, 1]
        for est, (lab, ls, mk) in EST_STYLE.items():
            draw(ax, res, est, "execute", roi, lab, ls, mk,
                 col if est == "per_repeat" else "#555555")
        ax.set_title(f"{name}: how the map is built\n(correct loops)",
                     fontsize=10)
    for ax in axes.ravel():
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.set_xticklabels(["0°", "60°", "120°", "180°", "240°", "300°", "0°"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("CV r (session mean ± SEM)")
    for ax in axes[1]:
        ax.set_xlabel("Lag into the future")
    for j in range(2):
        h, l = axes[1, j].get_legend_handles_labels()
        lg = axes[1, j].legend(
            h, l, loc="upper center", bbox_to_anchor=(0.5, -0.25),
            frameon=False, fontsize=8, ncol=1,
            title=("filled marker: lag > 0 at the session unit,\n"
                   "one-sided p < .05, uncorrected; open: n.s.") if j == 0
            else None)
        if j == 0:
            lg.get_title().set_fontsize(8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"loopsets_lag_profiles.{ext}"),
                    dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
