#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for per_lag_encoding_encode_vs_execute.py (plots result.json as is).

Lag profiles as in the gradient figure (panel c): mean CV r per lag, 95%
t-interval band, encode vs execute (the published profile) and execute with
the encode loop count (dotted). Rows: CELL unit, SESSION unit. Columns: HC
mid, mPFC. Under each panel, per lag: filled marker = BH q < .05 across the
12 lags, open marker = uncorrected p < .05 (vs 0: one-sided > 0, as
published; differences: two-sided).

    python scripts/per_lag_encoding_encode_vs_execute_figures.py <folder>

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
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import era_brewer

CM = 1 / 2.54
_p = era_brewer.era_brew("Showgirl2", n=7)
ROI_COL = {"HC_mid": _p[2], "mPFC": _p[1]}
ROI_L = {"HC_mid": "mid HC", "mPFC": "mPFC"}
ENC = "#3b3b45"
LAGS = list(range(0, 360, 30))
RASTER = (("encode", "encode vs 0"), ("execute", "execute vs 0"),
          ("execute_matched", "matched vs 0"),
          ("encode - execute", "enc − exe"),
          ("encode - execute_matched", "enc − matched"))


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False})


def sig_row(R, key, roi, unit):
    """(p, q) per lag for one raster row."""
    if " - " in key:
        rows = R["diff"][f"{key} | {roi}"][unit]
        return ([x["p_two_sided"] for x in rows],
                [x["q_two_sided_lags"] for x in rows])
    T = pd.DataFrame(R["vs0"][key])
    T = T[(T.roi == roi) & (T.analysis_unit == unit)].sort_values("lag_deg")
    return T.p_one_sided.tolist(), T.p_one_sided_fdr_lags.tolist()


def draw(d):
    _rc()
    R = json.load(open(os.path.join(d, "result.json")))
    fig = plt.figure(figsize=(17 * CM, 15 * CM))
    outer = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.42)
    for row, unit in enumerate(("cell", "subject")):
        for col, roi in enumerate(("HC_mid", "mPFC")):
            gs = outer[row, col].subgridspec(2, 1, height_ratios=[3, 1.35],
                                             hspace=0.08)
            ax, ar = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
            for s, colr, ls, lw, band in (
                    ("execute", ROI_COL[roi], "-", 1.6, True),
                    ("encode", ENC, "--", 1.6, True),
                    ("execute_matched", ROI_COL[roi], ":", 1.1, False)):
                c = R["per_unit_curves"][f"{s} | {roi} | {unit}"]
                m, sd, n = (np.array(c[k], float) for k in ("mean_r", "sd_r", "n"))
                ax.plot(LAGS, m, color=colr, ls=ls, lw=lw, marker="o",
                        ms=2.5 if band else 0, zorder=3)
                if band:
                    h = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)
                    ax.fill_between(LAGS, m - h, m + h, color=colr,
                                    alpha=0.18, lw=0, zorder=2)
            ax.axhline(0, color="0.3", lw=0.7, ls="--", zorder=1)
            ax.set_xlim(-12, 342)
            ax.set_xticks(LAGS[::2])
            ax.set_xticklabels([])
            n_u = R["per_unit_curves"][f"execute | {roi} | {unit}"]["n"][0]
            ax.set_title(f"{'abcd'[2 * row + col]}  {ROI_L[roi]}, "
                         f"{'cells' if unit == 'cell' else 'sessions'} "
                         f"(n = {n_u})", loc="left")
            ax.set_ylabel("mean CV r")
            for i, (key, lab) in enumerate(RASTER):
                p, q = sig_row(R, key, roi, unit)
                y = len(RASTER) - 1 - i
                colr = ENC if key == "encode" else (
                    ROI_COL[roi] if key.startswith("execute") else "0.15")
                for lag, pv, qv in zip(LAGS, p, q):
                    if np.isfinite(qv) and qv < .05:
                        ar.plot(lag, y, "o", ms=4.5, color=colr)
                    elif np.isfinite(pv) and pv < .05:
                        ar.plot(lag, y, "o", ms=4.5, mfc="white", mec=colr,
                                mew=1.0)
                ar.axhline(y, color="0.9", lw=0.6, zorder=0)
            ar.set_yticks(range(len(RASTER)))
            ar.set_yticklabels([lab for _, lab in RASTER][::-1], fontsize=7)
            ar.set_ylim(-0.6, len(RASTER) - 0.4)
            ar.set_xlim(-12, 342)
            ar.set_xticks(LAGS[::2])
            ar.set_xticklabels([f"{l}°" for l in LAGS[::2]])
            ar.spines["left"].set_visible(False)
            ar.tick_params(axis="y", length=0)
            ar.set_xlabel("lag (future →)")
    h = [Line2D([], [], color="0.35", lw=1.6, label="execute (published)"),
         Line2D([], [], color=ENC, lw=1.6, ls="--", label="encode"),
         Line2D([], [], color="0.35", lw=1.1, ls=":",
                label="execute, encode loop count"),
         Line2D([], [], marker="o", ls="", color="0.15", label="q < .05 (12 lags)"),
         Line2D([], [], marker="o", ls="", mfc="white", mec="0.15",
                label="p < .05 uncorrected")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.035),
               ncol=3, frameon=False, handlelength=2.0, columnspacing=1.2)
    ok = R["reproduces_published"]
    cap = ("The published per-lag location encoding (per_lag_encoding.py, "
           "noctrl: phase-residualised, repeats averaged per configuration, "
           "leave-one-configuration-out dwell-weighted r), run on the "
           "ENCODE loops (every loop before a block's first correct one = "
           "the rows the published run excludes) and on the published "
           "execute set; 'execute, encode loop count' = the first n execute "
           "loops of each configuration, n = its encode loops. Lines = mean "
           "CV r over cells (a, b) or over sessions (c, d; session = mean "
           "over its cells), bands = 95% t-interval. Tests on Fisher z "
           "(lagwise_aggregation): vs 0 one-sided, differences two-sided "
           "(paired within cell). Check: execute reproduces the published "
           f"per-cell r (max |diff| {ok['max_abs_diff']:.0e}, "
           f"{ok['n_cells']} cells).")
    fig.text(0.02, -0.06, textwrap.fill(cap, 150), fontsize=7.5, va="top",
             color="0.2")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"per_lag_encode_vs_execute.{e}"),
                    dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(d, "per_lag_encode_vs_execute.png"))


if __name__ == "__main__":
    draw(sys.argv[1])
