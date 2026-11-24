#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure for `lag0_tuning_vs_ripple_firing.py` (plots result.json and
cells.csv as they are; nothing is recomputed).

  a-c  r across cells (Pearson, 95% CI) of lag-0 CV r with each ripple-firing
       measure, for every tuning version (estimator x loop set) and ROI;
       star = within-session permutation p < .05
  d    reliability ceiling: per-cell r0 from encoding loops vs correct loops
  e    mPFC, claim 2c as originally tested (mode, correct loops x firing after
       discovery with vs without a ripple)
  f    mPFC, the direct pairing (per-repeat, encoding loops x firing in
       encoding ripples)

    python scripts/lag0_tuning_vs_ripple_firing_figures.py <folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE

CM = 1 / 2.54
COL = {"mPFC": SHOWGIRL2_DISCRETE[1], "HC_mid": SHOWGIRL2_DISCRETE[2],
       "HC_anterior": "#23677E"}
NAME = {"mPFC": "mPFC", "HC_mid": "mid HC", "HC_anterior": "anterior HC"}
VERSIONS = [(e, s) for e in ("mode", "per_repeat") for s in ("execute", "all", "encode")]
SET_NAME = {"execute": "correct", "all": "all", "encode": "encoding"}
EST_NAME = {"mode": "mode maps", "per_repeat": "per-repeat maps"}
Y_NAME = {"storage_rip_minus_norip": "claim 2c's measure:\nafter discovery, ripple − no ripple",
          "ripmod_encode": "firing in encoding ripples\n(in − out) / (in + out)",
          "ripmod_execute": "firing in correct-loop ripples\n(in − out) / (in + out)"}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, pad=1)


def corr_bar(ax, x, d, col):
    lo, hi = d["ci95"]
    ax.bar(x, d["r"], 0.24, color=col, edgecolor="white", lw=0.4)
    ax.errorbar(x, d["r"], yerr=[[d["r"] - lo], [hi - d["r"]]], color="k",
                lw=0.6, capsize=1)
    if d["p_perm"] < .05:
        ax.text(x, hi, "*", ha="center", va="bottom", fontsize=9)


def version_axis(ax):
    ax.set_xticks(range(len(VERSIONS)))
    ax.set_xticklabels([SET_NAME[s] for _, s in VERSIONS], fontsize=7.5,
                       rotation=35, ha="right", rotation_mode="anchor")
    for k, est in enumerate(("mode", "per_repeat")):
        ax.text(1 + 3 * k, -0.2, EST_NAME[est], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="0.3")
    ax.axvline(2.5, color="0.8", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(-0.55, len(VERSIONS) - 0.45)


def scatter(ax, T, R, xcol, ycol, key, title):
    g = T[T.roi == "mPFC"]
    ax.scatter(g[xcol], g[ycol], s=7, color=COL["mPFC"], alpha=0.7, lw=0)
    d = R["correlations"][key]
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(0, color="0.7", lw=0.5)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.text(0.03, 0.97, f"r = {d['r']:+.2f}, p = {d['p_perm']:.3f}\nn = {d['n']} cells, "
            f"{d['n_sessions']} sessions", transform=ax.transAxes, va="top", fontsize=7.5)
    clean(ax)


def main(folder):
    R = json.load(open(os.path.join(folder, "result.json")))
    T = pd.read_csv(os.path.join(folder, "cells.csv"))
    W, H = 25, 14.4
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax_at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    top = [ax_at(1.8 + 7.8 * i, 8.0, 6.6, 4.6) for i in range(3)]
    for ax, y, l in zip(top, Y_NAME, "abc"):
        for j, roi in enumerate(COL):
            for i, (e, s) in enumerate(VERSIONS):
                corr_bar(ax, i + (j - 1) * 0.26,
                         R["correlations"][f"{roi} | {e} | {s} | {y} | pearson"], COL[roi])
        version_axis(ax)
        ax.set_ylim(-0.42, 0.52)
        ax.set_title(Y_NAME[y], fontsize=8.5, pad=3)
        clean(ax)
        fig.text(ax.get_position().x0 - 1.5 / W, ax.get_position().y1 + 0.15 / H, l,
                 fontsize=12, fontweight="bold", va="bottom")
    top[0].set_ylabel("r across cells: lag-0 CV r\nvs ripple firing (95% CI)")
    for ax in top[1:]:
        ax.set_yticklabels([])
    hs = [plt.Rectangle((0, 0), 1, 1, color=COL[r]) for r in COL]
    fig.legend(hs, [f"{NAME[r]} ({R['n_cells'][r]['with_ripple_firing']} cells)" for r in COL],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               fontsize=8, title="tuning: r_lag000 per cell; * within-session permutation "
               "p < .05", title_fontsize=7.5)

    ax_d = ax_at(1.8, 1.4, 5.0, 4.2)
    for i, est in enumerate(("mode", "per_repeat")):
        for j, roi in enumerate(COL):
            corr_bar(ax_d, i + (j - 1) * 0.26,
                     R["reliability"][f"{roi} | {est} | encode vs execute"], COL[roi])
    ax_d.axhline(0, color="k", lw=0.6)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels([EST_NAME["mode"], EST_NAME["per_repeat"]], fontsize=8)
    ax_d.set_ylim(-0.42, 0.52)
    ax_d.set_ylabel("ρ across cells:\nencoding r0 vs correct r0 (95% CI)")
    ax_d.set_title("reliability ceiling: same cell,\nindependent loops", fontsize=8.5, pad=3)
    clean(ax_d)
    ax_e = ax_at(9.6, 1.4, 6.0, 4.2)
    scatter(ax_e, T, R, "r0 | mode | execute", "storage_rip_minus_norip",
            "mPFC | mode | execute | storage_rip_minus_norip | pearson",
            "mPFC, claim 2c as published:\nmode maps, correct loops")
    ax_e.set_xlabel("lag-0 CV r")
    ax_e.set_ylabel("after discovery:\nripple − no ripple")
    ax_f = ax_at(18.4, 1.4, 6.0, 4.2)
    scatter(ax_f, T, R, "r0 | per_repeat | encode", "ripmod_encode",
            "mPFC | per_repeat | encode | ripmod_encode | pearson",
            "mPFC, the direct pairing:\nper-repeat maps, encoding loops")
    ax_f.set_xlabel("lag-0 CV r")
    ax_f.set_ylabel("firing in encoding ripples\n(in − out) / (in + out)")
    for ax, l in ((ax_d, "d"), (ax_e, "e"), (ax_f, "f")):
        fig.text(ax.get_position().x0 - 1.6 / W, ax.get_position().y1 + 0.15 / H, l,
                 fontsize=12, fontweight="bold", va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"lag0_tuning_vs_ripple_firing.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("wrote", folder)


if __name__ == "__main__":
    main(sys.argv[1])
