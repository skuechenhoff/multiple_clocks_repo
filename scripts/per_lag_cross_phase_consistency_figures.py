#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for per_lag_cross_phase_consistency.py (plot result.json and
cells_q3.csv as is).

    cross_phase_consistency       Q2: per lag, across cells, rho of a cell's
                                  r in encode vs execute, and the split-half
                                  reliabilities that cap it
    lag_tuning_vs_ripple_firing   Q3: per lag, across cells, r of lag tuning
                                  (execute / encode / all repeats) with three
                                  ripple-firing measures
    claim2c_drivers               Q3: mPFC lag 0 x claim 2c's ripple contrast,
                                  and what carries the correlation

Filled marker = within-session permutation p < .05 (uncorrected); grey band
= +-1.96 / sqrt(n - 3), the range a correlation of this n reaches by chance.

    python scripts/per_lag_cross_phase_consistency_figures.py <folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import era_brewer

CM = 1 / 2.54
_p = era_brewer.era_brew("Showgirl2", n=7)
ROI_COL = {"HC_mid": _p[2], "mPFC": _p[1]}
ROI_L = {"HC_mid": "mid HC", "mPFC": "mPFC"}
LAGS = list(range(0, 360, 30))
ENC = "#3b3b45"


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False})


def ptxt(p):
    return "p<.001" if p < .001 else f"p={p:.3f}".replace("0.", ".")


def line(ax, recs, col, ls, lab, x_off=0):
    r = np.array([x.get("r", np.nan) for x in recs])
    p = np.array([x.get("p_perm", np.nan) for x in recs])
    xs = np.array(LAGS) + x_off
    ax.plot(xs, r, color=col, ls=ls, lw=1.3, label=lab, zorder=2)
    for x, v, pv in zip(xs, r, p):
        sig = np.isfinite(pv) and pv < .05
        ax.plot(x, v, "o", ms=4, mfc=col if sig else "white", mec=col,
                mew=1.0, zorder=3)


def null_band(ax, n):
    h = 1.96 / np.sqrt(n - 3)
    ax.axhspan(-h, h, color="0.92", lw=0, zorder=0)
    ax.axhline(0, color="0.3", lw=0.7, ls="--", zorder=1)


def save(fig, d, name):
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{e}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(d, name + ".png"))


def draw_q2(d, R):
    fig, axes = plt.subplots(1, 2, figsize=(18 * CM, 7 * CM), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    for ax, roi in zip(axes, ("HC_mid", "mPFC")):
        Q = R["Q2"]
        get = lambda pair: [Q[f"{roi} | {pair} | {l}"] for l in LAGS]
        null_band(ax, get("encode vs execute")[0]["n"])
        line(ax, get("encode vs execute"), ENC, "-", "encode vs execute")
        line(ax, get("encode vs execute_matched"), ENC, ":",
             "encode vs execute (encode loop count)")
        line(ax, get("execute_odd vs execute_even"), ROI_COL[roi], "-",
             "execute: odd vs even repeats")
        line(ax, get("encode_odd vs encode_even"), ROI_COL[roi], "--",
             "encode: odd vs even repeats")
        n = get("encode vs execute")[0]["n"]
        ax.set_title(f"{'ab'[axes.tolist().index(ax)]}  {ROI_L[roi]} "
                     f"({n} cells)", loc="left")
        ax.set_xticks(LAGS[::2])
        ax.set_xticklabels([f"{l}°" for l in LAGS[::2]])
        ax.set_xlabel("lag")
        ax.set_ylim(-0.35, 0.35)
    axes[0].set_ylabel("ρ across cells")
    h = [Line2D([], [], color=ENC, lw=1.3, label="encode vs execute"),
         Line2D([], [], color=ENC, ls=":", lw=1.3,
                label="encode vs execute, encode loop count"),
         Line2D([], [], color="0.45", lw=1.3,
                label="execute: odd vs even repeats (reliability)"),
         Line2D([], [], color="0.45", ls="--", lw=1.3,
                label="encode: odd vs even repeats (reliability)"),
         Line2D([], [], marker="o", ls="", color="0.3",
                label="within-session permutation p < .05")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=False)
    cap = ("Per cell, the published per-lag r (per_lag_encoding.py functions, "
           "noctrl) from encode loops, execute loops, the first n execute "
           "loops (n = encode loops) and odd vs even repeats of every "
           "configuration; Spearman ρ across cells per lag; p from permuting "
           "one variable within session (10,000). Grey band = ±1.96/√(n − 3). "
           "Two cell properties can correlate at most √(reliability of one × "
           "reliability of the other); the odd/even lines are those reliabilities.")
    fig.text(0.01, -0.25, textwrap.fill(cap, 160), fontsize=7.5, va="top",
             color="0.2")
    save(fig, d, "cross_phase_consistency")


def draw_q3(d, R):
    ys = (("storage_rip_minus_norip", "after first discoveries:\nripple − no ripple (claim 2c)"),
          ("ripmod_encode", "in ripples vs outside,\nwhole encode phase"),
          ("ripmod_execute", "in ripples vs outside,\nwhole execute phase"))
    xs = (("execute", ENC, "-", "tuning: execute (published)"),
          ("all", "#a30d6c", "-", "tuning: all repeats"),
          ("encode", "#9a9aa6", "--", "tuning: encode"))
    fig, axes = plt.subplots(2, 3, figsize=(19 * CM, 11 * CM), sharey=True,
                             sharex=True,
                             gridspec_kw=dict(hspace=0.45, wspace=0.1))
    for i, roi in enumerate(("HC_mid", "mPFC")):
        for j, (y, ylab) in enumerate(ys):
            ax = axes[i, j]
            recs = {x: [R["Q3"][f"{roi} | {x} | {y} | {l}"] for l in LAGS]
                    for x, *_ in xs}
            null_band(ax, recs["execute"][0]["n"])
            for k, (x, col, ls, lab) in enumerate(xs):
                line(ax, recs[x], col, ls, lab, x_off=(k - 1) * 4)
            if i == 0:
                ax.set_title(ylab, fontsize=8.5)
            ax.set_ylim(-0.35, 0.35)
            ax.set_xticks(LAGS[::3])
            ax.set_xticklabels([f"{l}°" for l in LAGS[::3]])
            if i == 1:
                ax.set_xlabel("lag of the tuning")
        axes[i, 0].set_ylabel(f"{ROI_L[roi]} (n = "
                              f"{R['Q3'][f'{roi} | execute | ripmod_encode | 0']['n']})"
                              "\nr across cells")
    h = [Line2D([], [], color=c, ls=ls, lw=1.3, label=lab)
         for _, c, ls, lab in xs] + [
        Line2D([], [], marker="o", ls="", color="0.3",
               label="within-session permutation p < .05")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, frameon=False)
    cap = ("Pearson r across cells (as claim 2c) of each cell's per-lag "
           "tuning with its firing around ripples. Left: claim 2c's measure "
           "(storage_vs_planning_2026-10-01): modulation 0.05–1.0 s after a "
           "first reward discovery with a ripple within 0.5 s minus without, "
           "against the cell's encode baseline (arrival-locked). Middle, "
           "right: (rate in ripples − rate > 1 s from any ripple) / sum over "
           "the whole phase (lag_group_ripple_2026-10-01). 72 correlations "
           "per region, uncorrected: about 3–4 filled markers per region are "
           "expected by chance.")
    fig.text(0.01, -0.1, textwrap.fill(cap, 170), fontsize=7.5, va="top",
             color="0.2")
    save(fig, d, "lag_tuning_vs_ripple_firing")


def draw_drivers(d, R):
    C = pd.read_csv(os.path.join(d, "cells_q3.csv"))
    D = R["drivers"]
    fig, axes = plt.subplots(1, 2, figsize=(20 * CM, 7.5 * CM),
                             gridspec_kw=dict(width_ratios=[1, 1.3],
                                              wspace=1.0))
    ax = axes[0]
    g = C[C.roi == "mPFC"].dropna(subset=["r0_execute",
                                          "storage_rip_minus_norip"])
    ax.scatter(g.r0_execute, g.storage_rip_minus_norip, s=9,
               color=ROI_COL["mPFC"], alpha=0.6, lw=0)
    out = g[g.neuron == D["execute"]["without the most influential cell"]
            ["dropped"]]
    ax.scatter(out.r0_execute, out.storage_rip_minus_norip, s=40,
               facecolor="none", edgecolor="k", lw=1.0,
               label="most influential cell")
    mis = g[g.roi_ripple_pipeline != "mPFC"]
    ax.scatter(mis.r0_execute, mis.storage_rip_minus_norip, s=40, marker="s",
               facecolor="none", edgecolor="#a30d6c", lw=1.0,
               label="labelled mOFC in the\nripple pipeline")
    b = np.polyfit(g.r0_execute, g.storage_rip_minus_norip, 1)
    xx = np.linspace(g.r0_execute.min(), g.r0_execute.max(), 5)
    ax.plot(xx, np.polyval(b, xx), color=ROI_COL["mPFC"], lw=1.4)
    ax.axhline(0, color="0.3", lw=0.7, ls="--")
    ax.axvline(0, color="0.3", lw=0.7, ls="--")
    ax.set_xlabel("lag-0 tuning (CV r, execute)")
    ax.set_ylabel("after first discoveries:\nripple − no ripple")
    pr = D["execute"]["pearson"]
    ax.set_title(f"a  mPFC, n = {pr['n']}: r = {pr['r']:+.2f}, "
                 f"{ptxt(pr['p_perm'])}", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1)
    ax = axes[1]
    rows = [("claim 2c as published\n(ripple-pipeline labels, n = 148)",
             R["labels"]["claim 2c, ripple-pipeline labels"], "p"),
            ("per-lag labels (n = 145)", D["execute"]["pearson"], "p_perm"),
            ("Spearman", D["execute"]["spearman"], "p_perm"),
            ("within session", D["execute"]["within-session (demeaned)"], "p_perm"),
            ("between sessions (n = 31)",
             D["execute"]["between-session (session means)"], "p_plain"),
            ("| firing rate", D["execute"]["partial | firing rate"], "p_perm"),
            ("| event counts", D["execute"]["partial | n events (rip, no rip)"],
             "p_perm"),
            ("without 1 cell", D["execute"]["without the most influential cell"],
             "p_perm"),
            ("tuning from all repeats", D["all"]["pearson"], "p_perm"),
            ("tuning from encode", D["encode"]["pearson"], "p_perm")]
    for i, (lab, r, pk) in enumerate(rows):
        n = r["n"]
        h = 1.96 / np.sqrt(n - 3)
        ax.barh(i, r["r"], 0.6, color=ROI_COL["mPFC"] if r[pk] < .05
                else "white", edgecolor=ROI_COL["mPFC"], lw=1.0)
        ax.plot([r["r"] - h, r["r"] + h], [i, i], color="0.15", lw=0.9)
        ax.text(1.02, i, ptxt(r[pk]), va="center", fontsize=7, color="0.4",
                transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_xlim(-0.35, 0.6)
    ax.set_xlabel("r (approx. 95% CI)")
    ls = D["execute"]["leave-one-session-out"]
    ax.set_title(f"b  what carries it (leave one session out: "
                 f"{ls['min']:+.2f} to {ls['max']:+.2f})", loc="left",
                 fontsize=9)
    cap = ("mPFC cells: lag-0 tuning (published r_lag000_noctrl) vs claim "
           "2c's ripple contrast after first discoveries. Rows 2–10 use the "
           "per-lag ROI labels; p from permuting within session, except the "
           "published row (plain Pearson) and between sessions (n = 31). "
           "'| x' = partial correlation; within session = session-demeaned; "
           "without 1 cell = largest Cook's distance removed. a: squares = the 3 cells "
           "the ripple pipeline labels mOFC; the published version instead included 6 "
           "cells the per-lag table labels mOFC.")
    fig.text(0.01, -0.3, textwrap.fill(cap, 175), fontsize=7.5, va="top",
             color="0.2")
    save(fig, d, "claim2c_drivers")


if __name__ == "__main__":
    _rc()
    d = sys.argv[1]
    R = json.load(open(os.path.join(d, "result.json")))
    draw_q2(d, R)
    draw_q3(d, R)
    draw_drivers(d, R)
