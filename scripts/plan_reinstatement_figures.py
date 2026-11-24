#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the first-run D - A effect in medial frontal LFP specific to the ripple band?

    python scripts/plan_reinstatement_figures.py

Reads the per-session tables written by `cells_plan_reinstatement.py` (one run
per signal) and shows, for the medial frontal contacts:
    a  Δ at each discovery A-D, first run of each layout: ripple band vs HFB
    b  first-run D - A per session, every signal (+ mPFC spikes)
    c  the pre-registered primary (all runs) D - A, every signal
    d  positive control (execute Δ, all runs), every signal
    e  per-session first-run D - A: ripple band vs HFB
    f  ripple band first-run D - A by region (MedialFrontal / HPC / Visual)

Statistics are computed first, with the same `signflip_t` as the analysis, and
written to `figure_stats.json`; the plotting code only displays them.

@author: Svenja Kuchenhoff
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.plan_reinstatement as pr
from mc.plotting.cell_results import STATE_QUADRANT_COLORS

DATE = "2026-10-01"
GROUP = os.path.join(rrsa._derivatives(), "group")
OUT = os.path.join(GROUP, f"plan_reinstatement_figures_{DATE}")
os.makedirs(OUT, exist_ok=True)

# one colour per signal, from era_brewer Showgirl2; spikes in neutral grey
SIGNALS = {"theta": ("theta\n4-8 Hz", "#CCB178"),
           "beta": ("beta\n13-30 Hz", "#7BB594"),
           "hfb": ("HFB\n70-150 Hz", "#448363"),
           "ripple": ("ripple band\n80-120 Hz", "#B74C2D")}
SPIKES = ("mPFC\nspikes", "#4F4F4F")
REGIONS = ["MedialFrontal", "HPC", "Visual"]


def table(signal):
    return pd.read_csv(os.path.join(
        GROUP, f"plan_reinstatement_{signal}_{DATE}", "per_session.csv"))


# ---------------------------------------------------------------------------
# data and statistics
T = {s: table(s) for s in list(SIGNALS) + ["spikes"]}
MF = {s: T[s][T[s].roi == "MedialFrontal"] for s in SIGNALS}
MF["spikes"] = T["spikes"][T["spikes"].roi == "mPFC"]

stat = {}
for s, g in MF.items():
    for col in ("D_minus_A_run0", "D_minus_A_all", "execute_all_mean",
                "execute_run0_mean"):
        stat[f"{s}|{col}"] = pr.signflip_t(g[col])
for r in REGIONS:
    g = T["ripple"][T["ripple"].roi == r]
    stat[f"ripple|{r}|D_minus_A_run0"] = pr.signflip_t(g.D_minus_A_run0)
pair = MF["ripple"][["session", "D_minus_A_run0"]].merge(
    MF["hfb"][["session", "D_minus_A_run0"]], on="session",
    suffixes=("_ripple", "_hfb")).dropna()
r_, p_ = stats.pearsonr(pair.D_minus_A_run0_ripple, pair.D_minus_A_run0_hfb)
stat["ripple_vs_hfb_run0_pearson"] = dict(n=int(len(pair)), r=float(r_),
                                          p_two_sided=float(p_))
json.dump(stat, open(os.path.join(OUT, "figure_stats.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
# plotting
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def ptxt(d):
    p = d["p_one_sided"]
    return f"t = {d['t']:+.2f}\nP = {p:.3f}" if p >= .001 else \
        f"t = {d['t']:+.2f}\nP < .001"


def strip(ax, keys, col, title, ylabel):
    for i, s in enumerate(keys):
        lab, c = SPIKES if s == "spikes" else SIGNALS[s]
        v = MF[s][col].dropna().to_numpy()
        jit = np.random.default_rng(pr.SEED).uniform(-0.15, 0.15, len(v))
        ax.scatter(i + jit, v, s=10, color=c, alpha=0.35, lw=0)
        d = stat[f"{s}|{col}"]
        ax.errorbar(i, d["mean"], d["sem"], fmt="o", ms=7, color=c,
                    mec="k", mew=0.6, capsize=3, zorder=3)
        ax.text(i, 1.0, ptxt(d), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([SPIKES[0] if s == "spikes" else SIGNALS[s][0]
                        for s in keys], fontsize=8)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=34, fontsize=10)


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
keys = list(SIGNALS) + ["spikes"]

# a -- Δ by discovered state, first run, ripple band vs HFB
ax = axes[0, 0]
x = np.arange(4)
for s, dx in (("hfb", -0.08), ("ripple", 0.08)):
    lab, c = SIGNALS[s]
    M = MF[s][[f"discovery_run0_{k}" for k in pr.STATES]].to_numpy()
    n = np.isfinite(M).sum(axis=0)
    mu = np.nanmean(M, axis=0)
    se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(n)
    ax.plot(x + dx, mu, color=c, lw=1.5, label=lab.replace("\n", " "))
    for k in range(4):
        ax.errorbar(x[k] + dx, mu[k], se[k], fmt="o", ms=7, capsize=3,
                    color=c, mfc=STATE_QUADRANT_COLORS[k], mec=c, mew=1.5)
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{k}\n{i} known" for i, k in enumerate(pr.STATES)])
ax.set_xlabel("reward discovered (future rewards known)")
ax.set_ylabel("Δ similarity, same − other layout\n(z·z / contact)")
ax.set_title("a  medial frontal LFP, first run of each layout\n"
             "(marker fill = state; mean ± SEM over sessions)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")

strip(axes[0, 1], keys, "D_minus_A_run0",
      "b  first run only: D − A (secondary)", "D − A of Δ, per session")
strip(axes[0, 2], keys, "D_minus_A_all",
      "c  all runs: D − A (pre-registered primary;\n"
      "Bonferroni over 4 signals: P < .0125)", "D − A of Δ, per session")
strip(axes[1, 0], keys, "execute_all_mean",
      "d  positive control: execution Δ, all runs\n"
      "(is there a layout-specific pattern at all?)", "execute Δ, per session")

# e -- ripple vs HFB, per session
ax = axes[1, 1]
ax.scatter(pair.D_minus_A_run0_hfb, pair.D_minus_A_run0_ripple, s=18,
           color=SIGNALS["ripple"][1], alpha=0.7, lw=0)
lim = np.nanmax(np.abs(pair[["D_minus_A_run0_hfb",
                             "D_minus_A_run0_ripple"]].to_numpy())) * 1.1
ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, ls="--")
ax.axhline(0, color="0.4", lw=0.6)
ax.axvline(0, color="0.4", lw=0.6)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
d = stat["ripple_vs_hfb_run0_pearson"]
ax.set_title(f"e  same sessions, first-run D − A\n"
             f"r = {d['r']:+.2f}, P = {d['p_two_sided']:.1e} (n = {d['n']})",
             fontsize=10)
ax.set_xlabel("HFB 70-150 Hz: D − A")
ax.set_ylabel("ripple band 80-120 Hz: D − A")

# f -- ripple band, by region
ax = axes[1, 2]
for i, r in enumerate(REGIONS):
    v = T["ripple"][T["ripple"].roi == r].D_minus_A_run0.dropna().to_numpy()
    jit = np.random.default_rng(pr.SEED).uniform(-0.15, 0.15, len(v))
    c = SIGNALS["ripple"][1] if r == "MedialFrontal" else "0.55"
    ax.scatter(i + jit, v, s=10, color=c, alpha=0.35, lw=0)
    d = stat[f"ripple|{r}|D_minus_A_run0"]
    ax.errorbar(i, d["mean"], d["sem"], fmt="o", ms=7, color=c, mec="k",
                mew=0.6, capsize=3, zorder=3)
    ax.text(i, 1.0, ptxt(d) + f"\nn = {d['n']}",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=8)
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(range(len(REGIONS)))
ax.set_xticklabels(["medial\nfrontal", "hippocampus", "visual\n(confound check)"],
                   fontsize=8)
ax.set_xlim(-0.6, len(REGIONS) - 0.4)
ax.set_ylabel("D − A of Δ, per session")
ax.set_title("f  ripple band, first run: D − A by region", pad=44,
             fontsize=10)

fig.suptitle("Plan reinstatement in medial frontal LFP: is the first-run D − A "
             "effect specific to 80-120 Hz?\nMedialFrontal = mPFC + mOFC "
             "contacts within |MNI x| ≤ 20 mm (no lateral OFC). One-sided "
             "sign-flip P, n = sessions. EXPLORATORY.", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "plan_reinstatement_band_specificity.jpeg"),
            dpi=200)
fig.savefig(os.path.join(OUT, "plan_reinstatement_band_specificity.pdf"))
print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                  for k, v in stat.items()}, indent=1))
print(f"-> {OUT}")
plt.show()
