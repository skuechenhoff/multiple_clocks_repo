#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why does the F2 lag profile (2026-11-14) look cleaner than the published one?
(2026-11-23, descriptive)

Same mPFC cells, same sessions, execution loops, session unit (mean Fisher z
over a session's cells), two estimators:
    published  per_lag_encoding.py: repeats averaged per configuration, per-bin
               MODE location (per_lag_encode_vs_execute_2026-11-23, 'execute';
               identical to the published r_lag*_noctrl)
    F2 pooled  per_lag_encoding_by_phase.py: every (repeat, bin) pooled, each
               repeat's OWN location (2026-11-14 run, r_exe)
and two error-bar definitions (+-SEM as in F2, 95% t-interval as in the new
figures). Plus the per-cell split-half reliability (odd vs even execution
repeats) of each estimator, a criterion that does not look at the effect.

    python scripts/per_lag_estimator_comparison.py

@author: Svenja Kuchenhoff
"""
import os
import json
import datetime
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

GROUP = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
F2 = os.path.join(GROUP, "per_lag_encoding_by_phase", "2026-11-14_18-29-12",
                  "per_cell.csv")
PUB = os.path.join(GROUP, "per_lag_encode_vs_execute_2026-11-23", "per_cell.csv")
SPLIT = os.path.join(GROUP, "per_lag_cross_phase_2026-11-23", "per_cell_extra.csv")
OUT = os.path.join(GROUP, f"per_lag_estimator_comparison_{datetime.date.today()}")
LAGS = list(range(0, 360, 30))
CM = 1 / 2.54
COL = {"published": "#3D7A5E", "F2 pooled": "#D7657F"}


def fz(r):
    return np.arctanh(np.clip(r, -0.9999999, 0.9999999))


def session_profile(X, sess):
    Z = pd.DataFrame(fz(X), columns=LAGS)
    Z["s"] = sess
    S = Z.groupby("s").mean()
    n = S.notna().sum().to_numpy()
    m, se = S.mean().to_numpy(), S.sem().to_numpy()
    t, p = stats.ttest_1samp(S, 0, nan_policy="omit")
    return dict(mean=m.tolist(), sem=se.tolist(), n=n.tolist(),
                ci_half=(stats.t.ppf(.975, n - 1) * se).tolist(),
                p_two_sided=np.asarray(p).tolist())


def main():
    os.makedirs(OUT, exist_ok=True)
    f = pd.read_csv(F2)
    f = f[f.roi == "mPFC"].set_index("neuron")
    P = pd.read_csv(PUB)
    P = P[(P.set == "execute") & (P.roi == "mPFC")].set_index("neuron")
    both = f.index.intersection(P.index)
    sess = P.loc[both, "subject_id"].to_numpy()
    X = {"published": P.loc[both, [f"r_lag{l:03d}" for l in LAGS]].to_numpy(),
         "F2 pooled": f.loc[both, [f"r_exe_lag{l:03d}" for l in LAGS]].to_numpy()}
    res = {k: session_profile(v, sess) for k, v in X.items()}
    res["cells"] = int(len(both))
    res["sessions"] = int(len(np.unique(sess)))
    res["rho_between_estimators_per_lag"] = [
        float(stats.spearmanr(X["published"][:, j], X["F2 pooled"][:, j],
                              nan_policy="omit")[0]) for j in range(12)]
    # split-half reliability of the per-cell r
    sp = pd.read_csv(SPLIT)
    sp = sp[sp.roi == "mPFC"]
    o = sp[sp.set == "execute_odd"].set_index("neuron")
    e = sp[sp.set == "execute_even"].set_index("neuron")
    res["split_half_published"] = {
        l: float(stats.spearmanr(o[f"r_lag{l:03d}"], e.loc[o.index, f"r_lag{l:03d}"],
                                 nan_policy="omit")[0]) for l in LAGS}
    res["split_half_F2_pooled"] = {
        l: float(stats.spearmanr(f[f"r_exe_half1_lag{l:03d}"],
                                 f[f"r_exe_half2_lag{l:03d}"],
                                 nan_policy="omit")[0]) for l in (0, 60)}
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(19 * CM, 6.5 * CM),
                             gridspec_kw=dict(wspace=0.55,
                                              width_ratios=[1, 1, 0.9]))
    for ax, (band, lab) in zip(axes[:2], (("sem", "± SEM (as F2)"),
                                          ("ci_half", "95% CI (new figures)"))):
        for k in ("published", "F2 pooled"):
            m, h = np.array(res[k]["mean"]), np.array(res[k][band])
            ax.plot(LAGS, m, color=COL[k], marker="o", ms=3, lw=1.5, label=k)
            ax.fill_between(LAGS, m - h, m + h, color=COL[k], alpha=0.18, lw=0)
        ax.axhline(0, color="0.3", lw=0.7, ls="--")
        ax.set_xticks(LAGS[::3])
        ax.set_xlabel("lag (°)")
        ax.set_ylim(-0.13, 0.17)
        ax.set_title(f"{'ab'[axes.tolist().index(ax)]}  mPFC execute\n{lab}",
                     loc="left", fontsize=9)
    axes[0].set_ylabel("session mean Fisher z")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[2]
    sh = res["split_half_published"]
    ax.bar(np.arange(12), [sh[l] for l in LAGS], 0.7, color=COL["published"],
           label="published")
    sf = res["split_half_F2_pooled"]
    ax.plot([0, 2], [sf[0], sf[60]], "D", color=COL["F2 pooled"], ms=6,
            label="F2 pooled (0°, 60° only)")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xticks(range(0, 12, 3)); ax.set_xticklabels(LAGS[::3])
    ax.set_xlabel("lag (°)")
    ax.set_ylabel("ρ, odd vs even repeats")
    ax.set_title("c  split-half reliability\nper cell (n = 155)", loc="left",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    cap = ("Same 155 mPFC cells and 32 sessions, execution loops. published "
           "= per_lag_encoding.py (repeats averaged, mode location); F2 pooled "
           "= per_lag_encoding_by_phase.py (each repeat's own location). Lines "
           "= mean over sessions of the session-mean Fisher z; bands = ± SEM "
           "(a) or 95% t-interval (b). c: Spearman across cells of r from odd "
           "vs even execution repeats of every configuration.")
    fig.text(0.01, -0.12, textwrap.fill(cap, 165), fontsize=7.5, va="top",
             color="0.2")
    for e_ in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"estimator_comparison.{e_}"), dpi=300,
                    bbox_inches="tight")
    print(json.dumps({k: res[k] for k in ("split_half_F2_pooled", "cells",
                                          "sessions")}))


if __name__ == "__main__":
    main()
