"""
Figure for `per_lag_encoding_by_phase.py` (F2), drawn from its saved outputs.

    from per_lag_encoding_by_phase_figures import draw
    draw(out_dir)

@author: Svenja Kuchenhoff
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "exploratory"))
from plot_gradient_claims import (plt, layout, caption, save, dots, zero,
                                  _clean, _p, COL, SEED)

MEAS_C = {"r_exe": "#5C1027", "r_disc": "#D7657F", "r_x": "#0e3d3a",
          "r_x_n": "#9ec6b8"}
MEAS_L = {"r_exe": "execution → execution",
          "r_disc": "discovery → discovery",
          "r_x": "execution → discovery",
          "r_x_n": "execution → execution,\nsame n repeats"}
GROUPS = ("mPFC", "mOFC", "HC")


def _profile(ax, S, R, g, letter):
    for m in ("r_exe", "r_disc"):
        s = S[(S.group == g) & (S.measure == m)]
        mu = s.groupby("lag").value.mean()
        se = s.groupby("lag").value.sem()
        ax.plot(mu.index, mu, color=MEAS_C[m], lw=1.4, marker="o", ms=2.5,
                label=MEAS_L[m])
        ax.fill_between(mu.index, mu - se, mu + se, color=MEAS_C[m],
                        alpha=0.2, lw=0)
        for lag in mu.index:
            p = R[f"{g} | {m} | lag {lag}"]["session"].get("p_perm", 1)
            if p < 0.05:
                ax.text(lag, mu[lag] + se[lag], "*", ha="center",
                        color=MEAS_C[m], fontsize=9)
    zero(ax)
    ax.set_xticks([0, 90, 180, 270])
    ax.set_xlabel("future lag (°)")
    ax.set_ylabel("spatial consistency\n(CV r, Fisher z)")
    n = R[f"{g} | r_exe | lag 0"]["session"]["n"]
    ax.set_title(f"{letter}  {g}: lag profile by phase\n({n} sessions)",
                 pad=12)
    _clean(ax)


def draw(d):
    T = pd.read_csv(os.path.join(d, "per_cell.csv"))
    S = pd.read_csv(os.path.join(d, "per_session.csv"))
    R = json.load(open(os.path.join(d, "result.json")))["tests"]
    fig, axes = layout([1, 1, 1, 2, 1, 2, 1])
    for ax, g, letter in zip(axes[:3], GROUPS, "abc"):
        _profile(ax, S, R, g, letter)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    # d: lag 0, the four estimates per group
    ax = axes[3]
    rng = np.random.default_rng(SEED)
    for j, g in enumerate(GROUPS):
        for i, m in enumerate(("r_exe", "r_disc", "r_x", "r_x_n")):
            s = S[(S.group == g) & (S.measure == m) & (S.lag == 0)]
            p = R[f"{g} | {m} | lag 0"]["session"].get("p_perm")
            dots(ax, j * 5 + i, s.value, MEAS_C[m], p, rng=rng,
                 py=1.0 + 0.07 * (i % 2))
        ch = R[f"{g} | change | lag 0"]["session"]
        ax.text(j * 5 + 2.5, -0.33, f"change {ch['mean']:+.3f}, "
                f"{_p(ch['p_perm'])}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=6.5)
        ax.text(j * 5 + 1.5, -0.2, g, transform=ax.get_xaxis_transform(),
                ha="center", color=COL.get(g, "k"), fontsize=8)
    for m in MEAS_L:
        ax.plot([], [], "o", color=MEAS_C[m], label=MEAS_L[m])
    ax.legend(fontsize=6.3, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.45, -0.4))
    zero(ax)
    ax.set_xticks([])
    ax.set_ylabel("lag-0 consistency\n(CV r, Fisher z)")
    ax.set_title("d  lag 0 within and across phases (dots = sessions)",
                 pad=22)
    _clean(ax)
    # e: same cells, both phases (independent repeats)
    ax = axes[4]
    for g in ("mPFC", "mOFC"):
        G = T[T.roi == g]
        ax.scatter(G.r_exe_lag000, G.r_disc_lag000, s=7, color=COL[g],
                   alpha=0.6, lw=0, label=g)
    r = R["mPFC | r_disc vs r_exe, lag 0 (across cells)"]
    r2 = R["mOFC | r_disc vs r_exe, lag 0 (across cells)"]
    zero(ax)
    ax.axvline(0, color="k", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel("lag-0 CV r, execution")
    ax.set_ylabel("lag-0 CV r, discovery")
    ax.set_title(f"e  same cells, both phases\nmPFC ρ = {r['rho']:+.2f}, "
                 f"{_p(r['p_perm'])}; mOFC ρ = {r2['rho']:+.2f}, "
                 f"{_p(r2['p_perm'])}", pad=12, fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    _clean(ax)
    # f: the change index at every lag
    ax = axes[5]
    for g in GROUPS:
        s = S[(S.group == g) & (S.measure == "change")]
        mu = s.groupby("lag").value.mean()
        se = s.groupby("lag").value.sem()
        c = COL.get(g, "k")
        ax.plot(mu.index, mu, color=c, lw=1.4, marker="o", ms=2.5, label=g)
        ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.15, lw=0)
        for lag in mu.index:
            p = R[f"{g} | change | lag {lag}"]["session"].get("p_perm", 1)
            if p < 0.05:
                ax.text(lag, mu[lag] + se[lag] + 0.003, f"* {_p(p)}",
                        ha="center", color=c, fontsize=6.5)
    zero(ax)
    ax.set_xticks(range(0, 360, 30))
    ax.set_xlabel("future lag (°)")
    ax.set_ylabel("change index (r)\n> 0: discovery departs")
    ax.set_title("f  does discovery firing follow the execution tuning? "
                 "(change per lag)", pad=12)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=3)
    _clean(ax)
    # g: is 'lag-0 cell' a stable property? odd vs even execution repeats
    ax = axes[6]
    lab = []
    for g, members in (("mPFC", ["mPFC"]), ("mOFC", ["mOFC"]),
                       ("HC", ["HC_anterior", "HC_mid"])):
        G = T[T.roi.isin(members)]
        ax.scatter(G.r_exe_half1_lag000, G.r_exe_half2_lag000, s=5,
                   color=COL.get(g, "k"), alpha=0.5, lw=0, label=g)
        r = R[f"{g} | reliability: odd vs even execution repeats, lag 0"]
        lab.append(f"{g} ρ = {r['rho']:+.2f}, {_p(r['p_perm'])}")
    zero(ax)
    ax.axvline(0, color="k", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel("lag-0 CV r, odd execution repeats")
    ax.set_ylabel("lag-0 CV r, even repeats")
    ax.set_title("g  is a cell's lag-0 tuning reliable?\n" + "\n".join(lab),
                 pad=12, fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", markerscale=2)
    _clean(ax)
    fig.suptitle("F2 — do cells change their lag tuning between discovery "
                 "and execution?", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.0)
    caption(fig, "Per cell, the per-lag rate-map estimator of the manuscript "
            "(leave-one-configuration-out, dwell-weighted r, phase "
            "residualised), with rate maps pooled over each configuration's "
            "repeats. Discovery = the first traversal and every repeat before "
            "the first fully correct one; execution = the rest (the published "
            "set). a–c: mean ± SEM over sessions of the session-mean Fisher z; "
            "* sign-flip p < .05, uncorrected. d: lag 0. 'execution → "
            "discovery' = execution maps of the other configurations predict "
            "the held-out configuration's discovery loops; 'same n repeats' = "
            "the same training maps predict as many execution repeats as that "
            "configuration had discovery repeats (50 draws). change = the "
            "second minus the first, on the same folds: > 0 means discovery "
            "firing departs from the execution tuning. e: per cell, lag-0 r in "
            "each phase (independent repeats); ρ across cells, null = "
            "execution r permuted within session. f: the change index at "
            "every lag, mean ± SEM over sessions; * sign-flip p < .05, "
            "uncorrected over 12 lags. g: the same cell's execution lag-0 r "
            "from odd vs even execution repeats of every configuration — the "
            "ceiling for any analysis that sorts cells by lag-0 tuning.",
            below=0.03)
    save(fig, d, "F2_tuning_by_phase")
