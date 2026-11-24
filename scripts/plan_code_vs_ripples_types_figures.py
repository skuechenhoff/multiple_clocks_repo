"""
Figure for `plan_code_vs_ripples_types.py` (B1b), from its saved outputs.

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
from plan_code_vs_ripples_types import SUBSETS, LAGS

SUB_C = {"L": "#448363", "E": "#5C1027", "B": "#7E6B8F"}
SHORT = {"L_all": "all learning\n(= B1)", "L_FT": "first\ntraversal",
         "L_rest": "rest of\nlearning", "L_postCorrect": "after\ncorrect",
         "L_postFD": "after first\ndiscovery", "L_postError": "after\nincorrect",
         "L_other": "not after\nan uncover", "E_first3": "execution,\nfirst 3 loops",
         "E_all": "whole\nexecution", "B_all": "whole\nblock"}


def _sub_dots(ax, S, R, col, rng):
    for i, sub in enumerate(SUBSETS):
        g = S[S.subset == sub]
        r = R[f"{sub} | {col}"]
        c = SUB_C[sub[0]]
        dots(ax, i, g[col].dropna(), c, rng=rng,
             marker="D" if sub == "L_all" else "o")
        lab = _p(r["p_perm"]) + (f"\nq={r['q_bh_family']:.2f}"
                                 if "q_bh_family" in r else "")
        ax.text(i, 1.01, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.8, color=c)
    zero(ax)
    ax.set_xticks(range(len(SUBSETS)), [SHORT[s] for s in SUBSETS],
                  fontsize=6.5)
    _clean(ax)


def draw(d):
    S = pd.read_csv(os.path.join(d, "session_betas.csv"))
    R = json.load(open(os.path.join(d, "result.json")))["tests"]
    rng = np.random.default_rng(SEED)
    fig, axes = layout([1, 2, 3, 3])
    fig.subplots_adjust(wspace=1.0)
    # a: the crude 'future on average' test for learning ripples
    ax = axes[0]
    g = S[S.subset == "L_all"]
    for i, (col, lab) in enumerate((("F_all", "code\n30–330°"),
                                    ("mean_beta_30_330", "mean β\n30–330°"),
                                    ("F_near", "code\n30–90°"),
                                    ("F_30-240", "code 30–240°\n(post hoc)"))):
        r = R[f"L_all | {col}"]
        dots(ax, i, g[col].dropna(), "#448363" if i < 3 else "#9A9A9A",
             rng=rng)
        ax.text(i, 1.01, _p(r["p_perm"]), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6)
    zero(ax)
    ax.set_xticks(range(4), ["code\n30–330°", "mean β\n30–330°",
                             "code\n30–90°", "code\n30–240°\n(post hoc)"],
                  fontsize=5.8)
    ax.set_ylabel("ripple β per session\n(learning ripples, mPFC)")
    ax.set_title("a  future lags collapsed", pad=18)
    _clean(ax)
    # b: heatmap subset x lag (mean beta), * p < .05
    ax = axes[1]
    M = np.array([[R[f"{sub} | {l}"].get("mean", np.nan) for l in LAGS]
                  for sub in SUBSETS])
    Pm = np.array([[R[f"{sub} | {l}"].get("p_perm", 1) for l in LAGS]
                   for sub in SUBSETS])
    v = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if Pm[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(LAGS)), [f"{l}" for l in LAGS], fontsize=6.5)
    ax.set_yticks(range(len(SUBSETS)),
                  [SHORT[s].replace("\n", " ") for s in SUBSETS], fontsize=6)
    ax.set_xlabel("lag of the plan code (°)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean ripple β", fontsize=7)
    ax.set_title("b  which ripples, which lag? (* p < .05, uncorrected)",
                 pad=10)
    # c: 60 deg per subset
    ax = axes[2]
    _sub_dots(ax, S, R, "60", rng)
    ax.set_ylabel("ripple β per session\n(mPFC 60° code)")
    ax.set_title("c  60° code, by ripple type (q = BH over the 20-test family)",
                 pad=22)
    # d: collapsed future per subset
    ax = axes[3]
    _sub_dots(ax, S, R, "F_all", rng)
    ax.set_ylabel("ripple β per session\n(mPFC code, 30–330° mean)")
    ax.set_title("d  all future lags collapsed (30–330°), by ripple type",
                 pad=22)
    fig.suptitle("B1b (exploratory) — future lags collapsed, and which "
                 "ripples carry the B1 relation", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", y=1.0)
    caption(fig, "mPFC, own reading. Per block, the plan code of B1 (first 3 "
            "execution loops; mean Fisher z over cells), and per ripple type "
            "the count in that block. Per session, OLS of the code on "
            "z-scored log(1 + count) with log duration of the window the "
            "ripples come from, number of learning loops, block order and run "
            "index; dots = sessions' ripple β (diamond = all learning ripples, the "
            "B1 predictor), p = sign-flip over sessions. Green = learning "
            "ripples by context (pause = press to min(stillness, 2 s)); "
            "bordeaux = execution ripples; purple = the whole block. a: the "
            "code averaged over the 11 future lags (30–330°, the manuscript's "
            "'beyond-now' set) before the regression; the mean of the 11 "
            "per-lag βs; 30–90°; and 30–240°, the range read off B1 panel d "
            "(post hoc, circular). Per-lag βs share blocks and cells, so they "
            "are not independent tests.", below=0.02)
    save(fig, d, "B1b_future_and_ripple_types")
