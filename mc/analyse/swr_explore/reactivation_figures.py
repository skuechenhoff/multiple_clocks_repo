"""
Figure for `reactivation` (C1), drawn from its saved outputs.

@author: Svenja Kuchenhoff
"""

import os
import sys
import json

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
from plot_gradient_claims import (plt, layout, caption, save, dots, zero,
                                  _clean, _p, COL, SEED)

READ_C = {"own": "#0e3d3a", "matched": "#a30d6c"}
READ_L = {"own": "own flanks", "matched": "stillness-matched"}
TPL_L = {"own": "own block's\nexecution", "other": "other configs'\nexecution",
         "specific": "specific\n(own − other)", "all": "all execution\n(general)"}
ROIS = ("mPFC", "mOFC", "mPFC+mOFC", "HC_all")
RC = dict(COL, **{"mPFC+mOFC": "#7E6B8F"})


def _pair_dots(ax, T, R, roi, ripples, tpls, rng, x0=0.0, step=2.5):
    for i, k in enumerate(tpls):
        for j, rd in enumerate(("own", "matched")):
            g = T[(T.roi == roi) & (T.ripples == ripples) & (T.template == k)
                  & (T.reading == rd)]
            r = R.get(f"{roi} | {ripples} | {k} | {rd}", {})
            if not len(g) or not r.get("n"):
                continue
            x = x0 + i * step + j
            dots(ax, x, g.value.dropna(), READ_C[rd], rng=rng)
            ax.text(x, 1.0 + 0.07 * j, _p(r["p_perm"]),
                    transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=5.8, color=READ_C[rd])


def draw(d):
    T = pd.read_csv(os.path.join(d, "per_session.csv"))
    R = json.load(open(os.path.join(d, "result.json")))["tests"]
    rng = np.random.default_rng(SEED)
    fig, axes = layout([2, 1, 2, 1])
    # a: mPFC, learning ripples, every template
    ax = axes[0]
    tpls = ("own", "other", "specific", "all")
    _pair_dots(ax, T, R, "mPFC", "learning", tpls, rng)
    zero(ax)
    ax.set_xticks([i * 2.5 + 0.5 for i in range(len(tpls))],
                  [TPL_L[k] for k in tpls], fontsize=6.5)
    for rd in READ_C:
        ax.plot([], [], "o", color=READ_C[rd], label=f"ripple − {READ_L[rd]}")
    ax.legend(fontsize=6.3, frameon=False, loc="lower left")
    n = R.get("mPFC | learning | specific | matched", {}).get("n")
    ax.set_ylabel("reactivation strength,\nripple − control (a.u.)")
    ax.set_title(f"a  mPFC ({n} sessions), learning ripples, by template",
                 pad=22)
    _clean(ax)
    # b: the primary contrast, per ROI
    ax = axes[1]
    for i, roi in enumerate(ROIS):
        for j, rd in enumerate(("own", "matched")):
            g = T[(T.roi == roi) & (T.ripples == "learning")
                  & (T.template == "specific") & (T.reading == rd)]
            r = R.get(f"{roi} | learning | specific | {rd}", {})
            if not r.get("n"):
                continue
            x = i * 2.5 + j
            dots(ax, x, g.value.dropna(), READ_C[rd], rng=rng)
            ax.text(x, 1.0 + 0.07 * j, _p(r["p_perm"]),
                    transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=5.8, color=READ_C[rd])
    zero(ax)
    ax.set_xticks([i * 2.5 + 0.5 for i in range(len(ROIS))],
                  [r.replace("_all", "").replace("+", "+\n") for r in ROIS],
                  fontsize=6.5)
    ax.set_ylabel("specific reactivation\n(own − other configs)")
    ax.set_title("b  configuration-specific,\nlearning ripples, by region",
                 pad=22)
    _clean(ax)
    # c: general reactivation by region
    ax = axes[2]
    for i, roi in enumerate(ROIS):
        _pair_dots(ax, T, R, roi, "learning", ("all",), rng, x0=i * 2.5)
    zero(ax)
    ax.set_xticks([i * 2.5 + 0.5 for i in range(len(ROIS))],
                  [r.replace("_all", "") for r in ROIS], fontsize=7)
    ax.set_ylabel("general reactivation\n(all execution co-firing)")
    ax.set_title("c  do learning ripples replay execution co-firing at all?",
                 pad=22)
    _clean(ax)
    # d: ripples after a first discovery
    ax = axes[3]
    for i, roi in enumerate(("mPFC", "mPFC+mOFC", "HC_all")):
        _pair_dots(ax, T, R, roi, "post_FD", ("specific",), rng, x0=i * 2.5)
    zero(ax)
    ax.set_xticks([0.5, 3.0, 5.5], ["mPFC", "mPFC+\nmOFC", "HC"], fontsize=6.5)
    ax.set_ylabel("specific reactivation")
    ax.set_title("d  only ripples after a\nfirst discovery", pad=22)
    _clean(ax)
    fig.suptitle("C1 — do learning ripples reactivate the co-firing of the "
                 "block's later execution?", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.06)
    caption(fig, "No tuning curves, maps or cell classes: per session, every "
            "cell's spike counts in 250 ms bins (z-scored with session-wide "
            "statistics); template = cell × cell correlation over a block's "
            "EXECUTION bins (bins within 0.5 s of a ripple dropped, diagonal "
            "0). Reactivation strength R = zᵀCz / (n(n−1)) for the 250 ms "
            "window centred on each ripple in the block's LEARNING window "
            "(Peyrache et al. 2009). 'own block' uses that block's template, "
            "'other configs' the mean template of blocks with a different "
            "configuration, 'specific' their difference (co-firing belonging "
            "to this configuration), 'all execution' the session's pooled "
            "execution template. Readings: ripple minus its own flanks (±0.5–"
            "0.75 s) or minus ripple-free moments of the same learning window "
            "with the same time since the last and to the next press. Dots = "
            "sessions (≥ 3 cells); p = sign-flip over sessions, uncorrected; "
            "primary = mPFC, specific, stillness-matched.", below=0.02)
    save(fig, d, "C1_reactivation")
