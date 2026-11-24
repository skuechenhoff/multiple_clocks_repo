"""
Figure for `swr_hfb_direction_context.py` (D1, D2), from its saved outputs.

@author: Svenja Kuchenhoff
"""

import os
import json
import pickle

import numpy as np
import pandas as pd

from plot_gradient_claims import (plt, layout, caption, save, dots, zero,
                                  _clean, _p, COL, SEED)
from swr_hfb_direction_context import prepared, ROIS, SETS, SET_L

SET_STYLE = {"U_L": ("#D7657F", "-"), "U_E": ("#5C1027", "-"),
             "A_L": ("#D7657F", "--"), "A_E": ("#5C1027", "--")}
READ_C = {"own": "#0e3d3a", "matched": "#a30d6c"}


def draw(d):
    D = pickle.load(open(os.path.join(d, "d1_curves.pkl"), "rb"))
    R1 = json.load(open(os.path.join(d, "d1_result.json")))["tests"]
    V = pd.read_csv(os.path.join(d, "d1_session_values.csv"))
    B = pd.read_csv(os.path.join(d, "d2_betas.csv"))
    R2 = json.load(open(os.path.join(d, "d2_result.json")))
    Ev = pd.read_csv(os.path.join(d, "d2_events.csv"))
    rng = np.random.default_rng(SEED)
    fig, axes = layout([1, 1, 1, 1, 1, 1, 2, 1])
    fig.subplots_adjust(hspace=1.75)
    t_ms = D["t_ms"]
    # row 1: time courses, own reading
    for j, roi in enumerate(ROIS):
        ax = axes[j]
        for key in SETS:
            C = prepared(D["curves"], t_ms, key, roi, "own")
            if len(C) < 5:
                continue
            X = np.stack(list(C.values()))
            m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
            col, ls = SET_STYLE[key]
            ax.plot(t_ms, m, color=col, ls=ls, lw=1.3,
                    label=SET_L[key])
            ax.fill_between(t_ms, m - se, m + se, color=col, alpha=0.1, lw=0)
        zero(ax)
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.axvspan(-250, 0, color="0.93", lw=0, zorder=0)
        ax.set_xlim(-750, 750)
        ax.set_xlabel("time from ripple peak (ms)")
        ax.set_title(f"{'abc'[j]}  {roi}: ripple-locked HFB",
                     pad=8, color=COL[roi], fontsize=8.5)
        if j == 0:
            ax.legend(fontsize=6, frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, -0.3), ncol=2)
        _clean(ax)
    axes[0].set_ylabel("HFB (z, real − shifted)")
    # row 2: direction index per set, both readings
    for j, roi in enumerate(ROIS):
        ax = axes[3 + j]
        for i, key in enumerate(SETS):
            for k, rd in enumerate(("own", "matched")):
                g = V[(V.roi == roi) & (V.set == key) & (V.reading == rd)]
                r = R1.get(f"{roi} | {rd} | {key}", {}).get("direction", {})
                if not len(g) or not r.get("n"):
                    continue
                x = i * 2.5 + k
                dots(ax, x, g.direction, READ_C[rd], rng=rng)
                ax.text(x, 1.0 + 0.07 * k, _p(r["p_perm"]),
                        transform=ax.get_xaxis_transform(), ha="center",
                        fontsize=5.8, color=READ_C[rd])
        zero(ax)
        ax.set_xticks([i * 2.5 + 0.5 for i in range(len(SETS))],
                      [SET_L[k].replace(", ", "\n") for k in SETS], fontsize=6)
        pr = R1.get(f"{roi} | own | U_L - A_E"
                    + (" PRIMARY" if roi != "mPFC" else ""), {})
        ax.set_title(f"{'def'[j]}  {roi}: direction index\nU_L − A_E "
                     f"{pr.get('mean', np.nan):+.3f}, {_p(pr.get('p_perm', 1))}"
                     f"{' (primary)' if roi != 'mPFC' else ''}", pad=20,
                     fontsize=8, color=COL[roi])
        _clean(ax)
    axes[3].set_ylabel("post − pre ±250 ms\n(< 0: cortex leads)")
    for rd in READ_C:
        axes[5].plot([], [], "o", color=READ_C[rd], label=rd)
    axes[5].legend(fontsize=6, frameon=False, loc="lower right")
    # row 3: D2 betas
    ax = axes[6]
    for i, roi in enumerate(ROIS):
        for j, sub in enumerate(("learning", "execution")):
            for k, rd in enumerate(("own", "stillness")):
                g = B[(B.roi == roi) & (B.subset == sub) & (B.reading == rd)]
                key = f"{roi} | {sub} | {rd}" + (
                    " PRIMARY" if sub == "learning" and rd == "own"
                    and roi != "mPFC" else "")
                r = R2.get(key, {})
                if not len(g) or not r.get("n"):
                    continue
                x = i * 5.5 + j * 2.2 + k
                col = READ_C["own" if rd == "own" else "matched"]
                dots(ax, x, g.beta.dropna(), col, rng=rng)
                ax.text(x, 1.0 + 0.07 * k, _p(r["p_perm"]),
                        transform=ax.get_xaxis_transform(), ha="center",
                        fontsize=5.8, color=col)
        ax.text(i * 5.5 + 1.6, -0.3, roi, transform=ax.get_xaxis_transform(),
                ha="center", color=COL[roi])
    zero(ax)
    ax.set_xticks([i * 5.5 + j * 2.2 + 0.5 for i in range(3) for j in range(2)],
                  ["learning", "execution"] * 3, fontsize=6.5)
    ax.set_ylabel("logistic β of uncover HFB\non a ripple 0.6–2.5 s later")
    ax.set_title("g  D2: does the cortical uncover response predict the "
                 "ripple that follows?", pad=22)
    _clean(ax)
    # row 3, right: P(ripple) by HFB quintile, learning uncovers
    ax = axes[7]
    for roi in ROIS:
        g = Ev[(Ev.roi == roi) & (Ev.learning == 1)].copy()
        if not len(g):
            continue
        g["zq"] = g.groupby("session").hfb.transform(
            lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
        pr = g.groupby(["session", "zq"]).y.mean().unstack()
        m, se = pr.mean(0), pr.std(0, ddof=1) / np.sqrt(len(pr))
        ax.errorbar(np.arange(5) + 1, m, se, color=COL[roi], marker="o", ms=3,
                    lw=1.2, capsize=2, label=f"{roi} ({len(pr)} sessions)")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("uncover HFB response quintile\n(within session)")
    ax.set_ylabel("P(ripple 0.6–2.5 s later)")
    ax.set_title("h  learning uncovers, raw view", pad=10)
    ax.legend(fontsize=6, frameon=False)
    _clean(ax)
    fig.suptitle("D1, D2 — who leads around a ripple, and does the cortical "
                 "uncover response predict it?", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", y=1.0)
    caption(fig, "Ripple contexts: uncover = in the pause after an uncover "
            "press (≤ 2 s); away = no uncover press within −2…+0.5 s; "
            "learning = before the first fully correct repeat, execution = "
            "after (U_L = uncover ripples in learning, A_E = away ripples in "
            "execution; sessions per set differ, A_E fewest). a–c: ripple-locked HFB (real − 20 shifted nulls, minus "
            "|t| ≥ 750 ms), mean ± SEM over sessions; grey = the pre-ripple "
            "window. d–f: direction index (post (0, 250] − pre [−250, 0) ms; "
            "< 0 means the cortical rise comes first), dots = sessions; own = "
            "as a–c, matched = ripple minus ripple-free moments of the same "
            "context and phase with the same time since the last press and to "
            "the next. Primary (mOFC, lOFC): U_L − A_E < 0. g: per session, "
            "logistic regression of a ripple 0.6–2.5 s after an uncover on "
            "that uncover's 0–500 ms HFB response (z), with window length, "
            "correctness and first traversal (stillness: + pause and time "
            "since the previous press); uncovers with a ripple before +0.6 s "
            "excluded; sessions need ≥ 10 uncovers with and without a ripple. "
            "h: the raw relation for learning uncovers. p = sign-flip over "
            "sessions, uncorrected.", below=0.02)
    save(fig, d, "D1_D2_direction_context")
