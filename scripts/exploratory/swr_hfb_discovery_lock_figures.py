"""
Figures for `swr_hfb_discovery_lock.py` (F4), drawn from its saved outputs.

    from swr_hfb_discovery_lock_figures import draw
    draw(out_dir)

F4_epochs  mOFC vs mPFC HFB, ripple- and uncover-locked, both readings
F4_glm     the competing-regressor GLM: uncover or ripple?

@author: Svenja Kuchenhoff
"""

import os
import json
import pickle

import numpy as np
import pandas as pd

from plot_gradient_claims import (plt, layout, caption, save, dots, zero,
                                  _clean, _p, COL, SEED)
from swr_hfb_discovery_lock import (TITLE, LOCK, ROIS, REGS, WINDOW_MS,
                                    prepared)

READ = {"own": "own baseline", "matched": "stillness-matched"}
RLAB = {"rip_postFD": "ripple\nafter FD", "rip_FT_other": "other\nFT ripple",
        "rip_later": "later\nripple", "FD": "first\ndiscovery",
        "FT_error": "FT\nerror", "correct_later": "later\ncorrect",
        "error_later": "later\nerror", "move": "navig.\npress"}
SHORT = {"R_FT": "ripples, first traversal", "R_FD": "ripples after a FD",
         "U_FT": "uncovers, first traversal", "U_FD": "first discoveries (FD)"}


def _curve_panel(ax, curves, t_ms, res, key, reading, letter):
    lock = LOCK[key]
    tt = t_ms[lock]
    for k, roi in enumerate(ROIS):
        C = prepared(curves, t_ms, key, roi, reading)
        if len(C) < 5:
            continue
        X = np.stack(list(C.values()))
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        r = res[f"{key} | {reading} | {roi}"]
        w = r["window"]
        ax.plot(tt, m, color=COL[roi], lw=1.3,
                label=f"{roi} (n = {len(X)}): window {w['mean']:+.4f}, "
                      f"{_p(w['p_perm'])}")
        ax.fill_between(tt, m - se, m + se, color=COL[roi], alpha=0.2, lw=0)
        for c in r["clusters"]:
            if c["p"] < 0.05:
                y = 0.97 - 0.06 * k
                ax.plot([c["start_ms"], c["stop_ms"]], [y, y], color=COL[roi],
                        lw=3, transform=ax.get_xaxis_transform(),
                        solid_capstyle="butt")
                ax.text(c["stop_ms"] + 20, y, _p(c["p"]) + ("" if c["sign"] ==
                        "increase" else " (−)"), fontsize=5.8, va="center",
                        color=COL[roi], transform=ax.get_xaxis_transform())
    a, b = WINDOW_MS[lock]
    ax.axvspan(a, b, color="0.92", zorder=0, lw=0)
    zero(ax)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(tt[0], tt[-1])
    ax.set_xlabel(f"time from {'ripple' if lock == 'ripple' else 'reward press'}"
                  " (ms)")
    pr = res.get(f"{key} | {reading} | mOFC - mPFC", {})
    sub = ""
    if pr:
        sub = (f"\nmOFC − mPFC {pr['unpaired']['diff']:+.4f}, "
               f"{_p(pr['unpaired']['p_perm'])}\n(paired, n = "
               f"{pr['paired'].get('n')}: {_p(pr['paired'].get('p_perm', 1))})")
    ax.set_title(f"{letter}  {SHORT[key]}, {READ[reading]}{sub}",
                 fontsize=7, pad=8)
    ax.legend(fontsize=5.6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.3))
    _clean(ax)


def _ripsplit_panel(ax, W, res, reading, letter, rng):
    for j, roi in enumerate(ROIS):
        for i, key in enumerate(("U_FD_norip", "U_FD_rip")):
            g = W[(W.set == key) & (W.roi == roi) & (W.reading == reading)]
            r = res.get(f"{key} | {reading} | {roi}")
            if not len(g) or r is None:
                continue
            dots(ax, j * 2.4 + i, g.window, COL[roi], r["window"]["p_perm"],
                 rng=rng, py=1.0 + 0.08 * i)
    zero(ax)
    ax.set_xticks([0, 1, 2.4, 3.4], ["no ripple", "ripple"] * 2, fontsize=6.5)
    for x, roi in ((0.5, "mOFC"), (2.9, "mPFC")):
        ax.text(x, -0.25, roi, transform=ax.get_xaxis_transform(),
                ha="center", color=COL[roi])
    ax.set_ylabel("HFB, 0–500 ms after\nthe press (z)")
    ax.set_title(f"{letter}  first discoveries without vs\nwith a "
                 f"following ripple, {READ[reading]}", fontsize=7, pad=20)
    _clean(ax)


def draw_epochs(d):
    D = pickle.load(open(os.path.join(d, "curves.pkl"), "rb"))
    res = json.load(open(os.path.join(d, "result_epochs.json")))["tests"]
    W = pd.read_csv(os.path.join(d, "window_values.csv"))
    rng = np.random.default_rng(SEED)
    fig, axes = layout([1] * 12)
    order = (("R_FT", "own"), ("R_FD", "own"), None,
             ("U_FT", "own"), ("U_FD", "own"), ("split", "own"),
             ("R_FT", "matched"), ("R_FD", "matched"), None,
             ("U_FT", "matched"), ("U_FD", "matched"), ("split", "matched"))
    for i, (ax, o) in enumerate(zip(axes, order)):
        if o is None:
            continue
        L = "abcdefghijkl"[i]
        if o[0] == "split":
            _ripsplit_panel(ax, W, res, o[1], L, rng)
        else:
            _curve_panel(ax, D["curves"], D["t_ms"], res, o[0], o[1], L)
    # the two free slots: the window values of every set, mOFC vs mPFC
    for ax, reading, L in ((axes[2], "own", "c"), (axes[8], "matched", "i")):
        keys = ("R_FT", "R_FD", "U_FT", "U_FD")
        for i, key in enumerate(keys):
            for j, roi in enumerate(ROIS):
                g = W[(W.set == key) & (W.roi == roi) & (W.reading == reading)]
                if len(g):
                    dots(ax, i * 2.6 + j, g.window, COL[roi], rng=rng)
            pr = res.get(f"{key} | {reading} | mOFC - mPFC")
            if pr:
                ax.text(i * 2.6 + 0.5, 1.03, _p(pr["unpaired"]["p_perm"]),
                        transform=ax.get_xaxis_transform(), ha="center",
                        fontsize=6.3)
        zero(ax)
        ax.set_xticks([i * 2.6 + 0.5 for i in range(4)],
                      ["ripples,\nFT", "ripples\nafter FD", "uncovers,\nFT",
                       "first\ndiscoveries"], fontsize=6.3)
        for roi in ROIS:
            ax.plot([], [], "o", color=COL[roi], label=roi)
        ax.legend(fontsize=6.3, frameon=False, loc="lower left")
        ax.set_ylabel("window mean (z)")
        ax.set_title(f"{L}  window values, {READ[reading]}\n(p: mOFC vs "
                     f"mPFC, label permutation)", fontsize=7, pad=16)
        _clean(ax)
    axes[0].set_ylabel("ripple-locked HFB (z)")
    axes[3].set_ylabel("press-locked HFB (z)")
    axes[6].set_ylabel("ripple − matched\nmoments (z)")
    axes[9].set_ylabel("uncover − matched\npress (z)")
    fig.suptitle("F4 — mOFC vs mPFC HFB at discovery: locked to the ripple "
                 "or to the reward uncover?", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.03)
    caption(fig, "Partition ROIs as claim 1a; one curve per session, mean ± "
            "SEM, 50 ms sliding mean. Top two rows, own reading: real minus "
            "the mean of 20 shifted nulls, minus the epoch's own baseline "
            "(ripples: |t| ≥ 750 ms; presses: −1.5…−1.0 s). Bottom two rows, "
            "stillness-matched: uncovers minus arrow presses followed by the "
            "same stillness; ripples minus ripple-free moments of the same "
            "period (first-traversal time, or post-discovery pauses) with the "
            "same time since the last press and to the next. FT = first "
            "traversal; FD = first discovery (correct uncover in the first "
            "traversal); 'ripples after FD' = in the pause after it, ≤ 2 s. "
            "Grey band = the tested window (ripples ±250 ms, presses 0–500 "
            "ms); bars = clusters p < .05 (sign-flip over sessions, 10,000). "
            "mOFC − mPFC: unpaired label permutation and paired over sessions "
            "with both. Uncorrected across sets.", below=0.0)
    save(fig, d, "F4_epochs")


def draw_glm(d):
    S = pd.read_csv(os.path.join(d, "glm_session_betas.csv"))
    res = json.load(open(os.path.join(d, "result_glm.json")))
    rng = np.random.default_rng(SEED)
    fig, axes = layout([2, 1, 2, 1])
    for row, variant in enumerate(("own", "still")):
        ax = axes[row * 2]
        for i, k in enumerate(REGS):
            for j, roi in enumerate(ROIS):
                g = S[(S.variant == variant) & (S.roi == roi)]
                if f"b_{k}" not in g:
                    continue
                r = res[f"{variant} | {roi} | {k}"]
                dots(ax, i * 2.6 + j, g[f"b_{k}"].dropna(), COL[roi], rng=rng)
                ax.text(i * 2.6 + j, 1.0 + 0.08 * j, _p(r["p_perm"]),
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=6, color=COL[roi])
        zero(ax)
        ax.set_xticks([i * 2.6 + 0.5 for i in range(len(REGS))],
                      [RLAB[k] for k in REGS], fontsize=6.5)
        for roi in ROIS:
            ax.plot([], [], "o", color=COL[roi], label=roi)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        ax.set_ylabel("β per event: window −\nbaseline arm (summed z)")
        ax.set_title(f"{'ac'[row]}  every event type competing — "
                     f"{'events + drift' if variant == 'own' else '+ pause-state regressors (stillness)'}",
                     pad=22)
        _clean(ax)
        ax = axes[row * 2 + 1]
        for j, roi in enumerate(ROIS):
            g = S[(S.variant == variant) & (S.roi == roi)]
            v = (g.b_FD - g.b_rip_postFD).dropna()
            key = f"{variant} | {roi} | FD - rip_postFD" + (
                " PRIMARY" if variant == "own" else "")
            dots(ax, j, v, COL[roi], res[key].get("p_perm"), rng=rng)
        pr = res[f"{variant} | mOFC - mPFC | FD - rip_postFD"]
        zero(ax)
        ax.set_xticks([0, 1], list(ROIS))
        ax.set_ylabel("β(first discovery)\n− β(ripple after it)")
        ax.set_title(f"{'bd'[row]}  uncover (> 0) or ripple (< 0)?\n"
                     f"mOFC − mPFC {pr['unpaired']['diff']:+.3f}, "
                     f"{_p(pr['unpaired']['p_perm'])}", pad=22)
        _clean(ax)
    fig.suptitle("F4 — GLM: is mOFC / mPFC HFB explained by the reward "
                 "uncover or by the ripple?", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.06)
    caption(fig, "Continuous HFB per derivation (partition mOFC / mPFC, "
            "different shaft from the hippocampal contact), one differenced "
            "regressor per event type (boxcar −250…+500 ms minus a "
            "−1500…−750 ms arm, so β = response minus own baseline), all "
            "event types in one model so that they compete for the same "
            "variance, plus intercept and 3rd-order Legendre drift. Ripples "
            "split into those in the pause after a first discovery (≤ 2 s), "
            "other first-traversal ripples and later ones; uncovers into first "
            "discoveries, first-traversal errors, later correct and later "
            "incorrect. c, d add three boxcars for pause state (0.5–1, 1–2, "
            "> 2 s since the last press). Dots = sessions (mean over the "
            "session's derivations); p = sign-flip over sessions; b, d: the "
            "pre-registered contrast, mOFC − mPFC by label permutation.",
            below=-0.06)
    save(fig, d, "F4_glm")


def draw(d):
    draw_epochs(d)
    draw_glm(d)
