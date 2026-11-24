#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two summary figures from `swr_hfb_discovery_lock` (F4 + the 2026-11-14 (b)
summary sets): what mPFC and mOFC HFB are locked to -- hippocampal ripples,
uncovers, or both. Same layout for both regions so they can be compared:

    row 1  ripple-locked HFB (every in-task ripple; first-traversal ripples),
           own reading | stillness-matched | GLM ripple betas
    row 2  press-locked HFB (every incorrect / incorrect in the first traversal
           / every correct / first discoveries), own reading |
           stillness-matched | GLM uncover and navigation betas

Plots saved results only. Usage:
    python scripts/exploratory/plot_ripple_vs_uncover_locking.py

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_gradient_claims import (plt, layout, caption, save, dots, zero,
                                  _clean, _p, COL, SEED, SWR)
from swr_hfb_discovery_lock import prepared, WINDOW_MS

D = os.path.join(SWR, "hfb_discovery_lock_2026-11-14")
READ = {"own": "own baseline", "matched": "stillness-matched"}
VAR_C = {"own": "#0e3d3a", "still": "#a30d6c"}
VAR_L = {"own": "GLM, events + drift", "still": "GLM + stillness regressors"}
EV_C = {"U_ERR": "#d98bb8", "U_FT_ERR": "#a30d6c", "U_COR": "#9ec6b8",
        "U_FD": "#0e3d3a"}
EV_L = {"U_ERR": "every incorrect uncover",
        "U_FT_ERR": "incorrect, first traversal",
        "U_COR": "every correct uncover", "U_FD": "first discovery"}
RIP_L = {"R_ALL": "every ripple", "R_FT": "first-traversal ripples"}
GLM_RIP = (("rip_postFD", "after a first\ndiscovery"),
           ("rip_FT_other", "other first-\ntraversal"), ("rip_later", "later"))
GLM_EV = (("FD", "first\ndisc."), ("FT_error", "FT\nerror"),
          ("correct_later", "later\ncorrect"), ("error_later", "later\nerror"),
          ("move", "navig.\npress"))


def _light(c, f=0.5):
    from matplotlib.colors import to_rgb
    r = np.array(to_rgb(c))
    return tuple(r + (1 - r) * f)


def curves(ax, Dc, res, roi, keys, colours, labels, reading, xlab):
    lock_t = Dc["t_ms"]["ripple" if keys[0].startswith("R") else "event"]
    for k, key in enumerate(keys):
        C = prepared(Dc["curves"], Dc["t_ms"], key, roi, reading)
        r = res.get(f"{key} | {reading} | {roi}")
        if len(C) < 5 or r is None:
            continue
        X = np.stack(list(C.values()))
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        w = r["window"]
        sig = [c for c in r["clusters"] if c["p"] < 0.05]
        cl = ("\n   cluster " + ", ".join(
            f"{c['start_ms']:+.0f}…{c['stop_ms']:+.0f} ms {_p(c['p'])}"
            + ("" if c["sign"] == "increase" else " (−)") for c in sig)) \
            if sig else ""
        ax.plot(lock_t, m, color=colours[key], lw=1.3,
                label=f"{labels[key]}: {w['mean']:+.4f}, {_p(w['p_perm'])}{cl}")
        ax.fill_between(lock_t, m - se, m + se, color=colours[key], alpha=0.18,
                        lw=0)
        for c in sig:
            y = 0.98 - 0.045 * k
            ax.plot([c["start_ms"], c["stop_ms"]], [y, y], color=colours[key],
                    lw=2.5, solid_capstyle="butt",
                    transform=ax.get_xaxis_transform())
    lk = "ripple" if keys[0].startswith("R") else "event"
    ax.axvspan(*WINDOW_MS[lk], color="0.92", zorder=0, lw=0)
    zero(ax)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(lock_t[0], lock_t[-1])
    ax.set_xlabel(xlab)
    ax.legend(fontsize=5.6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), title="window mean, p (bars: "
              "clusters p < .05)", title_fontsize=5.6)
    _clean(ax)


def glm_panel(ax, S, res, roi, regs, rng):
    for i, (k, _) in enumerate(regs):
        for j, var in enumerate(("own", "still")):
            g = S[(S.variant == var) & (S.roi == roi)]
            if f"b_{k}" not in g:
                continue
            r = res[f"{var} | {roi} | {k}"]
            x = i * 2.5 + j
            dots(ax, x, g[f"b_{k}"].dropna(), VAR_C[var], rng=rng)
            ax.text(x, 1.0 + 0.07 * j, _p(r["p_perm"]),
                    transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=5.8, color=VAR_C[var])
    zero(ax)
    ax.set_xticks([i * 2.5 + 0.5 for i in range(len(regs))],
                  [lab for _, lab in regs], fontsize=6.3)
    ax.set_ylabel("GLM β per event\n(summed z)")
    _clean(ax)


def figure(roi, title, name):
    Dc = pickle.load(open(os.path.join(D, "curves.pkl"), "rb"))
    res = json.load(open(os.path.join(D, "result_epochs.json")))["tests"]
    S = pd.read_csv(os.path.join(D, "glm_session_betas.csv"))
    rg = json.load(open(os.path.join(D, "result_glm.json")))
    rng = np.random.default_rng(SEED)
    rip_c = {"R_ALL": COL[roi], "R_FT": _light(COL[roi])}
    fig, axes = layout([1] * 6)
    for j, rd in enumerate(("own", "matched")):
        curves(axes[j], Dc, res, roi, ("R_ALL", "R_FT"), rip_c, RIP_L, rd,
               "time from ripple (ms)")
        axes[j].set_title(f"{'ab'[j]}  ripple-locked, {READ[rd]}", pad=10)
        curves(axes[3 + j], Dc, res, roi, ("U_ERR", "U_FT_ERR", "U_COR", "U_FD"),
               EV_C,
               EV_L, rd, "time from the uncover press (ms)")
        axes[3 + j].set_title(f"{'de'[j]}  uncover-locked, {READ[rd]}",
                              pad=10)
    axes[0].set_ylabel(f"{roi} HFB (z)")
    axes[1].set_ylabel("ripple − matched\nmoments (z)")
    axes[3].set_ylabel(f"{roi} HFB (z)")
    axes[4].set_ylabel("uncover − matched\narrow press (z)")
    glm_panel(axes[2], S, rg, roi, GLM_RIP, rng)
    axes[2].set_title("c  GLM: ripples, all events competing", pad=22)
    glm_panel(axes[5], S, rg, roi, GLM_EV, rng)
    axes[5].set_title("f  GLM: uncovers and presses", pad=22)
    for var in VAR_C:
        axes[2].plot([], [], "o", color=VAR_C[var], label=VAR_L[var])
    axes[2].legend(fontsize=6, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.32))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold",
                 y=1.03)
    caption(fig, f"{roi} contacts of the 2026-09-28 partition (different shaft "
            "from the hippocampal contact); one curve per session, mean ± SEM "
            "over sessions, 50 ms sliding mean. Own baseline: real minus the "
            "mean of 20 time-shifted nulls, minus the epoch's own baseline "
            "(ripples |t| ≥ 750 ms; presses −1.5…−1.0 s). Stillness-matched: "
            "ripples minus ripple-free moments of the same period with the "
            "same time since the last press and to the next; uncovers minus "
            "arrow presses followed by the same stillness. Grey band = the "
            "tested window (ripples ±250 ms, presses 0–500 ms); bars = "
            "clusters p < .05 (sign-flip over sessions, 10,000). c, f: one GLM "
            "on the continuous HFB with every event type at once (differenced "
            "kernel −250…+500 ms minus a −1500…−750 ms arm), so each β is the "
            "response to that event after the others are accounted for; dots "
            "= sessions; p = sign-flip, uncorrected.", below=-0.16)
    out = os.path.join(SWR, "gradient_claims_followup_2026-11-14")
    save(fig, out, name)
    return out


def main():
    figure("mPFC", "mPFC HFB: locked to hippocampal ripples, not to uncovers",
           "F5_mPFC_ripples_not_uncovers")
    figure("mOFC", "mOFC HFB: locked to hippocampal ripples and to uncovers "
                   "(incorrect ones most)", "F6_mOFC_uncovers_and_ripples")


if __name__ == "__main__":
    main()
