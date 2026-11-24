#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for location_estimator_reconcile.py (plots result.json as is).

    a, b  HC current-square signal in ripples, learning vs execution, for each
          template source; a = the ripple's own duration (reported window),
          b = +-125 ms (D3 / D4c window)
    c     learning - execution, ripple counts matched (the reported contrast)

Bars = mean over sessions, whiskers = 95% t-interval, sign-flip p printed.
Filled = p < .05, open = not.

    python scripts/exploratory/location_estimator_reconcile_figure.py <folder>

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import textwrap

from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CM = 1 / 2.54
TPL = ("cross_config", "cross_config_phase", "same_block", "same_block_phase")
TPL_L = {"cross_config": "other\nconfigs\n(reported)",
         "cross_config_phase": "other\nconfigs,\nsame\nphase",
         "same_block": "same\nblock\n(D3)",
         "same_block_phase": "same\nblock,\nsame\nphase"}
PH = (("z_encode", "learning", "#3b3b45"),
      ("z_execute", "execution", "#9a9aa6"))
WIN = (("duration", "ripple duration", "#0e3d3a"),
       ("pm125", "±125 ms", "#5b9b8d"))


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 7.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False})


def ptxt(p):
    return "p<.001" if p < .001 else f"p={p:.3f}".replace("0.", ".")


def bars(ax, R, series, key, title, ylab, ylim):
    """series: (name, label, colour); key(template, name) -> result.json key."""
    ax.set_ylim(*ylim)
    k = len(series)
    w = 0.8 / k
    lab = []
    for i, t in enumerate(TPL):
        for j, (nm, _, col) in enumerate(series):
            r = R[key(t, nm)]
            x = i + (j - (k - 1) / 2) * w
            sig = r["p_perm"] < .05
            ax.bar(x, r["mean"], w * 0.9, color=col if sig else "white",
                   edgecolor=col, linewidth=1.0, zorder=2)
            h = stats.t.ppf(.975, r["n"] - 1) * r["sem"]
            ax.plot([x, x], [r["mean"] - h, r["mean"] + h], color="0.15",
                    lw=0.9, zorder=3)
            lab.append((x, max(r["mean"] + h, 0), r["p_perm"], sig))
    lo, hi = ax.get_ylim()
    T = hi + 0.4 * (hi - lo)
    for x, y, p, sig in lab:
        ax.text(x, y + 0.03 * (hi - lo), ptxt(p), rotation=90, ha="center",
                va="bottom", fontsize=7, color="0.1" if sig else "0.5")
    ax.set_ylim(lo, T)
    ax.axhline(0, color="0.2", lw=0.8, zorder=1)
    ax.set_xticks(range(len(TPL)))
    ax.set_xticklabels([TPL_L[t] for t in TPL])
    ax.set_xlim(-0.6, len(TPL) - 0.4)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")


def draw(d):
    _rc()
    J = json.load(open(os.path.join(d, "result.json")))
    R = J["tests"]
    fig, axes = plt.subplots(1, 3, figsize=(21 * CM, 7.5 * CM),
                             gridspec_kw=dict(wspace=0.35))
    for ax, (win, wl, _), letter in zip(axes[:2], WIN, "ab"):
        bars(ax, R, PH, lambda t, nm: f"{t} | {win} | {nm}",
             f"{letter}  window: {wl}", "HC current square\nin ripples (z)"
             if letter == "a" else "", (-0.3, 1.0))
    bars(axes[2], R, WIN, lambda t, nm: f"{t} | {nm} | d_z",
         "c  learning − execution\n(ripple counts matched)",
         "difference (z)", (-0.95, 0.8))
    h = [Patch(facecolor=c, edgecolor=c, label=l) for _, l, c in PH]
    h += [Patch(facecolor=c, edgecolor=c, label=f"c: {l}") for _, l, c in WIN]
    h += [Patch(facecolor="white", edgecolor="0.4", label="open: p ≥ .05")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2),
               ncol=len(h), frameon=False, handlelength=1.2,
               columnspacing=0.9)
    mix = R["same_block: execution share of the template, learning ripples"]
    chk = R["reproduces reported"]
    cap = (f"Same ripples, cells (HC anterior + mid) and sessions as the "
           f"reported phase contrast (swr_phase_contrast); only the template "
           f"source (x axis) and the spike window (a vs b) change. Statistic: "
           f"current square minus the other eight, z against 1000 square-label "
           f"permutations per session; a, b: all ripples per phase (as Fig "
           f"20); c: learning − execution with ripple counts matched (1000 "
           f"draws), the reported test. 'same block' = each cell's map from "
           f"the other traversals of the ripple's own block: for learning "
           f"ripples a median {100 * mix['median']:.0f}% of that map's time "
           f"comes from EXECUTION traversals. 'same phase' = only traversals "
           f"of the ripple's phase. Bars = mean over sessions, 95% CI; p = "
           f"sign-flip, uncorrected. Check: 'other configs' x ripple duration "
           f"reproduces the reported per-session values (max |diff| "
           f"{max(chk['z_encode'], chk['z_execute'], chk['d_z']):.1e}).")
    fig.text(0.01, -0.29, textwrap.fill(cap, 190), fontsize=7.5, va="top",
             color="0.2")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"location_estimator_reconcile.{e}"),
                    dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(d, "location_estimator_reconcile.png"))


if __name__ == "__main__":
    draw(sys.argv[1])
