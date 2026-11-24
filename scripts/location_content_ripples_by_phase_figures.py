#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for location_content_ripples_by_phase.py (plot result.json as is).

    current_square_by_phase   (request 1) HC current square, encode vs known,
                              in ripples and in ordinary moments, count-matched;
                              + the all-data per-lag estimator at 0 deg
    squares_in_ripples        (requests 3, 4) the path behind / ahead and the
                              reward squares, distance-controlled, ripples vs
                              ordinary moments, encode vs known

Bars = mean over sessions, whiskers = 95% t-interval, p = one-sample t-test
across sessions (as the reported analysis); filled = p < .05.

    python scripts/location_content_ripples_by_phase_figures.py <folder> [<per-lag folder>]

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import era_brewer

CM = 1 / 2.54
_p = era_brewer.era_brew("Showgirl2", n=7)
RIP, CTL = "#5b9b8d", "#7E6B8F"
ENC, EXE = "#3b3b45", "#9a9aa6"
ROI_L = {"HC_all": "HC pooled", "HC_anterior": "aHP", "HC_mid": "mHP"}
PH_L = {"encode": "encode", "execute": "known"}
ROLE_L = {"past3": "past\n3", "past2": "past\n2", "past1": "past\n1",
          "fut1": "next\n1", "fut2": "next\n2", "fut3": "next\n3",
          "NEXT_REW": "next\nreward", "OTHER_REW": "other\nrewards",
          "ANY_REW": "any\nreward"}


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False})


def ptxt(p):
    return "p<.001" if p < .001 else f"p={p:.3f}".replace("0.", ".")


def bar(ax, x, r, col, w, labels):
    if r is None or r.get("mean") is None:
        return
    sig = r["p"] < .05
    ax.bar(x, r["mean"], w * 0.9, color=col if sig else "white", edgecolor=col,
           lw=1.0, zorder=2)
    h = stats.t.ppf(.975, r["n"] - 1) * r["sem"]
    ax.plot([x, x], [r["mean"] - h, r["mean"] + h], color="0.15", lw=0.9,
            zorder=3)
    labels.append((x, max(r["mean"] + h, 0), r["p"], sig))


def finish(ax, labels, ylim):
    ax.set_ylim(*ylim)
    lo, hi = ylim
    T = hi + 0.42 * (hi - lo)
    for x, y, p, sig in labels:
        ax.text(x, min(y + 0.03 * (hi - lo), T - 0.3 * (T - lo)), ptxt(p),
                rotation=90, ha="center", va="bottom", fontsize=6.5,
                color="0.1" if sig else "0.5")
    ax.set_ylim(lo, T)
    ax.axhline(0, color="0.2", lw=0.8, zorder=1)


def caption(fig, text, y, width):
    fig.text(0.01, y, textwrap.fill(text, width), fontsize=7.5, va="top",
             color="0.2")


def save(fig, d, name):
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{e}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(d, name + ".png"))


# ── request 1 ───────────────────────────────────────────────────────────
def draw_current(d, R, lag_dir):
    rois = ("HC_all", "HC_anterior", "HC_mid")
    fig = plt.figure(figsize=(19 * CM, 13.5 * CM))
    gs = fig.add_gridspec(2, 3, hspace=0.8, wspace=0.45,
                          width_ratios=[1.5, 1, 1])
    # a: levels
    ax = fig.add_subplot(gs[0, :])
    lab, xt, xl = [], [], []
    for i, roi in enumerate(rois):
        for j, ph in enumerate(("encode", "execute")):
            x0 = i * 2.6 + j
            for k, (ev, col) in enumerate((("rip", RIP), ("ctl", CTL))):
                bar(ax, x0 + (k - 0.5) * 0.38,
                    R.get(f"{roi} | CURRENT | zm_{ev}_{ph}"), col, 0.38, lab)
            xt.append(x0); xl.append(f"{PH_L[ph]}")
        ax.text(i * 2.6 + 0.5, -0.17, ROI_L[roi], ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9)
    finish(ax, lab, (-0.3, 1.0))
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_xlim(-0.6, 2 * 2.6 + 1.6)
    ax.set_ylabel("current square (z)")
    ax.set_title("a  HC current-square signal, ripples vs ordinary moments "
                 "(same n per session)", loc="left")
    # b: encode - known
    ax = fig.add_subplot(gs[1, 0])
    lab = []
    for i, roi in enumerate(rois):
        for k, (ev, col) in enumerate((("rip", RIP), ("ctl", CTL))):
            bar(ax, i + (k - 0.5) * 0.38, R.get(f"{roi} | CURRENT | dz_{ev}"),
                col, 0.38, lab)
    finish(ax, lab, (-0.4, 1.0))
    ax.set_xticks(range(3)); ax.set_xticklabels([ROI_L[r] for r in rois])
    ax.set_ylabel("encode − known (z)")
    ax.set_title("b  encode − known", loc="left")
    # c: ripple - ordinary
    ax = fig.add_subplot(gs[1, 1])
    lab = []
    for i, roi in enumerate(rois):
        for k, (ph, col) in enumerate((("encode", ENC), ("execute", EXE))):
            bar(ax, i + (k - 0.5) * 0.38,
                R.get(f"{roi} | CURRENT | rip-ctl_{ph}"), col, 0.38, lab)
    finish(ax, lab, (-0.8, 0.8))
    ax.set_xticks(range(3)); ax.set_xticklabels([ROI_L[r] for r in rois])
    ax.set_ylabel("ripple − ordinary (z)")
    ax.set_title("c  ripple − ordinary", loc="left")
    # d: all data, per-lag estimator at 0 deg (from its own result.json)
    ax = fig.add_subplot(gs[1, 2])
    lab = []
    if lag_dir:
        L = json.load(open(os.path.join(lag_dir, "result.json")))
        for k, (s, col) in enumerate((("encode", ENC), ("execute", EXE),
                                      ("execute_matched", "white"))):
            T = pd.DataFrame(L["vs0"][s])
            for i, unit in enumerate(("cell", "subject")):
                t = T[(T.roi == "HC_mid") & (T.analysis_unit == unit)
                      & (T.lag_deg == 0)].iloc[0]
                c = L["per_unit_curves"][f"{s} | HC_mid | {unit}"]
                n, m, sd = c["n"][0], c["mean_r"][0], c["sd_r"][0]
                r = dict(n=n, mean=m, sem=sd / np.sqrt(n), p=t.p_one_sided)
                bar(ax, i + (k - 1) * 0.28, r,
                    EXE if s == "execute_matched" else col, 0.28, lab)
                if s == "execute_matched":
                    ax.patches[-1].set_hatch("////")
        finish(ax, lab, (-0.02, 0.1))
        ax.set_xticks(range(2)); ax.set_xticklabels(["cells", "sessions"])
        ax.set_ylabel("CV r at 0°")
        ax.set_title("d  mHP, all data\n(per-lag estimator, 0°)", loc="left")
    h = [Patch(fc=RIP, ec=RIP, label="ripples"),
         Patch(fc=CTL, ec=CTL, label="ordinary moments"),
         Patch(fc=ENC, ec=ENC, label="encode"),
         Patch(fc=EXE, ec=EXE, label="known / execute"),
         Patch(fc="white", ec=EXE, hatch="////",
               label="execute, encode loop count (d)"),
         Patch(fc="white", ec="0.4", label="open: p ≥ .05")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=3, frameon=False, handlelength=1.2)
    chk = R["reproduces reported (max |diff|)"]
    mx = max(v for r in chk.values() for k, v in r.items() if k != "n")
    caption(fig, "a–c: the reported ripple analysis (swr_phase_contrast): "
            "each HC cell's map from the OTHER configurations, spikes in the "
            "ripple's own duration, current square minus the other eight, z "
            "against 1000 square-label permutations per session. Ordinary "
            "moments: windows of the session's median ripple duration tiled "
            "every 50 ms inside visits, no ripple within 0.5 s, same maps and "
            "statistic. Every bar is count-matched: per session the same n "
            "(the reported matched count) for ripples and ordinary moments in "
            "both phases, 1000 draws. encode = before a block's first "
            "error-free loop; known = from it on. d: the published per-lag "
            "estimator (warped loops, all data) at 0°, from "
            "per_lag_encode_vs_execute (one-sided p, as published). Bars = "
            "mean, 95% CI; p = one-sample t-test across sessions (cells in d, "
            "left), uncorrected. Check: ripple values reproduce the reported "
            f"per-session values (max |diff| {mx:.0e}).", -0.06, 165)
    save(fig, d, "current_square_by_phase")


# ── requests 3, 4 ───────────────────────────────────────────────────────
def draw_squares(d, R, roi="HC_all"):
    roles = ("past3", "past2", "past1", "fut1", "fut2", "fut3", "NEXT_REW",
             "OTHER_REW")
    fig = plt.figure(figsize=(19 * CM, 18 * CM))
    gs = fig.add_gridspec(3, 4, hspace=0.85, wspace=0.6,
                          width_ratios=[1, 1, 1, 0.55])
    for row, ph in enumerate(("encode", "execute")):
        ax = fig.add_subplot(gs[row, :3])
        lab = []
        for i, st in enumerate(roles):
            for k, (ev, col) in enumerate((("rip", RIP), ("ctl", CTL))):
                bar(ax, i + (k - 0.5) * 0.38,
                    R.get(f"{roi} | {st} | zm_{ev}_{ph}"), col, 0.38, lab)
        finish(ax, lab, (-0.8, 0.8))
        ax.axvline(2.5, color="0.75", lw=0.8, ls=":")
        ax.axvline(5.5, color="0.75", lw=0.8, ls=":")
        ax.set_xticks(range(len(roles)))
        ax.set_xticklabels([ROLE_L[r] for r in roles])
        ax.set_xlim(-0.6, len(roles) - 0.4)
        ax.set_ylabel("evidence vs same-\ndistance squares (z)")
        ax.set_title(f"{'ab'[row]}  {PH_L[ph]}: path behind, path ahead, "
                     "rewards", loc="left")
        # REW - NOW
        ax = fig.add_subplot(gs[row, 3])
        lab = []
        for k, (ev, col) in enumerate((("rip", RIP), ("ctl", CTL))):
            bar(ax, (k - 0.5) * 0.5, R.get(f"{roi} | REW-NOW | zm_{ev}_{ph}"),
                col, 0.5, lab)
        finish(ax, lab, (-1.5, 0.5))
        ax.set_xticks([0]); ax.set_xticklabels(["rewards −\ncurrent"])
        ax.set_xlim(-0.7, 0.7)
        ax.set_title(f"{'cd'[row]}", loc="left")
    # e: ripple - ordinary per role and phase
    ax = fig.add_subplot(gs[2, :3])
    lab = []
    for i, st in enumerate(("CURRENT",) + roles):
        for k, (ph, col) in enumerate((("encode", ENC), ("execute", EXE))):
            bar(ax, i + (k - 0.5) * 0.38, R.get(f"{roi} | {st} | rip-ctl_{ph}"),
                col, 0.38, lab)
    finish(ax, lab, (-0.8, 0.8))
    ax.axvline(0.5, color="0.75", lw=0.8, ls=":")
    ax.set_xticks(range(len(roles) + 1))
    ax.set_xticklabels(["current\nsquare"] + [ROLE_L[r] for r in roles])
    ax.set_xlim(-0.6, len(roles) + 0.6)
    ax.set_ylabel("ripple − ordinary (z)")
    ax.set_title("e  what ripples add: ripple − ordinary moments",
                 loc="left")
    # f: next reward vs other rewards
    ax = fig.add_subplot(gs[2, 3])
    lab = []
    for k, (ph, col) in enumerate((("encode", ENC), ("execute", EXE))):
        bar(ax, (k - 0.5) * 0.5, R.get(f"{roi} | NEXT-OTHER_REW | zm_rip_{ph}"),
            col, 0.5, lab)
    finish(ax, lab, (-0.8, 0.8))
    ax.set_xticks([0]); ax.set_xticklabels(["next − other\nrewards,\nripples"])
    ax.set_xlim(-0.7, 0.7)
    ax.set_title("f", loc="left")
    h = [Patch(fc=RIP, ec=RIP, label="ripples"),
         Patch(fc=CTL, ec=CTL, label="ordinary moments"),
         Patch(fc=ENC, ec=ENC, label="encode (e, f)"),
         Patch(fc=EXE, ec=EXE, label="known (e, f)"),
         Patch(fc="white", ec="0.4", label="open: p ≥ .05")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False, handlelength=1.2)
    caption(fig, f"{ROI_L[roi]} cells; the reported estimator (maps from the "
            "OTHER configurations, ripple-duration window, 1000 square-label "
            "permutations, count-matched, 1000 draws). a, b: evidence for a "
            "square minus the mean evidence at the other squares at the SAME "
            "distance from the current square (the place code is spatially "
            "smooth). past / next 1–3 = the previous / next distinct squares "
            "of the block's visit sequence; next reward = the next reward of "
            "the sequence that is not the current square; other rewards = the "
            "remaining non-current reward squares. c, d: mean evidence at the "
            "non-current reward squares minus the current square (> 0 = "
            "rewards more than the current square). e: ripple − ordinary "
            "moments, same n. f: next reward − other rewards in ripples. "
            "Bars = mean over sessions, 95% CI; p = one-sample t-test, "
            "uncorrected.", -0.05, 165)
    save(fig, d, f"squares_in_ripples_{roi}")


if __name__ == "__main__":
    _rc()
    d = sys.argv[1]
    lag_dir = sys.argv[2] if len(sys.argv) > 2 else None
    R = json.load(open(os.path.join(d, "result.json")))["tests"]
    draw_current(d, R, lag_dir)
    for roi in ("HC_all", "HC_anterior", "HC_mid"):
        draw_squares(d, R, roi)
