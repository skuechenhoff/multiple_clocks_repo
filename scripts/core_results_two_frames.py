#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overview figure of the core single-unit results, in their two frames
(2026-11-24; replaces the 2026-11-23 version of this figure).

  TASK TIME (future angles; per-cell lag CV r, SK's measure)
    a-d  CV r at the 12 future lags, mid HC and mPFC overlaid,
         session mean +- SEM;
         rows = rate-map estimator (mode = published Fig 3c, per-repeat =
         new Fig 3c), columns = encoding loops / correct loops.
         0 deg is drawn again at 360 deg, faded, to show the loop closing.
         Star = the Fig 3c window test (window minus the other lags,
         one-sided, session unit; mPFC 30+60 deg, HC 0+330 deg).
  REAL TIME (hippocampal ripples; population read-out)
    e    HC-mPFC co-firing across events: ripples with their own duration,
         ripples with a fixed width, ordinary moments (mean, 95% CI)
    f    HC now x mPFC now content coupling (fixed width, per-event gain
         removed) in ripples vs ordinary moments (mean, 95% CI)

Plots the stored results as they are; nothing is recomputed:
    a-d  per_lag_loopsets_2026-11-23/result.json ["curves"] and
         <estimator>/<set>/overlay_double_dissociation_noctrl/
         overlay_target_window_tests.csv
    e-f  swr/cofiring_window_controls_2026-11-23/result.json

    python scripts/core_results_two_frames.py
    -> group/core_results_two_frames_<date>/

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE

G = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
LAGRUN = os.path.join(G, "per_lag_loopsets_2026-11-23")
SRC = dict(
    curves=os.path.join(LAGRUN, "result.json"),
    windows=os.path.join(LAGRUN, "{est}", "{ph}", "overlay_double_dissociation_noctrl",
                         "overlay_target_window_tests.csv"),
    counts=os.path.join(os.path.dirname(G), "cell_counts.json"),
    cofiring=os.path.join(G, "swr", "cofiring_window_controls_2026-11-24", "result.json"),
)
OUT = os.path.join(G, f"core_results_two_frames_{datetime.date.today()}")
CM = 1 / 2.54
COL = {"mPFC": SHOWGIRL2_DISCRETE[1], "HC_mid": SHOWGIRL2_DISCRETE[2]}   # CLAUDE.md
NAME = {"mPFC": "mPFC", "HC_mid": "mid HC"}
GREY = "#a0a0a0"
LAGS = list(range(0, 360, 30))
ESTIMATORS = {"mode": "mode maps\n(published Fig 3c)",
              "per_repeat": "per-repeat maps\n(new Fig 3c)"}
PHASES = {"encode": "encoding loops", "execute": "correct loops"}
WRAP_ALPHA = 0.35
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def ptxt(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, pad=1)


def lag_curve(R, est, ph, roi):
    c = R["curves"][f"{est} | {ph} | {roi} | subject"]
    return np.array(c["mean"]), np.array(c["sem"])


def panel_lags(ax, R, W, est, ph, ylim, bottom):
    x = np.array(LAGS + [360])
    for roi in ("HC_mid", "mPFC"):
        m, se = lag_curve(R, est, ph, roi)
        m, se = np.append(m, m[0]), np.append(se, se[0])           # 360 = 0 deg
        c = COL[roi]
        ax.fill_between(x[:-1], (m - se)[:-1], (m + se)[:-1], color=c, alpha=0.2, lw=0)
        ax.fill_between(x[-2:], (m - se)[-2:], (m + se)[-2:], color=c,
                        alpha=0.2 * WRAP_ALPHA, lw=0)
        ax.plot(x[:-1], m[:-1], "-o", color=c, lw=1.5, ms=3)
        ax.plot(x[-2:], m[-2:], "-", color=c, lw=1.5, alpha=WRAP_ALPHA)
        ax.plot(x[-1], m[-1], "o", color=c, ms=3, alpha=WRAP_ALPHA)
        row = W[(W.weighting == "subject") & (W.roi == roi)
                & (W.test == "target_vs_other_lags")].iloc[0]
        lags = [int(v) for v in row.target_lags_deg.split("+")]
        best = max((LAGS.index(l) for l in lags), key=lambda i: m[i])
        s = stars(row.p_one_sided)
        for i in [best] + ([len(LAGS)] if best == 0 else []):
            ax.text(x[i], m[i] + se[i] + 0.006, s, ha="center", va="bottom",
                    fontsize=10 if s != "n.s." else 7.5, color=c,
                    alpha=WRAP_ALPHA if i == len(LAGS) else 1)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_ylim(*ylim)
    ax.set_xlim(-12, 372)
    ax.set_xticks(x)
    ax.set_xticks(x[:-1] + 15, minor=True)
    ax.tick_params(axis="x", which="minor", length=0)
    if bottom:
        ax.set_xticklabels([f"{l}" for l in LAGS] + ["0"], fontsize=7.5)
        ax.get_xticklabels()[-1].set_alpha(WRAP_ALPHA + 0.15)
    else:
        ax.set_xticklabels([])
    clean(ax)


def ci_t(d):
    h = stats.t.ppf(0.975, d["n"] - 1) * d["sem"]
    return d["mean"] - h, d["mean"] + h


# (window, ripple kind, its baseline, hatch): each ripple window next to the
# ordinary moments with the same window-length structure
COUPLING = "HC now x mPFC now"           # same read-out as Fig 4h (gain kept)
WINDOW_PAIRS = (("own duration", "own", "ordinary_matched", "////"),
                ("fixed width", "fixed", "ordinary", None))


def pair_bars(C, ax, col, ylabel):
    """Per phase: [ripples, own | ordinary, duration-matched] and [ripples,
    fixed | ordinary, fixed]. Bars = session mean, 95% CI; star above a bar
    = vs 0; bracket = ripple - ordinary (paired across sessions)."""
    T, K = C["tests"], C["contrasts"]
    MP, w = COL["mPFC"], 0.19
    out, tops = {}, []
    for i, ph in enumerate(PHASES):
        for j, (wname, rip, base, hatch) in enumerate(WINDOW_PAIRS):
            xs = [i - 0.31 + j * 0.42, i - 0.31 + j * 0.42 + w]
            top = -np.inf
            for x, kind, face in zip(xs, (rip, base), (MP, GREY)):
                d = T[f"{col} | {kind} | {ph}"]
                lo, hi = ci_t(d)
                lab = (f"ripples, {wname}" if kind == rip else
                       "ordinary, duration-matched" if kind == "ordinary_matched"
                       else "ordinary, fixed width")
                ax.bar(x, d["mean"], w * 0.92, color=face if hatch is None else "white",
                       edgecolor=face, hatch=hatch, lw=0.8,
                       label=lab if i == 0 else None)
                ax.errorbar(x, d["mean"], yerr=[[d["mean"] - lo], [hi - d["mean"]]],
                            color="k", lw=0.7, capsize=1.2)
                s = stars(d["p"])
                if s != "n.s.":
                    ax.annotate(s, (x, hi), xytext=(0, 0), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)
                top = max(top, hi)
                out[f"{kind} | {ph}"] = dict(n=d["n"], mean=d["mean"], ci95=[lo, hi],
                                             p=d["p"], p_perm=d.get("p_perm"))
            k = K[f"{col} | {rip} - {base} | {ph}"]
            out[f"{rip} - {base} | {ph}"] = dict(n=k["n"], mean=k["mean"], t=k.get("t"),
                                                p=k["p"], p_perm=k.get("p_perm"))
            tops.append((xs, top, k["p"]))
    ax.axhline(0, color="k", lw=0.6)
    lo, hi = ax.get_ylim()
    span = hi - lo
    for xs, top, p in tops:
        y = top + 0.19 * span
        ax.plot([xs[0], xs[0], xs[1], xs[1]], [y - 0.03 * span, y, y, y - 0.03 * span],
                color="k", lw=0.6)
        ax.text(np.mean(xs), y + 0.01 * span, f"Δ {ptxt(p)}", ha="center",
                va="bottom", fontsize=6.5, rotation=90)
    ax.set_ylim(lo - 0.03 * span, hi + 0.7 * span)
    ax.set_xticks(range(len(PHASES)))
    ax.set_xticklabels([v.split()[0] for v in PHASES.values()])
    ax.set_xlim(-0.5, len(PHASES) - 0.5)
    ax.set_ylabel(ylabel)
    clean(ax)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    R = json.load(open(SRC["curves"]))
    counts = json.load(open(SRC["counts"]))
    C = json.load(open(SRC["cofiring"]))
    Wt = {(e, p): pd.read_csv(SRC["windows"].format(est=e, ph=p))
          for e in ESTIMATORS for p in PHASES}

    # one y range for all four lag panels (curves +- SEM, room for stars)
    ext = [f(m + s * sg) for e in ESTIMATORS for p in PHASES for r in COL
           for m, s in [lag_curve(R, e, p, r)] for f, sg in ((np.min, -1), (np.max, 1))]
    ylim = (min(ext) - 0.01, max(ext) + 0.03)

    W, H = 30.5, 12.4
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax_at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    pw, ph_, xs, ys = 6.5, 3.7, (2.5, 9.5), (6.8, 2.2)
    letters = iter("abcd")
    plotted = {"lags": {}}
    for r_i, est in enumerate(ESTIMATORS):
        for c_i, ph in enumerate(PHASES):
            ax = ax_at(xs[c_i], ys[r_i], pw, ph_)
            panel_lags(ax, R, Wt[(est, ph)], est, ph, ylim, bottom=r_i == 1)
            if c_i == 0:
                fig.text(0.15 / W, (ys[r_i] + ph_ / 2) / H, ESTIMATORS[est],
                         rotation=90, ha="left", va="center", fontsize=9,
                         fontweight="bold", color="0.25", multialignment="center")
            if r_i == 0:
                ax.set_title(PHASES[ph], fontsize=9.5, pad=3)
            if c_i == 0:
                ax.set_ylabel("CV r (mean ± SEM)", fontsize=9)
            else:
                ax.set_yticklabels([])
            if r_i == 1:
                ax.set_xlabel("future lag (°; 90° = one reward ahead)", labelpad=2)
            fig.text((xs[c_i] - (1.55 if c_i == 0 else 0.35)) / W,
                     (ys[r_i] + ph_ + 0.05) / H, next(letters), fontsize=12,
                     fontweight="bold", va="bottom")
            for roi in COL:
                m, se = lag_curve(R, est, ph, roi)
                w = Wt[(est, ph)]
                w = w[(w.weighting == "subject") & (w.roi == roi)
                      & (w.test == "target_vs_other_lags")].iloc[0]
                plotted["lags"][f"{est} | {ph} | {roi}"] = dict(
                    mean=m.tolist(), sem=se.tolist(),
                    window=w.target_lags_deg, window_minus_other_t=float(w.t_fisher_z),
                    df=int(w.df), p_one_sided=float(w.p_one_sided))
    # ROI legend for a-d, under the bottom row
    hs = [plt.Line2D([], [], color=COL[r], marker="o", ms=3, lw=1.5) for r in COL]
    n = counts["per_lag_fig3c"]
    fig.legend(hs, [f"{NAME[r]} ({n[r]['sessions']} sessions, {n[r]['cells']} cells)"
                    for r in COL], loc="lower left", bbox_to_anchor=(xs[0] / W, 0.35 / H),
               ncol=2, frameon=False, fontsize=8, handlelength=1.6)

    rx, rw, rh = 19.1, 6.4, 3.3
    ax_e = ax_at(rx, ys[0], rw, rh)
    plotted["cofiring"] = pair_bars(C, ax_e, "r_pop",
                                    "HC–mPFC co-firing\n(r of population z, 95% CI)")
    ax_e.set_title("all-cell co-firing", fontsize=9.5, pad=3)
    ax_e.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False,
                fontsize=7.5, handlelength=1.4, borderaxespad=0)
    ax_e.text(1.03, 0.02, "* vs 0;  Δ p: ripples −\nordinary, paired",
              transform=ax_e.transAxes, fontsize=7, color="0.3", va="bottom")
    ax_f = ax_at(rx, ys[1], rw, rh)
    plotted["coupling"] = pair_bars(C, ax_f, COUPLING,
                                    "HC now × mPFC now\n(r − shuffle r, 95% CI)")
    ax_f.set_title("current-square coupling", fontsize=9.5, pad=3)
    for ax, l in ((ax_e, "e"), (ax_f, "f")):
        fig.text((rx - 1.75) / W, (ax.get_position().y1 * H + 0.05) / H, l,
                 fontsize=12, fontweight="bold", va="bottom")

    fig.text(xs[0] / W, 11.75 / H, "Task time: each cell's map at every future lag",
             fontsize=11, fontweight="bold", va="bottom")
    fig.text(xs[0] / W, 11.3 / H, "per-cell cross-validated consistency across layouts; "
             "session = unit", fontsize=8, color="0.3", va="bottom")
    fig.text((rx - 1.75) / W, 11.75 / H, "Real time: hippocampal ripples",
             fontsize=11, fontweight="bold", va="bottom")
    fig.text((rx - 1.75) / W, 11.3 / H,
             f"{C['n_sessions']} sessions with HC and mPFC cells; session = unit",
             fontsize=8, color="0.3", va="bottom")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"core_results_two_frames.{ext}"),
                    dpi=300, bbox_inches="tight")
    json.dump(dict(sources=SRC, n_sessions_ripple=C["n_sessions"], plotted=plotted),
              open(os.path.join(OUT, "result.json"), "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
