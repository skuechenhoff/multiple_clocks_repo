"""
Figures for `lag0_cells` (F1, F3), drawn from its saved outputs only.

    from mc.analyse.swr_explore.lag0_cells_figures import draw
    draw(out_dir)

F1_claim2ab_lag0_cells   claim 2 a/b with all cells vs the top-30% lag-0 cells,
                         arrival-locked (as published) and reward-locked
F3_lag0_cells_ripples    are lag-0 cells ripple-locked (F3a), and do they carry
                         the square just discovered (F3b)?

@author: Svenja Kuchenhoff
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
from plot_gradient_claims import (plt, layout, caption, save, dots, zero,
                                  _clean, _p, COL, SEED, SWR)
from mc.analyse.swr_explore.lag0_cells import R_CENTRES, R_TEST

ROIS = ("mPFC", "mOFC", "HC_all")
EVENTS = (("FD", "first discovery"), ("RC", "correct,\nalready known"),
          ("INC", "incorrect"))
SETS = (("R_FT", "first-traversal ripples"),
        ("R_FD", "ripples after a first discovery"),
        ("R_EXE", "execution ripples"))
READ = {"own": "own flanks (0.5–1 s)", "matched": "stillness-matched"}


def _light(c, f=0.55):
    r = np.array(to_rgb(c))
    return tuple(r + (1 - r) * f)


def _roi_lab(ax, xs, rois, y=-0.3):
    for x, roi in zip(xs, rois):
        ax.text(x, y, roi.replace("_all", ""), transform=ax.get_xaxis_transform(),
                ha="center", color=COL[roi], fontsize=6.5)


def _events_panel(ax, get, letter, title, ylab, rng, legend=False):
    """Per event block: per ROI, all cells (light) then top-30% (dark)."""
    for i, (ev, lab) in enumerate(EVENTS):
        for j, roi in enumerate(ROIS):
            for k, cset in enumerate(("all", "top30")):
                v, p = get(ev, roi, cset)
                if v is None:
                    continue
                x = i * 7 + j * 2 + k * 0.85
                col = COL[roi] if cset == "top30" else _light(COL[roi], 0.4)
                dots(ax, x, v, col, rng=rng, marker="s" if k else "o")
                if p is not None:
                    ax.text(x, 1.0 + 0.08 * ((j * 2 + k) % 2), _p(p),
                            transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=5.8, color=col)
    zero(ax)
    ax.set_xticks([i * 7 + 2.4 for i in range(len(EVENTS))],
                  [lab for _, lab in EVENTS], fontsize=7)
    if legend:
        for roi in ROIS:
            ax.plot([], [], "o", color=_light(COL[roi], 0.4),
                    label=f"{roi.replace('_all', '')}, all cells")
            ax.plot([], [], "s", color=COL[roi],
                    label=f"{roi.replace('_all', '')}, top-30% lag-0")
        ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.3))
    ax.set_ylabel(ylab)
    ax.set_title(f"{letter}  {title}", pad=24)
    _clean(ax)


def draw_f1(d):
    rng = np.random.default_rng(SEED)
    A_all = pd.read_csv(os.path.join(SWR, "consolidation_2026-10-01",
                                     "per_session.csv"))
    A_top = pd.read_csv(os.path.join(d, "F1_arrival_locked_top30",
                                     "per_session.csv"))
    SA_all = pd.read_csv(os.path.join(SWR, "consolidation_2026-10-01",
                                      "summary.csv"))
    SA_top = pd.read_csv(os.path.join(d, "F1_arrival_locked_top30",
                                      "summary.csv"))
    for S in (SA_all, SA_top):
        S["measure"] = S.measure.str.strip()
    E = pd.read_csv(os.path.join(d, "F1_reward_locked_per_session.csv"))
    SE = pd.read_csv(os.path.join(d, "F1_reward_locked_summary.csv"))
    ev_arr = {"FD": "first_discovery", "RC": "repeat_correct",
              "INC": "incorrect"}

    def arr(ev, roi, cset, col="z_all", meas="represented at all"):
        P, S = (A_all, SA_all) if cset == "all" else (A_top, SA_top)
        g = P[(P.roi == roi) & (P.event == ev_arr[ev]) & (P.phase == "all")]
        s = S[(S.roi == roi) & (S.event == ev_arr[ev]) & (S.phase == "all")
              & (S.measure == meas)]
        return (g[col].dropna(), s.p.iloc[0] if len(s) else None) \
            if len(g) else (None, None)

    def rew(col):
        def f(ev, roi, cset):
            g = E[(E.roi == roi) & (E.cells == cset)]
            c = col.format(ev=ev)
            s = SE[(SE.roi == roi) & (SE.cells == cset) & (SE.measure == c)]
            if c not in g or not g[c].notna().sum():
                return None, None
            return g[c].dropna(), s.p_perm.iloc[0] if len(s) else None
        return f

    fig, axes = layout([2, 1, 2, 1, 2, 1])
    _events_panel(axes[0], arr, "a", "arrival-locked, as published: "
                  "which squares are held after an uncover?",
                  "evidence for the square\njust uncovered (z)", rng)

    def ripple_split(ax, getter, letter, title):
        for i, roi in enumerate(("mPFC", "mOFC")):
            a, pa = getter(roi, "without")
            b, pb = getter(roi, "with")
            if a is None or b is None:
                continue
            sh = sorted(set(a.index) & set(b.index))
            for s in sh:
                ax.plot([2 * i, 2 * i + 0.8], [a[s], b[s]], color="0.82",
                        lw=0.6, zorder=1)
            dots(ax, 2 * i, a.values, COL[roi], pa, rng=rng)
            dots(ax, 2 * i + 0.8, b.values, COL[roi], pb, rng=rng, py=1.14)
        zero(ax)
        ax.set_xticks([0, 0.8, 2, 2.8], ["no\nripple", "ripple"] * 2,
                      fontsize=6.5)
        _roi_lab(ax, [0.4, 2.4], ("mPFC", "mOFC"), y=-0.34)
        ax.set_ylabel("evidence for the new square (z)")
        ax.set_title(f"{letter}  {title}", pad=24)
        _clean(ax)

    def arr_rip(roi, which):
        g = A_top[(A_top.roi == roi) & (A_top.event == "first_discovery")
                  & (A_top.phase == "all")].set_index("session")
        col = f"z_{which}"
        if not len(g) or not g[col].notna().sum():
            return None, None
        s = SA_top[(SA_top.roi == roi) & (SA_top.event == "first_discovery")
                   & (SA_top.phase == "all")
                   & (SA_top.measure == f"trials {which.upper()}"
                      + (" a ripple" if which == "with" else ""))]
        return g[col].dropna(), s.p.iloc[0] if len(s) else None
    ripple_split(axes[1], arr_rip, "b", "top-30% lag-0, arrival-locked:\n"
                 "first discovery without vs with a ripple")
    _events_panel(axes[2], rew("z_{ev}"), "c", "reward-locked (0.05–1.0 s "
                  "after the press), raw", "evidence for the square\n"
                  "just uncovered (z)", rng)

    def rew_rip(roi, which):
        g = E[(E.roi == roi) & (E.cells == "top30")].set_index("session")
        c = "z_FD_norip" if which == "without" else "z_FD_rip"
        s = SE[(SE.roi == roi) & (SE.cells == "top30") & (SE.measure == c)]
        if c not in g or not g[c].notna().sum():
            return None, None
        return g[c].dropna(), s.p_perm.iloc[0] if len(s) else None
    ripple_split(axes[3], rew_rip, "d", "top-30% lag-0, reward-locked:\n"
                 "first discovery without vs with a ripple")
    _events_panel(axes[4], rew("z_{ev}_own"), "e", "reward-locked, own "
                  "reading (after − before the same press)",
                  "change in evidence for the\nsquare at the press (z)", rng,
                  legend=True)
    # f: the contrasts, top-30% only, both readings
    ax = axes[5]
    cons = (("z_FD-RC_own", "FD−RC\nown"),
            ("z_FD-RC_matched", "FD−RC\nmatched"),
            ("z_rip-norip_own", "rip−none\nown"),
            ("z_rip-norip_matched", "rip−none\nmatched"))
    for i, (c, lab) in enumerate(cons):
        for j, roi in enumerate(("mPFC", "mOFC")):
            g = E[(E.roi == roi) & (E.cells == "top30")]
            s = SE[(SE.roi == roi) & (SE.cells == "top30") & (SE.measure == c)]
            if c in g and g[c].notna().sum():
                dots(ax, i * 2.5 + j, g[c].dropna(), COL[roi], rng=rng)
                if len(s):
                    ax.text(i * 2.5 + j, 1.0 + 0.08 * j, _p(s.p_perm.iloc[0]),
                            transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=5.8, color=COL[roi])
    zero(ax)
    ax.set_xticks([i * 2.5 + 0.5 for i in range(len(cons))],
                  [lab for _, lab in cons], fontsize=6)
    for roi in ("mPFC", "mOFC"):
        ax.plot([], [], "o", color=COL[roi], label=roi)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_ylabel("contrast (z)")
    ax.set_title("f  top-30% lag-0: the contrasts,\nboth readings", pad=24)
    _clean(ax)
    n_top = json.load(open(os.path.join(d, "result.json")))["settings"]
    fig.suptitle("F1 — claim 2 a/b in the cells that are lag-0 during "
                 "execution", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.0)
    caption(fig, "Population evidence for the square just uncovered "
            "(leave-one-configuration-out place templates, z vs 1000 square-"
            "label permutations), one dot per session. Light circles: all "
            "cells of the ROI; dark squares: only the top 30% of cells by "
            "execution lag-0 tuning (r_lag000_noctrl ≥ "
            + ", ".join(f"{k.replace('_all', '')} {v:.2f}"
                        for k, v in n_top["thresholds"].items())
            + f"; {n_top['n_top_cells']} cells); sessions need ≥ 2 such cells, "
            "so fewer sessions enter. a, b: the published consolidation run "
            "(locked to ARRIVAL on the square, window +0.05…+1.0 s, ripple = "
            "any in that window), unchanged but for the cell filter. c–f: "
            "locked to the REWARD PRESS (as N2), ripple = one in the "
            "post-reward pause. e: post-press minus the same events' "
            "−0.5…−0.05 s evidence. f: FD − RC and ripple − no ripple (first "
            "discoveries), own reading and stillness-matched (1:1 on the "
            "following stillness). p = sign-flip over sessions, uncorrected; "
            "p labels take the colour of their column.", below=0.02)
    save(fig, d, "F1_claim2ab_lag0_cells")


def _psth(ax, T, col, title, letter):
    S = T[T[col].apply(lambda v: isinstance(v, np.ndarray))
          & np.isfinite(T.lag0)]
    lo, hi = np.nanpercentile(S.lag0, [30, 70])
    for lab, m, c, ls in (("top 30% lag-0", S.lag0 >= hi, "#0e3d3a", "-"),
                          ("bottom 30%", S.lag0 <= lo, "#9ec6b8", "-")):
        X = np.stack(S[m][col])
        mu, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        ax.plot(R_CENTRES, mu, color=c, ls=ls, lw=1.3,
                label=f"{lab} ({len(X)})")
        ax.fill_between(R_CENTRES, mu - se, mu + se, color=c, alpha=0.2, lw=0)
    zero(ax)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("time from ripple (s)")
    ax.set_title(f"{letter}  {title}", pad=10, fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    _clean(ax)


def _bars(ax, r, col, y):
    for c in (r or {}).get("clusters", []):
        if c["p"] < 0.05:
            ax.plot([c["start_s"], c["stop_s"]], [y, y], color=col, lw=3,
                    transform=ax.get_xaxis_transform(), solid_capstyle="butt")
            ax.text(c["stop_s"], y, f" {_p(c['p'])}", fontsize=6, color=col,
                    va="center", transform=ax.get_xaxis_transform())


def draw_f3(d):
    T = pd.read_pickle(os.path.join(d, "F3a_cells.pkl"))
    R = json.load(open(os.path.join(d, "result.json")))
    f3a, f3b = R["f3a"], R["f3b"]
    PC = pd.read_csv(os.path.join(d, "F3b_per_cell.csv"))
    fig, axes = layout([1] * 12)
    for i, rd in enumerate(("own", "matched")):
        for j, (k, lab) in enumerate(SETS):
            ax = axes[i * 3 + j]
            _psth(ax, T, f"{rd}_{k}", f"{lab}\n{READ[rd]}", "abcdef"[i * 3 + j])
            r = f3a.get(f"{rd} | {k} | modulation ±150 ms vs 0 | mPFC+mOFC", {})
            if r.get("n"):
                ax.text(0.98, 0.04, f"all cells ±150 ms: {r['mean']:+.3f}, "
                        f"{_p(r['p_perm'])}\n({r['n']} sessions)",
                        transform=ax.transAxes, ha="right", fontsize=5.8)
    axes[0].set_ylabel("firing change around ripple\n(fraction of mean rate)")
    axes[3].set_ylabel("ripple − matched moments\n(fraction of mean rate)")
    for j, (k, lab) in enumerate(SETS):
        ax = axes[6 + j]
        sub = []
        for rd, col, y in (("own", "#0e3d3a", 0.96), ("matched", "#a30d6c",
                                                       0.88)):
            r = f3a.get(f"{rd} | {k} | lag0 (time) | mPFC+mOFC")
            if not r:
                continue
            w = f3a[f"{rd} | {k} | lag0 (±150 ms) | mPFC+mOFC"]
            ax.plot(r["t_s"], r["rho"], color=col, lw=1.3, label=READ[rd])
            _bars(ax, r, col, y)
            sub.append(f"{rd} {w['rho']:+.2f} ({_p(w['p_perm'])})")
        zero(ax)
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlim(*R_TEST)
        ax.set_xlabel("time from ripple (s)")
        ax.set_title(f"{'ghi'[j]}  {lab}:\nlag-0 vs response, ±150 ms ρ\n"
                     f"{', '.join(sub)}", pad=10, fontsize=7)
        if j == 0:
            ax.legend(fontsize=6, frameon=False, loc="lower left")
        _clean(ax)
    axes[6].set_ylabel("Spearman ρ with lag-0\n(across mPFC + mOFC cells)")
    for j, (col, lab) in enumerate((("FD_own", "first discovery, after − "
                                     "before the press"),
                                    ("FD-RC_matched", "first discovery − "
                                     "known square,\nstillness-matched"),
                                    ("rip-norip_matched", "discovery with − "
                                     "without a ripple,\nstillness-matched"))):
        ax = axes[9 + j]
        for roi in ("mPFC", "mOFC"):
            g = PC[PC.roi == roi]
            ax.scatter(g.lag0, g[col], s=6, color=COL[roi], alpha=0.6, lw=0,
                       label=roi)
        r = f3b.get(f"{col} | mPFC+mOFC", {})
        zero(ax)
        ax.axvline(0, color="k", lw=0.6, ls="--", zorder=0)
        ax.set_xlabel("lag-0 tuning (CV r, execution)")
        ax.set_title(f"{'jkl'[j]}  {lab}\nρ = {r.get('rho', np.nan):+.2f}, "
                     f"{_p(r.get('p_perm', 1))}, {r.get('n')} cells", pad=10,
                     fontsize=7.5)
        _clean(ax)
    axes[9].set_ylabel("cell's evidence for the\ndiscovered square (z)")
    axes[9].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("F3 — are lag-0 cells ripple-locked, and do they carry the "
                 "square at discovery?", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.0)
    caption(fig, "mPFC + mOFC cells; lag-0 tuning = execution r_lag000_noctrl "
            "(top/bottom 30% for display only; tests use the continuous "
            "value). a–c: per cell, firing in 50 ms bins around hippocampal "
            "ripples minus its own 0.5–1.0 s flanks, divided by its mean rate; "
            "mean ± SEM over cells. d–f: ripple minus ripple-free moments of "
            "the same period with the same time since the last press and to "
            "the next (stillness-matched). Corner: all cells, ±150 ms, sign-flip "
            "over session means. g–i: across cells, Spearman of lag-0 with the "
            "response per bin; bars = clusters p < .05 over −0.5…+0.5 s, null = "
            "lag-0 permuted within session (10,000). j–l: per cell, its term of "
            "the population evidence for the square just discovered (reward-"
            "locked, 0.05–1.0 s), z vs square-label permutations; ρ with lag-0, "
            "within-session permutation. NB: a cell's lag-0 r barely replicates "
            "across odd vs even execution repeats (F2), which caps every "
            "correlation here.", below=0.02)
    save(fig, d, "F3_lag0_cells_ripples")


def draw(d):
    draw_f1(d)
    draw_f3(d)
