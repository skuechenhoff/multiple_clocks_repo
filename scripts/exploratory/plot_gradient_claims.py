#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One figure per claim of SK's gradient hypothesis, each panel showing ONE piece
of evidence directly -- per-session (or per-cell) values as they were saved,
with the statistic that analysis reported.

  claim 1  the ventral end (mOFC/OFC) is ripple-coupled in encoding, mPFC in
           execution
  claim 2  'now' neurons fire when a correct location is identified
  claim 3  in execution mPFC plays the immediate future, independent of ripples
  claim 4  lOFC encodes reward events, before / independent of ripples
  claim 5  the representation changes from exploration to execution

PLOT WHAT YOU GET: every value and every p comes from a saved results file;
nothing is re-tested here. Where a panel needs per-session window values from
saved curves (lOFC uncover response), they are taken with the same function
the analysis used, and the means are checked against the saved statistic.
Panels for analyses that have not been run yet are skipped.

    conda activate env_multiple_clocks
    python scripts/exploratory/plot_gradient_claims.py

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
import mc.analyse.swr_io as swr_io
from swr_hfb_direction import ROI_COL

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
CM = 1 / 2.54
SEED = 42
SWR = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr")
GROUP = os.path.dirname(SWR)
PVS = os.path.join(SWR, "plan_vs_store_2026-10-29")
COL = dict(ROI_COL, HC="#4F4F4F", HC_all="#4F4F4F")
PH = {"first_traversal": "#D7657F", "execution": "#5C1027"}   # phase ramp
GRAD = {"ventral": "#F7B32B", "dorsal": "#E8702A", "outside": "#9A9A9A"}


# ── helpers ──────────────────────────────────────────────────────────────
def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _p(p):
    return "p<.001" if p < 0.001 else f"p={p:.3f}".replace("0.", ".")


def dots(ax, x, v, col, p=None, marker="o", rng=None, py=1.07):
    """Sessions (or cells) as dots, mean ± SEM, the saved p above."""
    rng = rng or np.random.default_rng(SEED)
    v = np.asarray([u for u in v if np.isfinite(u)], float)
    if not v.size:
        return
    ax.scatter(x + rng.uniform(-0.13, 0.13, v.size), v, s=8, color=col,
               alpha=0.5, lw=0, zorder=2)
    ax.errorbar(x, v.mean(), v.std(ddof=1) / np.sqrt(v.size), fmt=marker,
                ms=5, mfc=col, mec="#1a1a1a", ecolor="#1a1a1a", capsize=2,
                zorder=3)
    if p is not None:
        ax.text(x, py, _p(p), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5)


def paired(ax, a, b, x=(0, 1), col="#448363", p=None, labels=None, col2=None):
    """Two values per session joined by a line; p of the paired contrast."""
    sh = sorted(set(a) & set(b))
    for s in sh:
        ax.plot(x, [a[s], b[s]], color="0.82", lw=0.6, zorder=1)
    rng = np.random.default_rng(SEED)
    dots(ax, x[0], [a[s] for s in sh], col, rng=rng)
    dots(ax, x[1], [b[s] for s in sh], col2 or col, rng=rng)
    if p is not None:
        ax.plot(x, [1.03, 1.03], transform=ax.get_xaxis_transform(),
                color="k", lw=0.7, clip_on=False)
        ax.text(np.mean(x), 1.05, _p(p), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5)
    if labels:
        ax.set_xticks(list(x), labels)


def zero(ax):
    ax.axhline(0, color="k", lw=0.6, ls="--", zorder=0)


PANEL_W, PANEL_H, NCOL = 6.2, 5.8, 3      # cm per column / row


def layout(spans):
    """Panels on a 3-column grid; a span of 2 gives a panel two columns."""
    from matplotlib.gridspec import GridSpec
    rows, cur = [], []
    for s in spans:
        if sum(cur) + s > NCOL:
            rows.append(cur)
            cur = []
        cur.append(s)
    rows.append(cur)
    fig = plt.figure(figsize=(NCOL * PANEL_W * CM, len(rows) * PANEL_H * CM))
    gs = GridSpec(len(rows), NCOL, figure=fig, hspace=1.0, wspace=0.6)
    axes = []
    for r, row in enumerate(rows):
        c = 0
        for s in row:
            axes.append(fig.add_subplot(gs[r, c:c + s]))
            c += s
    return fig, axes


def caption(fig, text, y=None, below=-0.02):
    import textwrap
    w = int(fig.get_figwidth() * 2.54 / 0.125)        # ~chars per line
    fig.text(0.01, below, textwrap.fill(text, w), fontsize=6.8, color="0.25",
             va="top")


def save(fig, d, name):
    for e in (".png", ".pdf"):
        fig.savefig(os.path.join(d, name + e), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _exists(*p):
    return os.path.exists(os.path.join(*p))


# ── claim 1 ──────────────────────────────────────────────────────────────
def claim1(d):
    have_g = _exists(SWR, "gradient_hfb_2026-10-30", "result.json")
    have_c = _exists(SWR, "cell_ripple_depth_2026-10-30", "sites.csv")
    fig, axes = layout([1, 1, 1, 1, 2] + [1] * (have_g + have_c))
    # a, b: ripple-locked HFB, mOFC vs mPFC, one axis each phase
    for ax, ph, lab in ((axes[0], "first_traversal", "first traversal"),
                        (axes[1], "execution", "execution")):
        r = json.load(open(os.path.join(SWR, "hfb_mofc_mpfc_phases_2026-10-08",
                                        ph, "result.json")))
        t = np.asarray(r["t_ms"])
        for k, roi in enumerate(("mOFC", "mPFC")):
            q = r["results"][roi]
            m, se = np.asarray(q["mean"]), np.asarray(q["sem"])
            ax.plot(t, m, color=COL[roi], lw=1.3,
                    label=f"{roi} (n = {q['n_units']})")
            ax.fill_between(t, m - se, m + se, color=COL[roi], alpha=0.2, lw=0)
            for c in q["clusters"]:
                if c["p"] < 0.05 and c["sign"] == "increase":
                    y = -0.012 - 0.004 * k
                    ax.plot([c["start_ms"], c["stop_ms"]], [y, y],
                            color=COL[roi], lw=3, solid_capstyle="butt")
                    ax.text(c["stop_ms"] + 20, y, _p(c["p"]), fontsize=6,
                            va="center", color=COL[roi])
        zero(ax)
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlim(-750, 750)
        ax.set_xlabel("time from ripple (ms)")
        ax.set_title(f"{'a' if ph == 'first_traversal' else 'b'}  {lab}")
        _clean(ax)
    axes[0].set_ylabel("ripple-locked HFB (z, real − null)")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    # c: before vs after the ripple, first traversal, per session
    L = pd.read_csv(os.path.join(SWR, "hfb_ripple_coupling_chain_2026-10-07",
                                 "chain_links_per_session.csv"))
    T = pd.read_csv(os.path.join(SWR, "hfb_ripple_coupling_chain_2026-10-07",
                                 "chain_tests.csv"))
    ax = axes[2]
    rng = np.random.default_rng(SEED)
    for i, roi in enumerate(("mOFC", "mPFC")):
        g = L[L.roi == roi].set_index("session")
        for s in g.index:
            ax.plot([2 * i, 2 * i + 0.8], [g.pre[s], g.post[s]], color="0.85",
                    lw=0.5, zorder=1)
        tp = T[(T.roi == roi) & (T.measure == "pre")].p_perm.iloc[0]
        tq = T[(T.roi == roi) & (T.measure == "post")].p_perm.iloc[0]
        dots(ax, 2 * i, g.pre, COL[roi], tp, rng=rng)
        dots(ax, 2 * i + 0.8, g.post, COL[roi], tq, rng=rng, py=1.08)
    zero(ax)
    ax.set_xticks([0, 0.8, 2, 2.8], ["before", "after", "before", "after"],
                  fontsize=7)
    ax.text(0.4, -0.3, "mOFC", transform=ax.get_xaxis_transform(),
            ha="center", color=COL["mOFC"])
    ax.text(2.4, -0.3, "mPFC", transform=ax.get_xaxis_transform(),
            ha="center", color=COL["mPFC"])
    ax.set_ylabel("HFB, 250 ms window (z)")
    ax.set_title("c  first traversal:\nbefore vs after ripple", pad=20)
    _clean(ax)
    # d: same-ripple coupling
    C = pd.read_csv(os.path.join(SWR, "hfb_ripple_coupling_chain_2026-10-07",
                                 "chain_coupling_per_session.csv"))
    ax = axes[3]
    for i, (k, m_) in enumerate((("fwd", "fwd"), ("rev", "rev"),
                                 ("asym", "asym"))):
        p = T[T.measure == m_].p_perm.iloc[0]
        dots(ax, i, C[k], "#8C8C8C", p, rng=rng)
    zero(ax)
    ax.set_xticks([0, 1, 2], ["mOFC pre →\nmPFC post", "mPFC pre →\nmOFC post",
                              "asymmetry"], fontsize=6.5)
    ax.set_ylabel("same-ripple coupling\n(Fisher z, real − shifted)")
    ax.set_title("d  does mOFC-before predict\nmPFC-after, ripple by ripple?",
                 pad=20)
    _clean(ax)
    # e: cells' spatial code inside vs outside encode ripples
    S = pd.read_csv(os.path.join(SWR, "ripple_templates_2026-10-07",
                                 "summary.csv"))
    P = pd.read_csv(os.path.join(SWR, "ripple_templates_2026-10-07",
                                 "per_session.csv"))
    P = P[P.template_phase == "any"]
    ax = axes[4]
    for i, roi in enumerate(("HC_all", "mPFC", "mOFC")):
        g = P[P.roi == roi].set_index("session")
        pr = S[(S.roi == roi) & (S.template_phase == "any")].set_index("measure")
        for s in g.index:
            ax.plot([2 * i, 2 * i + 0.8], [g.z_allT_flankW_encode[s],
                                           g.z_allT_ripW_encode[s]],
                    color="0.85", lw=0.5, zorder=1)
        dots(ax, 2 * i, g.z_allT_flankW_encode, COL[roi],
             pr.loc["z_allT_flankW_encode", "p_signflip"], rng=rng)
        dots(ax, 2 * i + 0.8, g.z_allT_ripW_encode, COL[roi],
             pr.loc["z_allT_ripW_encode", "p_signflip"], rng=rng, py=1.08)
        ax.text(2 * i + 0.4, -0.3, roi.replace("_all", ""),
                transform=ax.get_xaxis_transform(), ha="center", color=COL[roi])
    zero(ax)
    ax.set_xticks([0, 0.8, 2, 2.8, 4, 4.8], ["flank", "ripple"] * 3,
                  fontsize=6.5)
    ax.set_ylabel("evidence for the current\nsquare (z, cell population)")
    ax.set_title("e  cells, encoding: current square\noutside vs inside ripples",
                 pad=20)
    _clean(ax)
    k = 5
    if have_g:
        _g1_panel(axes[k], "f")
        k += 1
    if have_c:
        _c4_panel(axes[k], "g" if have_g else "f")
    fig.suptitle("Claim 1 — the ventral end (mOFC) is ripple-coupled in "
                 "encoding, mPFC in execution", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", y=1.07)
    caption(fig, "a, b: ripple-locked HFB, mean ± SEM over sessions (real − 6 "
            "shifted nulls, own baseline ≥ 750 ms, 50 ms sliding mean); bars: "
            "increase clusters (sign-flip). c: per-session HFB in [−250, 0) and "
            "(0, +250] ms around every first-traversal ripple, p vs 0. d: "
            "ripple-by-ripple Spearman (Fisher z) of one region's pre with the "
            "other's post, real − shifted; 11 sessions with both. e: cell "
            "population evidence for the current square with the usual "
            "(all-time) spatial template, in encode ripple windows vs their "
            "matched flanks (ripple_templates). Lines join sessions. f: G1, "
            "per medial HFB contact, ripple-locked HFB (±250 ms − 250–750 ms "
            "flanks, 20 shifted nulls) against MNI z, centred within session, "
            "pooled fits for display; the test is one slope per session. g: C4, "
            "cell ripple modulation (encoding − execution) per recording site.",
            below=-0.07)
    save(fig, d, "claim1_ventral_ripple_coupling")


def _g1_panel(ax, letter):
    R = pd.read_csv(os.path.join(SWR, "gradient_hfb_2026-10-30",
                                 "contact_values.csv"))
    res = json.load(open(os.path.join(SWR, "gradient_hfb_2026-10-30",
                                      "result.json")))
    m = R[R.medial & (R.lock == "ripple")]
    for ph in ("first_traversal", "execution"):
        g = m[m.set == ph]
        v = g.value - g.groupby("session").value.transform("mean")
        zc = g.z - g.groupby("session").z.transform("mean")
        ax.scatter(zc, v, s=7, color=PH[ph], alpha=0.6, lw=0,
                   label=ph.replace("_", " "))
        ok = np.isfinite(v) & np.isfinite(zc)
        b = np.polyfit(zc[ok], v[ok], 1)
        xx = np.linspace(zc.min(), zc.max(), 5)
        ax.plot(xx, np.polyval(b, xx), color=PH[ph], lw=1.5)
    zero(ax)
    r = res["tests"]["G1"]["PRIMARY first_traversal - execution, slope on z"]
    ax.set_xlabel("MNI z − session mean (mm)\nventral ← → dorsal")
    ax.set_ylabel("ripple-locked HFB, session-centred (z)")
    ax.set_title(f"{letter}  HFB contacts along the axis\n(first trav. − exec.) "
                 f"slope {_p(r['p_perm'])}, n = {r['n']}", pad=20)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.4), ncol=2)
    _clean(ax)


def _c4_panel(ax, letter):
    S = pd.read_csv(os.path.join(SWR, "cell_ripple_depth_2026-10-30",
                                 "sites.csv"))
    res = json.load(open(os.path.join(SWR, "cell_ripple_depth_2026-10-30",
                                      "result.json")))["tests"]
    for roi in ("mPFC", "mOFC"):
        g = S[S.roi == roi]
        ax.scatter(g.z, g.enc_minus_exe, s=6 + 3 * g.n_cells, color=COL[roi],
                   alpha=0.7, lw=0.3, edgecolor="0.2", label=roi)
    zero(ax)
    r = res["PRIMARY enc - exe vs z (mPFC + mOFC sites)"]
    ax.set_xlabel("MNI z (mm)\nventral ← → dorsal")
    ax.set_ylabel("cell ripple modulation,\nencoding − execution")
    ax.set_title(f"{letter}  cells per site along the axis\nρ = {r['rho']:+.2f}, "
                 f"{_p(r['p_perm'])}, {r['n_sites']} sites", pad=20)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.4), ncol=2)
    _clean(ax)


# ── claim 2 ──────────────────────────────────────────────────────────────
def claim2(d):
    have_n = _exists(SWR, "discovery_cells_2026-10-30", "result.json")
    fig, axes = layout([2, 1, 1, 2])
    C = pd.read_csv(os.path.join(SWR, "consolidation_2026-10-01",
                                 "per_session.csv"))
    S = pd.read_csv(os.path.join(SWR, "consolidation_2026-10-01",
                                 "summary.csv"))
    S["measure"] = S.measure.str.strip()
    rng = np.random.default_rng(SEED)
    # a: the just-uncovered square, by event type
    ax = axes[0]
    ev = ("first_discovery", "repeat_correct", "incorrect")
    for j, roi in enumerate(("mPFC", "mOFC", "HC_all")):
        for i, e in enumerate(ev):
            g = C[(C.roi == roi) & (C.event == e) & (C.phase == "all")]
            p = S[(S.roi == roi) & (S.event == e) & (S.phase == "all")
                  & (S.measure == "represented at all")].p.iloc[0]
            dots(ax, i * 4 + j, g.z_all, COL[roi], p, rng=rng,
                 py=1.0 + 0.08 * (j % 2))
    zero(ax)
    ax.set_xticks([1, 5, 9], ["first discovery\n(new reward)",
                              "correct,\nalready known", "incorrect"],
                  fontsize=7)
    ax.set_ylabel("evidence for the square\njust uncovered (z)")
    for roi in ("mPFC", "mOFC", "HC_all"):
        ax.plot([], [], "o", color=COL[roi], label=roi.replace("_all", ""))
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.set_title("a  which squares are held after an uncover?", pad=20)
    _clean(ax)
    # b: with vs without a ripple, first discovery
    ax = axes[1]
    for i, roi in enumerate(("mPFC", "mOFC")):
        g = C[(C.roi == roi) & (C.event == "first_discovery")
              & (C.phase == "all")].set_index("session")
        p = S[(S.roi == roi) & (S.event == "first_discovery") & (S.phase == "all")
              & (S.measure == "WITH minus WITHOUT")].p.iloc[0]
        paired(ax, g.z_without.dropna().to_dict(), g.z_with.dropna().to_dict(),
               x=(2 * i, 2 * i + 0.8), col=COL[roi], p=p)
        ax.text(2 * i + 0.4, -0.32, roi, transform=ax.get_xaxis_transform(),
                ha="center", color=COL[roi])
    zero(ax)
    ax.set_xticks([0, 0.8, 2, 2.8], ["no\nripple", "ripple"] * 2, fontsize=6.5)
    ax.set_ylabel("evidence for the new square (z)")
    ax.set_title("b  first discovery:\nwithout vs with a ripple", pad=22)
    _clean(ax)
    # c: lag-0 cells x storage ripple contrast
    ax = axes[2]
    PC = pd.read_csv(os.path.join(SWR, "storage_vs_planning_2026-10-01",
                                  "per_cell.csv"))
    SS = pd.read_csv(os.path.join(SWR, "storage_vs_planning_2026-10-01",
                                  "summary.csv"))
    g = PC[PC.roi == "mPFC"].dropna(subset=["lag0_strength",
                                            "storage_rip_minus_norip"])
    ax.scatter(g.lag0_strength, g.storage_rip_minus_norip, s=8,
               color=COL["mPFC"], alpha=0.6, lw=0)
    b = np.polyfit(g.lag0_strength, g.storage_rip_minus_norip, 1)
    xx = np.linspace(g.lag0_strength.min(), g.lag0_strength.max(), 5)
    ax.plot(xx, np.polyval(b, xx), color=COL["mPFC"], lw=1.4)
    r = SS[(SS.roi == "mPFC") & (SS.predictor == "lag0_strength")
           & (SS.measure == "storage_rip_minus_norip")].iloc[0]
    zero(ax)
    ax.set_xlabel("lag-0 tuning (CV r, execution)")
    ax.set_ylabel("firing after storage events:\nnear ripple − not")
    ax.set_title(f"c  mPFC cells (n = {int(r.n_cells)})\nr = {r.r:+.2f}, "
                 f"{_p(r.p)} (arrival-locked)", pad=20)
    _clean(ax)
    # d: lag profiles along the gradient
    ax = axes[3]
    M = pd.read_csv(os.path.join(GROUP, "cell_gradient_master",
                                 "2026-08-28_15-19-35", "per_cell_master.csv"))
    F = pd.read_csv(os.path.join(GROUP, "cell_gradient_master",
                                 "2026-08-28_15-19-35", "final_splits_2026-10-06",
                                 "final_splits_per_cell.csv"))
    LZ = pd.read_csv(os.path.join(GROUP, "cell_gradient_master",
                                  "2026-08-28_15-19-35", "final_splits_2026-10-06",
                                  "lagwise_vs_zero.csv"))
    M = M.merge(F[["neuron", "pc1_ventral_dorsal_group"]], on="neuron")
    lags = np.arange(0, 360, 30)
    cols = [f"r_lag{l:03d}_noctrl" for l in lags]
    for grp in ("ventral", "dorsal", "outside"):
        X = M[M.pc1_ventral_dorsal_group == grp][cols].to_numpy(float)
        m, se = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1) / np.sqrt(len(X))
        ax.plot(lags, m, color=GRAD[grp], lw=1.4, marker="o", ms=2.5,
                label=f"{grp} (n = {len(X)})")
        ax.fill_between(lags, m - se, m + se, color=GRAD[grp], alpha=0.18, lw=0)
        sig = LZ[(LZ.scheme == "pc1_ventral_dorsal") & (LZ.group == grp)
                 & (LZ.p_fdr_across_groups < 0.05)]
        for l in sig.lag_deg:
            ax.text(l, m[list(lags).index(l)] + se[list(lags).index(l)] + 0.005,
                    "*", ha="center", color=GRAD[grp], fontsize=9)
    zero(ax)
    ax.set_xticks([0, 90, 180, 270])
    ax.set_xlabel("future lag (°)")
    ax.set_ylabel("spatial consistency (CV r)")
    ax.set_title("d  mPFC cells along the fMRI gradient", pad=20)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    _clean(ax)
    fig.suptitle("Claim 2 — 'now' neurons fire when a correct location is "
                 "identified", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.07)
    caption(fig, "a: population evidence for the square just uncovered, "
            "window +0.05…+1.0 s (consolidation, 2026-10-01); dots = sessions; "
            "mPFC first discovery − repeat +0.77, P = .0035 (paired). b: the "
            "same for first discoveries with vs without a ripple in the window. "
            "c: per mPFC cell, lag-0 tuning against [firing after storage "
            "events with a ripple within ±0.5 s − without]; Pearson across "
            "cells. NB: a–c are locked to ARRIVAL on the square, not the reward "
            "press (re-run as N1/N2). d: mean ± SEM of the 12-lag spatial "
            "consistency per gradient group (PC1 median split); * q_FDR < .05.",
            y=-0.18)
    save(fig, d, "claim2_now_cells_at_discovery")
    if have_n:
        _claim2_new(d)


def _claim2_new(d):
    """N1/N2 figures are drawn by `discovery_cells_figures.draw` from their
    own saved outputs; copied here so claim 2 sits in one folder."""
    import shutil
    src = os.path.join(SWR, "discovery_cells_2026-10-30")
    for name, new in (("N1_discovery_cells", "claim2b_now_cells_reward_locked"),
                      ("N2_new_square", "claim2c_new_square_reward_locked")):
        for e in (".png", ".pdf"):
            shutil.copy(os.path.join(src, name + e), os.path.join(d, new + e))


# ── claim 3 ──────────────────────────────────────────────────────────────
def claim3(d):
    fig, axes = layout([2, 1, 2])
    rng = np.random.default_rng(SEED)
    # a: firing at the preferred square, all task time vs inside ripples
    PL = pd.read_csv(os.path.join(SWR, "preferred_lag_firing_2026-10-01",
                                  "per_cell.csv"))
    PS = pd.read_csv(os.path.join(SWR, "preferred_lag_firing_2026-10-01",
                                  "summary.csv"))
    ax = axes[0]
    for i, roi in enumerate(("HC", "mPFC", "mOFC")):
        for j, w in enumerate(("all", "ripple")):
            g = PL[(PL.roi == roi) & (PL.phase == "execute") & (PL.window == w)
                   & (PL.pref_lag != 0)]
            p = PS[(PS.roi == roi) & (PS.phase == "execute") & (PS.window == w)
                   & (PS.group == "pref lag != 0")].p.iloc[0]
            dots(ax, 2.5 * i + j, g.sel, COL[roi], p, rng=rng, py=1.0 + 0.08 * j)
        ax.text(2.5 * i + 0.5, -0.3, roi, transform=ax.get_xaxis_transform(),
                ha="center", color=COL[roi])
    zero(ax)
    ax.set_xticks([0, 1, 2.5, 3.5, 5, 6], ["task", "ripple"] * 3, fontsize=6.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("selectivity for the cell's\npreferred square (execution)")
    ax.set_title("a  future-tuned cells (pref. lag ≠ 0):\nat their square in "
                 "task time vs inside ripples", pad=20)
    _clean(ax)
    # b: future_shift -- gate and primary, mPFC units and medial HFB vs HC
    FS = pd.read_csv(os.path.join(SWR, "future_shift_2026-10-07",
                                  "per_session.csv"))
    FSS = pd.read_csv(os.path.join(SWR, "future_shift_2026-10-07",
                                   "summary.csv"))
    ax = axes[1]
    sets = (("spikes", "mPFC", "mPFC\nunits"), ("hfb", "MedialFrontal",
                                                 "medial\nfrontal HFB"),
            ("spikes", "HC_all", "HC\nunits"))
    for i, (sig, roi, lab) in enumerate(sets):
        base = FS[(FS.signal == sig) & (FS.roi == roi) & (FS.ref == "block")
                  & (FS.phase == "all")]
        sb_ = FSS[(FSS.signal == sig) & (FSS.roi == roi) & (FSS.ref == "block")
                  & (FSS.phase == "all")]
        gate = base[base.quantity == "now"].z_level_ctrl
        pg = sb_[(sb_.quantity == "now") & (sb_.measure == "z_level_ctrl")].p
        dots(ax, 2.5 * i, gate, COL.get(roi, "#888"),
             pg.iloc[0] if len(pg) else None, rng=rng, marker="s")
        I = base[base.quantity == "I"].z_diff
        pI = sb_[(sb_.quantity == "I") & (sb_.measure == "z_diff")].p
        dots(ax, 2.5 * i + 1, I, COL.get(roi, "#888"),
             pI.iloc[0] if len(pI) else None, rng=rng, py=1.16)
        ax.text(2.5 * i + 0.5, -0.42, lab, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=6.5)
    zero(ax)
    ax.set_xticks([0, 1, 2.5, 3.5, 5, 6], ["now\n(gate)", "shift"] * 3,
                  fontsize=6)
    ax.set_ylabel("z vs label permutation")
    ax.set_title("b  does a ripple move activity\ntowards the next squares?",
                 pad=20)
    _clean(ax)
    # c: T4 mPFC -- path content of mPFC ripples, execution
    T4 = pd.read_csv(os.path.join(T4_DIR, "per_session.csv"))
    T4S = pd.read_csv(os.path.join(T4_DIR, "summary.csv"))
    ax = axes[2]
    for i, roi in enumerate(("HC_all", "mPFC")):
        for j, q in enumerate(("PAST", "now", "FUT")):
            g = T4[(T4.roi == roi) & (T4.subset == "all")
                   & (T4.phase == "execute") & (T4.quantity == q)]
            p = T4S[(T4S.roi == roi) & (T4S.subset == "all")
                    & (T4S.phase == "execute") & (T4S.quantity == q)
                    & (T4S.measure == "z_diff")].p_perm.iloc[0]
            dots(ax, 4 * i + j, g.z_diff, COL[roi], p, rng=rng,
                 py=1.0 + 0.08 * (j % 2))
        ax.text(4 * i + 1, -0.24, roi.replace("_all", ""),
                transform=ax.get_xaxis_transform(), ha="center",
                color=COL[roi])
    zero(ax)
    ax.set_xticks([0, 1, 2, 4, 5, 6], ["past", "now", "next"] * 2, fontsize=6.5)
    ax.set_ylabel("ripple − matched still moment (z)")
    ax.set_title("c  execution ripples: which squares?\n(T4, all ripples)",
                 pad=20)
    _clean(ax)
    fig.suptitle("Claim 3 — in execution, mPFC plays the immediate future, "
                 "independent of ripples", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.07)
    caption(fig, "a: per cell, selectivity for the preferred square (chosen on "
            "training layouts, measured on the held-out one), execution, in "
            "all task time vs inside ripple windows; cells with a future "
            "preferred lag; p across cells. b: future_shift (pre-registered "
            "2026-10-07): the gate = current square in ordinary still moments "
            "(squares); the primary = ripple − matched change towards the next "
            "three squares (circles); dots = sessions. c: T4 spike evidence in "
            "execution ripples for the previous 3, current and next 3 squares, "
            "ripple − matched still moment.", y=-0.2)
    save(fig, d, "claim3_execution_future_without_ripples")


# ── claim 4 ──────────────────────────────────────────────────────────────
def claim4(d):
    from swr_plan_vs_store import _base_smooth
    have_g = _exists(SWR, "gradient_hfb_2026-10-30", "result.json")
    fig, axes = layout([2, 1, 1] + [1, 1] * have_g)
    rng = np.random.default_rng(SEED)
    z = np.load(os.path.join(PVS, "T1b_hfb_state", "curves.npz"))
    st = json.load(open(os.path.join(PVS, "T1b_hfb_state", "result.json")))["stats"]
    t = z["t_event_ms"]
    ax = axes[0]
    vals = {}
    for key, lab, col in (("FT correct", "first traversal", "#0e3d3a"),
                          ("exec correct", "execution", "#9ec6b8")):
        X = np.stack([_base_smooth(c, t) for c in z[f"event|{key}|lOFC"]])
        m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
        ax.plot(t, m, color=col, lw=1.3, label=f"correct, {lab} (n = {len(X)})")
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.25, lw=0)
        # per-session window values exactly as T1b computed them
        raw = z[f"event|{key}|lOFC"]
        b_ = (t >= -1500) & (t < -1000)
        w_ = (t >= 0) & (t < 500)
        v = np.array([(c - c[b_].mean())[w_].mean() for c in raw])
        assert np.isclose(v.mean(), st[f"event | lOFC | {key}"]["mean"])
        vals[key] = dict(zip(z[f"event|{key}|lOFC|sessions"], v))
    ax.axvspan(-1500, -1000, color="0.92", lw=0)
    zero(ax)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("time from correct uncover (ms)")
    ax.set_ylabel("lOFC HFB (z, real − null)")
    ax.set_title("a  lOFC response to a correct uncover")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    _clean(ax)
    ax = axes[1]
    for j, key in enumerate(("FT correct", "exec correct")):
        dots(ax, j, list(vals[key].values()),
             "#0e3d3a" if j == 0 else "#9ec6b8",
             st[f"event | lOFC | {key}"]["p_perm"], rng=rng)
    zero(ax)
    ax.set_xticks([0, 1], ["first\ntraversal", "execution"], fontsize=7)
    ax.set_ylabel("0–0.5 s response (z)")
    ax.set_title("b  per session", pad=20)
    _clean(ax)
    ax = axes[2]
    L = pd.read_csv(os.path.join(SWR, "hfb_ripple_coupling_chain_lOFC_2026-10-07",
                                 "chain_links_per_session.csv"))
    T = pd.read_csv(os.path.join(SWR, "hfb_ripple_coupling_chain_lOFC_2026-10-07",
                                 "chain_tests.csv"))
    g = L[L.roi == "lOFC"].set_index("session")
    paired(ax, g.pre.to_dict(), g.post.to_dict(), col=COL["lOFC"],
           p=T[(T.roi == "lOFC") & (T.measure == "direction")].p_perm.iloc[0],
           labels=["before", "after"])
    zero(ax)
    ax.set_ylabel("lOFC HFB, 250 ms window (z)")
    ax.set_title("c  first-traversal ripples:\nlOFC before vs after", pad=22)
    _clean(ax)
    if have_g:
        _g2_panel(axes[3], "d")
        _g3_panel(axes[4], "e")
    fig.suptitle("Claim 4 — lOFC encodes reward events, before / independent "
                 "of ripples", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.07)
    caption(fig, "a: uncover-locked lOFC HFB (T1b, 20 shifted nulls, own "
            "baseline −1.5…−1.0 s in grey, 50 ms sliding mean), mean ± SEM. "
            "b: 0–0.5 s per session; p = T1b sign-flip vs 0. c: per-session "
            "lOFC HFB in [−250, 0) and (0, +250] ms around every first-traversal "
            "ripple; p = direction index (after − before) vs 0. d: G2, "
            "per frontal HFB contact, response to correct first-traversal "
            "uncovers (0–0.5 s) against MNI z; test = one slope per session. "
            "e: G3a, per session, lOFC HFB in [−250, 0) ms before ripples with "
            "no press in the preceding 2 s (quiet) and before ripples that "
            "follow a press; p vs 0.", below=-0.07)
    save(fig, d, "claim4_lofc_reward_events")


def _g2_panel(ax, letter):
    R = pd.read_csv(os.path.join(SWR, "gradient_hfb_2026-10-30",
                                 "contact_values.csv"))
    res = json.load(open(os.path.join(SWR, "gradient_hfb_2026-10-30",
                                      "result.json")))["tests"]["G2"]
    ev = R[(R.lock == "event") & (R.set == "FT correct")]
    for lab, m, col, mk in (("medial", ev[ev.medial], COL["mPFC"], "o"),
                            ("lOFC", ev[ev.roi == "lOFC"], COL["lOFC"], "s"),
                            ("other frontal", ev[~ev.medial & (ev.roi != "lOFC")],
                             "#8C8C8C", "^")):
        ax.scatter(m.z, m.value, s=9, color=col, marker=mk, alpha=0.7, lw=0,
                   label=lab)
    zero(ax)
    r = res["PRIMARY FT correct, slope on z"]
    ax.set_xlabel("MNI z (mm)\nventral ← → dorsal")
    ax.set_ylabel("first-traversal correct uncover,\n0–0.5 s (z)")
    ax.set_title(f"{letter}  reward response along the axis\nwithin-session "
                 f"slope {_p(r['p_perm'])}, n = {r['n']}", pad=20)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.4), ncol=3)
    _clean(ax)


def _g3_panel(ax, letter):
    R = pd.read_csv(os.path.join(SWR, "gradient_hfb_2026-10-30",
                                 "contact_values.csv"))
    res = json.load(open(os.path.join(SWR, "gradient_hfb_2026-10-30",
                                      "result.json")))["tests"]["G3"]
    lof = R[R.lofc]
    rng = np.random.default_rng(SEED)
    pre = {k: lof[(lof.lock == "ripple") & (lof.set == k)].groupby("session")
           .pre.mean() for k in ("quiet", "after press")}
    dots(ax, 0, pre["quiet"], COL["lOFC"],
         res["PRIMARY quiet ripples, pre-ripple [-250, 0)"]["p_perm"], rng=rng)
    dots(ax, 1, pre["after press"], "#d4749f",
         res["after-press ripples, pre-ripple [-250, 0)"]["p_perm"], rng=rng)
    zero(ax)
    ax.set_xticks([0, 1], ["quiet\nripples", "after-press\nripples"],
                  fontsize=7)
    ax.set_ylabel("lOFC HFB before the ripple\n[−250, 0) ms (z)")
    ax.set_title(f"{letter}  is the lOFC lead\nthe press response?", pad=20)
    _clean(ax)


# ── claim 5 ──────────────────────────────────────────────────────────────
T4_DIR = os.path.join(SWR, "plan_vs_store_2026-10-30", "T4_path_content")
READINGS = (("z_diff", "stillness-matched (pre-registered)"),
            ("z_ownflank", "same-event (own flanks)"))


def claim5(d):
    """T4 path content in both readings: each ripple against stratum-matched
    still moments (pre-registered) and against its own flanks on the same
    square."""
    T4 = pd.read_csv(os.path.join(T4_DIR, "per_session.csv"))
    T4S = pd.read_csv(os.path.join(T4_DIR, "summary.csv"))
    PC = pd.read_csv(os.path.join(T4_DIR, "phase_contrast_posthoc.csv"))
    fig, axes = layout([2, 1, 2, 1, 1, 1])
    k = 0
    for meas, lab in READINGS:
        # past / current / next, post-correct, first traversal vs execution
        ax = axes[k]
        for j, q in enumerate(("PAST", "now", "FUT")):
            g = T4[(T4.roi == "HC_all") & (T4.subset == "post_correct")
                   & (T4.quantity == q)]
            a = g[g.phase == "first_traversal"].set_index("session")[meas]
            b = g[g.phase == "execute"].set_index("session")[meas]
            pc = PC[(PC.roi == "HC_all") & (PC.subset == "post_correct")
                    & (PC.quantity == q) & (PC.measure == meas)].p_perm
            paired(ax, a.dropna().to_dict(), b.dropna().to_dict(),
                   x=(3 * j, 3 * j + 1), col=PH["first_traversal"],
                   col2=PH["execution"], p=pc.iloc[0] if len(pc) else None)
            ax.text(3 * j + 0.5, -0.33, {"PAST": "past 3", "now": "current",
                                         "FUT": "next 3"}[q],
                    transform=ax.get_xaxis_transform(), ha="center")
        zero(ax)
        ax.set_xticks([0, 1, 3, 4, 6, 7], ["first\ntrav.", "exec."] * 3,
                      fontsize=6)
        ax.set_ylabel("ripple − reference (z)")
        n = int(T4[(T4.roi == "HC_all") & (T4.subset == "post_correct")
                   & (T4.phase == "first_traversal")
                   & (T4.quantity == "FUT")][meas].notna().sum())
        ax.set_title(f"{'ac'[k // 2]}  HC, post-correct ripples — {lab}\n"
                     f"({n} sessions in the first traversal)", pad=22)
        _clean(ax)
        # past - next, all ripples and post-correct, vs 0 per phase
        ax = axes[k + 1]
        rng = np.random.default_rng(SEED)
        for i, sub in enumerate(("post_correct", "all")):
            for j, ph in enumerate(("first_traversal", "execute")):
                g = T4[(T4.roi == "HC_all") & (T4.subset == sub)
                       & (T4.phase == ph) & (T4.quantity == "PAST-FUT")]
                r = T4S[(T4S.roi == "HC_all") & (T4S.subset == sub)
                        & (T4S.phase == ph) & (T4S.quantity == "PAST-FUT")
                        & (T4S.measure == meas)].p_perm
                dots(ax, 2.5 * i + j, g[meas],
                     PH["first_traversal" if j == 0 else "execution"],
                     r.iloc[0] if len(r) else None, rng=rng, py=1.0 + 0.08 * j)
            ax.text(2.5 * i + 0.5, -0.33, "post-correct" if i == 0 else
                    "all ripples", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=7)
        zero(ax)
        ax.set_xticks([0, 1, 2.5, 3.5], ["first\ntrav.", "exec."] * 2,
                      fontsize=6)
        ax.set_ylabel("past − next (z)")
        ax.set_title(f"{'bd'[k // 2]}  HC past − next,\n{lab.split(' (')[0]}",
                     pad=22)
        _clean(ax)
        k += 2
    # mPFC, both readings, past - next post-correct
    for ax, (meas, lab), letter in zip(axes[4:6], READINGS, "ef"):
        rng = np.random.default_rng(SEED)
        for j, ph in enumerate(("first_traversal", "execute")):
            g = T4[(T4.roi == "mPFC") & (T4.subset == "post_correct")
                   & (T4.phase == ph) & (T4.quantity == "PAST-FUT")]
            r = T4S[(T4S.roi == "mPFC") & (T4S.subset == "post_correct")
                    & (T4S.phase == ph) & (T4S.quantity == "PAST-FUT")
                    & (T4S.measure == meas)].p_perm
            dots(ax, j, g[meas], PH["first_traversal" if j == 0
                                    else "execution"],
                 r.iloc[0] if len(r) else None, rng=rng)
        zero(ax)
        ax.set_xticks([0, 1], ["first\ntraversal", "execution"], fontsize=7)
        ax.set_ylabel("past − next (z)")
        ax.set_title(f"{letter}  mPFC post-correct,\n{lab.split(' (')[0]}",
                     pad=22)
        _clean(ax)
    fig.suptitle("Claim 5 — the representation changes from exploration to "
                 "execution (both readings)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", y=1.07)
    caption(fig, "T4 (pre-registered 2026-10-29), HC and mPFC units: spike "
            "evidence in the ripple window (±125 ms) for the squares just "
            "walked (past 3), the current one and the next 3, against "
            "same-block references and a square-label permutation. Two "
            "readings. Stillness-matched (a, b, e; the pre-registered one): "
            "minus stratum-matched still moments without a ripple in the same "
            "post-uncover pauses. Same-event (c, d, f; added 2026-10-30): minus "
            "the same ripple's own flanks (0.25–0.5 s either side, same square, "
            "no other ripple within them), so fewer sessions qualify. Lines "
            "join sessions; bracket p = first traversal − execution (POST HOC, "
            "paired sign-flip); p above dots = vs 0.", below=-0.03)
    save(fig, d, "claim5_explore_to_execute")


def main():
    d = os.path.join(SWR, f"gradient_claims_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    for f in (claim1, claim2, claim3, claim4, claim5):
        f(d)
        print(f"  {f.__name__} done")
    print(f"-> {d}")


if __name__ == "__main__":
    main()
