#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics for the ripple-triggered RSA: how much data there is, what supports
each model, what the nulls actually look like, and whether the effect is
time-locked.

Four figures, none of them a result:

  coverage_sweep      ripples and spikes per RDM condition as the inclusion
                      rule is loosened, from +-0.5 s around the press to the
                      whole inter-uncover interval. Also how many runs of each
                      configuration a session contributes.
  model_support       what the two models are actually built from -- how many
                      configuration pairs share a reward location, and at what
                      rank offset. This is the ceiling on what either model can
                      ever explain.
  null_distributions  the nulls as distributions rather than error bars, with
                      the observed value drawn on top.
  early_vs_late       the same fit computed on ripples 0-1 s after the uncover
                      and on ripples from 1 s to the end of the interval. A
                      genuinely time-locked effect is present in the first and
                      absent in the second; a sign flip between them is what
                      noise looks like.

    python scripts/swr_ripple_rsa_diagnostics.py run

@author: Svenja Kuchenhoff
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripple_rsa as rrsa
from mc.plotting.cell_results import (roi_display, get_roi_colour,
                                      STATE_QUADRANT_COLORS, OBSERVED_VALUE_COLOR)

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)
STAMP = datetime.now().strftime("%Y-%m-%d")

CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 1.0, "lines.linewidth": 2.2, "savefig.dpi": 300,
})

ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]
# inclusion rules, loosest last
# All post-press, so the sweep is comparable with the analysis (which uses
# post = [0, +1] s). A symmetric window would mix the pre and post conditions.
RULES = [("0–0.5 s", "window", (0.0, 0.5)), ("0–1 s", "window", (0.0, 1.0)),
         ("0–2 s", "window", (0.0, 2.0)), ("interval", "interval", None)]
SEED = 42


def _out(data_root=None):
    d = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr",
                     f"ripple_rsa_diagnostics_{STAMP}")
    os.makedirs(d, exist_ok=True)
    return d


def run(bundle=None, data_root=None, n_perm=2000, n_surrogate=200, out=None):
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_diagnostics")
    bundle_dir = bundle or os.path.join(swr_io.derivatives_dir(data_root),
                                        "group", "swr", "bundle")
    bd = rrsa.load_bundle(bundle_dir)
    events = rrsa.discovery_events(bd)
    roi_tab = rrsa.cell_roi_table()
    roi_tab = roi_tab[roi_tab.roi.isin(ROIS)]
    spikes = rrsa.load_spike_times(verbose=False)
    sessions = sorted(set(roi_tab.session))

    cov = _coverage_table(bd, events, roi_tab, spikes, out_dir)
    _figure_coverage(cov, bd, events, out_dir)
    _figure_model_support(out_dir)
    el = _early_late(bd, events, roi_tab, spikes, sessions, n_perm, out_dir)
    _figure_early_late(el, out_dir)
    _figure_nulls(bd, events, roi_tab, spikes, sessions, n_perm, n_surrogate,
                  out_dir)
    print(f"\nwritten to {out_dir}")


# =============================================================================
def _coverage_table(bd, events, roi_tab, spikes, out_dir):
    rows = []
    for label, scheme, win in RULES:
        cache = rrsa.cache_ripple_rates(bd, events, roi_tab, spikes, win,
                                        scheme=scheme)
        pack = rrsa.patterns_from_cache(cache)
        iu = np.triu_indices(rrsa.N_CONFIG, 1)
        # spikes per cell per condition, from the cached rates
        for blk in cache:
            pass
        for roi in ROIS:
            sel = pack["roi"] == roi
            for si, state in enumerate(rrsa.STATES):
                rdm, n_used = rrsa.rdm_for(pack, roi, state)
                P = pack["patterns"][sel][:, :, si]
                rows.append({
                    "rule": label, "roi": roi, "state": state,
                    "n_ripples": int(np.nansum(pack["counts"][:, :, si])),
                    "n_cells": int(sel.sum()),
                    "pairs_missing": int((~np.isfinite(rdm[iu])).sum()),
                    "cells_per_pair": float(np.median(n_used[iu])),
                    "cond_observed": float(np.isfinite(P).mean() * 100),
                })
    cov = pd.DataFrame(rows)
    cov.to_csv(os.path.join(out_dir, "coverage_sweep.csv"), index=False)
    print("COVERAGE SWEEP (ripples per state, pooled over sessions)")
    print(cov.pivot_table(index="rule", columns="state", values="n_ripples",
                          aggfunc="first").to_string())
    return cov


def _figure_coverage(cov, bd, events, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(2 * 8.0 * CM, 2 * 6.4 * CM))
    order = [r[0] for r in RULES]
    x = np.arange(len(order))

    ax = axes[0][0]
    for si, state in enumerate(rrsa.STATES):
        y = [cov[(cov.rule == r) & (cov.state == state)].n_ripples.iloc[0]
             for r in order]
        ax.plot(x, y, "o-", color=STATE_QUADRANT_COLORS[si], label=state, ms=6)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("ripples (all sessions)")
    ax.set_title("a  Ripples per reward, by inclusion rule", fontsize=10, pad=4)
    ax.legend(frameon=False, fontsize=8, ncol=4, title="uncovered reward",
              title_fontsize=8)

    ax = axes[0][1]
    for roi in ROIS:
        y = [cov[(cov.rule == r) & (cov.roi == roi)].cells_per_pair.median()
             for r in order]
        ax.plot(x, y, "o-", color=get_roi_colour(roi), label=roi_display(roi),
                ms=6)
    ax.axhline(rrsa.MIN_CELLS_PER_PAIR, color="#b03030", ls="--", lw=1.6)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("cells behind each RDM cell")
    ax.set_title("b  Cells contributing per config pair", fontsize=10, pad=4)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1][0]
    for roi in ROIS:
        y = [cov[(cov.rule == r) & (cov.roi == roi)].pairs_missing.mean()
             for r in order]
        ax.plot(x, y, "o-", color=get_roi_colour(roi), ms=6)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("config pairs NOT estimable\n(of 28)")
    ax.set_title("c  Loosening the rule completes the RDM", fontsize=10, pad=4)

    # runs per configuration per session
    ax = axes[1][1]
    runs = (events.drop_duplicates(["session", "grid_no"])
                  .groupby(["session", "cfg"]).size())
    ax.hist(runs.values, bins=np.arange(0.5, runs.max() + 1.5), color="#7BB594",
            edgecolor="k", linewidth=0.8)
    ax.set_xlabel("grids of the same configuration per session")
    ax.set_ylabel("count")
    ax.set_title(f"d  Runs per configuration\n(median {runs.median():.0f}; "
                 f"these are what a cross-run RDM would split)",
                 fontsize=10, pad=4)

    fig.suptitle("How much data reaches each cell of the 8 × 8 RDM", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"coverage_sweep.{e}"),
                    bbox_inches="tight")


def _figure_model_support(out_dir):
    """What the models are built from -- the ceiling on what they can explain."""
    iu = np.triu_indices(rrsa.N_CONFIG, 1)
    fig, axes = plt.subplots(1, 3, figsize=(3 * 7.0 * CM, 6.2 * CM))

    ax = axes[0]
    width = 0.2
    for si, state in enumerate(rrsa.STATES):
        k = rrsa.STATES.index(state)
        sh = np.array([[len(set(rrsa.CONFIGS[i][:k + 1])
                            & set(rrsa.CONFIGS[j][:k + 1]))
                        for j in range(8)] for i in range(8)])[iu]
        counts = [np.sum(sh == v) for v in range(4)]
        ax.bar(np.arange(4) + (si - 1.5) * width, counts, width,
               color=STATE_QUADRANT_COLORS[si], edgecolor="k", linewidth=0.6,
               label=state)
    ax.set_xticks(range(4))
    ax.set_xlabel("shared reward locations")
    ax.set_ylabel("configuration pairs (of 28)")
    ax.set_title("a  known_set: what carries the model\n"
                 "at A nothing is shared — model is flat", fontsize=10, pad=4)
    ax.legend(frameon=False, fontsize=8, ncol=4, title="after uncovering",
              title_fontsize=8)

    ax = axes[1]
    for si, state in enumerate(rrsa.STATES):
        k = rrsa.STATES.index(state)
        offs = []
        for i in range(8):
            for j in range(i + 1, 8):
                offs += [abs(x - y) for x in range(k + 1) for y in range(k + 1)
                         if rrsa.CONFIGS[i][x] == rrsa.CONFIGS[j][y]]
        if not offs:
            continue
        vals, cnt = np.unique(offs, return_counts=True)
        ax.bar(vals + (si - 1.5) * width, cnt, width,
               color=STATE_QUADRANT_COLORS[si], edgecolor="k", linewidth=0.6)
    ax.set_xlabel("rank offset of the shared location")
    ax.set_ylabel("shared-location events")
    ax.set_title("b  known_seq: offset 0 NEVER occurs\n"
                 "so a strict order model has no data", fontsize=10, pad=4)

    ax = axes[2]
    rho = []
    for state in rrsa.STATES:
        k = rrsa.STATES.index(state)
        a = rrsa.known_set_rdm(k)[iu]
        b = rrsa.known_seq_rdm(k)[iu]
        rho.append(np.nan if a.std() == 0 else np.corrcoef(a, b)[0, 1])
    ax.bar(range(4), rho, 0.6, color=[STATE_QUADRANT_COLORS[i] for i in range(4)],
           edgecolor="k", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(4)); ax.set_xticklabels(rrsa.STATES)
    ax.set_xlabel("after uncovering")
    ax.set_ylabel("r(known_set, known_seq)")
    ax.set_title("c  The two models are near-identical\n"
                 "— not separable over 28 pairs", fontsize=10, pad=4)

    fig.suptitle("What the models are actually built from", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"model_support.{e}"),
                    bbox_inches="tight")


def _cache_lag(bd, events, roi_tab, spikes, sessions, lo, hi):
    """Ripples whose lag from the uncover press is in [lo, hi), within interval."""
    out = []
    for s in sessions:
        rip = rrsa.ripples_in_intervals(bd, s, events)
        if rip.empty:
            continue
        lag = rip.t_peak_s.values - rip.press_t_s.values
        rip = rip[(lag >= lo) & (lag < hi)]
        cells = roi_tab[roi_tab.session == s]
        if rip.empty or cells.empty:
            continue
        half = rip.duration_s.values / 2.0
        rates = np.empty((len(cells), len(rip)))
        for row, (_, c) in enumerate(cells.iterrows()):
            rates[row] = rrsa.spike_counts_in_windows(
                spikes[s]["spikes"][int(c.cell)], rip.t_peak_s.values,
                half) / (2 * half)
        out.append({"session": s, "rates": rates,
                    "ci": np.array([rrsa.CONFIG_LABELS.index(c) for c in rip.cfg]),
                    "si": np.array([rrsa.STATES.index(x) for x in rip.state]),
                    "roi": cells.roi.to_numpy()})
    return out


def _early_late(bd, events, roi_tab, spikes, sessions, n_perm, out_dir):
    rng = np.random.default_rng(SEED)
    perms = [{s: rng.permutation(rrsa.N_CONFIG) for s in sessions}
             for _ in range(n_perm)]
    rows = []
    for label, (lo, hi) in [("early 0–1 s", (0, 1)),
                            ("late 1 s–end", (1, np.inf)),
                            ("whole interval", (0, np.inf))]:
        cache = _cache_lag(bd, events, roi_tab, spikes, sessions, lo, hi)
        pack = rrsa.patterns_from_cache(cache)
        npacks = [rrsa.patterns_from_cache(cache, config_perm=p) for p in perms]
        n_rip = int(np.nansum(pack["counts"]))
        for roi in ROIS:
            for state in ("B", "C", "D"):
                M = rrsa.model_rdms(state)["known_set"]
                rdm, _ = rrsa.rdm_for(pack, roi, state)
                o = rrsa.fit_rho(rdm, M)
                if not np.isfinite(o):
                    continue
                null = np.array([rrsa.fit_rho(rrsa.rdm_for(p, roi, state)[0], M)
                                 for p in npacks])
                null = null[np.isfinite(null)]
                rows.append({"subset": label, "n_ripples": n_rip, "roi": roi,
                             "state": state, "rho": o,
                             "p": float((null >= o).mean()),
                             "null_sd": float(null.std())})
    el = pd.DataFrame(rows)
    el.to_csv(os.path.join(out_dir, "early_vs_late.csv"), index=False)
    print("\nEARLY vs LATE (known_set)")
    print(el.pivot_table(index=["roi", "state"], columns="subset",
                         values="rho").round(3).to_string())
    return el


def _figure_early_late(el, out_dir):
    subs = ["early 0–1 s", "late 1 s–end", "whole interval"]
    fig, axes = plt.subplots(1, 3, figsize=(3 * 6.6 * CM, 6.4 * CM), sharey=True)
    for ax, sub in zip(axes, subs):
        g = el[el.subset == sub]
        n = g.n_ripples.iloc[0] if len(g) else 0
        for i, roi in enumerate(ROIS):
            gg = g[g.roi == roi]
            for _, row in gg.iterrows():
                off = 0.26 * (["B", "C", "D"].index(row.state) - 1)
                ax.plot(i + off, row.rho, "o", color=get_roi_colour(roi), ms=7,
                        markeredgecolor="white", markeredgewidth=0.6)
                if row.state == "D":
                    ax.annotate("D", (i + off, row.rho), fontsize=7,
                                ha="center", va="top",
                                textcoords="offset points", xytext=(0, -9))
        sd = g.null_sd.median() if len(g) else np.nan
        ax.axhspan(-sd, sd, color="#cccccc", alpha=0.45, lw=0)
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xticks(range(len(ROIS)))
        ax.set_xticklabels([roi_display(r) for r in ROIS], rotation=35,
                           ha="right")
        ax.set_title(f"{sub}\n({n} ripples)", fontsize=10, pad=4)
    axes[0].set_ylabel("known_set fit (Spearman $\\rho$)")
    fig.suptitle("Is the fit time-locked to the discovery?\n"
                 "grey band = ±1 SD of the config-relabel null. A real "
                 "time-locked effect is positive on the left and ~0 in the "
                 "middle; a sign flip is noise.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"early_vs_late.{e}"),
                    bbox_inches="tight")


def _figure_nulls(bd, events, roi_tab, spikes, sessions, n_perm, n_surrogate,
                  out_dir):
    """The nulls as distributions, for the two inclusion rules side by side."""
    rng = np.random.default_rng(SEED)
    perms = [{s: rng.permutation(rrsa.N_CONFIG) for s in sessions}
             for _ in range(n_perm)]
    srng = np.random.default_rng(SEED + 1)
    fig, axes = plt.subplots(2, 3, figsize=(3 * 7.0 * CM, 2 * 6.0 * CM))
    for r, (label, scheme, win) in enumerate(
            [("post 0–1 s window", "window", (0.0, 1.0)),
             ("inter-uncover interval", "interval", None)]):
        cache = rrsa.cache_ripple_rates(bd, events, roi_tab, spikes, win,
                                        scheme=scheme)
        pack = rrsa.patterns_from_cache(cache)
        npacks = [rrsa.patterns_from_cache(cache, config_perm=p) for p in perms]
        spacks = [rrsa.patterns_from_cache(
            rrsa.cache_ripple_rates(bd, events, roi_tab, spikes, win,
                                    scheme=scheme, surrogate_rng=srng))
            for _ in range(n_surrogate)]
        for c, roi in enumerate(["mPFC", "PCC", "HC_anterior"]):
            ax = axes[r][c]
            M = rrsa.model_rdms("D")["known_set"]
            rdm, _ = rrsa.rdm_for(pack, roi, "D")
            o = rrsa.fit_rho(rdm, M)
            n1 = np.array([rrsa.fit_rho(rrsa.rdm_for(p, roi, "D")[0], M)
                           for p in npacks])
            n2 = np.array([rrsa.fit_rho(rrsa.rdm_for(p, roi, "D")[0], M)
                           for p in spacks])
            n1, n2 = n1[np.isfinite(n1)], n2[np.isfinite(n2)]
            bins = np.linspace(-0.8, 0.8, 41)
            ax.hist(n1, bins=bins, color="#b7b0a6", alpha=0.85,
                    label="config relabel", density=True)
            ax.hist(n2, bins=bins, histtype="step", color="#5C1027", lw=2.0,
                    label="surrogate window", density=True)
            if np.isfinite(o):
                ax.axvline(o, color=OBSERVED_VALUE_COLOR, lw=2.6)
                ax.text(0.03, 0.97,
                        f"observed {o:+.2f}\np(relabel) = {np.mean(n1 >= o):.3f}\n"
                        f"p(surrogate) = {np.mean(n2 >= o):.3f}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        color=OBSERVED_VALUE_COLOR)
            ax.set_title(f"{roi_display(roi)} · uncover D\n{label}",
                         fontsize=10, pad=4)
            ax.set_xlabel("known_set fit (Spearman $\\rho$)")
            if c == 0:
                ax.set_ylabel("null density")
            if r == 0 and c == 2:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("The nulls, as distributions\n"
                 "config relabel: each session's 8 configurations are renamed "
                 "at random, so cross-session alignment is destroyed.  "
                 "surrogate: the window is moved off the ripple.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"null_distributions.{e}"),
                    bbox_inches="tight")


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
