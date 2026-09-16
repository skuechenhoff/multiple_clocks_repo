#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the ripple-triggered single-unit RSA powered at all? -- answered on RAW
spike times, before any RSA is run.

Reads spike times from `abcd_passed.mat` (NOT the 25 ms binned per-grid
matrices, which cannot form a window shorter than one bin) and ROI labels from
`neurons_with_ROI_labels.csv` (`atlas_roi`). Counts, for every ripple falling
around a first reward discovery, how many spikes each simultaneously recorded
neuron fires inside the ripple.

The quantity that decides everything is SPIKES PER NEURON PER RIPPLE. A human
unit fires at a few Hz; a ripple lasts ~60 ms. If that product is far below 1,
then most (neuron, ripple) pairs are a zero, the per-condition pattern is a
count of a handful of spikes, and no similarity structure can be estimated no
matter how the conditions are arranged.

Windows, all three reported side by side:
    peak +- 10 ms          the approximation asked for (20 ms)
    peak +- duration/2     the ripple's own extent, from `duration_s`
    peak +- 100 ms         a deliberately generous comparison

    python scripts/swr_ripple_rsa_power.py run
    python scripts/swr_ripple_rsa_power.py run --no_figures

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripple_rsa as rrsa
from mc.plotting.cell_results import (roi_display, get_roi_colour,
                                      STATE_QUADRANT_COLORS)

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

STAMP = datetime.now().strftime("%Y-%m-%d")

# ROIs worth reporting, in the project's fixed order. EC is carried even though
# `atlas_roi` assigns only 3 cells to it in these sessions -- that disagreement
# with `neurons_MNI_latest.csv` (51) is itself a finding and should be visible.
ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC", "EC"]

WINDOWS = {
    "peak +- 10 ms": ("fixed", 0.010),
    "peak +- duration/2": ("duration", None),
    "peak +- 100 ms": ("fixed", 0.100),
}


def _out(data_root=None):
    d = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr",
                     f"ripple_rsa_power_{STAMP}")
    os.makedirs(d, exist_ok=True)
    return d


def run(bundle=None, data_root=None, figures=True, out=None):
    out_dir = out or _out(data_root)
    swr_io.start_log(out_dir, "swr_ripple_rsa_power")
    bundle_dir = bundle or os.path.join(swr_io.derivatives_dir(data_root),
                                        "group", "swr", "bundle")
    print(f"bundle : {bundle_dir}\noutput : {out_dir}\n")

    bd = rrsa.load_bundle(bundle_dir)
    events = rrsa.discovery_events(bd)
    roi_tab = rrsa.cell_roi_table()
    print(f"{len(events)} discovery events in {events.session.nunique()} sessions")
    print(f"{len(roi_tab)} cells; ROI counts:\n"
          f"{roi_tab.roi.value_counts().to_string()}\n")

    print("loading raw spike times")
    spk = rrsa.load_spike_times()

    # ---- one row per (cell, ripple) ----------------------------------------
    rows = []
    for wname, press_win in rrsa.PRESS_WINDOWS.items():
        for s in rrsa.DSR_SESSIONS:
            rip = rrsa.ripples_near_events(bd, s, events, press_win)
            if rip.empty:
                continue
            cells = roi_tab[roi_tab.session == s]
            spikes = spk[s]["spikes"]
            centres = rip.t_peak_s.values
            dur = rip.duration_s.values
            for _, c in cells.iterrows():
                st = spikes[int(c.cell)]
                rec = {"window": wname, "session": s, "cell": int(c.cell),
                       "roi": c.roi, "n_ripples": len(rip)}
                for lab, (kind, val) in WINDOWS.items():
                    hw = val if kind == "fixed" else dur / 2.0
                    n = rrsa.spike_counts_in_windows(st, centres, hw)
                    width = (2 * val if kind == "fixed" else dur)
                    rec[f"spikes|{lab}"] = float(n.mean())
                    rec[f"rate|{lab}"] = float((n / width).mean())
                    rec[f"pzero|{lab}"] = float((n == 0).mean())
                # per-condition bookkeeping uses the ripple's own labels
                rec["n_cond_covered"] = int(
                    rip.groupby(["cfg", "state"]).ngroups)
                rows.append(rec)
    cells_df = pd.DataFrame(rows)
    cells_df.to_csv(os.path.join(out_dir, "per_cell_ripple_firing.csv"),
                    index=False)

    # ---- the headline table -------------------------------------------------
    print("\nSPIKES PER NEURON PER RIPPLE -- the number that decides the design")
    summ = []
    for (wname, roi), g in cells_df.groupby(["window", "roi"]):
        if roi not in ROIS:
            continue
        for lab in WINDOWS:
            summ.append({
                "press_window": wname, "roi": roi, "ripple_window": lab,
                "n_cells": len(g),
                "mean_spikes_per_ripple": g[f"spikes|{lab}"].mean(),
                "mean_rate_hz": g[f"rate|{lab}"].mean(),
                "pct_zero": 100 * g[f"pzero|{lab}"].mean(),
            })
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out_dir, "power_summary.csv"), index=False)
    show = summ[summ.ripple_window == "peak +- duration/2"]
    print(show.to_string(index=False))

    # ---- how many spikes reach one RDM condition ---------------------------
    cond = _condition_budget(bd, events, roi_tab, spk, out_dir)

    # ---- settings -----------------------------------------------------------
    swr_io.write_settings(out_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__),
        "spike_source": "abcd_passed.mat (raw spikeTimes, NOT 25 ms bins)",
        "roi_source": f"{rrsa.ROI_TABLE} column '{rrsa.ROI_COLUMN}'",
        "roi_join": "positional on `cell idx`; verified against mat "
                    "electrodeLabel for all 28 sessions",
        "bundle": bundle_dir,
        "sessions": rrsa.DSR_SESSIONS,
        "events": "correct & is_discovery & explore stage, 8 shared configs",
        "n_events": int(len(events)),
        "press_windows_s": {k: list(v) for k, v in rrsa.PRESS_WINDOWS.items()},
        "ripple_windows": {k: (v[1] if v[0] == "fixed" else "duration/2")
                           for k, v in WINDOWS.items()},
        "ripple_extent_note":
            "the bundle keeps only t_peak_s and duration_s; t_start_s/t_end_s "
            "already exist in the per-session ripple_events.csv and only need "
            "re-exporting -- no re-detection",
        "seed": 42,
    })

    if figures:
        _figure_power(cells_df, cond, out_dir)
        plt.show(block=False)
    print(f"\nwritten to {out_dir}")


def _condition_budget(bd, events, roi_tab, spk, out_dir):
    """Spikes reaching one (config x state) RDM cell, pooled across sessions.

    This is the quantity the RSA actually consumes: for one config and one
    state, every ripple from every session contributes its cells' spikes.
    """
    rows = []
    for wname, press_win in rrsa.PRESS_WINDOWS.items():
        per = {}
        for s in rrsa.DSR_SESSIONS:
            rip = rrsa.ripples_near_events(bd, s, events, press_win)
            if rip.empty:
                continue
            cells = roi_tab[roi_tab.session == s]
            for (cfg, state), g in rip.groupby(["cfg", "state"]):
                for roi in ROIS:
                    sel = cells[cells.roi == roi]
                    if sel.empty:
                        continue
                    tot = 0
                    for _, c in sel.iterrows():
                        n = rrsa.spike_counts_in_windows(
                            spk[s]["spikes"][int(c.cell)],
                            g.t_peak_s.values, g.duration_s.values / 2.0)
                        tot += int(n.sum())
                    k = (cfg, state, roi)
                    d = per.setdefault(k, {"ripples": 0, "cells": 0,
                                           "spikes": 0, "sessions": 0})
                    d["ripples"] += len(g)
                    d["cells"] += len(sel)
                    d["spikes"] += tot
                    d["sessions"] += 1
        for (cfg, state, roi), d in per.items():
            rows.append({"press_window": wname, "cfg": cfg, "state": state,
                         "roi": roi, **d,
                         "spikes_per_cell": d["spikes"] / max(d["cells"], 1)})
    cond = pd.DataFrame(rows)
    cond.to_csv(os.path.join(out_dir, "condition_budget.csv"), index=False)
    print("\nPER RDM CONDITION (one config x one state), pooled over sessions, "
          "ripple extent = duration")
    agg = (cond.groupby(["press_window", "roi"])
               .agg(median_ripples=("ripples", "median"),
                    median_cells=("cells", "median"),
                    median_spikes=("spikes", "median"),
                    median_spikes_per_cell=("spikes_per_cell", "median"))
               .reset_index())
    print(agg.to_string(index=False))
    agg.to_csv(os.path.join(out_dir, "condition_budget_summary.csv"),
               index=False)
    return cond


# =============================================================================
def _roi_colour(roi):
    return get_roi_colour(roi)


def _tick_labels(cells_df, rois):
    """ROI name with the number of cells behind it -- EC is 3 cells here and
    must not be read as if it were comparable to HC."""
    n = {r: int((cells_df[cells_df.window == "post"].roi == r).sum())
         for r in rois}
    return [f"{roi_display(r)}\n(n={n[r]})" for r in rois]


def _figure_power(cells_df, cond, out_dir):
    """Four panels: is there enough spiking to build these RDMs?"""
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    rois = [r for r in ROIS if r in set(cells_df.roi)]

    # -- a. firing rate in the ripple, pre vs post, by ROI --------------------
    ax = fig.add_subplot(gs[0, 0])
    w = 0.36
    for i, roi in enumerate(rois):
        for j, press in enumerate(["pre", "post"]):
            g = cells_df[(cells_df.roi == roi) & (cells_df.window == press)]
            if g.empty:
                continue
            v = g["rate|peak +- duration/2"]
            ax.bar(i + (j - 0.5) * w, v.mean(), w, yerr=v.sem(),
                   color=_roi_colour(roi), alpha=1.0 if press == "post" else 0.45,
                   edgecolor="k", linewidth=0.5, capsize=2)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(_tick_labels(cells_df, rois), rotation=30, fontsize=8,
                       ha="right")
    ax.set_ylabel("firing rate during the ripple (Hz)", fontsize=10)
    ax.set_title("a  Firing rate inside ripples, by ROI\n"
                 "faint = before the press, solid = after", fontsize=11)
    ax.legend(handles=[Line2D([], [], color="#666", lw=6, alpha=0.45,
                              label="pre  [-0.35, 0] s"),
                       Line2D([], [], color="#666", lw=6,
                              label="post [+0.15, +0.70] s")],
              fontsize=8, frameon=False)

    # -- b. THE number: spikes per neuron per ripple ---------------------------
    ax = fig.add_subplot(gs[0, 1])
    labs = list(WINDOWS)
    for i, roi in enumerate(rois):
        g = cells_df[(cells_df.roi == roi) & (cells_df.window == "post")]
        if g.empty:
            continue
        y = np.array([g[f"spikes|{l}"].mean() for l in labs])
        # a log axis cannot show an exact zero (EC: no spike at all in any
        # +-10 ms window); plot the finite points and mark the zero explicitly
        ok = y > 0
        ax.plot(np.flatnonzero(ok), y[ok], "o-", color=_roi_colour(roi),
                label=f"{roi_display(roi)} (n={len(g)})", markersize=6)
        for k in np.flatnonzero(~ok):
            ax.annotate("0", (k, y[ok].min() if ok.any() else 1e-3),
                        color=_roi_colour(roi), fontsize=9, ha="center",
                        va="top", fontweight="bold")
    ax.axhline(1.0, color=STATE_QUADRANT_COLORS[0], ls="--", lw=1.2)
    ax.text(0.02, 1.05, "1 spike per neuron per ripple", fontsize=8,
            color=STATE_QUADRANT_COLORS[0], transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([l.replace("peak +- ", "±") for l in labs], fontsize=9)
    ax.set_ylabel("spikes per neuron per ripple", fontsize=10)
    ax.set_title("b  The number that decides the design\n"
                 "(post-press ripples; log scale)", fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # -- c. proportion of (neuron, ripple) pairs that are zero ----------------
    ax = fig.add_subplot(gs[1, 0])
    for i, roi in enumerate(rois):
        for j, press in enumerate(["pre", "post"]):
            g = cells_df[(cells_df.roi == roi) & (cells_df.window == press)]
            if g.empty:
                continue
            v = 100 * g["pzero|peak +- duration/2"]
            ax.bar(i + (j - 0.5) * w, v.mean(), w, yerr=v.sem(),
                   color=_roi_colour(roi), alpha=1.0 if press == "post" else 0.45,
                   edgecolor="k", linewidth=0.5, capsize=2)
    ax.axhline(90, color="#555", ls=":", lw=1)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(_tick_labels(cells_df, rois), rotation=30, fontsize=8,
                       ha="right")
    ax.set_ylabel("(neuron, ripple) pairs with NO spike (%)", fontsize=10)
    ax.set_title("c  How often a neuron is silent during a ripple\n"
                 "ripple extent = its own duration", fontsize=11)

    # -- d. spikes reaching one RDM condition ---------------------------------
    ax = fig.add_subplot(gs[1, 1])
    sub = cond[cond.press_window == "post"]
    for i, roi in enumerate(rois):
        g = sub[sub.roi == roi]
        if g.empty:
            continue
        ax.scatter(np.full(len(g), i) + np.random.RandomState(42)
                   .uniform(-0.18, 0.18, len(g)),
                   g.spikes_per_cell, s=16, color=_roi_colour(roi), alpha=0.65,
                   edgecolor="none")
        ax.plot([i - 0.28, i + 0.28], [g.spikes_per_cell.median()] * 2,
                color="k", lw=2)
    ax.axhline(1.0, color=STATE_QUADRANT_COLORS[0], ls="--", lw=1.2)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(_tick_labels(cells_df, rois), rotation=30, fontsize=8,
                       ha="right")
    ax.set_ylabel("spikes per neuron, per RDM condition", fontsize=10)
    ax.set_title("d  What one cell of the 8 x 8 RDM is built from\n"
                 "one dot = one (config x state) condition, bar = median",
                 fontsize=11)

    fig.suptitle("Power for the ripple-triggered single-unit RSA — raw spike "
                 "times, first reward discoveries, 28 shared-config sessions",
                 fontsize=13)
    fig.savefig(os.path.join(out_dir, "ripple_rsa_power_overview.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "ripple_rsa_power_overview.pdf"),
                bbox_inches="tight")
    print("figure written: ripple_rsa_power_overview.png / .pdf")


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
