#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does a cell's current-location tuning (lag-0 CV r) predict how it fires in
ripples? Claim 2c redone with the final ROI labels, every cell, and every
tuning version of the loop-set run. (2026-11-24, exploratory)

x  per-cell r_lag000_noctrl from `per_lag_loopsets_2026-11-23`
   (final labels, all cells):
       estimator  mode (published) | per_repeat (new Fig 3c)
       loop set   execute (correct loops, published) | all | encode
y  per-cell ripple firing, as before (unchanged files, joined on session x
   cell; their own ROI column is ignored):
       storage_rip_minus_norip  claim 2c's y: firing 0.05-1.0 s after a first
                                reward discovery with a ripple within 0.5 s
                                minus without (`storage_vs_planning_2026-10-01`)
       ripmod_encode            (rate in ripples - rate outside) / sum over
       ripmod_execute           the encoding / correct-loop phase; outside =
                                > 1 s from any ripple (`lag_group_ripple_2026-10-01`)

Test, per ROI x tuning version x y: Pearson r across cells (as claim 2c),
p from permuting y within session (10,000, seed 42) -- the imported
`perm_corr` of `per_lag_cross_phase_consistency.py`, unchanged; Spearman
with the same function alongside.

Reliability ceiling, same function: across cells, Spearman of r0 from
encoding loops vs r0 from correct loops (independent loops of the same
cell), per estimator.

    python scripts/lag0_tuning_vs_ripple_firing.py
    -> group/lag0_tuning_vs_ripple_firing_<date>/

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from per_lag_cross_phase_consistency import perm_corr, SEED, N_PERM

G = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
LAGRUN = os.path.join(G, "per_lag_loopsets_2026-11-23")
SP = os.path.join(G, "swr", "storage_vs_planning_2026-10-01", "per_cell.csv")
LG = os.path.join(G, "swr", "lag_group_ripple_2026-10-01", "per_cell.csv")
OUT = os.path.join(G, f"lag0_tuning_vs_ripple_firing_{datetime.date.today()}")
ESTIMATORS = ("mode", "per_repeat")
SETS = ("execute", "all", "encode")
YS = ("storage_rip_minus_norip", "ripmod_encode", "ripmod_execute")
ROIS = ("mPFC", "HC_mid", "HC_anterior")


def cells():
    """One row per cell: final label, session, r0 for every version, y."""
    T = None
    for est in ESTIMATORS:
        for s in SETS:
            d = pd.read_csv(os.path.join(LAGRUN, est, s, "per_cell_ALL_ROIs.csv"),
                            usecols=["neuron", "roi", "subject_id", "cell_idx",
                                     "r_lag000_noctrl"])
            d = d.rename(columns={"r_lag000_noctrl": f"r0 | {est} | {s}"})
            T = d if T is None else T.merge(
                d, on=["neuron", "roi", "subject_id", "cell_idx"], how="outer")
    T["session"], T["cell"] = T.subject_id.astype(int), T.cell_idx.astype(int) - 1
    sp = pd.read_csv(SP)[["session", "cell", "storage_rip_minus_norip"]]
    lg = pd.read_csv(LG).pivot_table(index=["session", "cell"], columns="phase",
                                     values="ripmod")
    lg.columns = [f"ripmod_{c}" for c in lg.columns]
    T = T.merge(sp, on=["session", "cell"], how="left")
    T = T.merge(lg.reset_index(), on=["session", "cell"], how="left")
    return T[T.roi.isin(ROIS)].reset_index(drop=True)


def with_ci(d):
    """Add the Fisher-z 95% CI of r to a perm_corr result."""
    if "r" in d and d["n"] > 3:
        z, h = np.arctanh(d["r"]), 1.96 / np.sqrt(d["n"] - 3)
        d["ci95"] = [float(np.tanh(z - h)), float(np.tanh(z + h))]
    return d


def main():
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    T = cells()
    T.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    res = {"correlations": {}, "reliability": {}, "n_cells": {}}
    for roi in ROIS:
        g = T[T.roi == roi]
        res["n_cells"][roi] = dict(cells=int(len(g)), sessions=int(g.session.nunique()),
                                   with_ripple_firing=int(g.ripmod_encode.notna().sum()))
        for est in ESTIMATORS:
            for s in SETS:
                x = g[f"r0 | {est} | {s}"].to_numpy(float)
                for y in YS:
                    for meth in ("pearson", "spearman"):
                        res["correlations"][f"{roi} | {est} | {s} | {y} | {meth}"] = \
                            with_ci(perm_corr(x, g[y].to_numpy(float), g.session, meth))
            res["reliability"][f"{roi} | {est} | encode vs execute"] = with_ci(perm_corr(
                g[f"r0 | {est} | encode"], g[f"r0 | {est} | execute"], g.session))
    res["settings"] = dict(created=str(datetime.date.today()),
                           script="scripts/lag0_tuning_vs_ripple_firing.py",
                           x=LAGRUN, y=[SP, LG], rois=ROIS, n_perm=N_PERM, seed=SEED,
                           test="perm_corr (per_lag_cross_phase_consistency.py): "
                                "y permuted within session",
                           labels="final (atlas_roi), from the loop-set tables")
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1, default=float)
    for k, v in res["correlations"].items():
        if k.endswith("pearson"):
            print(f"{k:62s} r {v.get('r', np.nan):+.3f}  p {v.get('p_perm', np.nan):.3f}  n {v['n']}")
    for k, v in res["reliability"].items():
        print(f"{k:62s} rho {v.get('r', np.nan):+.3f}  p {v.get('p_perm', np.nan):.3f}  n {v['n']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
