#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-lag location encoding (`per_lag_encoding.py`) for three loop sets and
both rate-map estimators, with the FINAL ROI labels. (2026-11-23)

Loop sets (rows of the session's behaviour table):
    execute   correct != 0 or rep_correct != 0  = the published
              `all_minus_explore` set (Fig 3c)
    all       every loop, including the encoding loops before a block's
              first correct one
    encode    correct == 0 and rep_correct == 0 (only the encoding loops)

Estimators (`per_lag_encoding.MAP_ESTIMATOR`):
    per_repeat   each repeat with its own visited locations (SK's choice)
    mode         average repeats, per-bin mode location (2026-06-30 run)

Per cell, set and estimator, the pipeline's own functions:
    `_residualise_phase` (cosine) on that set's rows ->
    `_build_per_cfg_sequences` -> `_fit_variant(with_controls=False)` at the
    12 lags = leave-one-configuration-out dwell-weighted r (`r_lag*_noctrl`).
    A session needs >= 5 configurations in that set, as published.

Cells: `cs.load_cells('all_in_roi_table', ROIS_TO_RUN)` = what a fresh
pipeline run selects, with the final ROI labels (ROI_assignment, Aug 28).

Deviations from the full pipeline, none of which changes r:
    * no per-cell permutations (the group tests do not use them);
    * noctrl and the 12 single lags only.
Check: `mode` + `execute` must reproduce the published r (written to
result.json).

Output: group/per_lag_loopsets_<date>/<estimator>/<set>/per_cell_ALL_ROIs.csv
in the published per-cell format, so `overlay_double_dissociation.py` and
the other readers take it as is.

    python scripts/per_lag_encoding_loopsets.py

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import mc.analyse.helpers_human_cells as hh
import mc.analyse.cell_selection as cs
import mc.analyse.lagwise_aggregation as la
from mc.analyse.roi_labels import read_per_cell_table, FINAL_ROI_TABLE
from mc.analyse.swr_explore.preferred_lag_firing import PER_CELL
import per_lag_encoding as ple

SETS = ("execute", "all", "encode")
ESTIMATORS = ("per_repeat", "mode")
STATS_ROIS = ("HC_mid", "HC_anterior", "mPFC", "mOFC", "EC", "PCC")
SEED = ple.RANDOM_SEED
LAG_COLS = [f"r_lag{l:03d}_noctrl" for l in ple.LAGS_DEG]
OUT = os.path.join(ple.DATA_DIR, "group",
                   f"per_lag_loopsets_{datetime.date.today()}")


def row_sets(beh):
    enc = ((beh.correct == 0) & (beh.rep_correct == 0)).to_numpy()
    return {"execute": ~enc, "all": np.ones(len(beh), bool), "encode": enc}


def profile(arr, beh, locs, rows, estimator):
    """r at the 12 lags for one cell, loop set and estimator."""
    b = beh[rows].reset_index(drop=True)
    _, idx_cfg = np.unique(b[["loc_A", "loc_B", "loc_C", "loc_D"]].to_numpy(),
                           axis=0, return_inverse=True)
    idx_cfg = idx_cfg.ravel()
    n_cfg = len(np.unique(idx_cfg))
    if n_cfg < 5:
        return np.full(len(ple.LAGS_DEG), np.nan), n_cfg
    ple.MAP_ESTIMATOR = estimator
    y = ple._residualise_phase(arr[rows], basis=ple.PHASE_RESIDUALISE)
    L = locs[rows]
    Y_cfg, loc_cfg, btn_cfg = ple._build_per_cfg_sequences(
        y, idx_cfg, L, np.zeros(L.shape, int), n_cfg)
    rng = np.random.default_rng(SEED)
    r = [ple._fit_variant(Y_cfg, loc_cfg, btn_cfg, [lag], with_controls=False,
                          include_loc_now_in_ctrl=False, n_perms=0, rng=rng)[0]
         for lag in ple.LAGS_DEG]
    return np.array(r), n_cfg


def session(sub_str, cells):
    d = hh.load_norm_data(ple.DATA_DIR, [sub_str], res_data=False)
    d = d[f"sub-{sub_str}"]
    beh = d["beh"].reset_index(drop=True)
    locs = d["locations"].to_numpy(float)
    masks = row_sets(beh)
    rows = []
    for nid, n_df in d["normalised_neurons"].items():
        _, ci = cs.parse_neuron_label(nid)
        c = cells[cells.cell_idx == ci]
        if c.empty:
            continue
        c = c.iloc[0]
        arr = n_df.to_numpy(float)
        for est in ESTIMATORS:
            for s in SETS:
                r, n_cfg = profile(arr, beh, locs, masks[s], est)
                rec = dict(estimator=est, set=s, neuron=nid, roi=c.roi,
                           subject_id=int(sub_str), cell_idx=int(ci),
                           MNI_x=c.MNI_x, MNI_y=c.MNI_y, MNI_z=c.MNI_z,
                           n_cfg=int(n_cfg), n_loops=int(masks[s].sum()))
                rec.update(dict(zip(LAG_COLS, r)))
                rows.append(rec)
    print(f"  sub-{sub_str}: {len(rows) // (len(SETS) * len(ESTIMATORS))} cells",
          flush=True)
    return rows


def run():
    cells = cs.load_cells(cell_set="all_in_roi_table", rois_keep=ple.ROIS_TO_RUN)
    cells = cells.copy()
    cells["subject_id"] = cells["subject_id"].astype(str).str.zfill(2)
    jobs = [(s, g) for s, g in cells.groupby("subject_id")]
    out = Parallel(n_jobs=8, verbose=0)(delayed(session)(s, g) for s, g in jobs)
    return pd.DataFrame([r for rows in out for r in rows])


def lagwise(df, est, s):
    """Each ROI's profile vs 0 at the cell and session unit (published tests)."""
    g = df[(df.estimator == est) & (df.set == s)]
    cb = {roi: (g[g.roi == roi][LAG_COLS].to_numpy(float),
                g[g.roi == roi].subject_id.to_numpy())
          for roi in STATS_ROIS if (g.roi == roi).any()}
    return la.lagwise_tests_both_units(cb, ple.LAGS_DEG, f"{est} | {s}")


def diff_tests(df, a, b, roi):
    """Per-lag test of profile a - profile b (Fisher z, same cells), both
    units. a, b = (estimator, set)."""
    g = df[df.roi == roi]
    A = g[(g.estimator == a[0]) & (g.set == a[1])].set_index("neuron")
    B = g[(g.estimator == b[0]) & (g.set == b[1])].set_index("neuron")
    both = A.index.intersection(B.index)
    D = (la.fisher_z(A.loc[both, LAG_COLS].to_numpy(float))
         - la.fisher_z(B.loc[both, LAG_COLS].to_numpy(float)))
    out = {}
    for unit in ("cell", "subject"):
        X = la.aggregate_curves(D, A.loc[both].subject_id.to_numpy(),
                                unit=unit, fisher=False)
        t, p1, p2, n = la._t_gt0(X)
        q = la.bh_fdr(p2)
        out[unit] = [dict(lag=lag, n=int(n[j]), mean_dz=float(np.nanmean(X[:, j])),
                          t=float(t[j]), p_two_sided=float(p2[j]),
                          q_two_sided_lags=float(q[j]))
                     for j, lag in enumerate(ple.LAGS_DEG)]
    return out


def summarise(df):
    res = {"vs0": {}, "diff": {}, "n": {}}
    for est in ESTIMATORS:
        for s in SETS:
            T = lagwise(df, est, s)
            res["vs0"][f"{est} | {s}"] = T.to_dict(orient="records")
            g = df[(df.estimator == est) & (df.set == s)]
            for roi in STATS_ROIS:
                h = g[(g.roi == roi)].dropna(subset=["r_lag000_noctrl"])
                res["n"][f"{est} | {s} | {roi}"] = dict(
                    cells=int(len(h)), sessions=int(h.subject_id.nunique()))
        for a, b in (("all", "execute"), ("encode", "execute")):
            for roi in ("HC_mid", "mPFC"):
                res["diff"][f"{est} | {a} - {b} | {roi}"] = diff_tests(
                    df, (est, a), (est, b), roi)
    for s in SETS:
        for roi in ("HC_mid", "mPFC"):
            res["diff"][f"per_repeat - mode | {s} | {roi}"] = diff_tests(
                df, ("per_repeat", s), ("mode", s), roi)
    # curves the figure draws: mean and SEM per lag, per unit
    res["curves"] = {}
    for est in ESTIMATORS:
        for s in SETS:
            for roi in STATS_ROIS:
                g = df[(df.estimator == est) & (df.set == s) & (df.roi == roi)]
                for unit in ("cell", "subject"):
                    X = la.aggregate_curves(g[LAG_COLS].to_numpy(float),
                                            g.subject_id.to_numpy(),
                                            unit=unit, fisher=False)
                    n = np.isfinite(X).sum(0)
                    res["curves"][f"{est} | {s} | {roi} | {unit}"] = dict(
                        mean=np.nanmean(X, 0).tolist(),
                        sem=(np.nanstd(X, 0, ddof=1) / np.sqrt(n)).tolist(),
                        n=n.tolist())
    # check: mode + execute reproduces the published r (2026-06-30 compute)
    pub = read_per_cell_table(PER_CELL).set_index("neuron")
    E = df[(df.estimator == "mode") & (df.set == "execute")].set_index("neuron")
    common = E.index.intersection(pub.index)
    dif = np.abs(E.loc[common, LAG_COLS].to_numpy(float)
                 - pub.loc[common, LAG_COLS].to_numpy(float))
    res["reproduces_published"] = dict(
        table=PER_CELL, n_cells_common=int(len(common)),
        n_cells_here=int(len(E)), n_cells_published=int(len(pub)),
        max_abs_diff=float(np.nanmax(dif)),
        n_nan_mismatch=int((np.isfinite(E.loc[common, LAG_COLS].to_numpy(float))
                            != np.isfinite(pub.loc[common, LAG_COLS]
                                           .to_numpy(float))).sum()),
        roi_label_agrees=bool((E.loc[common, "roi"].astype(str)
                               == pub.loc[common, "roi"].astype(str)).all()))
    return res


if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "per_cell_long.csv")
    df = pd.read_csv(f) if os.path.exists(f) else run()
    df.to_csv(f, index=False)
    for (est, s), g in df.groupby(["estimator", "set"]):
        d = os.path.join(OUT, est, s)
        os.makedirs(d, exist_ok=True)
        g.drop(columns=["estimator", "set"]).to_csv(
            os.path.join(d, "per_cell_ALL_ROIs.csv"), index=False)
    res = summarise(df)
    res["settings"] = dict(
        created=str(datetime.date.today()),
        script="scripts/per_lag_encoding_loopsets.py",
        roi_labels=FINAL_ROI_TABLE, sets=SETS, estimators=ESTIMATORS,
        lags_deg=ple.LAGS_DEG, phase_residualise=ple.PHASE_RESIDUALISE,
        min_dwell_bins=ple.MIN_DWELL_BINS, min_shared_locs=ple.MIN_SHARED_LOCS,
        weighted_correlation=ple.WEIGHTED_CORRELATION, min_configs=5,
        n_perms_per_cell=0, seed=SEED)
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res["reproduces_published"], indent=1))
