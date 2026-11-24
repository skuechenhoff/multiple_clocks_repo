#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The published per-lag location encoding (`per_lag_encoding.py`), run on the
ENCODING loops and compared with the published EXECUTION run. (2026-11-23)

Nothing in the estimator changes; only the loops that go in:

    execute          rows with correct != 0 or rep_correct != 0
                     = the published `all_minus_explore` set
    encode           rows with correct == 0 and rep_correct == 0
                     = exactly the rows the published run excludes (every
                     loop of a block before its first correct one)
    execute_matched  per configuration, the first n execute rows, n = that
                     configuration's number of encode rows (same
                     configurations and loop count as encode)

Per cell, per set (the published functions, imported):
    `_residualise_phase` (cosine) on that set's rows ->
    `_build_per_cfg_sequences` (average repeats per configuration, per-bin
    mode location) -> `_fit_variant(..., with_controls=False)` at the 12 lags
    = leave-one-configuration-out, dwell-weighted r (`r_lag*_noctrl`).
    A session needs >= 5 configurations in that set, as published.

Cells and ROI labels = the published per-cell table (`PER_CELL`); the
execute run must reproduce its r_lag*_noctrl (checked, written to
result.json).

Deviations from the published run, all without effect on r:
    * no per-cell permutations (n_perms = 0): the per-cell p is not used by
      the tests below, and r does not depend on it;
    * only `noctrl` and only the 12 single lags (no ctrl, no dsr variants).

Group tests (`mc.analyse.lagwise_aggregation`, the module the published
numbers come from), at the CELL and the SESSION unit:
    each profile vs 0   lagwise_tests_both_units (Fisher z, one-sided > 0,
                        two-sided also given; BH across the 12 lags)
    encode - execute    per-cell difference of Fisher z, aggregated with
    encode - matched    aggregate_curves (session = mean over its cells), t
                        test; two-sided p, BH across the 12 lags

    python scripts/per_lag_encoding_encode_vs_execute.py

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import mc.analyse.helpers_human_cells as hh
import mc.analyse.lagwise_aggregation as la
from mc.analyse.swr_explore.preferred_lag_firing import PER_CELL
import per_lag_encoding as ple

ROIS = ("HC_mid", "mPFC")
SETS = ("encode", "execute", "execute_matched")
SEED = ple.RANDOM_SEED
OUT = os.path.join(ple.DATA_DIR, "group",
                   f"per_lag_encode_vs_execute_{datetime.date.today()}")


def row_sets(beh):
    """Boolean row masks per set."""
    enc = ((beh.correct == 0) & (beh.rep_correct == 0)).to_numpy()
    exe = ~enc
    cfg = [tuple(x) for x in beh[["loc_A", "loc_B", "loc_C", "loc_D"]]
           .to_numpy()]
    matched = np.zeros(len(beh), bool)
    for c in set(cfg):
        rows = np.array([i for i, x in enumerate(cfg) if x == c])
        n = enc[rows].sum()
        matched[rows[exe[rows]][:n]] = True
    return {"encode": enc, "execute": exe, "execute_matched": matched}


def profile(arr, beh, locs, rows):
    """r at the 12 lags for one cell and one row set (published functions)."""
    b = beh[rows].reset_index(drop=True)
    _, idx_cfg = np.unique(b[["loc_A", "loc_B", "loc_C", "loc_D"]]
                           .to_numpy(), axis=0, return_inverse=True)
    idx_cfg = idx_cfg.ravel()
    n_cfg = len(np.unique(idx_cfg))
    if n_cfg < 5:
        return None, n_cfg
    y = ple._residualise_phase(arr[rows], basis=ple.PHASE_RESIDUALISE)
    L = locs[rows]
    Y_cfg, loc_cfg, btn_cfg = ple._build_per_cfg_sequences(
        y, idx_cfg, L, np.zeros(L.shape, int), n_cfg)
    rng = np.random.default_rng(SEED)
    r = [ple._fit_variant(Y_cfg, loc_cfg, btn_cfg, [lag], with_controls=False,
                          include_loc_now_in_ctrl=False, n_perms=0,
                          rng=rng)[0] for lag in ple.LAGS_DEG]
    return np.array(r), n_cfg


def run():
    pub = ple._canonicalize_roi_names(read_per_cell_table(PER_CELL))
    pub = pub[pub.roi.isin(ROIS)]
    rows = []
    for sub, g in pub.groupby("subject_id"):
        sub_str = f"{int(sub):02d}"
        d = hh.load_norm_data(ple.DATA_DIR, [sub_str], res_data=False)
        d = d[f"sub-{sub_str}"]
        beh = d["beh"].reset_index(drop=True)
        locs = d["locations"].to_numpy(float)
        masks = row_sets(beh)
        for _, c in g.iterrows():
            arr = d["normalised_neurons"][c.neuron].to_numpy(float)
            for s in SETS:
                r, n_cfg = profile(arr, beh, locs, masks[s])
                rec = dict(neuron=c.neuron, roi=c.roi, subject_id=int(sub),
                           set=s, n_cfg=n_cfg, n_loops=int(masks[s].sum()))
                for lag, v in zip(ple.LAGS_DEG,
                                  r if r is not None else [np.nan] * 12):
                    rec[f"r_lag{lag:03d}"] = v
                rows.append(rec)
        print(f"  sub-{sub_str}: {len(g)} cells", flush=True)
    return pd.DataFrame(rows)


def curves(df, s, roi):
    g = df[(df.set == s) & (df.roi == roi)].set_index("neuron")
    return g[[f"r_lag{l:03d}" for l in ple.LAGS_DEG]], g.subject_id


def diff_tests(df, a, b, roi):
    """Per-lag test of profile a - profile b, at both units."""
    A, sa = curves(df, a, roi)
    B, _ = curves(df, b, roi)
    both = A.index.intersection(B.index)
    D = la.fisher_z(A.loc[both].to_numpy()) - la.fisher_z(B.loc[both].to_numpy())
    subj = sa.loc[both].to_numpy()
    out = {}
    for unit in ("cell", "subject"):
        X = la.aggregate_curves(D, subj, unit=unit, fisher=False)
        t, p1, p2, n = la._t_gt0(X)
        q = la.bh_fdr(p2)
        out[unit] = [dict(lag=lag, n=int(n[j]), mean_dz=float(np.nanmean(X[:, j])),
                          t=float(t[j]), p_two_sided=float(p2[j]),
                          q_two_sided_lags=float(q[j]))
                     for j, lag in enumerate(ple.LAGS_DEG)]
    return out


def summarise(df):
    res = {"vs0": {}, "diff": {}, "per_unit_curves": {}}
    for s in SETS:
        cb = {}
        for roi in ROIS:
            C, subj = curves(df, s, roi)
            cb[roi] = (C.to_numpy(), subj.to_numpy())
            for unit in ("cell", "subject"):
                X = la.aggregate_curves(C.to_numpy(), subj.to_numpy(),
                                        unit=unit, fisher=False)
                n = np.isfinite(X).sum(0)
                res["per_unit_curves"][f"{s} | {roi} | {unit}"] = dict(
                    mean_r=np.nanmean(X, 0).tolist(),
                    sd_r=np.nanstd(X, 0, ddof=1).tolist(), n=n.tolist())
        T = la.lagwise_tests_both_units(cb, ple.LAGS_DEG, s)
        res["vs0"][s] = T.to_dict(orient="records")
    for a, b in (("encode", "execute"), ("encode", "execute_matched"),
                 ("execute_matched", "execute")):
        for roi in ROIS:
            res["diff"][f"{a} - {b} | {roi}"] = diff_tests(df, a, b, roi)
    # check: execute reproduces the published r_lag*_noctrl
    pub = ple._canonicalize_roi_names(read_per_cell_table(PER_CELL)).set_index("neuron")
    E, _ = curves(df, "execute", ROIS[0])
    E = pd.concat([curves(df, "execute", r)[0] for r in ROIS])
    P = pub.loc[E.index, [f"r_lag{l:03d}_noctrl" for l in ple.LAGS_DEG]].to_numpy()
    dif = np.abs(E.to_numpy() - P)
    res["reproduces_published"] = dict(
        max_abs_diff=float(np.nanmax(dif)),
        n_cells=int(len(E)),
        n_nan_mismatch=int((np.isfinite(E.to_numpy()) != np.isfinite(P)).sum()))
    # sample sizes
    res["n"] = {f"{s} | {roi}": dict(
        cells=int(np.isfinite(curves(df, s, roi)[0].to_numpy()).any(1).sum()),
        sessions=int(df[(df.set == s) & (df.roi == roi)]
                     .dropna(subset=["r_lag000"]).subject_id.nunique()))
        for s in SETS for roi in ROIS}
    return res


if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "per_cell.csv")
    df = pd.read_csv(f) if os.path.exists(f) else run()
    df.to_csv(f, index=False)
    res = summarise(df)
    res["settings"] = dict(
        created=str(datetime.date.today()),
        script="scripts/per_lag_encoding_encode_vs_execute.py",
        estimator="per_lag_encoding.py, noctrl, imported functions",
        cells=PER_CELL, rois=ROIS, sets={
            "execute": "correct != 0 or rep_correct != 0 (published "
                       "all_minus_explore)",
            "encode": "correct == 0 and rep_correct == 0",
            "execute_matched": "first n execute rows per configuration, n = "
                               "its encode rows"},
        min_cfg=5, n_perms=0, lags=ple.LAGS_DEG,
        tests="lagwise_aggregation: Fisher z t-tests at cell and session "
              "unit; vs 0 one-sided (two-sided given); differences "
              "two-sided; BH across 12 lags")
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res["reproduces_published"]), json.dumps(res["n"]))
