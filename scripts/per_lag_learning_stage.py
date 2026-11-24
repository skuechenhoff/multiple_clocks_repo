#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHEN in a block do mPFC and mid HC carry each lag? Lag profiles by learning
stage, with the per-repeat estimator and the final ROI labels. (2026-11-23)

Stages (loops of one block = one `grid_no`, in order):
    encode    loops before the block's first correct one
              (correct == 0 and rep_correct == 0)
    exe 1-2   the first two loops from the first correct one on
    exe 3-5   loops 3-5 from then on
    exe 6+    the rest

Per cell (pipeline functions from `per_lag_encoding.py`,
MAP_ESTIMATOR = 'per_repeat'):
    y = `_residualise_phase` (cosine) over ALL the session's loops.
    Reference for held-out configuration c and lag k: each other
    configuration's map from ALL its loops (`_lag_shifted_rate_map`, every
    repeat with its own locations), dwell-weighted mean over them -- as in
    `_ratemap_cv_one_setup`.
    Observed: configuration c's map from its loops in ONE stage.
    Score: dwell-weighted r with the published keep rule (both dwells >=
    MIN_DWELL_BINS, >= MIN_SHARED_LOCS squares); mean over configurations
    = the cell's CV r for that stage and lag.
    The reference never changes with the stage, so stage differences come
    from the test loops only. Session needs >= 5 configurations.
    Two versions (column `matched`):
      False  every loop of the stage
      True   loop count matched across stages: per configuration, each
             stage contributes n = that configuration's smallest stage
             count, drawn N_DRAWS times (seeded, shared by all cells of the
             session); r averaged over draws. More loops give a less noisy
             map and so a higher r, so only this version compares stages
             fairly.

Group (`mc.analyse.lagwise_aggregation`, cell and session units):
    each stage's profile vs 0 (one-sided)
    stage differences per lag (Fisher z, two-sided, BH across lags):
        exe 1-2 - encode, exe 6+ - encode, exe 6+ - exe 1-2
    focused: lag 0 and the Fig 3c window (mPFC 30+60, HC 0+330) per stage;
    trend = per-unit OLS slope of the value over the 4 stages, t-test.

    python scripts/per_lag_learning_stage.py

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
from mc.analyse.roi_labels import FINAL_ROI_TABLE
import per_lag_encoding as ple

ple.MAP_ESTIMATOR = "per_repeat"
STAGES = ("encode", "exe 1-2", "exe 3-5", "exe 6+")
ROIS = ("HC_mid", "mPFC")
WINDOWS = {"HC_mid": (0, 330), "mPFC": (30, 60)}
SEED = ple.RANDOM_SEED
N_DRAWS = 20
LAG_COLS = [f"r_lag{l:03d}" for l in ple.LAGS_DEG]
OUT = os.path.join(ple.DATA_DIR, "group",
                   f"per_lag_learning_stage_{datetime.date.today()}")


def stage_labels(beh):
    """Stage of every loop (row) of the session."""
    enc = ((beh.correct == 0) & (beh.rep_correct == 0)).to_numpy()
    k = np.zeros(len(beh), int)
    for _, rows in beh.groupby("grid_no").groups.items():
        rows = np.sort(np.asarray(rows))
        exe = rows[~enc[rows]]
        k[exe] = np.arange(len(exe))
    lab = np.where(k < 2, "exe 1-2", np.where(k < 5, "exe 3-5", "exe 6+"))
    return np.where(enc, "encode", lab).astype(object)


def score(pred, tot, obs, d_obs):
    """The published fold score (`_ratemap_cv_one_setup`)."""
    w = np.minimum(tot, d_obs)
    keep = (w >= ple.MIN_DWELL_BINS) & np.isfinite(pred) & np.isfinite(obs)
    if keep.sum() < ple.MIN_SHARED_LOCS:
        return np.nan
    return ple._weighted_pearson_r(pred[keep], obs[keep], w[keep])


def loop_subsets(cfg, stage, matched, rng):
    """{stage: {config: [row-index arrays]}}: every loop, or N_DRAWS
    count-matched draws."""
    out = {s: {} for s in STAGES}
    for c in np.unique(cfg):
        rows = {s: np.flatnonzero((cfg == c) & (stage == s)) for s in STAGES}
        n = min(len(r) for r in rows.values())
        for s in STAGES:
            if not matched:
                if len(rows[s]):
                    out[s][c] = [rows[s]]
            elif n > 0:
                out[s][c] = [np.sort(rng.choice(rows[s], n, replace=False))
                             for _ in range(N_DRAWS)]
    return out


def cell_stages(y, L, cfg, subsets):
    """{stage: (12,) CV r} for one cell."""
    cfgs = np.unique(cfg)
    out = {s: np.full(len(ple.LAGS_DEG), np.nan) for s in STAGES}
    for j, lag in enumerate(ple.LAGS_DEG):
        full = {c: ple._lag_shifted_rate_map(y[cfg == c], L[cfg == c], lag)
                for c in cfgs}
        fold = {s: [] for s in STAGES}
        for c in cfgs:
            R = np.stack([full[o][0] for o in cfgs if o != c])
            D = np.stack([full[o][1] for o in cfgs if o != c])
            tot = D.sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                pred = np.nansum(R * D, 0) / np.where(tot > 0, tot, np.nan)
            for s in STAGES:
                draws = [score(pred, tot, *ple._lag_shifted_rate_map(
                    y[m], L[m], lag)) for m in subsets[s].get(c, [])]
                if draws and np.isfinite(draws).any():
                    fold[s].append(np.nanmean(draws))
        for s in STAGES:
            v = np.array(fold[s], float)
            if np.isfinite(v).any():
                out[s][j] = np.nanmean(v)
    return out


def session(sub_str, cells):
    d = hh.load_norm_data(ple.DATA_DIR, [sub_str], res_data=False)
    d = d[f"sub-{sub_str}"]
    beh = d["beh"].reset_index(drop=True)
    L = d["locations"].to_numpy(float)
    _, cfg = np.unique(beh[["loc_A", "loc_B", "loc_C", "loc_D"]].to_numpy(),
                       axis=0, return_inverse=True)
    cfg = cfg.ravel()
    if len(np.unique(cfg)) < 5:
        return []
    stage = stage_labels(beh)
    rng = np.random.default_rng([SEED, int(sub_str)])
    subsets = {m: loop_subsets(cfg, stage, m, rng) for m in (False, True)}
    rows = []
    for nid, n_df in d["normalised_neurons"].items():
        _, ci = cs.parse_neuron_label(nid)
        c = cells[cells.cell_idx == ci]
        if c.empty:
            continue
        y = ple._residualise_phase(n_df.to_numpy(float),
                                   basis=ple.PHASE_RESIDUALISE)
        for matched in (False, True):
            for s, r in cell_stages(y, L, cfg, subsets[matched]).items():
                rec = dict(neuron=nid, roi=c.roi.iloc[0],
                           subject_id=int(sub_str), matched=matched, stage=s,
                           n_loops=int(sum(len(v[0]) for v in
                                           subsets[matched][s].values())))
                rec.update(dict(zip(LAG_COLS, r)))
                rows.append(rec)
    print(f"  sub-{sub_str}: {len(rows) // (2 * len(STAGES))} cells", flush=True)
    return rows


def run():
    cells = cs.load_cells(cell_set="all_in_roi_table", rois_keep=list(ROIS)).copy()
    cells["subject_id"] = cells["subject_id"].astype(str).str.zfill(2)
    out = Parallel(n_jobs=8)(delayed(session)(s, g)
                             for s, g in cells.groupby("subject_id"))
    return pd.DataFrame([r for rows in out for r in rows])


def unit_values(df, roi, stage, cols, unit):
    """(n_units,) mean Fisher z over `cols` per cell or session."""
    g = df[(df.roi == roi) & (df.stage == stage)]
    Z = la.fisher_z(g[cols].to_numpy(float))
    v = np.nanmean(Z, 1) if len(cols) > 1 else Z[:, 0]
    return la.aggregate_curves(v[:, None], g.subject_id.to_numpy(), unit=unit,
                               fisher=False)[:, 0], g


def summarise(df_all):
    res = {}
    for matched in (False, True):
        res["matched" if matched else "all_loops"] = summarise_version(
            df_all[df_all.matched == matched])
    return res


def summarise_version(df):
    res = {"vs0": {}, "diff": {}, "focused": {}, "curves": {}, "n": {}}
    for s in STAGES:
        cb = {roi: (df[(df.roi == roi) & (df.stage == s)][LAG_COLS]
                    .to_numpy(float),
                    df[(df.roi == roi) & (df.stage == s)].subject_id.to_numpy())
              for roi in ROIS}
        res["vs0"][s] = la.lagwise_tests_both_units(
            cb, ple.LAGS_DEG, s).to_dict(orient="records")
        for roi in ROIS:
            g = df[(df.roi == roi) & (df.stage == s)]
            res["n"][f"{s} | {roi}"] = dict(
                cells=int(g[LAG_COLS].notna().any(axis=1).sum()),
                sessions=int(g.dropna(subset=LAG_COLS, how="all")
                             .subject_id.nunique()),
                loops_per_session_median=float(
                    g.groupby("subject_id").n_loops.first().median()))
            for unit in ("cell", "subject"):
                X = la.aggregate_curves(g[LAG_COLS].to_numpy(float),
                                        g.subject_id.to_numpy(), unit=unit,
                                        fisher=False)
                n = np.isfinite(X).sum(0)
                res["curves"][f"{s} | {roi} | {unit}"] = dict(
                    mean=np.nanmean(X, 0).tolist(),
                    sem=(np.nanstd(X, 0, ddof=1) / np.sqrt(n)).tolist(),
                    n=n.tolist())
    for a, b in (("exe 1-2", "encode"), ("exe 6+", "encode"),
                 ("exe 6+", "exe 1-2")):
        for roi in ROIS:
            A = df[(df.roi == roi) & (df.stage == a)].set_index("neuron")
            B = df[(df.roi == roi) & (df.stage == b)].set_index("neuron")
            D = (la.fisher_z(A[LAG_COLS].to_numpy(float))
                 - la.fisher_z(B.loc[A.index, LAG_COLS].to_numpy(float)))
            out = {}
            for unit in ("cell", "subject"):
                X = la.aggregate_curves(D, A.subject_id.to_numpy(), unit=unit,
                                        fisher=False)
                t, _, p2, n = la._t_gt0(X)
                q = la.bh_fdr(p2)
                out[unit] = [dict(lag=lag, n=int(n[j]),
                                  mean_dz=float(np.nanmean(X[:, j])),
                                  t=float(t[j]), p_two_sided=float(p2[j]),
                                  q_two_sided_lags=float(q[j]))
                             for j, lag in enumerate(ple.LAGS_DEG)]
            res["diff"][f"{a} - {b} | {roi}"] = out
    # focused: lag 0 and the Fig 3c window, per stage, plus the trend
    for roi in ROIS:
        for name, lags in (("lag 0", (0,)), ("window", WINDOWS[roi])):
            cols = [f"r_lag{l:03d}" for l in lags]
            for unit in ("cell", "subject"):
                V = np.column_stack([unit_values(df, roi, s, cols, unit)[0]
                                     for s in STAGES])
                rows = []
                for j, s in enumerate(STAGES):
                    v = V[:, j][np.isfinite(V[:, j])]
                    t, p1, p2, n = la._t_gt0(v[:, None])
                    rows.append(dict(stage=s, n=int(n[0]),
                                     mean_z=float(v.mean()),
                                     ci95=float(1.96 * v.std(ddof=1)
                                                / np.sqrt(len(v))),
                                     t=float(t[0]), p_one_sided=float(p1[0]),
                                     p_two_sided=float(p2[0])))
                ok = np.isfinite(V).all(1)
                x = np.arange(len(STAGES)) - 1.5
                slope = (V[ok] - V[ok].mean(1, keepdims=True)) @ x / (x @ x)
                t, _, p2, n = la._t_gt0(slope[:, None])
                res["focused"][f"{roi} | {name} {lags} | {unit}"] = dict(
                    stages=rows,
                    trend=dict(n=int(n[0]), mean_slope_z_per_stage=float(
                        slope.mean()), t=float(t[0]), p_two_sided=float(p2[0])))
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
        script="scripts/per_lag_learning_stage.py", roi_labels=FINAL_ROI_TABLE,
        stages=STAGES, map_estimator="per_repeat",
        reference="other configurations, all loops", windows=WINDOWS,
        lags_deg=ple.LAGS_DEG, phase_residualise=ple.PHASE_RESIDUALISE,
        min_dwell_bins=ple.MIN_DWELL_BINS, min_shared_locs=ple.MIN_SHARED_LOCS,
        min_configs=5, seed=SEED)
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for k, v in res["matched"]["focused"].items():
        print("matched |", k, " | ".join(f"{r['stage']}: {r['mean_z']:+.3f} (p={r['p_one_sided']:.3f})"
                            for r in v["stages"]),
              f"| trend {v['trend']['mean_slope_z_per_stage']:+.4f} p={v['trend']['p_two_sided']:.3f}")
