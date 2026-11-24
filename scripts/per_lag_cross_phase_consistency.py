#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is a cell's lag tuning the same in encode and execute, and does it relate to
its firing in ripples? (2026-11-23, exploratory)

Builds on `per_lag_encoding_encode_vs_execute.py` (the published per-lag
functions, imported; its per_cell.csv is reused for encode / execute /
execute_matched). Added per cell, same functions, same cells (mid HC, mPFC):

    all            every repeat (encode + execute)
    execute_odd    every other execute repeat of each configuration,
    execute_even   and the rest  (split-half reliability of the published r)
    encode_odd / encode_even    the same within encode

Q2  across cells, per lag: Spearman rho of r_encode vs r_execute, of
    r_encode vs r_execute_matched, and the split-half reliabilities
    (odd vs even) that cap them.
Q3  across cells, per lag: Pearson r (as claim 2c) of lag tuning (execute,
    encode or all repeats) with
      storage_rip_minus_norip   claim 2c's y: firing modulation 0.05-1.0 s
                                after a first reward discovery with a ripple
                                within 0.5 s minus without (arrival-locked;
                                `storage_vs_planning_2026-10-01`)
      ripmod_encode / _execute  (rate in ripples - rate outside) / sum over
                                the whole phase, outside = > 1 s from any
                                ripple (`lag_group_ripple_2026-10-01`)
    and, for mPFC lag 0 x claim 2c's y, a decomposition: Spearman, within-
    and between-session, partial on firing rate and on event counts,
    without the most influential cell, leave-one-session-out range, and the
    lag-0 reliabilities by firing-rate tertile.

Every correlation is tested by permuting one variable WITHIN session (10,000,
seed 42), the same function throughout, so session differences cannot
create a cell-level correlation.

ROI labels = the published per-lag table. Claim 2c used the ripple
pipeline's labels (rrsa.cell_roi_table), which differ for some cells; both
versions of claim 2c are reported.

    python scripts/per_lag_cross_phase_consistency.py

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import mc.analyse.helpers_human_cells as hh
import per_lag_encoding_encode_vs_execute as pe

SEED, N_PERM = 42, 10000
LAGS = pe.ple.LAGS_DEG
ROIS = pe.ROIS
EXTRA = ("all", "execute_odd", "execute_even", "encode_odd", "encode_even")
GROUP = pe.ple.DATA_DIR + "/group"
BASE = os.path.join(GROUP, "per_lag_encode_vs_execute_2026-11-23")
SP = os.path.join(GROUP, "swr", "storage_vs_planning_2026-10-01", "per_cell.csv")
LG = os.path.join(GROUP, "swr", "lag_group_ripple_2026-10-01", "per_cell.csv")
OUT = os.path.join(GROUP, f"per_lag_cross_phase_{datetime.date.today()}")
PAIRS = (("encode", "execute"), ("encode", "execute_matched"),
         ("execute_odd", "execute_even"), ("encode_odd", "encode_even"))
YS = ("storage_rip_minus_norip", "ripmod_encode", "ripmod_execute")
XS = ("execute", "encode", "all")


def extra_masks(beh):
    base = pe.row_sets(beh)
    cfg = np.array(["-".join(map(str, x)) for x in
                    beh[["loc_A", "loc_B", "loc_C", "loc_D"]].to_numpy()])
    out = {"all": np.ones(len(beh), bool)}
    for nm in ("execute", "encode"):
        odd, even = np.zeros(len(beh), bool), np.zeros(len(beh), bool)
        for c in np.unique(cfg):
            rows = np.flatnonzero((cfg == c) & base[nm])
            odd[rows[0::2]] = True
            even[rows[1::2]] = True
        out[f"{nm}_odd"], out[f"{nm}_even"] = odd, even
    return out


def run_extra():
    pub = pe.ple._canonicalize_roi_names(read_per_cell_table(pe.PER_CELL))
    pub = pub[pub.roi.isin(ROIS)]
    rows = []
    for sub, g in pub.groupby("subject_id"):
        sub_str = f"{int(sub):02d}"
        d = hh.load_norm_data(pe.ple.DATA_DIR, [sub_str], res_data=False)
        d = d[f"sub-{sub_str}"]
        beh = d["beh"].reset_index(drop=True)
        locs = d["locations"].to_numpy(float)
        masks = extra_masks(beh)
        for _, c in g.iterrows():
            arr = d["normalised_neurons"][c.neuron].to_numpy(float)
            for s in EXTRA:
                r, n_cfg = pe.profile(arr, beh, locs, masks[s])
                rec = dict(neuron=c.neuron, roi=c.roi, subject_id=int(sub),
                           set=s, n_cfg=n_cfg, n_loops=int(masks[s].sum()))
                for lag, v in zip(LAGS, r if r is not None else [np.nan] * 12):
                    rec[f"r_lag{lag:03d}"] = v
                rows.append(rec)
        print(f"  sub-{sub_str}", flush=True)
    return pd.DataFrame(rows)


def perm_corr(x, y, sess, method="spearman", seed=SEED):
    """Correlation across cells, p from permuting y within session."""
    x, y, sess = map(np.asarray, (x, y, sess))
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, sess = x[ok], y[ok], sess[ok]
    if len(x) < 8:
        return dict(n=int(len(x)))
    if method == "spearman":
        x, y = stats.rankdata(x), stats.rankdata(y)
    xc = (x - x.mean()) / x.std()
    yc = (y - y.mean()) / y.std()          # invariant under permutation
    obs = float(np.mean(xc * yc))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(sess == s) for s in np.unique(sess)]
    idx = np.tile(np.arange(len(y)), (N_PERM, 1))
    for g in groups:
        idx[:, g] = g[np.argsort(rng.random((N_PERM, len(g))), axis=1)]
    null = (yc[idx] * xc).mean(1)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + N_PERM)
    p_plain = (stats.spearmanr(x, y)[1] if method == "spearman"
               else stats.pearsonr(x, y)[1])
    return dict(n=int(len(x)), r=obs, p_perm=float(p), p_plain=float(p_plain),
                n_sessions=int(len(groups)))


def wide(df):
    """(neuron x (set, lag)) table + session and roi per neuron."""
    cols = [f"r_lag{l:03d}" for l in LAGS]
    W = df.pivot_table(index="neuron", columns="set", values=cols)
    meta = df.drop_duplicates("neuron").set_index("neuron")[["roi",
                                                             "subject_id"]]
    return W, meta.reindex(W.index)        # same row order as W


def cell_table(meta):
    """Per-cell ripple variables and firing rate, joined on (session, cell)."""
    pub = read_per_cell_table(pe.PER_CELL)[["neuron", "subject_id", "cell_idx"]]
    pub["session"], pub["cell"] = pub.subject_id, pub.cell_idx - 1
    sp = pd.read_csv(SP)
    lg = pd.read_csv(LG)
    lg = lg.pivot_table(index=["session", "cell"], columns="phase",
                        values=["ripmod", "rate_out"])
    lg.columns = [f"{a}_{b}" for a, b in lg.columns]
    T = pub.merge(sp[["session", "cell", "roi", "storage_rip_minus_norip",
                      "n_storage_rip", "n_storage_norip"]].rename(
                          columns={"roi": "roi_ripple_pipeline"}),
                  on=["session", "cell"], how="left")
    T = T.merge(lg.reset_index(), on=["session", "cell"], how="left")
    return T.set_index("neuron").reindex(meta.index)


def drivers(x, y, sess, T):
    """mPFC lag 0 x claim-2c y: what carries the correlation."""
    d = pd.DataFrame(dict(x=x, y=y, s=sess, neuron=T.index.to_numpy(),
                          rate=np.log(T.rate_out_encode.to_numpy(float) + 0.01),
                          n_rip=np.log1p(T.n_storage_rip.to_numpy(float)),
                          n_norip=np.log1p(T.n_storage_norip.to_numpy(float))))
    d = d.dropna()
    out = {"pearson": perm_corr(d.x, d.y, d.s, "pearson"),
           "spearman": perm_corr(d.x, d.y, d.s, "spearman")}
    dm = d.groupby("s")[["x", "y"]].transform(lambda v: v - v.mean())
    out["within-session (demeaned)"] = perm_corr(dm.x, dm.y, d.s, "pearson")
    sm = d.groupby("s")[["x", "y"]].mean()
    r, p = stats.pearsonr(sm.x, sm.y)
    out["between-session (session means)"] = dict(n=int(len(sm)), r=float(r),
                                                  p_plain=float(p))

    def resid(v, cov):
        X = np.column_stack([np.ones(len(d))] + [d[c] for c in cov])
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    for lab, cov in (("partial | firing rate", ["rate"]),
                     ("partial | n events (rip, no rip)", ["n_rip", "n_norip"]),
                     ("partial | rate + n events", ["rate", "n_rip", "n_norip"])):
        out[lab] = perm_corr(resid(d.x.to_numpy(), cov),
                             resid(d.y.to_numpy(), cov), d.s, "pearson")
    # most influential cell (Cook's distance of y ~ x)
    X = np.column_stack([np.ones(len(d)), d.x])
    b = np.linalg.lstsq(X, d.y, rcond=None)[0]
    e = d.y - X @ b
    h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    cook = e ** 2 / (2 * e.var()) * h / (1 - h) ** 2
    keep = np.argsort(cook.to_numpy())[:-1]
    dd = d.iloc[keep]
    out["without the most influential cell"] = perm_corr(dd.x, dd.y, dd.s,
                                                         "pearson")
    out["without the most influential cell"]["dropped"] = str(
        d.neuron.iloc[np.argmax(cook.to_numpy())])
    loso = [stats.pearsonr(d.x[d.s != s], d.y[d.s != s])[0]
            for s in d.s.unique()]
    out["leave-one-session-out"] = dict(min=float(np.min(loso)),
                                        max=float(np.max(loso)),
                                        n_sessions=int(d.s.nunique()))
    out["firing rate vs x"] = perm_corr(d.rate, d.x, d.s, "pearson")
    out["firing rate vs y"] = perm_corr(d.rate, d.y, d.s, "pearson")
    return out


def summarise(df):
    W, meta = wide(df)
    T = cell_table(meta)
    # alignment check: execute lag 0 must equal the published value per cell
    pub = read_per_cell_table(pe.PER_CELL).set_index("neuron")
    assert np.allclose(W[("r_lag000", "execute")].to_numpy(),
                       pub.loc[W.index, "r_lag000_noctrl"].to_numpy(),
                       equal_nan=True)
    assert (T.index == W.index).all()
    res = {"Q2": {}, "Q3": {}, "drivers": {}, "tertiles": {}, "labels": {}}
    for roi in ROIS:
        m = (meta.roi == roi).to_numpy()
        sess = meta.subject_id.to_numpy()[m]
        for a, b in PAIRS:
            for lag in LAGS:
                c = f"r_lag{lag:03d}"
                res["Q2"][f"{roi} | {a} vs {b} | {lag}"] = perm_corr(
                    W[(c, a)].to_numpy()[m], W[(c, b)].to_numpy()[m], sess)
        for y in YS:
            yv = T[y].to_numpy(float)[m]
            for xs in XS:
                for lag in LAGS:
                    c = f"r_lag{lag:03d}"
                    res["Q3"][f"{roi} | {xs} | {y} | {lag}"] = perm_corr(
                        W[(c, xs)].to_numpy()[m], yv, sess, "pearson")
        # lag-0 reliability and cross-phase agreement by firing-rate tertile
        rate = T.rate_out_encode.to_numpy(float)[m]
        q = np.nanquantile(rate, [1 / 3, 2 / 3])
        tert = np.digitize(rate, q)
        for k, nm in enumerate(("low", "mid", "high")):
            t = (tert == k) & np.isfinite(rate)
            for a, b in PAIRS[:3]:
                res["tertiles"][f"{roi} | {nm} rate | {a} vs {b} | 0"] = \
                    perm_corr(W[("r_lag000", a)].to_numpy()[m][t],
                              W[("r_lag000", b)].to_numpy()[m][t], sess[t])
    m = (meta.roi == "mPFC").to_numpy()
    for xs in ("execute", "all", "encode"):
        res["drivers"][xs] = drivers(W[("r_lag000", xs)].to_numpy()[m],
                                     T.storage_rip_minus_norip.to_numpy(float)[m],
                                     meta.subject_id.to_numpy()[m], T[m])
    # claim 2c as published (ripple-pipeline labels) vs per-lag labels
    sp = pd.read_csv(SP)
    g = sp[sp.roi == "mPFC"].dropna(subset=["lag0_strength",
                                            "storage_rip_minus_norip"])
    r, p = stats.pearsonr(g.lag0_strength, g.storage_rip_minus_norip)
    res["labels"]["claim 2c, ripple-pipeline labels"] = dict(n=len(g), r=r, p=p)
    res["labels"]["claim 2c, per-lag labels"] = res["drivers"]["execute"]["pearson"]
    cells = pd.DataFrame({f"r0_{xs}": W[("r_lag000", xs)] for xs in XS})
    cells = cells.join(meta).join(T[["roi_ripple_pipeline", "rate_out_encode"]
                                    + list(YS)])
    cells.to_csv(os.path.join(OUT, "cells_q3.csv"))
    res["labels"]["label mismatch (ripple pipeline x per-lag)"] = pd.crosstab(
        T.roi_ripple_pipeline, meta.roi).to_dict()
    return res


if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "per_cell_extra.csv")
    X = pd.read_csv(f) if os.path.exists(f) else run_extra()
    X.to_csv(f, index=False)
    df = pd.concat([pd.read_csv(os.path.join(BASE, "per_cell.csv")), X])
    res = summarise(df)
    res["settings"] = dict(
        created=str(datetime.date.today()),
        script="scripts/per_lag_cross_phase_consistency.py",
        base=BASE, storage=SP, lag_group=LG, n_perm=N_PERM, seed=SEED,
        correlation_test="permute one variable within session",
        Q2="Spearman", Q3="Pearson (as claim 2c)")
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res["labels"], indent=1, default=float))
