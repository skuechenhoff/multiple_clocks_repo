#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1: do blocks with more hippocampal ripples while LEARNING show a stronger mPFC
plan code once the sequence is EXECUTED?

Pre-registered in CHANGELOG.md, 2026-11-14 (b). Built around the two limits found
in F1-F4: no cell is classified (per-cell lag tuning does not replicate), and
every learning-phase ripple is used (not only the few after first discoveries).

UNIT = one block (one run of a configuration). Per block:
    learning   grid onset -> start of the first fully correct repeat
               (the agreed exclusive boundary)
    ripples    hippocampal ripples in the learning window (bundle_v2, pooled
               over HC derivations, de-duplicated)
    plan code  per cell, the execution rate maps (F2's pooled estimator) of all
               OTHER configurations predict this block's first 3 execution
               loops; Fisher z of the dwell-weighted r, averaged over the
               session's cells of the ROI (>= 2 finite)

Per session (>= 8 blocks), OLS on z-scored predictors:
    code ~ log1p(ripples) + log(duration) + n learning loops + block order
           + run index (re-encounter of the configuration)
    own reading       covariate = log learning duration
    matched reading   covariate = log learning STILL time (sum over inter-step
                      gaps of max(0, gap - 0.5 s)) in place of the duration
The ripple beta per session is tested across sessions by sign-flip.

PRIMARY: mPFC, 60 deg, own reading, beta > 0.

Usage:
    conda activate env_multiple_clocks
    python scripts/plan_code_vs_ripples.py

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
import mc.analyse.swr_location as swl
from mc.analyse.future_spatial_peaks import _residualise_phase
from per_lag_encoding_by_phase import (pooled_map, predicted, score, fz,
                                       sign_flip, PER_CELL, ROI_NAME_MAP,
                                       DATA_DIR, LAGS_DEG)

SEED = 42
N_FIRST = 3
STILL_GAP_S = 0.5
MIN_BLOCKS = 8
MIN_CELLS = 2
GROUPS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"], "HC": ["HC_anterior", "HC_mid"]}
STACK = (30, 60, 90)                    # the dsrinf lags, stacked
PRIMARY = ("mPFC", "lag060", "first3", "own")
OUT = os.path.join(DATA_DIR, "group", "swr",
                   f"plan_code_vs_ripples_{datetime.date.today()}")


# ── blocks ───────────────────────────────────────────────────────────────
def block_table(beh, t_rip, t_steps):
    """Per block: learning / execution windows, ripples, still time, loops."""
    rows, seen = [], {}
    beh = beh.reset_index(drop=True)
    cfg = beh[["loc_A", "loc_B", "loc_C", "loc_D"]].astype(int).astype(str)\
        .agg("-".join, axis=1)
    for order, (g, gb) in enumerate(beh.groupby("grid_no", sort=True)):
        gb = gb.sort_values("rep_overall")
        ok = np.flatnonzero(gb.correct.to_numpy(int) == 1)
        if not len(ok):
            continue
        t0 = float(gb.new_grid_onset.iloc[0])
        ex0 = max(float(gb.new_grid_onset.iloc[ok[0]]), float(gb.t_D.iloc[0]))
        end = float(gb.t_D.max())
        c = cfg[gb.index[0]]
        seen[c] = seen.get(c, 0) + 1
        st = t_steps[(t_steps >= t0) & (t_steps < ex0)]
        still = float(np.clip(np.diff(np.r_[t0, st, ex0]) - STILL_GAP_S, 0,
                              None).sum())
        rows.append(dict(grid_no=int(g), order=order, cfg=c, run=seen[c],
                         n_learn_loops=max(int(ok[0]), 1),
                         t_learn0=t0, t_exe0=ex0, t_end=end,
                         learn_s=ex0 - t0, still_s=still, exe_s=end - ex0,
                         n_rip_learn=int(((t_rip >= t0) & (t_rip < ex0)).sum()),
                         n_rip_exe=int(((t_rip >= ex0) & (t_rip < end)).sum())))
    B = pd.DataFrame(rows)
    B["prev_rip_exe"] = B.n_rip_exe.shift(1)
    B["prev_exe_s"] = B.exe_s.shift(1)
    return B


def cell_block_codes(Y, L, cfg_id, exe, grid, B):
    """{(grid_no, lag tag, which): r} for one cell. `exe` marks execution rows;
    training = execution rows of every OTHER configuration."""
    out = {}
    rows_c = {c: np.flatnonzero((cfg_id == c) & exe) for c in np.unique(cfg_id)}
    tags = [(f"lag{l:03d}", [l]) for l in LAGS_DEG] + [("stack30-90",
                                                        list(STACK))]
    for tag, lags in tags:
        def pmap(r):
            ms = [pooled_map(Y[r], L[r], l) for l in lags]
            return np.concatenate([m[0] for m in ms]), \
                np.concatenate([m[1] for m in ms])
        me = {c: pmap(r) for c, r in rows_c.items() if len(r)}
        for _, b in B.iterrows():
            # execution loops of this block; its first traversal is learning
            # even when it happens to be correct (exclusive boundary)
            rb = np.flatnonzero((grid == b.grid_no) & exe)
            rb = rb[rb != np.flatnonzero(grid == b.grid_no)[0]]
            c_b = cfg_id[rb[0]] if len(rb) else None
            tr = [c for c in me if c != c_b]
            if not len(rb) or not tr:
                continue
            pe = predicted(me, tr)
            out[(int(b.grid_no), tag, "first3")] = score(*pe, *pmap(rb[:N_FIRST]))
            out[(int(b.grid_no), tag, "all")] = score(*pe, *pmap(rb))
    return out


def session(sub, cells, rip, steps):
    s = int(sub)
    d = hh.load_norm_data(DATA_DIR, [f"{s:02d}"], res_data=False)
    if not d:
        return None, None
    S = d[f"sub-{s:02d}"]
    beh = S["beh"].reset_index(drop=True)
    L = S["locations"].to_numpy(float)
    exe = ~((beh.rep_correct == 0) & (beh.correct == 0)).to_numpy()
    _, cfg_id = np.unique(beh[["loc_A", "loc_B", "loc_C", "loc_D"]].to_numpy(),
                          axis=0, return_inverse=True)
    cfg_id = np.asarray(cfg_id).ravel()
    grid = beh.grid_no.to_numpy()
    t_rip = np.unique(np.round(rip[rip.session == s].t_peak_s.to_numpy(float),
                               3))
    t_st = np.sort(steps[steps.session == s].t_s.to_numpy(float))
    B = block_table(beh, t_rip, t_st)
    B["session"] = s
    keep = cells[cells.subject_id == s].set_index("neuron")
    codes = []
    for nid, df in S["normalised_neurons"].items():
        if nid not in keep.index:
            continue
        X = df.to_numpy(float)
        Y = np.full_like(X, np.nan)
        Y[exe] = _residualise_phase(X[exe], basis="cosine")
        for (g, tag, which), r in cell_block_codes(Y, L, cfg_id, exe, grid,
                                                   B).items():
            codes.append(dict(session=s, neuron=nid, roi=keep.loc[nid, "roi"],
                              grid_no=g, lag=tag, which=which, r=r))
    return B, pd.DataFrame(codes)


# ── stats ────────────────────────────────────────────────────────────────
def design(D, reading, predictor="ripples"):
    if predictor == "ripples":
        x = np.log1p(D.n_rip_learn)
        cov = [np.log(D.learn_s if reading == "own" else D.still_s.clip(0.1))]
    else:                                   # negative control
        x = np.log1p(D.prev_rip_exe)
        cov = [np.log(D.prev_exe_s), np.log(D.learn_s if reading == "own"
                                            else D.still_s.clip(0.1))]
    cov += [D.n_learn_loops, D.order, D.run]
    X = np.column_stack([x] + cov).astype(float)
    return X


def session_beta(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    if len(y) < MIN_BLOCKS:
        return np.nan
    sd = X.std(0)
    keep = sd > 0
    if not keep[0]:
        return np.nan
    Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    Z = np.column_stack([np.ones(len(y)), Z])
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        return np.nan
    b = np.linalg.lstsq(Z, (y - y.mean()) / (y.std() or 1), rcond=None)[0]
    return float(b[1])


def tests(B, C):
    res, rows = {}, []
    for gname, members in GROUPS.items():
        G = C[C.roi.isin(members)].copy()
        G["z"] = fz(G.r)
        P = G.groupby(["session", "grid_no", "lag", "which"]).z.agg(
            ["mean", "count"]).reset_index()
        P = P[P["count"] >= MIN_CELLS]
        for (lag, which), g in P.groupby(["lag", "which"]):
            D = B.merge(g, on=["session", "grid_no"])
            for reading in ("own", "matched"):
                for pred in ("ripples", "prev_exe_ripples"):
                    betas = {s: session_beta(d["mean"].to_numpy(float),
                                             design(d, reading, pred))
                             for s, d in D.groupby("session")}
                    for s, v in betas.items():
                        rows.append(dict(group=gname, lag=lag, which=which,
                                         reading=reading, predictor=pred,
                                         session=s, beta=v))
                    res[f"{gname} | {lag} | {which} | {reading} | {pred}"] = \
                        sign_flip(list(betas.values()))
    return res, pd.DataFrame(rows)


def mixed(B, C, gname="mPFC", lag="lag060", which="first3"):
    import statsmodels.formula.api as smf
    G = C[C.roi.isin(GROUPS[gname]) & (C.lag == lag) & (C.which == which)].copy()
    G["z"] = fz(G.r)
    P = G.groupby(["session", "grid_no"]).z.agg(["mean", "count"]).reset_index()
    D = B.merge(P[P["count"] >= MIN_CELLS], on=["session", "grid_no"])
    D = D.assign(lrip=np.log1p(D.n_rip_learn), ldur=np.log(D.learn_s))
    for c in ("mean", "lrip", "ldur", "n_learn_loops", "order", "run"):
        D[c] = (D[c] - D.groupby("session")[c].transform("mean"))
    f = smf.mixedlm("mean ~ lrip + ldur + n_learn_loops + order + run", D,
                    groups=D.session).fit()
    return dict(n_blocks=int(len(D)), n_sessions=int(D.session.nunique()),
                beta=float(f.params["lrip"]), se=float(f.bse["lrip"]),
                p=float(f.pvalues["lrip"]))


def main():
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    cells = read_per_cell_table(PER_CELL)
    cells["roi"] = cells.roi.map(lambda r: ROI_NAME_MAP.get(r, r))
    cells = cells[cells.roi.isin(sum(GROUPS.values(), []))]
    rip = pd.read_csv(os.path.join(DATA_DIR, "group", "swr", "bundle_v2",
                                   "ripples.csv"), usecols=["session", "t_peak_s"])
    subs = sorted(cells.subject_id.unique())
    steps = swl.load([int(s) for s in subs])
    Bs, Cs = [], []
    for s in subs:
        B, C = session(s, cells, rip, steps)
        if B is None:
            continue
        Bs.append(B)
        Cs.append(C)
        print(f"  s{int(s):02d}: {len(B)} blocks, {C.neuron.nunique() if len(C) else 0} cells",
              flush=True)
    B, C = pd.concat(Bs, ignore_index=True), pd.concat(Cs, ignore_index=True)
    B.to_csv(os.path.join(OUT, "blocks.csv"), index=False)
    C.to_csv(os.path.join(OUT, "cell_block_codes.csv"), index=False)
    res, betas = tests(B, C)
    betas.to_csv(os.path.join(OUT, "session_betas.csv"), index=False)
    res["MixedLM | mPFC | lag060 | first3 | own"] = mixed(B, C)
    res["MixedLM | mPFC | lag000 | first3 | own"] = mixed(B, C, lag="lag000")
    res["MixedLM | HC | lag000 | first3 | own"] = mixed(B, C, "HC", "lag000")
    key = " | ".join(PRIMARY[:3]) + f" | {PRIMARY[3]} | ripples"
    print(f"\n  PRIMARY {key}: {res[key]}")
    for k in res:
        if k.startswith(("mPFC | lag000 |", "mPFC | lag060 |", "mPFC | stack",
                         "HC | lag000 | first3", "mOFC | lag060 | first3",
                         "MixedLM")):
            print(f"  {k:52s} {res[k]}")
    settings = dict(analysis="plan_code_vs_ripples (B1)",
                    created=str(datetime.date.today()),
                    script=os.path.relpath(__file__, REPO), seed=SEED,
                    cells=PER_CELL, groups=GROUPS, n_first=N_FIRST,
                    still_gap_s=STILL_GAP_S, min_blocks=MIN_BLOCKS,
                    min_cells=MIN_CELLS, stack=STACK, primary=PRIMARY,
                    learning="grid onset -> start of first fully correct repeat",
                    estimator="F2 pooled rate maps; train = execution of all "
                              "other configurations; test = this block's first "
                              "3 (or all) execution loops; Fisher z, mean over "
                              "cells",
                    model="per session OLS, z-scored: code ~ log1p(ripples) + "
                          "log(duration | still time) + n loops + order + run;"
                          " ripple beta sign-flipped across sessions",
                    negative_control="previous block's execution ripples",
                    preregistration="CHANGELOG.md 2026-11-14 (b) B1")
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(dict(settings=settings, tests=res), f, indent=1, default=str)
    from plan_code_vs_ripples_figures import draw
    draw(OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
