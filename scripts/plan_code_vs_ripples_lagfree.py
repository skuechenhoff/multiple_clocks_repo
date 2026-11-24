#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1 follow-up (2026-11-23, exploratory): learning-phase hippocampal ripples vs
the mPFC plan code in execution, measured on ALL execution loops of a block as
well as the first 3, and with LAG-FREE per-cell codes.

Everything is read from B1's saved per-cell block codes
(`plan_code_vs_ripples_2026-11-14/cell_block_codes.csv`: per cell, block,
lag and loop set, the dwell-weighted r between the lag-k rate map of this
block's execution loops and the execution maps of all OTHER configurations,
F2's pooled estimator) and analysed with B1's own functions (`design`,
`session_beta`: per session OLS of the z-scored block code on z-scored
log(1 + learning ripples) + log learning duration + n learning loops + block
order + run index; `sign_flip` over sessions).

Block code variants (mPFC; per block = mean Fisher z over the session's
mPFC cells, >= 2):
    lag 60        B1's pre-registered primary lag
    future mean   mean over the 11 lags 30..330 (>= 8 finite), as in
                  plan_code_vs_ripples_learning_figure.py
    best lag      per cell, the future lag with the highest mean code over the
                  cell's OTHER blocks (same loop set), read in this block:
                  each cell's own lag, chosen without this block
    max lag       per cell and block, the maximum over the 11 future lags:
                  the literal 'highest consistency'; noise-inflated, but the
                  same operation in every block
for the first 3 and for all execution loops. Also the pooled partial
correlation (blocks, within session, after the covariates), as in the
learning figure's panels c, d.

    python scripts/plan_code_vs_ripples_lagfree.py

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "exploratory"))
from plan_code_vs_ripples import session_beta, design
from per_lag_encoding_by_phase import fz, sign_flip, DATA_DIR
from plan_code_vs_ripples_learning_figure import partial_xy

SEED = 42
FUTURE = list(range(30, 360, 30))
MIN_CELLS, MIN_LAGS = 2, 8
WHICH = ("first3", "all")
VARIANTS = ("lag 60", "future mean", "best lag", "max lag")
B1_DIR = os.path.join(DATA_DIR, "group", "swr", "plan_code_vs_ripples_2026-11-14")
OUT = os.path.join(DATA_DIR, "group", "swr",
                   f"plan_code_vs_ripples_lagfree_{datetime.date.today()}")


def cell_codes(C, which):
    """(neuron, session, grid_no) x lag table of Fisher z, mPFC, one loop set."""
    G = C[(C.roi == "mPFC") & (C.which == which)
          & C.lag.str.startswith("lag")].copy()
    G["z"] = fz(G.r)
    G["lagdeg"] = G.lag.str[3:].astype(int)
    return G.pivot_table(index=["session", "neuron", "grid_no"],
                         columns="lagdeg", values="z")


def block_codes(Z):
    """Per (session, grid_no): the four variants, mean over cells."""
    F = Z[FUTURE]
    out = pd.DataFrame(index=Z.index)
    out["lag 60"] = Z[60]
    out["future mean"] = F.mean(1).where(F.notna().sum(1) >= MIN_LAGS)
    out["max lag"] = F.max(1).where(F.notna().sum(1) >= MIN_LAGS)
    best = np.full(len(Z), np.nan)
    for (s, n), g in Z.groupby(level=["session", "neuron"]):
        Fg = g[FUTURE].to_numpy()
        pos = [Z.index.get_loc(i) for i in g.index]
        for k in range(len(g)):
            other = np.delete(Fg, k, axis=0)
            if not len(other) or np.isnan(other).all():
                continue
            lag_i = np.nanargmax(np.nanmean(other, 0))
            best[pos[k]] = Fg[k, lag_i]
    out["best lag"] = best
    A = out.groupby(level=["session", "grid_no"]).agg(["mean", "count"])
    res = pd.DataFrame({v: A[(v, "mean")].where(A[(v, "count")] >= MIN_CELLS)
                        for v in VARIANTS})
    return res.reset_index()


def main():
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    B = pd.read_csv(os.path.join(B1_DIR, "blocks.csv"))
    C = pd.read_csv(os.path.join(B1_DIR, "cell_block_codes.csv"))
    res, betas = {}, []
    for w in WHICH:
        D = B.merge(block_codes(cell_codes(C, w)), on=["session", "grid_no"])
        D.to_csv(os.path.join(OUT, f"blocks_with_codes_{w}.csv"), index=False)
        for v in VARIANTS:
            bs = {}
            for s, d in D.groupby("session"):
                bs[s] = session_beta(d[v].to_numpy(float),
                                     design(d, "own", "ripples"))
                betas.append(dict(session=int(s), which=w, variant=v,
                                  beta=bs[s], n_blocks=len(d)))
            res[f"{v} | {w} | beta"] = sign_flip(list(bs.values()))
            x, y = partial_xy(D.rename(columns={v: "_code"}), "_code")
            r, p = stats.pearsonr(x, y)
            res[f"{v} | {w} | partial r (pooled blocks)"] = dict(
                r=float(r), p=float(p), n_blocks=int(len(x)))
    S = pd.DataFrame(betas)
    S.to_csv(os.path.join(OUT, "session_betas.csv"), index=False)
    for v in VARIANTS:
        P = S[S.variant == v].pivot(index="session", columns="which",
                                    values="beta").dropna()
        res[f"{v} | first3 - all | beta"] = sign_flip(
            (P["first3"] - P["all"]).to_numpy())
    res["settings"] = dict(
        created=str(datetime.date.today()),
        script="scripts/plan_code_vs_ripples_lagfree.py", source=B1_DIR,
        model="B1 session_beta / design (own reading); sign_flip over "
              "sessions", variants=VARIANTS, future_lags=FUTURE,
        min_cells=MIN_CELLS, min_lags=MIN_LAGS)
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for k, v in res.items():
        if k != "settings":
            print(f"  {k:42s}", {a: (round(b, 4) if isinstance(b, float) else b)
                                 for a, b in v.items()
                                 if a in ("n", "mean", "p_perm", "r", "p",
                                          "n_blocks")})
    return OUT


if __name__ == "__main__":
    main()
