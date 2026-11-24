#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1b (EXPLORATORY, CHANGELOG 2026-11-14 (b), written after seeing B1): the B1
block dose-response with

  1. the future lags collapsed -- F_all = mean block code over lags 30..330
     (all 11 'beyond-now' lags, the manuscript's existing set); F_near =
     30..90; F_30-240 (post hoc, the range read off B1 panel d); and the
     per-session mean of the 11 per-lag ripple betas;
  2. different ripples as the predictor -- learning ripples by context
     (first traversal / rest; after correct, first-discovery, incorrect
     uncovers; none of those), execution ripples, the whole block.

mPFC only, own reading only (log duration of the predictor's window as the
covariate, plus n learning loops, block order, run). Reuses B1's saved
per-cell block codes (first 3 execution loops); nothing is recomputed.

Usage:
    conda activate env_multiple_clocks
    python scripts/plan_code_vs_ripples_types.py

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
import mc.analyse.helpers_human_cells as hh
from plan_code_vs_ripples import block_table, session_beta, N_FIRST
from per_lag_encoding_by_phase import fz, sign_flip, DATA_DIR

SEED = 42
B1_DIR = os.path.join(DATA_DIR, "group", "swr", "plan_code_vs_ripples_2026-11-14")
OUT = os.path.join(DATA_DIR, "group", "swr",
                   f"plan_code_vs_ripples_types_{datetime.date.today()}")
MAX_PAUSE_S = 2.0
LAGS = list(range(0, 360, 30))
FUTURE = {"F_all": list(range(30, 360, 30)), "F_near": [30, 60, 90],
          "F_30-240": list(range(30, 270, 30))}
MIN_LAGS = 8
MIN_CELLS = 2
SUBSETS = {   # name: (window, covariate window, label)
    "L_all": ("learning", "learn", "all learning ripples (= B1)"),
    "L_FT": ("first traversal", "learn", "first traversal"),
    "L_rest": ("rest of learning", "learn", "rest of learning"),
    "L_postCorrect": ("after correct uncovers", "learn",
                      "after correct uncovers"),
    "L_postFD": ("after first discoveries", "learn",
                 "after first discoveries"),
    "L_postError": ("after incorrect uncovers", "learn",
                    "after incorrect uncovers"),
    "L_other": ("not after any uncover", "learn", "not after an uncover"),
    "E_first3": ("first 3 execution loops", "first3",
                 "execution, first 3 loops"),
    "E_all": ("whole execution", "exe", "whole execution"),
    "B_all": ("whole block", "block", "whole block"),
}


def in_windows(t, W):
    m = np.zeros(len(t), bool)
    for a, b in W:
        m |= (t > a) & (t <= b)
    return m


def block_ripples(s, P, t_rip):
    d = hh.load_norm_data(DATA_DIR, [f"{s:02d}"], res_data=False)
    beh = d[f"sub-{s:02d}"]["beh"].reset_index(drop=True)
    B = block_table(beh, t_rip, np.empty(0))
    exe = ~((beh.rep_correct == 0) & (beh.correct == 0)).to_numpy()
    unc = P[P.kind == "uncover"].dropna(subset=["still_next_s"])
    pauses = {}
    for name, q in (("correct", unc.valence == "correct"),
                    ("FD", (unc.valence == "correct")
                     & (unc.stage == "first uncovers")),
                    ("error", unc.valence == "error")):
        u = unc[q]
        pauses[name] = list(zip(u.t_s, u.t_s + np.minimum(u.still_next_s,
                                                          MAX_PAUSE_S)))
    any_pause = pauses["correct"] + pauses["error"]
    rows = []
    for _, b in B.iterrows():
        g = beh[beh.grid_no == b.grid_no]
        t_ft = float(g.t_D.iloc[0])
        r = np.flatnonzero((beh.grid_no == b.grid_no).to_numpy() & exe)
        r = r[r != g.index[0]][:N_FIRST]
        t_f3 = float(beh.t_D.iloc[r].max()) if len(r) else np.nan
        tr = t_rip
        learn = (tr >= b.t_learn0) & (tr < b.t_exe0)
        rec = dict(session=s, grid_no=int(b.grid_no), order=b.order,
                   run=b.run, n_learn_loops=b.n_learn_loops,
                   learn_s=b.learn_s, exe_s=b.exe_s,
                   block_s=b.t_end - b.t_learn0, first3_s=t_f3 - b.t_exe0)
        rec["L_all"] = learn.sum()
        rec["L_FT"] = (learn & (tr < t_ft)).sum()
        rec["L_rest"] = (learn & (tr >= t_ft)).sum()
        rec["L_postCorrect"] = (learn & in_windows(tr, pauses["correct"])).sum()
        rec["L_postFD"] = (learn & in_windows(tr, pauses["FD"])).sum()
        rec["L_postError"] = (learn & in_windows(tr, pauses["error"])).sum()
        rec["L_other"] = (learn & ~in_windows(tr, any_pause)).sum()
        rec["E_first3"] = ((tr >= b.t_exe0) & (tr < t_f3)).sum()
        rec["E_all"] = ((tr >= b.t_exe0) & (tr < b.t_end)).sum()
        rec["B_all"] = ((tr >= b.t_learn0) & (tr < b.t_end)).sum()
        rows.append(rec)
    return pd.DataFrame(rows)


def block_codes(C):
    """Per block and lag: mean Fisher z over mPFC cells (>= 2); plus the
    collapsed future measures."""
    G = C[(C.roi == "mPFC") & (C.which == "first3")
          & C.lag.str.startswith("lag")].copy()
    G["z"] = fz(G.r)
    G["lagdeg"] = G.lag.str[3:].astype(int)
    A = G.groupby(["session", "grid_no", "lagdeg"]).z.agg(["mean", "count"])
    A = A[A["count"] >= MIN_CELLS]["mean"].unstack("lagdeg")
    for name, lags in FUTURE.items():
        v = A[lags]
        A[name] = v.mean(1).where(v.notna().sum(1) >= min(MIN_LAGS, len(lags)))
    return A.reset_index()


def design(D, subset):
    cov = {"learn": D.learn_s, "first3": D.first3_s, "exe": D.exe_s,
           "block": D.block_s}[SUBSETS[subset][1]]
    return np.column_stack([np.log1p(D[subset]), np.log(cov.clip(0.1)),
                            D.n_learn_loops, D.order, D.run]).astype(float)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    C = pd.read_csv(os.path.join(B1_DIR, "cell_block_codes.csv"))
    A = block_codes(C)
    rip = pd.read_csv(os.path.join(DATA_DIR, "group", "swr", "bundle_v2",
                                   "ripples.csv"), usecols=["session", "t_peak_s"])
    P = pd.read_csv(os.path.join(DATA_DIR, "group", "swr",
                                 "press_categories.csv"))
    Bs = []
    for s in sorted(A.session.unique()):
        t_rip = np.unique(np.round(rip[rip.session == s].t_peak_s
                                   .to_numpy(float), 3))
        Bs.append(block_ripples(int(s), P[P.session == s], t_rip))
        print(f"  s{int(s):02d}", flush=True)
    B = pd.concat(Bs, ignore_index=True)
    D = B.merge(A, on=["session", "grid_no"])
    D.to_csv(os.path.join(OUT, "blocks_with_codes.csv"), index=False)
    outcomes = LAGS + list(FUTURE)
    rows = []
    for sub in SUBSETS:
        for s, d in D.groupby("session"):
            X = design(d, sub)
            rec = dict(subset=sub, session=int(s), n_blocks=len(d),
                       mean_count=float(d[sub].mean()))
            for o in outcomes:
                rec[str(o)] = session_beta(d[o].to_numpy(float), X)
            rec["mean_beta_30_330"] = np.nanmean([rec[str(l)] for l in
                                                  FUTURE["F_all"]]) \
                if np.isfinite([rec[str(l)] for l in FUTURE["F_all"]]).sum() \
                >= MIN_LAGS else np.nan
            rows.append(rec)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "session_betas.csv"), index=False)
    res = {}
    for sub in SUBSETS:
        g = S[S.subset == sub]
        for o in [str(x) for x in outcomes] + ["mean_beta_30_330"]:
            res[f"{sub} | {o}"] = sign_flip(g[o].to_numpy(float))
    fam = [f"{sub} | {o}" for sub in SUBSETS for o in ("60", "F_all")]
    q = bh([res[k].get("p_perm", 1.0) for k in fam])
    for k, qq in zip(fam, q):
        res[k]["q_bh_family"] = float(qq)
    print("\n  subset            60°                F_all              "
          "mean β 30-330      F_near             F_30-240 (post hoc)")
    for sub in SUBSETS:
        cells = []
        for o in ("60", "F_all", "mean_beta_30_330", "F_near", "F_30-240"):
            r = res[f"{sub} | {o}"]
            cells.append(f"{r.get('mean', np.nan):+.3f} p={r.get('p_perm', np.nan):.3f}"
                         + (f" q={r['q_bh_family']:.2f}" if 'q_bh_family' in r
                            else ""))
        print(f"  {sub:16s} " + "   ".join(f"{c:24s}" for c in cells))
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(dict(settings=dict(
            analysis="plan_code_vs_ripples_types (B1b, exploratory)",
            created=str(datetime.date.today()),
            script=os.path.relpath(__file__, REPO), seed=SEED, roi="mPFC",
            reading="own (log window duration + n loops + order + run)",
            codes=os.path.join(B1_DIR, "cell_block_codes.csv"),
            which="first 3 execution loops", future=FUTURE,
            min_lags=MIN_LAGS, min_cells=MIN_CELLS,
            subsets={k: v[0] for k, v in SUBSETS.items()},
            covariate_window={k: v[1] for k, v in SUBSETS.items()},
            pause=f"(press, press + min(still_next_s, {MAX_PAUSE_S}))",
            presses="group/swr/press_categories.csv",
            family="BH over 10 subsets x {60, F_all}",
            note="F_30-240 chosen after seeing B1 panel d: circular",
            preregistration="CHANGELOG.md 2026-11-14 (b) B1b (exploratory)"),
            tests=res), f, indent=1, default=str)
    from plan_code_vs_ripples_types_figures import draw
    draw(OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
