#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do mPFC and hippocampus carry the CURRENT square together, moment by
moment -- in ripples and ordinary moments, while encoding and executing?
(2026-11-23)

Events and estimator = the reported location analysis
(`mc/analyse/swr_phase_contrast.py`) as used in
`location_content_ripples_by_phase.py`, imported (`events`, `session`,
`occ_weights`, `stat_perm`):
    ripples   peak inside a visit; window = the ripple's own duration;
              counts z-scored per cell with the ripples' flanks
    ordinary  no ripple within 0.5 s; centres every 50 ms inside visits;
              width = the session's median ripple duration; <= 15,000
    templates each cell's map from the OTHER configurations
              (`swc.build_templates`, leave-one-grid_id-out)
    CURRENT   evidence at the current square minus the mean of the other 8
              (`pcn._stat`)
    NEXT      `fut1` of the same script: evidence at the next distinct square
              of the visit sequence, distance-controlled
ROIs: HC = HC_anterior + HC_mid (the reported `HC_all`) and mPFC; final ROI
labels (`rrsa.cell_roi_table`, checked against ROI_assignment).

1. LEVEL. mPFC (and HC) CURRENT and NEXT per event type x phase: the
   imported `session`, unchanged (count-matched z vs 1000 square
   permutations).
2. COUPLING. Sessions with >= 2 HC and >= 2 mPFC cells. Per event:
   h = HC CURRENT, m = mPFC CURRENT (or mPFC NEXT), observed values.
       r        Pearson r(h, m) across the events of one type x phase
       null     m shuffled among events with the SAME current square
                (same type x phase), N_PERM times, the same r function
       coupling r - mean(null r): what is left after removing the part
                that only reflects which squares both regions decode well
   Needs >= MIN_EVENTS events with both values.
Group, session = unit (`context_content.group`: t-test and sign-flip):
coupling vs 0 per type x phase; ripple - ordinary per phase; encode -
execute per type.

    python scripts/now_coupling_hc_mpfc.py

@author: Svenja Kuchenhoff
"""
import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import location_content_ripples_by_phase as lc
from mc.analyse.roi_labels import FINAL_ROI_TABLE
from mc.analyse.swr_explore.context_content import group

rrsa, swl, swc, pcn = lc.rrsa, lc.swl, lc.swc, lc.pcn
SEED = pcn.SEED
PHASES = pcn.PHASES
N_PERM = 1000
MIN_EVENTS = 20
PAIRS = {"HC now x mPFC now": ("CURRENT", "CURRENT"),
         "HC now x mPFC next": ("CURRENT", "fut1")}
OUT = os.path.join(rrsa._derivatives(), "group", "swr",
                   f"now_coupling_hc_mpfc_{datetime.date.today()}")


def corr_rows(h, M):
    """Pearson r between h (n,) and every row of M (k, n)."""
    hc = h - h.mean()
    Mc = M - M.mean(1, keepdims=True)
    return (Mc @ hc) / np.sqrt((Mc ** 2).sum(1) * (hc ** 2).sum())


def shuffle_within(strata, rng):
    """(N_PERM, n) index matrix: each row permutes within strata."""
    n = len(strata)
    idx = np.tile(np.arange(n), (N_PERM, 1))
    for g in np.unique(strata):
        mem = np.flatnonzero(strata == g)
        idx[:, mem] = rng.permuted(np.tile(mem, (N_PERM, 1)), axis=1)
    return idx


def coupling(h, m, square, rng):
    ok = np.isfinite(h) & np.isfinite(m)
    if ok.sum() < MIN_EVENTS:
        return None
    h, m, square = h[ok], m[ok], square[ok]
    r = corr_rows(h, m[None])[0]
    null = corr_rows(h, m[shuffle_within(square, rng)])
    return dict(n=int(ok.sum()), r=float(r), null_mean=float(null.mean()),
                coupling=float(r - null.mean()),
                z=float((r - null.mean()) / null.std()))


def level_session(s, spk_s, r, occ, t_rip, d_rip, steps_s, P):
    lc.ROIS = ("HC_all", "mPFC")
    lc.STATS = ("CURRENT", "fut1")
    return lc.session(s, spk_s, r, occ, t_rip, d_rip, steps_s, P)


def coupling_session(s, spk_s, r, occ, t_all, d_all, steps_s):
    ev = lc.events(s, occ, t_all, d_all, steps_s)
    if ev is None:
        return []
    lc.STATS = ("CURRENT", "fut1")
    OW = lc.occ_weights(steps_s, occ)
    P0 = np.arange(9)[None]                       # observed values only
    kc = ev["kc"]
    kinds = {"ripple": (ev["k"], ev["loc"], ev["masks"]),
             "ordinary": (kc, occ["loc"].to_numpy()[kc].astype(int),
                          ev["masks_c"])}
    V = {}
    for roi in ("HC_all", "mPFC"):
        cells = [c for c in r[r.roi.isin(pcn.ROI_SETS[roi])].cell.to_numpy()
                 if c < len(spk_s["spikes"])]
        if len(cells) < pcn.MIN_CELLS:
            return []
        _, loo = swc.build_templates(spk_s, cells, occ, ev["grids"])
        C_rip, _ = swc.zscore_cells([
            swc.window_counts(spk_s, cells, ev["t_rip"] - ev["half"],
                              ev["t_rip"] + ev["half"]),
            swc.window_counts(spk_s, cells, ev["t_fl"] - ev["w_fl"],
                              ev["t_fl"] + ev["w_fl"])])
        C_ctl, = swc.zscore_cells([
            swc.window_counts(spk_s, cells, ev["tc"] - ev["h"],
                              ev["tc"] + ev["h"])])
        E = {"ripple": swc.score_windows(C_rip, ev["t_rip"], ev["grid"], loo),
             "ordinary": swc.score_windows(
                 C_ctl, ev["tc"], occ.cv_group.to_numpy()[kc], loo)}
        for kind, (kk, _, _) in kinds.items():
            for stat in ("CURRENT", "fut1"):
                W, dfn = OW[stat]
                v = lc.stat_perm(E[kind], W[kk], P0)[:, 0]
                ok = np.isfinite(E[kind]).all(1) & dfn[kk]
                V[(roi, stat, kind)] = np.where(ok, v, np.nan)
    rows = []
    for kind, (kk, square, masks) in kinds.items():
        for ph in PHASES:
            for pair, (sh, sm) in PAIRS.items():
                rng = np.random.default_rng([SEED, s, PHASES.index(ph),
                                             list(kinds).index(kind)])
                mk = masks[ph]
                c = coupling(V[("HC_all", sh, kind)][mk],
                             V[("mPFC", sm, kind)][mk], square[mk], rng)
                if c is not None:
                    rows.append(dict(session=int(s), kind=kind, phase=ph,
                                     pair=pair, **c))
    return rows


def run():
    rng = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng.permutation(9) for _ in range(pcn.N_PERM)])
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    lev, cou = [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < pcn.MIN_RIPPLES:
            continue
        st = steps[steps.session == s]
        lev.append(delayed(level_session)(s, spk[s], r, occ, t_rip, d_rip,
                                          st, P))
        cou.append(delayed(coupling_session)(s, spk[s], r, occ, t_rip,
                                             d_rip, st))
    L = Parallel(n_jobs=6, verbose=5)(lev)
    C = Parallel(n_jobs=6, verbose=5)(cou)
    return (pd.DataFrame([x for o in L for x in o]),
            pd.DataFrame([x for o in C for x in o]))


def summarise(L, C):
    res = {"level": {}, "coupling": {}, "coupling_contrasts": {}}
    both = sorted(set(C.session))
    for roi in ("HC_all", "mPFC"):
        for stat in ("CURRENT", "fut1"):
            g = L[(L.roi == roi) & (L.stat == stat)]
            for col in ("zm_rip_encode", "zm_rip_execute", "zm_ctl_encode",
                        "zm_ctl_execute", "rip-ctl_encode", "rip-ctl_execute",
                        "dz_rip", "dz_ctl"):
                res["level"][f"{roi} | {stat} | {col} | all sessions"] = group(
                    g[col].to_numpy(float))
                res["level"][f"{roi} | {stat} | {col} | coupling sessions"] = \
                    group(g[g.session.isin(both)][col].to_numpy(float))
    for pair in PAIRS:
        W = C[C.pair == pair].pivot_table(index="session",
                                          columns=["kind", "phase"],
                                          values="coupling")
        for kind in ("ripple", "ordinary"):
            for ph in PHASES:
                if (kind, ph) in W:
                    g = group(W[(kind, ph)].to_numpy(float))
                    v = W[(kind, ph)].dropna().to_numpy()
                    if len(v) >= 3:
                        g["p_one_sided_gt0"] = float(
                            stats.ttest_1samp(v, 0, alternative="greater").pvalue)
                    res["coupling"][f"{pair} | {kind} | {ph}"] = g
            if (kind, "encode") in W and (kind, "execute") in W:
                res["coupling_contrasts"][f"{pair} | {kind} | encode - execute"] = \
                    group((W[(kind, "encode")] - W[(kind, "execute")]).to_numpy(float))
        for ph in PHASES:
            if ("ripple", ph) in W and ("ordinary", ph) in W:
                res["coupling_contrasts"][f"{pair} | {ph} | ripple - ordinary"] = \
                    group((W[("ripple", ph)] - W[("ordinary", ph)]).to_numpy(float))
    # does the coupling switch partner? (now x now - now x next), encode - execute
    W = C.pivot_table(index="session", columns=["pair", "kind", "phase"],
                      values="coupling")
    a, b = list(PAIRS)
    for kind in ("ripple", "ordinary"):
        d = {ph: W[(a, kind, ph)] - W[(b, kind, ph)] for ph in PHASES}
        for ph in PHASES:
            res["coupling_contrasts"][f"{a} - {b} | {kind} | {ph}"] = group(
                d[ph].to_numpy(float))
        res["coupling_contrasts"][f"({a} - {b}) encode - execute | {kind}"] = \
            group((d["encode"] - d["execute"]).to_numpy(float))
    res["n_sessions_coupling"] = len(both)
    res["n_events_median"] = {" | ".join(k): float(v) for k, v in
                              C.groupby(["pair", "kind", "phase"]).n.median().items()}
    return res


if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    fl, fc = os.path.join(OUT, "level_per_session.csv"), os.path.join(
        OUT, "coupling_per_session.csv")
    if os.path.exists(fl) and os.path.exists(fc):
        L, C = pd.read_csv(fl), pd.read_csv(fc)
    else:
        L, C = run()
    L.to_csv(fl, index=False)
    C.to_csv(fc, index=False)
    res = summarise(L, C)
    res["settings"] = dict(
        created=str(datetime.date.today()),
        script="scripts/now_coupling_hc_mpfc.py", roi_labels=FINAL_ROI_TABLE,
        rois={"HC_all": pcn.ROI_SETS["HC_all"], "mPFC": pcn.ROI_SETS["mPFC"]},
        pairs=PAIRS, n_perm_coupling=N_PERM, min_events=MIN_EVENTS,
        null="mPFC values shuffled among events with the same current square "
             "(same type x phase)", seed=SEED,
        level="location_content_ripples_by_phase.session (reported estimator)")
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for k, v in res["coupling"].items():
        print(k, {a: round(b, 4) if isinstance(b, float) else b
                  for a, b in v.items()})
    for k, v in res["coupling_contrasts"].items():
        print(k, {a: round(b, 4) if isinstance(b, float) else b
                  for a, b in v.items()})
