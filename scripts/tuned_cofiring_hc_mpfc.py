#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do the better-tuned cells of HC and mPFC fire together during ripples?
(2026-11-23)

SK's question, deliberately less precise than the location-content
coupling (`now_coupling_hc_mpfc.py`): no templates, no decoded square --
just the firing of the cells with higher lag-profile CV r.

Events (identical to the reported location analysis, `lc.events`):
    ripples   peak inside a visit, window = the ripple's own duration;
              counts z-scored per cell together with the ripple flanks
    ordinary  no ripple within 0.5 s, every 50 ms inside visits, width =
              the session's median ripple duration; z-scored on their own
    phase     encode (before a block's first correct loop) / execute

Tuning per cell: CV r of the per-lag analysis (per-repeat maps, correct
loops = the new Fig 3c table), at
    now      r at 0 deg                      (HC and mPFC)
    future   mean r at 30 and 60 deg         (mPFC, the Fig 3c window)

Per event and region:
    T = mean_i (r_i - mean_region r) * z_i
i.e. how much more the better-tuned cells fire than the less-tuned ones
in that event. No threshold, no cell selection.

Per session (>= 2 HC_all and >= 2 mPFC cells with a tuning value), per
event type x phase:
    r_tuned  Pearson r across events of T_HC and T_mPFC
    null     the same r with the tuning values shuffled among the cells of
             each region (N_PERM, the same function, the identity first)
    tuned coupling = r_tuned - mean(null)  -- what is specific to WHICH
             cells fire, beyond the two regions simply firing together
    r_pop    Pearson r of the plain mean z of all cells (context: general
             co-firing)
Pairs: HC now x mPFC now (the hypothesis); HC now x mPFC future.
Group, session = unit: t-test + sign-flip (`context_content.group`).

    python scripts/tuned_cofiring_hc_mpfc.py
    -> group/swr/tuned_cofiring_hc_mpfc_<date>/

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
import location_content_ripples_by_phase as lc
from now_coupling_hc_mpfc import corr_rows
from mc.analyse.roi_labels import read_per_cell_table
from mc.analyse.swr_explore.context_content import group

rrsa, swl, swc, pcn = lc.rrsa, lc.swl, lc.swc, lc.pcn
SEED = pcn.SEED
PHASES = pcn.PHASES
N_PERM = 1000
MIN_EVENTS = 20
MIN_CELLS = 2
PER_LAG = os.path.join(rrsa._derivatives(), "group", "per_lag_loopsets_2026-11-23",
                       "per_repeat", "execute", "per_cell_ALL_ROIs.csv")
TUNING = {"now": ["r_lag000_noctrl"],
          "future": ["r_lag030_noctrl", "r_lag060_noctrl"]}
PAIRS = {"HC now x mPFC now": ("now", "now"),
         "HC now x mPFC future": ("now", "future")}
OUT = os.path.join(rrsa._derivatives(), "group", "swr",
                   f"tuned_cofiring_hc_mpfc_{datetime.date.today()}")


def tuning_table():
    t = read_per_cell_table(PER_LAG)
    out = t[["subject_id", "cell_idx"]].astype(int).copy()
    for name, cols in TUNING.items():
        out[name] = t[cols].mean(1)
    return out


def tuned_index(Z, W):
    """Z (n_events, n_cells) z-counts; W (k, n_cells) centred tunings
    -> (k, n_events) tuned activity."""
    return (W @ Z.T) / Z.shape[1]


def weight_perms(w, rng):
    """(N_PERM + 1, n) centred weights; row 0 = observed."""
    w = w - w.mean()
    P = np.vstack([np.arange(len(w))] +
                  [rng.permutation(len(w)) for _ in range(N_PERM)])
    return w[P]


def tuned_r(Zh, Zm, Wh, Wm):
    """r between HC and mPFC tuned activity for every weight row."""
    Th, Tm = tuned_index(Zh, Wh), tuned_index(Zm, Wm)
    Th = Th - Th.mean(1, keepdims=True)
    Tm = Tm - Tm.mean(1, keepdims=True)
    return (Th * Tm).sum(1) / np.sqrt((Th ** 2).sum(1) * (Tm ** 2).sum(1))


def session(s, spk_s, r, occ, t_all, d_all, steps_s, tun):
    ev = lc.events(s, occ, t_all, d_all, steps_s)
    if ev is None:
        return []
    Z, Wt = {}, {}
    for roi in ("HC_all", "mPFC"):
        rr = r[r.roi.isin(pcn.ROI_SETS[roi])]
        rr = rr[rr.cell < len(spk_s["spikes"])]
        rr = rr.assign(cell_idx=rr.cell + 1).merge(     # cell is 0-based
            tun[tun.subject_id == s], on="cell_idx", how="inner")
        rr = rr[np.isfinite(rr[list(TUNING)]).all(1)]
        if len(rr) < MIN_CELLS:
            return []
        cells = rr.cell.to_numpy()
        C_rip, _ = swc.zscore_cells([
            swc.window_counts(spk_s, cells, ev["t_rip"] - ev["half"],
                              ev["t_rip"] + ev["half"]),
            swc.window_counts(spk_s, cells, ev["t_fl"] - ev["w_fl"],
                              ev["t_fl"] + ev["w_fl"])])
        C_ctl, = swc.zscore_cells([
            swc.window_counts(spk_s, cells, ev["tc"] - ev["h"],
                              ev["tc"] + ev["h"])])
        # swc returns (cells, windows); here events are rows
        Z[(roi, "ripple")], Z[(roi, "ordinary")] = C_rip.T, C_ctl.T
        Wt[roi] = {k: rr[k].to_numpy(float) for k in TUNING}
        Wt[roi + "_n"] = len(rr)
    masks = {"ripple": ev["masks"], "ordinary": ev["masks_c"]}
    rows = []
    for kind in ("ripple", "ordinary"):
        for ph in PHASES:
            mk = masks[kind][ph]
            Zh, Zm = Z[("HC_all", kind)][mk], Z[("mPFC", kind)][mk]
            ok = np.isfinite(Zh).all(1) & np.isfinite(Zm).all(1)
            if ok.sum() < MIN_EVENTS:
                continue
            Zh, Zm = Zh[ok], Zm[ok]
            r_pop = corr_rows(Zh.mean(1), Zm.mean(1)[None])[0]
            for pair, (th, tm) in PAIRS.items():
                rng = np.random.default_rng([SEED, s, PHASES.index(ph),
                                             ("ripple", "ordinary").index(kind),
                                             list(PAIRS).index(pair)])
                R = tuned_r(Zh, Zm, weight_perms(Wt["HC_all"][th], rng),
                            weight_perms(Wt["mPFC"][tm], rng))
                rows.append(dict(session=int(s), kind=kind, phase=ph, pair=pair,
                                 n_events=int(ok.sum()),
                                 n_hc=Wt["HC_all_n"], n_mpfc=Wt["mPFC_n"],
                                 r_pop=float(r_pop), r_tuned=float(R[0]),
                                 null_mean=float(np.nanmean(R[1:])),
                                 tuned_coupling=float(R[0] - np.nanmean(R[1:]))))
    return rows


def run():
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    tun = tuning_table()
    jobs = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < pcn.MIN_RIPPLES:
            continue
        jobs.append(delayed(session)(s, spk[s], r, occ, t_rip, d_rip,
                                     steps[steps.session == s], tun))
    out = Parallel(n_jobs=6, verbose=5)(jobs)
    return pd.DataFrame([x for o in out for x in o])


def summarise(D):
    res = {"n_sessions": int(D.session.nunique()), "cells": {}, "tests": {},
           "contrasts": {}}
    res["cells"] = {k: int(v) for k, v in
                    D.drop_duplicates("session")[["n_hc", "n_mpfc"]].sum().items()}
    for kind in ("ripple", "ordinary"):
        for ph in PHASES:
            g = D[(D.kind == kind) & (D.phase == ph)]
            res["tests"][f"population co-firing r | {kind} | {ph}"] = group(
                g.drop_duplicates("session").r_pop.to_numpy(float))
            for pair in PAIRS:
                gp = g[g.pair == pair]
                res["tests"][f"{pair} | {kind} | {ph}"] = group(
                    gp.tuned_coupling.to_numpy(float))
    piv = D.pivot_table(index="session", columns=["pair", "kind", "phase"],
                        values="tuned_coupling")
    for pair in PAIRS:
        for kind in ("ripple", "ordinary"):
            d = (piv[(pair, kind, "encode")] - piv[(pair, kind, "execute")]).dropna()
            res["contrasts"][f"{pair} | {kind} | encode - execute"] = group(d.to_numpy())
        for ph in PHASES:
            d = (piv[(pair, "ripple", ph)] - piv[(pair, "ordinary", ph)]).dropna()
            res["contrasts"][f"{pair} | ripple - ordinary | {ph}"] = group(d.to_numpy())
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    D = run()
    D.to_csv(os.path.join(OUT, "per_session.csv"), index=False)
    res = summarise(D)
    res["config"] = dict(per_lag_table=PER_LAG, tuning=TUNING, n_perm=N_PERM,
                         min_events=MIN_EVENTS, min_cells=MIN_CELLS, seed=SEED,
                         events="lc.events (reported location analysis)")
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1,
              default=float)
    print(json.dumps({k: res[k] for k in ("n_sessions", "cells")}, indent=1))
    for k, v in list(res["tests"].items()) + list(res["contrasts"].items()):
        print(f"{k:60s} mean {v.get('mean', np.nan):+.4f}  "
              f"p {v.get('p', np.nan):.3f}  n {v['n']}")
