#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why do D3 and the reported phase contrast disagree about the hippocampal
current-square signal in ripples? (2026-11-20, diagnostic)

    reported (swr_phase_contrast, Fig 20/21)   encode > execute
    D3 (context_content, current square)       execute > learning

The two differ in where each cell's location TEMPLATE comes from, and in the
spike window. Everything else is held at the reported pipeline (same ripples,
cells, sessions, statistic, 1000 square-label permutations, 1000 count-matched
draws), and one thing is swapped at a time:

  template  cross_config        other CONFIGURATIONS, all traversals (reported)
            cross_config_phase  other configurations, same phase only (as D4c)
            same_block          other traversals of the SAME BLOCK (as D3 / T4)
            same_block_phase    other traversals of the same block, same phase
  window    duration            the ripple's own detection duration (reported)
            pm125               +-125 ms around the peak (as D3 and D4c)

cross_config x duration must reproduce the reported per-session values.

Also reported: for learning ripples under `same_block`, the share of the
template's dwell time that comes from EXECUTION traversals.

    python scripts/exploratory/location_estimator_reconcile.py

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import datetime

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import mc.analyse.swr_phase_contrast as pcn
import mc.analyse.swr_explore.future_shift as fs
from mc.analyse.swr_explore.context_content import group

SEED = pcn.SEED
HALF = 0.125
PHASES = pcn.PHASES
TEMPLATES = ("cross_config", "cross_config_phase", "same_block",
             "same_block_phase")
WINDOWS = ("duration", "pm125")
CELLS = pcn.ROI_SETS["HC_all"]
OUT = os.path.join(rrsa._derivatives(), "group", "swr",
                   f"location_estimator_reconcile_{datetime.date.today()}")
REPORTED = os.path.join(rrsa._derivatives(), "group", "swr",
                        "phase_contrast_2026-09-30", "per_session.csv")


def same_block_E(spk_s, cells, occ, RT, Xw, ev, rows=None):
    """E (n, 9) and visited (n, 9) from the other traversals of the block,
    optionally only the occupancy rows `rows` (one phase)."""
    sums, _, min_expo = fs.spike_signal(spk_s, cells)
    S, X = sums(occ)
    if rows is not None:
        S, X = S[:, rows], X[:, rows]
        occ, RT = occ[rows].reset_index(drop=True), RT[rows].reset_index(drop=True)
    refs = fs.references(S, X, occ, RT, None, min_expo)
    E, V, _ = fs.evidence(Xw, ev, refs)
    return E, V


def levels(S, masks, ok, s):
    """The reported per-phase levels and count-matched contrast (copied from
    swr_phase_contrast.main, same seed)."""
    rec = {}
    for p_ in PHASES:
        sel = np.flatnonzero(masks[p_] & ok)
        rec[f"n_{p_}"] = len(sel)
        rec[f"z_{p_}"] = (float(pcn._z(S[sel].mean(axis=0)))
                          if len(sel) >= pcn.MIN_RIPPLES else np.nan)
    idx = {p_: np.flatnonzero(masks[p_] & ok) for p_ in PHASES}
    nmin = min(len(v) for v in idx.values())
    if nmin >= pcn.MIN_RIPPLES:
        rs = np.random.default_rng([SEED, s, 0, 7])
        zd = {}
        for p_ in PHASES:
            A = np.zeros((pcn.N_MATCH, len(S)))
            for di in range(pcn.N_MATCH):
                A[di, rs.choice(idx[p_], nmin, replace=False)] = 1.0 / nmin
            zd[p_] = pcn._z(A @ S)
        rec["d_z"] = float(np.nanmean(zd["encode"] - zd["execute"]))
        for p_ in PHASES:
            rec[f"z_{p_}_matched"] = float(np.nanmean(zd[p_]))
    return rec


def run():
    rng = np.random.default_rng(SEED)
    pcn._stat.P = np.vstack([np.arange(9)]
                            + [rng.permutation(9) for _ in range(pcn.N_PERM)])
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    rows, mix = [], []
    for s in sessions:
        if s not in spk:
            continue
        # ripple selection exactly as swr_phase_contrast.main
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < pcn.MIN_RIPPLES:
            continue
        st_s = steps[steps.session == s]
        ph_i, _, _, _ = swc.phase_labels(st_s, occ)
        o = occ.sort_values("start_s")
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = o.index.to_numpy()[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        if inside.sum() < pcn.MIN_RIPPLES:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        half = d_rip / 2.0
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        ph = ph_i[k]
        grids = np.unique(occ.cv_group.to_numpy())
        masks = {p_: (ph == p_) for p_ in PHASES}
        if min(m.sum() for m in masks.values()) < pcn.MIN_RIPPLES:
            continue
        cells = [c for c in r[r.roi.isin(CELLS)].cell.to_numpy()
                 if c < len(spk[s]["spikes"])]
        if len(cells) < pcn.MIN_CELLS:
            continue
        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=pcn.FLANK_GAP_S,
                                         max_per=pcn.MAX_FLANKS)
        w_fl = half[owner] if len(t_fl) else np.zeros(0)
        C_dur, _ = swc.zscore_cells([
            swc.window_counts(spk[s], cells, t_rip - half, t_rip + half),
            swc.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
        C_125, = swc.zscore_cells([
            swc.window_counts(spk[s], cells, t_rip - HALF, t_rip + HALF)])
        RT = fs.role_table(st_s, occ)
        ev = RT.iloc[k].reset_index(drop=True)
        occ_phase = {p_: ph_i == p_ for p_ in PHASES}
        loo = swc.build_templates(spk[s], cells, occ, grids)[1]
        loo_p = {p_: swc.build_templates(spk[s], cells, occ[occ_phase[p_]],
                                         grids)[1] for p_ in PHASES}

        # learning ripples, same_block: share of template dwell from execution
        dwell = (occ.stop_s - occ.start_s).to_numpy()
        key, blk = RT.key.to_numpy(), RT.block.to_numpy()
        fr = []
        for i in np.flatnonzero(masks["encode"]):
            other = (blk == ev.block[i]) & (key != ev.key[i])
            if dwell[other].sum() > 0:
                fr.append(dwell[other & (ph_i == "execute")].sum()
                          / dwell[other].sum())
        mix.append(dict(session=int(s), exec_share=float(np.mean(fr))
                        if fr else np.nan))

        for win, C in (("duration", C_dur), ("pm125", C_125)):
            for tpl in TEMPLATES:
                if tpl == "cross_config":
                    E = swc.score_windows(C, t_rip, grid, loo)
                    ok = np.isfinite(E).all(axis=1)
                elif tpl == "cross_config_phase":
                    E = np.full((len(t_rip), 9), np.nan)
                    for p_ in PHASES:
                        m = masks[p_]
                        E[m] = swc.score_windows(C[:, m], t_rip[m], grid[m],
                                                 loo_p[p_])
                    ok = np.isfinite(E).all(axis=1)
                else:
                    E = np.full((len(t_rip), 9), np.nan)
                    V = np.zeros((len(t_rip), 9), bool)
                    for p_ in PHASES:
                        m = masks[p_]
                        rows_ = None if tpl == "same_block" else occ_phase[p_]
                        E[m], V[m] = same_block_E(
                            spk[s], cells, occ, RT, C[:, m],
                            ev[m].reset_index(drop=True), rows_)
                    ok = np.isfinite(E).all(axis=1) & V[np.arange(len(loc)),
                                                         loc - 1]
                S = pcn._stat(E, loc - 1)
                rec = dict(session=int(s), template=tpl, window=win,
                           n_cells=len(cells))
                rec.update(levels(S, masks, ok, s))
                rows.append(rec)
        print(f"  s{s:02d}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(mix)


def summarise(R, M):
    res = {}
    for (tpl, win), g in R.groupby(["template", "window"]):
        for col in ("z_encode", "z_execute", "z_encode_matched",
                    "z_execute_matched", "d_z"):
            res[f"{tpl} | {win} | {col}"] = group(g[col])
    res["same_block: execution share of the template, learning ripples"] = \
        dict(median=float(M.exec_share.median()),
             q25=float(M.exec_share.quantile(.25)),
             q75=float(M.exec_share.quantile(.75)), n=int(M.exec_share.notna().sum()))
    # check: cross_config x duration against the reported per-session values
    rep = pd.read_csv(REPORTED)
    rep = rep[rep.roi == "HC_all"].set_index("session")
    mine = R[(R.template == "cross_config") & (R.window == "duration")] \
        .set_index("session")
    both = sorted(set(rep.index) & set(mine.index))
    res["reproduces reported"] = {
        col: float(np.nanmax(np.abs(rep.loc[both, col] - mine.loc[both, col])))
        for col in ("z_encode", "z_execute", "d_z")}
    res["reproduces reported"]["n_sessions"] = [len(both), len(rep), len(mine)]
    return res


if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "per_session.csv")
    if os.path.exists(f):
        R = pd.read_csv(f)
        M = pd.read_csv(os.path.join(OUT, "template_mix.csv"))
    else:
        R, M = run()
        R.to_csv(f, index=False)
        M.to_csv(os.path.join(OUT, "template_mix.csv"), index=False)
    res = summarise(R, M)
    settings = dict(created=str(datetime.date.today()),
                    script="scripts/exploratory/location_estimator_reconcile.py",
                    base="mc/analyse/swr_phase_contrast.py (ripples, cells, "
                         "statistic, permutations, count matching)",
                    cells=CELLS, templates=TEMPLATES, windows=WINDOWS,
                    half_s=HALF, seed=SEED, n_perm=pcn.N_PERM,
                    n_match=pcn.N_MATCH, reported=REPORTED,
                    p="sign-flip over sessions (context_content.group), "
                      "uncorrected")
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(dict(settings=settings, tests=res), fh, indent=1)
    for k, v in res.items():
        if "|" in k and v.get("mean") is not None:
            print(f"  {k:48s} n={v['n']:2d} {v['mean']:+.3f}  p={v['p_perm']:.4f}")
        else:
            print(" ", k, v)
    from location_estimator_reconcile_figure import draw
    draw(OUT)
