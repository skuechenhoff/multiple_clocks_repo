#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why is the evidence NEGATIVE in some regions? Is the template transferable?

    python scripts/swr_content_explore.py cross_task

The estimator scores a ripple against a place template built from the OTHER
configurations. If a cell's firing at a square is not stable across
configurations, that template is not merely weak -- it can be ANTI-correlated
with what the cell actually does in the held-out one, and the population match
then comes out reliably negative. That is a candidate explanation for mOFC
(ripple -0.544, P = 0.028, non-ripple +0.031) and it is checkable directly.

SK'S HYPOTHESIS, tested here: the same square is visited at a different point
in the sequence in different configurations -- location 5 is first in one
layout and half-way through another -- so a cell tuned to location x LAG fires
differently at the same square across tasks. If so, cross-task rate-map
similarity should DECREASE as the lag at which that square is visited diverges
between the two tasks.

THREE OUTPUTS
  1. the config x config correlation of each cell's rate map, averaged over
     cells -- a 'confusion matrix' over tasks rather than over locations;
  2. the mean off-diagonal value per ROI, which is what the leave-one-out
     template actually relies on. Near zero means the template carries no
     transferable information; NEGATIVE means it is actively misleading;
  3. whether that pairwise similarity is predicted by how similar the two
     configurations' LAG structure is -- the direct test of the hypothesis.

Lag of a square in a configuration = the circular mean of the warped angle
theta at which it is occupied (`swr_warp`), so 'visited early' vs 'visited
late' is on the same axis the per-lag pipeline uses.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import mc.analyse.swr_warp as sww
from mc.analyse.swr_content import LOCATIONS

SEED = 42
ROI_GROUPS = {"HC": ["HC_anterior", "HC_mid"], "mPFC": ["mPFC"],
              "mOFC": ["mOFC"]}
MIN_SHARED = 5            # locations shared between two configurations
MIN_CELLS = 2


def _circ_mean_deg(x):
    if not len(x):
        return np.nan
    r = np.deg2rad(x)
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def _restricted_map(stimes, occ_g, seg_lo, seg_hi):
    """Place map built only from time inside [seg_lo, seg_hi] segments.

    Used to ask whether the template transfers across configurations during
    RIPPLES specifically, versus outside them -- the two cannot be compared
    unless each is built from its own time base.
    """
    a = occ_g.start_s.to_numpy(float); b = occ_g.stop_s.to_numpy(float)
    loc = occ_g["loc"].to_numpy(float)
    out = np.full(9, np.nan)
    for i, L in enumerate(LOCATIONS):
        m = loc == L
        if not m.any():
            continue
        lo = np.maximum(seg_lo[:, None], a[m][None, :])
        hi = np.minimum(seg_hi[:, None], b[m][None, :])
        w = np.clip(hi - lo, 0, None)
        t = float(w.sum())
        if t <= 0:
            continue
        n = 0.0
        ii, jj = np.nonzero(w > 0)
        for i2, j2 in zip(ii, jj):
            n += ((stimes >= lo[i2, j2]) & (stimes <= hi[i2, j2])).sum()
        out[i] = n / t
    return out


def _circ_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def main():
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows, pairs, models = [], [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, _, _ = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r):
            continue
        st = steps[steps.session == s]
        grids = np.unique(occ.cv_group.to_numpy())
        if len(grids) < 3:
            continue
        # lag (warped theta) at which each square is occupied, per configuration
        bnd = sww.trial_boundaries(st)
        t_mid = (occ.start_s.to_numpy() + occ.stop_s.to_numpy()) / 2
        th, _ = sww.theta_of(t_mid, bnd)
        cvg = occ.cv_group.to_numpy()
        loc = occ["loc"].to_numpy().astype(int)
        # ripple and non-ripple segment edges for this session
        _t = rip[rip.session == s]
        _h = _t.duration_s.to_numpy(float) / 2.0
        rlo = _t.t_peak_s.to_numpy(float) - _h
        rhi = _t.t_peak_s.to_numpy(float) + _h
        edges = np.sort(np.concatenate([[occ.start_s.min() - 1],
                                        rlo - 1.0, rhi + 1.0,
                                        [occ.stop_s.max() + 1]]))
        nlo, nhi = edges[:-1][::2], edges[1:][::2]
        keep = nhi > nlo
        nlo, nhi = nlo[keep], nhi[keep]
        lag_of = {}
        for g in grids:
            v = np.full(9, np.nan)
            for L in range(1, 10):
                m = (cvg == g) & (loc == L) & np.isfinite(th)
                if m.sum() >= 3:
                    v[L - 1] = _circ_mean_deg(th[m])
            lag_of[g] = v

        # ---- MODEL similarity between configurations, for comparison
        for i, g1 in enumerate(grids):
            for g2 in grids[i + 1:]:
                mods = {}
                for nm, fn in (("occupancy", None), ("reward_overlap", None),
                               ("trajectory", None)):
                    pass
                o1 = occ[cvg == g1]; o2 = occ[cvg == g2]
                # (1) occupancy: time spent at each square
                d1 = np.array([ (o1.stop_s - o1.start_s)[o1["loc"] == L].sum()
                                for L in LOCATIONS], float)
                d2 = np.array([ (o2.stop_s - o2.start_s)[o2["loc"] == L].sum()
                                for L in LOCATIONS], float)
                occ_r = (np.corrcoef(d1, d2)[0, 1]
                         if d1.std() > 0 and d2.std() > 0 else np.nan)
                # (2) reward overlap: how many of the 4 rewards are shared
                def _rew(g):
                    k = occ[cvg == g].step_idx.to_numpy()
                    sv = st.sort_values("t_s").reset_index(drop=True)
                    r4 = sv[["rew_A", "rew_B", "rew_C", "rew_D"]].to_numpy(float)
                    return set(r4[k[0]]) if len(k) else set()
                ro = len(_rew(g1) & _rew(g2)) / 4.0
                # (3) trajectory: 9x9 step-transition matrices
                def _trans(o):
                    T = np.zeros((9, 9))
                    L = o["loc"].to_numpy().astype(int)
                    for x, y in zip(L[:-1], L[1:]):
                        if 1 <= x <= 9 and 1 <= y <= 9:
                            T[x - 1, y - 1] += 1
                    return T.ravel()
                t1, t2 = _trans(o1), _trans(o2)
                tr = (np.corrcoef(t1, t2)[0, 1]
                      if t1.std() > 0 and t2.std() > 0 else np.nan)
                models.append(dict(session=s, g1=float(g1), g2=float(g2),
                                   occupancy_sim=float(occ_r),
                                   reward_overlap=float(ro),
                                   trajectory_sim=float(tr)))

        for roiname, members in ROI_GROUPS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            # rate map per cell per configuration
            maps = {g: np.array([swc.place_map(spk[s]["spikes"][c],
                                               occ[cvg == g])
                                 for c in cells]) for g in grids}
            # the same maps built from RIPPLE time only, and from non-ripple
            # time only, so transferability can be compared between the two
            maps_w = {"rip": {}, "norip": {}}
            for g in grids:
                og = occ[cvg == g]
                maps_w["rip"][g] = np.array([
                    _restricted_map(spk[s]["spikes"][c], og, rlo, rhi)
                    for c in cells])
                maps_w["norip"][g] = np.array([
                    _restricted_map(spk[s]["spikes"][c], og, nlo, nhi)
                    for c in cells])
            for i, g1 in enumerate(grids):
                for g2 in grids[i + 1:]:
                    rs = []
                    for ci in range(len(cells)):
                        a, b = maps[g1][ci], maps[g2][ci]
                        ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() < MIN_SHARED or np.std(a[ok]) == 0 \
                           or np.std(b[ok]) == 0:
                            continue
                        rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
                    if len(rs) < MIN_CELLS:
                        continue
                    ld = _circ_diff(lag_of[g1], lag_of[g2])
                    pairs.append(dict(session=s, roi=roiname,
                                      g1=float(g1), g2=float(g2),
                                      n_cells=len(rs),
                                      map_r=float(np.mean(rs)),
                                      lag_div=float(np.nanmean(ld))))
                    for nm in ("rip", "norip"):
                        rs2 = []
                        for ci in range(len(cells)):
                            a_ = maps_w[nm][g1][ci]; b_ = maps_w[nm][g2][ci]
                            ok = np.isfinite(a_) & np.isfinite(b_)
                            if ok.sum() < MIN_SHARED or np.std(a_[ok]) == 0 \
                               or np.std(b_[ok]) == 0:
                                continue
                            rs2.append(np.corrcoef(a_[ok], b_[ok])[0, 1])
                        pairs[-1][f"map_r_{nm}"] = (float(np.mean(rs2))
                                                    if len(rs2) >= MIN_CELLS
                                                    else np.nan)
            # leave-one-out: each configuration against the MEAN of the others
            for g in grids:
                oth = [x for x in grids if x != g]
                rs = []
                for ci in range(len(cells)):
                    a = maps[g][ci]
                    b = np.nanmean(np.stack([maps[x][ci] for x in oth]), axis=0)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() < MIN_SHARED or np.std(a[ok]) == 0 \
                       or np.std(b[ok]) == 0:
                        continue
                    rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
                if len(rs) >= MIN_CELLS:
                    rows.append(dict(session=s, roi=roiname, grid=float(g),
                                     n_cells=len(rs), loo_r=float(np.mean(rs))))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows); PRS = pd.DataFrame(pairs)
    d = os.path.join(deriv, "group", "swr",
                     f"cross_task_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    MOD = pd.DataFrame(models)
    R.to_csv(os.path.join(d, "loo.csv"), index=False)
    PRS.to_csv(os.path.join(d, "pairs.csv"), index=False)
    MOD.to_csv(os.path.join(d, "models.csv"), index=False)
    PRS = PRS.merge(MOD, on=["session", "g1", "g2"], how="left")
    PRS.to_csv(os.path.join(d, "pairs.csv"), index=False)

    print("\n=== IS THE PLACE TEMPLATE TRANSFERABLE ACROSS CONFIGURATIONS? ===")
    print("    leave-one-out: each configuration's rate map vs the mean of the "
          "others\n")
    out, ov = [], {}
    for roiname in ROI_GROUPS:
        g = R[R.roi == roiname]
        v = g.groupby("session").loo_r.mean().dropna()
        if len(v) < 5:
            continue
        t, p = stats.ttest_1samp(v, 0)
        print(f"  {roiname:6s} n = {len(v):3d} sessions | mean r = "
              f"{v.mean():+.4f} | t = {t:+5.2f} | P = {p:.4g}"
              f"{' *' if p < .05 else ''}")
        rec = dict(roi=roiname, n_sessions=int(len(v)), loo_r=float(v.mean()),
                   t=float(t), p=float(p))
        out.append(rec); ov[f"{roiname}|loo"] = rec

    print("\n=== DOES LAG DIVERGENCE EXPLAIN IT? ===")
    print("    per configuration PAIR: rate-map similarity vs how differently "
          "the squares are timed\n")
    for roiname in ROI_GROUPS:
        g = PRS[PRS.roi == roiname].dropna(subset=["map_r", "lag_div"])
        if len(g) < 20:
            continue
        rr, pp = stats.pearsonr(g.lag_div, g.map_r)
        # also within session, then across sessions
        per = []
        for sess, gg in g.groupby("session"):
            if len(gg) >= 6 and gg.lag_div.std() > 0:
                per.append(stats.pearsonr(gg.lag_div, gg.map_r)[0])
        tt, p2 = stats.ttest_1samp(per, 0) if len(per) >= 5 else (np.nan, np.nan)
        print(f"  {roiname:6s} pairs = {len(g):4d} | pooled r(lag divergence, "
              f"map similarity) = {rr:+.3f}, P = {pp:.4g}"
              f"{' *' if pp < .05 else ''}")
        print(f"         within-session then across ({len(per)} sessions): "
              f"mean r = {np.mean(per) if per else np.nan:+.3f}, "
              f"t = {tt:+.2f}, P = {p2:.4g}{' *' if p2 < .05 else ''}")
        ov[f"{roiname}|lag"] = dict(roi=roiname, n_pairs=int(len(g)),
                                    r_pooled=float(rr), p_pooled=float(pp),
                                    r_within=float(np.mean(per)) if per else None,
                                    p_within=float(p2) if per else None)
    print("\n=== NEURAL vs TASK SIMILARITY (within session, then across) ===")
    print("    does a pair of configurations look alike neurally because the "
          "TASK is alike?\n")
    for roiname in ROI_GROUPS:
        g = PRS[PRS.roi == roiname]
        for mod in ("occupancy_sim", "reward_overlap", "trajectory_sim"):
            per = []
            for sess, gg in g.dropna(subset=["map_r", mod]).groupby("session"):
                if len(gg) >= 6 and gg[mod].std() > 0:
                    per.append(stats.pearsonr(gg[mod], gg.map_r)[0])
            if len(per) < 5:
                continue
            t, p = stats.ttest_1samp(per, 0)
            print(f"  {roiname:6s} {mod:16s} n = {len(per):3d} sessions | "
                  f"mean r = {np.mean(per):+.3f} | t = {t:+5.2f} | "
                  f"P = {p:.4g}{' *' if p < .05 else ''}")
            ov[f"{roiname}|model|{mod}"] = dict(
                roi=roiname, model=mod, n_sessions=int(len(per)),
                r=float(np.mean(per)), t=float(t), p=float(p))
        print()
    print("=== TRANSFERABILITY: RIPPLE vs NON-RIPPLE TIME ===\n")
    for roiname in ROI_GROUPS:
        g = PRS[PRS.roi == roiname]
        for c, nm in (("map_r_rip", "ripple time"),
                      ("map_r_norip", "non-ripple time")):
            v = g.groupby("session")[c].mean().dropna() if c in g else \
                pd.Series(dtype=float)
            if len(v) < 5:
                continue
            t, p = stats.ttest_1samp(v, 0)
            print(f"  {roiname:6s} {nm:16s} n = {len(v):3d} | r = "
                  f"{v.mean():+.4f} | t = {t:+5.2f} | P = {p:.4g}"
                  f"{' *' if p < .05 else ''}")
            ov[f"{roiname}|{c}"] = dict(roi=roiname, window=nm,
                                        r=float(v.mean()), t=float(t),
                                        p=float(p))
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, roi_groups=ROI_GROUPS, min_shared=MIN_SHARED,
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
