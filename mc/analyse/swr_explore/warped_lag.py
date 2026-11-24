#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST (c): the paper's 12-bin FUTURE-LAG code, measured with ripple spikes only.

    python scripts/swr_content_explore.py warped_lag

The per-lag encoding pipeline asks, in time-warped space, "does this cell fire
for location L when L is theta degrees into the future?". This asks the same
question of ripple spikes.

WHY THE TEMPLATE IS NOT BUILT FROM RIPPLE SPIKES. The literal version -- rebuild
the 9-point rate map per cell per layout from ripple spikes and cross-validate
across layouts -- was measured and is not viable: the median PFC cell has
**1.33 ripple spikes per rate-map cell**, so the held-out map is Poisson noise
(only 10 of 39 sessions reach even 3). Instead the template comes from all the
data, where it is well estimated, and ripple spikes supply the MEASUREMENT:

    template  = what this cell codes at lag theta        (all data, LOO)
    measured  = E(L) = sum_c n_c * T_c^theta(L) in the ripple window

That direction is also the only non-circular one: a template built from ripple
spikes and then tested on ripple spikes would be scoring itself.

The warp is needed only to say WHICH LOCATION is theta ahead -- `swr_warp`
supplies theta(t) from `state` + `t_A..t_D`, the same boundary set as
`save_humanABCDneurons_normalised.m`. No 360-bin matrices are needed.

⚠ RESOLUTION. One physical step spans ~18 deg (p90 = 47.5 deg), so lag bins are
not independent. Measured P(location at lag == location at lag 0): 30 deg
32.8%, 60 deg 12.3%, 90 deg 6.4%, 180 deg 7.7%, against 11% chance. So 60 deg
and beyond are clean, but **30 deg is a third redundant with the current
location** -- a positive there alone can be current-location bleed. The
a-priori mPFC window is 30-60 deg, so 60 deg is its interpretable end.

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
from mc.analyse.swr_content import phase_labels

SEED = 42
N_PERM = 1000
N_MATCH = 1000
ROI_SETS = {"mPFC": ["mPFC"], "mOFC": ["mOFC"],
            "HC_all": ["HC_anterior", "HC_mid"]}
LAGS_DEG = list(range(0, 360, 30))
PHASES = ("first_traversal", "encode_rest", "execute")
MIN_RIPPLES = 20
MIN_CELLS = 2
FLANK_GAP_S = 0.05
MAX_FLANKS = 2
A_PRIORI = {"mPFC": (30, 60), "HC_all": (0, 330)}     # per_lag_encoding.py:272


def _stat(E, li, P):
    E = np.nan_to_num(E)
    H = np.take_along_axis(E, P[:, li].T, axis=1)
    return H - (E.sum(axis=1)[:, None] - H) / 8.0


def _z(M):
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd, np.nan)


def main():
    rng = np.random.default_rng(SEED)
    P = np.vstack([np.arange(9)] + [rng.permutation(9) for _ in range(N_PERM)])
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        st_s = steps[steps.session == s]
        bnd = sww.trial_boundaries(st_s)
        if not len(bnd):
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        ph_i, first_i, _, _ = phase_labels(st_s, occ)

        # theta for every occupancy interval (its midpoint) and every ripple
        t_mid = (occ.start_s.to_numpy() + occ.stop_s.to_numpy()) / 2
        th_occ, wh_occ = sww.theta_of(t_mid, bnd)
        th_rip, wh_rip = sww.theta_of(t_rip, bnd)

        o = occ.sort_values("start_s")
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = o.index.to_numpy()[jj]
        inside = (j >= 0) & (t_rip <= b[jj]) & np.isfinite(th_rip)
        if inside.sum() < MIN_RIPPLES:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        th_rip, wh_rip = th_rip[inside], wh_rip[inside]
        half = d_rip / 2.0
        grid = occ.cv_group.to_numpy()[k]
        ph, fst = ph_i[k], first_i[k]
        grids = np.unique(occ.cv_group.to_numpy())
        masks = {"first_traversal": fst,
                 "encode_rest": (ph == "encode") & ~fst,
                 "execute": (ph == "execute")}

        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]

        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            C_rip, C_fl = swc.zscore_cells([
                swc.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                swc.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
            for lag in LAGS_DEG:
                # template condition: the square that is `lag` ahead of each
                # occupancy interval; target: the same for each ripple
                cond = sww.location_at_theta(th_occ, wh_occ, bnd, st_s, lag) - 1
                tgt = sww.location_at_theta(th_rip, wh_rip, bnd, st_s, lag) - 1
                if (cond >= 0).sum() < 50 or (tgt >= 0).sum() < MIN_RIPPLES:
                    continue
                loo = {gr: np.array([
                    swc.zscore_map(swc.cond_map(spk[s]["spikes"][c], occ, cond,
                                                9, exclude_grid=gr))
                    for c in cells]) for gr in grids}
                E_r = swc.score_windows(C_rip, t_rip, grid, loo)
                E_f = swc.score_windows(C_fl, t_fl, grid[owner], loo)
                ok_r = np.isfinite(E_r).all(axis=1) & (tgt >= 0)
                ok_f = np.isfinite(E_f).all(axis=1) & (tgt[owner] >= 0)
                S = _stat(E_r, np.clip(tgt, 0, 8), P)
                S_f = _stat(E_f, np.clip(tgt[owner], 0, 8), P)
                acc = np.zeros_like(S); cnt = np.zeros(len(S))
                np.add.at(acc, owner[ok_f], S_f[ok_f])
                np.add.at(cnt, owner[ok_f], 1.0)
                has = cnt > 0
                D = np.zeros_like(S)
                D[has] = S[has] - acc[has] / cnt[has, None]

                rec = dict(session=s, subject=subj, roi=roiname, lag=lag)
                for p_ in PHASES:
                    sel = np.flatnonzero(masks[p_] & ok_r)
                    rec[f"n_{p_}"] = len(sel)
                    rec[f"z_{p_}"] = (float(_z(S[sel].mean(axis=0)))
                                      if len(sel) >= MIN_RIPPLES else np.nan)
                    selc = np.flatnonzero(masks[p_] & ok_r & has)
                    rec[f"c1_{p_}"] = (float(_z(D[selc].mean(axis=0)))
                                       if len(selc) >= MIN_RIPPLES else np.nan)
                idx = {p_: np.flatnonzero(masks[p_] & ok_r) for p_ in PHASES}
                nmin = min(len(v) for v in idx.values())
                if nmin >= MIN_RIPPLES:
                    rs = np.random.default_rng([SEED, s, lag,
                                                abs(hash(roiname)) % 997])
                    zd = {}
                    for p_ in PHASES:
                        A = np.zeros((N_MATCH, len(S)))
                        for di in range(N_MATCH):
                            A[di, rs.choice(idx[p_], nmin,
                                            replace=False)] = 1.0 / nmin
                        zd[p_] = _z(A @ S)
                    rec["n_matched"] = int(nmin)
                    for p_ in PHASES:
                        rec[f"z_{p_}_matched"] = float(np.nanmean(zd[p_]))
                rows.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"warped_lag_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)

    print("\n=== FUTURE-LAG CODE IN RIPPLE SPIKES (warped theta) ===")
    print("    a-priori: mPFC 30-60 deg, HC 0/330 deg\n")
    out, ov = [], {}
    print(f"  {'ROI':10s} {'lag':>4s} {'test':24s} {'n':>3s} {'z':>7s} "
          f"{'t':>6s} {'P':>9s}")
    for roiname in ROI_SETS:
        for lag in LAGS_DEG:
            g = R[(R.roi == roiname) & (R.lag == lag)]
            for col, nm in ([(f"z_{p}_matched", f"{p} (matched)")
                             for p in PHASES]
                            + [(f"c1_{p}", f"ripple-flank, {p}")
                               for p in PHASES]):
                v = g[col].dropna() if col in g else pd.Series(dtype=float)
                if len(v) < 5:
                    continue
                t, p = stats.ttest_1samp(v, 0)
                ap = A_PRIORI.get(roiname, ())
                mark = " <-" if lag in ap else ""
                star = " *" if p < .05 else ""
                print(f"  {roiname:10s} {lag:4d} {nm:24s} {len(v):3d} "
                      f"{v.mean():+7.3f} {t:+6.2f} {p:9.4g}{star}{mark}")
                rec = dict(roi=roiname, lag=int(lag), test=nm,
                           n_sessions=int(len(v)), z=float(v.mean()),
                           t=float(t), df=int(len(v) - 1), p=float(p),
                           a_priori=bool(lag in ap))
                out.append(rec); ov[f"{roiname}|{lag}|{nm}"] = rec
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, n_match=N_MATCH,
                   roi_sets=ROI_SETS, lags_deg=LAGS_DEG, phases=list(PHASES),
                   a_priori=A_PRIORI, min_ripples=MIN_RIPPLES,
                   template="all data, leave-one-configuration-out",
                   measurement="ripple windows only",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
