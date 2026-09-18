#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the roles design collinear, and does `on_route` survive dropping its rivals?

    python scripts/swr_roles_collinearity.py

SK: *"a lot of the regressors are pretty overlapping, and worst case even fully
contain another regressor... How correlated are all regressors with each other?
What happens to their estimates if you remove some co-regressors?"*

The binary roles are mutually exclusive BY CONSTRUCTION -- `design()` strips
`adjacent` of the goals and the next step, strips `on_route` of rewards,
adjacency, goals and recency, and so on. So no indicator literally contains
another. But mutual exclusivity is not independence: with nine squares, an
intercept and eight square dummies, the roles between them cover almost every
square, and the REFERENCE category for `on_route` is whatever is left --
measured at about 1.5 squares per run. A thin reference makes the design
ill-conditioned even when no two columns are nested, and that would show up as a
large variance inflation factor rather than a large correlation.

Three things are reported:

  1. the correlation matrix over all (window x square) rows, pooled
  2. the variance inflation factor per regressor, which is the quantity that
     actually matters for whether a coefficient is trustworthy
  3. a LEAVE-ONE-REGRESSOR-OUT refit: the full model minus one term at a time,
     so the effect of every rival on `on_route` is visible directly

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
from mc.analyse.swr_explore.ripple_content_roles import (run_history, design, fit_all,
                                              TERMS, FLANK_GAP_S, MAX_FLANKS,
                                              MIN_WINDOWS, N_PERM, SEED)

ROI = "HC_mid"
MEMBERS = ["HC_mid"]
PHASE = "known"


def vif(A):
    """Variance inflation factor per column, via the inverse correlation matrix."""
    keep = A.std(axis=0) > 0
    C = np.corrcoef(A[:, keep], rowvar=False)
    try:
        d = np.diag(np.linalg.pinv(C))
    except np.linalg.LinAlgError:
        d = np.full(keep.sum(), np.nan)
    out = np.full(A.shape[1], np.nan)
    out[keep] = d
    return out


def main():
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(9) for _ in range(N_PERM)]
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    terms = TERMS[PHASE]

    Xall, rows = [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        (rew, known, err, vis, last, first, route, goal, nxt,
         g1, g2, corr, occ_sq, trial_n) = run_history(steps[steps.session == s],
                                                      occ)
        o = occ.sort_values("start_s")
        row_of = o.index.to_numpy()
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = row_of[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        m = ~first[k] if PHASE == "known" else first[k]
        if m.sum() < MIN_WINDOWS:
            continue
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        half = d_rip / 2.0
        X = design(PHASE, loc[m], rew[k][m], known[k][m], err[k][m], vis[k][m],
                   last[k][m], route[k][m], goal[k][m], nxt[k][m],
                   g1[k][m], g2[k][m], occ_sq[k][m])
        Xall.append(X.reshape(-1, X.shape[2]))

        # --- leave-one-regressor-out, same estimator as the main analysis
        cells = [c for c in r[r.roi.isin(MEMBERS)].cell.to_numpy()
                 if c < len(spk[s]["spikes"])]
        if len(cells) < 2:
            continue
        grids = np.unique(occ.cv_group.to_numpy())
        loo_t = swc.weighted_loo(spk[s], cells, occ, grids, "all")
        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]
        C_rip, C_fl = swc.zscore_cells([
            swc.window_counts(spk[s], cells, t_rip - half, t_rip + half),
            swc.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
        E_rip = swc.score_windows(C_rip, t_rip, grid, loo_t)
        E_fl = swc.score_windows(C_fl, t_fl, grid[owner], loo_t)
        mf = m[owner]
        if mf.sum() < MIN_WINDOWS:
            continue
        kf = k[owner][mf]
        Xf = design(PHASE, loc[owner][mf], rew[kf], known[kf], err[kf], vis[kf],
                    last[kf], route[kf], goal[kf], nxt[kf], g1[kf], g2[kf],
                    occ_sq[kf])
        i_route = terms.index("on_route")

        for drop in ["none"] + terms:
            if drop == "on_route":
                continue
            if drop == "none":
                keep = np.ones(X.shape[2], bool)
                idx = i_route
            else:
                di = terms.index(drop)
                keep = np.ones(X.shape[2], bool); keep[di] = False
                idx = i_route - (1 if di < i_route else 0)
            Br = fit_all(X[:, :, keep], E_rip[m], perms)
            Bf = fit_all(Xf[:, :, keep], E_fl[mf], perms)
            if Br is None or Bf is None:
                continue
            dv = Br[idx, 0] - Bf[idx, 0]
            dn = Br[idx, 1:] - Bf[idx, 1:]
            sd = np.nanstd(dn)
            rows.append(dict(session=s, dropped=drop,
                             z=(dv - np.nanmean(dn)) / sd if sd > 0 else np.nan))
        print(f"  s{s:02d}", flush=True)

    A = np.vstack(Xall)
    names = terms + [f"square_{i}" for i in range(2, 10)] + ["intercept"]
    v = vif(A)
    C = np.corrcoef(A[:, :len(terms)], rowvar=False)

    d = os.path.join(deriv, "group", "swr",
                     f"roles_collinearity_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(C, index=terms, columns=terms).to_csv(
        os.path.join(d, "correlation_matrix.csv"))
    pd.DataFrame(dict(term=names, vif=v)).to_csv(
        os.path.join(d, "vif.csv"), index=False)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(d, "leave_one_regressor_out.csv"), index=False)

    print(f"\n=== how often each role is ON (fraction of window x square rows) ===")
    for i, t in enumerate(terms):
        print(f"  {t:14s} {A[:, i].mean():.4f}"
              + ("   <- the reference is what is left over" if t == "on_route" else ""))
    print(f"\n=== variance inflation factor (>5 is a problem, >10 severe) ===")
    for n_, vv in zip(names, v):
        if np.isfinite(vv) and (vv > 2 or n_ in terms):
            print(f"  {n_:14s} {vv:7.2f}")
    print(f"\n=== largest absolute correlations between role regressors ===")
    pairs = [(abs(C[i, j]), terms[i], terms[j])
             for i in range(len(terms)) for j in range(i + 1, len(terms))]
    for a_, t1, t2 in sorted(pairs, reverse=True)[:8]:
        print(f"  {t1:14s} vs {t2:14s} r = {C[terms.index(t1), terms.index(t2)]:+.3f}")

    print(f"\n=== `on_route` ripple-minus-flank with each rival REMOVED ===")
    print(f"    {ROI}, {PHASE} phase\n")
    for drop in ["none"] + terms:
        g = R[R.dropped == drop].z.dropna()
        if len(g) < 5:
            continue
        p = stats.ttest_1samp(g, 0)[1]
        lab = "FULL MODEL" if drop == "none" else f"minus {drop}"
        print(f"  {lab:24s} n={len(g):3d}  {g.mean():+.3f}  p = {p:.4g}")
    json.dump(dict(roi=ROI, phase=PHASE, terms=terms, n_perm=N_PERM,
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
