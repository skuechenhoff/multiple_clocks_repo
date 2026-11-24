#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the mPFC 'holds a new reward' effect ride on a minority of good cells?

    python scripts/swr_content_explore.py template_quality

Two results sit in tension:
  * `consolidation` (fig29): after a FIRST discovery, mPFC represents the square
    just uncovered (+0.770, P = 5.6e-4), and only when the reward is new;
  * `cross_task` (fig36): the mPFC place template does not transfer across
    configurations at all (leave-one-out r = -0.002, P = 0.93).

If the template carries no transferable information, it is not obvious how the
population read-out reaches +0.77. The two are different measures, and the
obvious reconciliation is heterogeneity: the MEAN per-cell correlation can be
zero while a minority of cells transfer well and carry the population match.
That is testable rather than assumable.

    1. the DISTRIBUTION of per-cell transferability, not its mean;
    2. the consolidation statistic recomputed separately from the cells ABOVE
       and BELOW the median, within each session, so both groups face the same
       events and the same behaviour.

If the effect lives in the high group, the population read-out is being carried
by a transferring minority and both results stand. If it is equal in both, the
read-out does not depend on template quality and the tension is real.

⚠ Sessions carry few PFC cells (mPFC median ~5), so a median split leaves ~2-3
per group. Sessions with fewer than MIN_FOR_SPLIT cells are excluded from the
split and the n is reported.

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
from mc.analyse.swr_content import phase_labels

SEED = 42
N_PERM = 1000
ROI_GROUPS = {"HC": ["HC_anterior", "HC_mid"], "mPFC": ["mPFC"],
              "mOFC": ["mOFC"]}
POST = (0.05, 1.00)
MIN_SHARED = 5
MIN_FOR_SPLIT = 4          # cells needed in an ROI before a median split
MIN_EVENTS = 15


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
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    cells_out, split_out = [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, _, _ = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r):
            continue
        st = steps[steps.session == s].sort_values("t_s").reset_index(drop=True)
        cvg = occ.cv_group.to_numpy()
        grids = np.unique(cvg)
        if len(grids) < 3:
            continue
        # ---- first discoveries, as in `consolidation`
        t_all = st.t_s.to_numpy(float)
        unc = np.flatnonzero(st.is_uncover.to_numpy(float) == 1)
        if not len(unc):
            continue
        t_u = t_all[unc]; loc_u = st["loc"].to_numpy(float)[unc].astype(int)
        cu = st.correct_uncover.to_numpy(float)[unc]
        gn = st.grid_num.to_numpy(float)[unc]
        gid = st.grid_id.to_numpy(float)[unc]
        seen, isf = set(), np.zeros(len(t_u), bool)
        for i in np.flatnonzero(cu == 1):
            k = (gn[i], loc_u[i])
            if k not in seen:
                seen.add(k); isf[i] = True
        if isf.sum() < MIN_EVENTS:
            continue
        t_ev, loc_ev, grid_ev = t_u[isf], loc_u[isf], gid[isf]

        for roiname, members in ROI_GROUPS.items():
            cl = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                  if c < len(spk[s]["spikes"])]
            if len(cl) < 2:
                continue
            # ---- per-cell leave-one-configuration-out transferability
            maps = {g: {c: swc.place_map(spk[s]["spikes"][c], occ[cvg == g])
                        for c in cl} for g in grids}
            q = {}
            for c in cl:
                rs = []
                for g in grids:
                    a_ = maps[g][c]
                    b_ = np.nanmean(np.stack([maps[x][c] for x in grids
                                              if x != g]), axis=0)
                    ok = np.isfinite(a_) & np.isfinite(b_)
                    if ok.sum() < MIN_SHARED or np.std(a_[ok]) == 0 \
                       or np.std(b_[ok]) == 0:
                        continue
                    rs.append(np.corrcoef(a_[ok], b_[ok])[0, 1])
                q[c] = float(np.mean(rs)) if rs else np.nan
                cells_out.append(dict(session=s, cell=c, roi=roiname,
                                      quality=q[c]))
            good = [c for c in cl if np.isfinite(q[c])]
            if len(good) < MIN_FOR_SPLIT:
                continue
            med = np.median([q[c] for c in good])
            groups = {"high": [c for c in good if q[c] > med],
                      "low": [c for c in good if q[c] <= med]}
            for nm, sub in groups.items():
                if len(sub) < 2:
                    continue
                _, loo = swc.build_templates(spk[s], sub, occ, grids)
                C, = swc.zscore_cells([swc.window_counts(
                    spk[s], sub, t_ev + POST[0], t_ev + POST[1])])
                E = swc.score_windows(C, t_ev, grid_ev, loo)
                ok = np.isfinite(E).all(axis=1)
                if ok.sum() < MIN_EVENTS:
                    continue
                S = _stat(E[ok], np.clip(loc_ev[ok] - 1, 0, 8), P)
                split_out.append(dict(session=s, roi=roiname, group=nm,
                                      n_cells=len(sub),
                                      mean_quality=float(np.mean(
                                          [q[c] for c in sub])),
                                      z=float(_z(S.mean(axis=0)))))
        print(f"  s{s:02d}", flush=True)

    CE = pd.DataFrame(cells_out); SP = pd.DataFrame(split_out)
    d = os.path.join(deriv, "group", "swr",
                     f"template_quality_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    CE.to_csv(os.path.join(d, "per_cell_quality.csv"), index=False)
    SP.to_csv(os.path.join(d, "split.csv"), index=False)

    print("\n=== DISTRIBUTION of per-cell template transferability ===\n")
    ov = {}
    print(f"  {'ROI':6s} {'cells':>6s} {'mean':>8s} {'SD':>7s} {'% > 0':>7s} "
          f"{'% > 0.2':>8s} {'90th pct':>9s}")
    for rn in ROI_GROUPS:
        v = CE[CE.roi == rn].quality.dropna()
        if len(v) < 10:
            continue
        print(f"  {rn:6s} {len(v):6d} {v.mean():+8.4f} {v.std():7.3f} "
              f"{(v > 0).mean() * 100:6.1f}% {(v > .2).mean() * 100:7.1f}% "
              f"{v.quantile(.9):+9.3f}")
        ov[f"{rn}|dist"] = dict(roi=rn, n_cells=int(len(v)),
                                mean=float(v.mean()), sd=float(v.std()),
                                frac_pos=float((v > 0).mean()),
                                p90=float(v.quantile(.9)))

    print("\n=== fig29 EFFECT, cells SPLIT by template quality ===")
    print("    same events, same behaviour; only which cells form the "
          "population differs\n")
    print(f"  {'ROI':6s} {'group':6s} {'n':>3s} {'quality':>9s} {'z':>8s} "
          f"{'t':>6s} {'P':>9s}")
    for rn in ROI_GROUPS:
        g = SP[SP.roi == rn]
        vals = {}
        for nm in ("high", "low"):
            v = g[g.group == nm].z.dropna()
            if len(v) < 5:
                continue
            t, p = stats.ttest_1samp(v, 0)
            vals[nm] = v
            print(f"  {rn:6s} {nm:6s} {len(v):3d} "
                  f"{g[g.group == nm].mean_quality.mean():+9.4f} "
                  f"{v.mean():+8.3f} {t:+6.2f} {p:9.4g}"
                  f"{' *' if p < .05 else ''}")
            ov[f"{rn}|{nm}"] = dict(roi=rn, group=nm, n_sessions=int(len(v)),
                                    z=float(v.mean()), t=float(t), p=float(p))
        if len(vals) == 2:
            w = g.pivot_table(index="session", columns="group",
                              values="z").dropna()
            if len(w) >= 5:
                dd = w["high"] - w["low"]
                t, p = stats.ttest_1samp(dd, 0)
                print(f"  {rn:6s} {'HIGH minus LOW':12s} {len(dd):3d} "
                      f"{'':9s} {dd.mean():+8.3f} {t:+6.2f} {p:9.4g}"
                      f"{' *' if p < .05 else ''}")
                ov[f"{rn}|diff"] = dict(roi=rn, n_sessions=int(len(dd)),
                                        z=float(dd.mean()), t=float(t),
                                        p=float(p))
        print()
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, roi_groups=ROI_GROUPS,
                   post=POST, min_for_split=MIN_FOR_SPLIT,
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
