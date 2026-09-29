#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location content in ripples, encode vs execute. NO offsets, NO time course.

    python scripts/swr_content_main.py --step=phase

THIS IS THE REPORTED ANALYSIS. It is deliberately self-contained: spikes
are counted in the ripple window (and in its matched flanks, for the
specificity control) and nowhere else, so the Methods needs no mention of
sliding windows. `timecourse_by_phase` remains for the descriptive time course;
its fixed-window numbers differ slightly because there the per-unit count
normalisation pools across all 41 offsets.

Three tests per ROI, session as the unit of inference:
  * C0 per phase       -- is the occupied location decodable in ripples?
  * C0 contrast        -- encode minus execute, ripple counts matched
  * C1 per phase       -- ripple minus its matched flanks (ripple-specificity)

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
N_PERM = 1000           # template-label permutations per session
N_MATCH = 1000          # count-matched subsample draws
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_anterior": ["HC_anterior"], "HC_mid": ["HC_mid"]}
PHASES = ("encode", "execute")
MIN_RIPPLES = 20
MIN_CELLS = 2
FLANK_GAP_S = 0.05
MAX_FLANKS = 2


def _stat(E, li):
    """(n_windows, 1 + N_PERM): observed statistic then its permutation null."""
    E = np.nan_to_num(E)
    H = np.take_along_axis(E, _stat.P[:, li].T, axis=1)
    return H - (E.sum(axis=1)[:, None] - H) / 8.0


def _z(M):
    """z = (observed - mean(null)) / SD(null), column 0 vs the rest."""
    sd = M[..., 1:].std(axis=-1)
    return np.where(sd > 0, (M[..., 0] - M[..., 1:].mean(axis=-1)) / sd, np.nan)


def main():
    rng = np.random.default_rng(SEED)
    _stat.P = np.vstack([np.arange(9)]
                        + [rng.permutation(9) for _ in range(N_PERM)])
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s",
                               "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows, counts = [], []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        ph_i, _, _, _ = phase_labels(steps[steps.session == s], occ)
        o = occ.sort_values("start_s")
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(a) - 1)
        k = o.index.to_numpy()[jj]
        inside = (j >= 0) & (t_rip <= b[jj])      # ripple is inside a visit
        if inside.sum() < MIN_RIPPLES:
            continue
        t_rip, d_rip, k = t_rip[inside], d_rip[inside], k[inside]
        half = d_rip / 2.0
        loc = occ["loc"].to_numpy()[k].astype(int)
        grid = occ.cv_group.to_numpy()[k]
        ph = ph_i[k]
        grids = np.unique(occ.cv_group.to_numpy())
        masks = {p_: (ph == p_) for p_ in PHASES}
        for p_ in PHASES:
            counts.append(dict(session=s, phase=p_, n=int(masks[p_].sum())))
        if min(m.sum() for m in masks.values()) < MIN_RIPPLES:
            continue

        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        w_fl = half[owner] if len(t_fl) else np.zeros(0)

        for ri, (roiname, members) in enumerate(ROI_SETS.items()):
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            # counts in the ripple windows and in their flanks, z-scored per
            # unit across exactly those windows -- nothing else is analysed
            C_rip, C_fl = swc.zscore_cells([
                swc.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                swc.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
            E_rip = swc.score_windows(C_rip, t_rip, grid, loo)
            ok = np.isfinite(E_rip).all(axis=1)
            S = _stat(E_rip, loc - 1)

            # flank statistic, averaged within each ripple that has flanks
            E_fl = swc.score_windows(C_fl, t_fl, grid[owner], loo)
            ok_f = np.isfinite(E_fl).all(axis=1)
            S_fl = _stat(E_fl, loc[owner] - 1)
            acc = np.zeros_like(S); cnt = np.zeros(len(S))
            np.add.at(acc, owner[ok_f], S_fl[ok_f])
            np.add.at(cnt, owner[ok_f], 1.0)
            has = cnt > 0
            D = np.zeros_like(S)
            D[has] = S[has] - acc[has] / cnt[has, None]

            rec = dict(session=s, subject=subj, roi=roiname)
            for p_ in PHASES:
                sel = np.flatnonzero(masks[p_] & ok)
                rec[f"n_{p_}"] = len(sel)
                rec[f"z_{p_}"] = (float(_z(S[sel].mean(axis=0)))
                                  if len(sel) >= MIN_RIPPLES else np.nan)
                selc = np.flatnonzero(masks[p_] & ok & has)
                rec[f"c1_{p_}"] = (float(_z(D[selc].mean(axis=0)))
                                   if len(selc) >= MIN_RIPPLES else np.nan)

            # count-matched contrast: one selection matrix, 1000 draws, one
            # matmul per phase
            idx = {p_: np.flatnonzero(masks[p_] & ok) for p_ in PHASES}
            nmin = min(len(v) for v in idx.values())
            if nmin >= MIN_RIPPLES:
                rs = np.random.default_rng([SEED, s, ri, 7])
                zd = {}
                for p_ in PHASES:
                    A = np.zeros((N_MATCH, len(S)))
                    for di in range(N_MATCH):
                        A[di, rs.choice(idx[p_], nmin, replace=False)] = 1.0 / nmin
                    zd[p_] = _z(A @ S)
                dz = zd["encode"] - zd["execute"]
                rec.update(n_matched=int(nmin), d_z=float(np.nanmean(dz)),
                           d_z_sd_draws=float(np.nanstd(dz)))
            rows.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"phase_contrast_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(d, "ripple_counts.csv"),
                                index=False)

    print("\n=== LOCATION CONTENT IN RIPPLES, ENCODE vs EXECUTE ===")
    print("    ripple window only; session = unit of inference\n")
    out, ov = [], {}
    hdr = (f"  {'ROI':12s} {'test':22s} {'n':>3s} {'z':>7s} {'t':>7s} "
           f"{'P':>10s} {'95% CI':>18s}")
    print(hdr)
    for roiname in ROI_SETS:
        g = R[R.roi == roiname]
        for col, nm in (("z_encode", "encode"), ("z_execute", "execute"),
                        ("d_z", "encode - execute"),
                        ("c1_encode", "ripple - flank, enc"),
                        ("c1_execute", "ripple - flank, exe")):
            v = g[col].dropna() if col in g else pd.Series(dtype=float)
            if len(v) < 5:
                continue
            t, p = stats.ttest_1samp(v, 0)
            ci = stats.t.interval(.95, len(v) - 1, v.mean(), stats.sem(v))
            print(f"  {roiname:12s} {nm:22s} {len(v):3d} {v.mean():+7.3f} "
                  f"{t:+7.2f} {p:10.4g} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
            rec = dict(roi=roiname, test=nm, n_sessions=int(len(v)),
                       z=float(v.mean()), t=float(t), df=int(len(v) - 1),
                       p=float(p), ci95=[float(ci[0]), float(ci[1])],
                       sem=float(stats.sem(v)))
            out.append(rec); ov[f"{roiname}|{nm}"] = rec
        print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    C = pd.DataFrame(counts)
    keep = R[R.roi == "HC_all"].session.unique()
    med = {p_: float(C[(C.phase == p_) & (C.session.isin(keep))].n.median())
           for p_ in PHASES}
    print(f"  median ripples per analysed session: "
          f"encode {med['encode']:.0f}, execute {med['execute']:.0f}")
    json.dump(dict(seed=SEED, n_perm=N_PERM, n_match=N_MATCH,
                   roi_sets=ROI_SETS, phases=list(PHASES),
                   min_ripples=MIN_RIPPLES, min_cells=MIN_CELLS,
                   flank_gap_s=FLANK_GAP_S, max_flanks=MAX_FLANKS,
                   median_ripples_analysed=med,
                   window="each ripple's own detection duration; no offsets",
                   normalisation="counts z-scored per unit across the ripple "
                                 "windows and their flanks only",
                   unit="session",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
