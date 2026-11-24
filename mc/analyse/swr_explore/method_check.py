#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METHOD CHECK: does this estimator recover the KNOWN hippocampal result?

    python scripts/swr_content_explore.py method_check

The manuscript shows, by RSA and by per-lag encoding, that hippocampus encodes
the CURRENT location (lag 0) during execution. Every ripple analysis here says
current-location evidence in `execute` is weak (+0.21, P = 0.13). Those are not
necessarily in conflict -- ripples are ~3% of task time -- but until it is
checked, the alternative is that the estimator is simply wrong.

So: the SAME templates and the SAME statistic, applied to

    RIPPLE windows        -- each ripple's own duration
    NON-RIPPLE windows    -- tiled, >= 1 s from any ripple, same median width

within each phase. If the non-ripple windows recover a strong current-location
signal during execution, the method works and the ripple result is a real
statement about ripples. If they do NOT, the method is broken and every null in
this session is uninterpretable.

⚠ Counts are MATCHED between the two window types (z carries a sqrt(n) term and
tiled windows are far more numerous), and the window WIDTH is matched to the
median ripple duration, so neither can explain a difference.

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
N_MATCH = 200
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_anterior": ["HC_anterior"], "HC_mid": ["HC_mid"],
            "mPFC": ["mPFC"], "mOFC": ["mOFC"]}
PHASES = ("encode", "execute")
MIN_WIN = 30
MIN_CELLS = 2


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
        if not len(occ) or not len(r) or len(t_rip) < MIN_WIN:
            continue
        st = steps[steps.session == s]
        subj = rip[rip.session == s].subject_key.iloc[0]
        ph_i, _, _, _ = phase_labels(st, occ)
        width = float(np.median(d_rip))
        o = occ.sort_values("start_s")
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        idx_o = o.index.to_numpy()

        def _place(t0):
            """(occupancy row, location, cv_group, phase) for each window start."""
            j = np.searchsorted(a, t0, side="right") - 1
            jj = np.clip(j, 0, len(a) - 1)
            ok = (j >= 0) & (t0 <= b[jj])
            k = idx_o[jj]
            return k, ok

        # ---- ripple windows
        k_r, ok_r0 = _place(t_rip)
        half = d_rip / 2.0
        # ---- non-ripple windows, tiled, >=1 s from any ripple
        t_nr, loc_nr = swc.tiled_windows(occ, width, avoid=t_rip,
                                         avoid_pad_s=1.0)
        if len(t_nr) < MIN_WIN:
            continue
        k_n, ok_n0 = _place(t_nr)
        grids = np.unique(occ.cv_group.to_numpy())
        LOC = occ["loc"].to_numpy().astype(int)
        CVG = occ.cv_group.to_numpy()

        for roiname, members in ROI_SETS.items():
            cells = [c for c in r[r.roi.isin(members)].cell.to_numpy()
                     if c < len(spk[s]["spikes"])]
            if len(cells) < MIN_CELLS:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            C_r, C_n = swc.zscore_cells([
                swc.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                swc.window_counts(spk[s], cells, t_nr - width / 2,
                                  t_nr + width / 2)])
            E_r = swc.score_windows(C_r, t_rip, CVG[k_r], loo)
            E_n = swc.score_windows(C_n, t_nr, CVG[k_n], loo)
            S_r = _stat(E_r, np.clip(LOC[k_r] - 1, 0, 8), P)
            S_n = _stat(E_n, np.clip(LOC[k_n] - 1, 0, 8), P)
            fin_r = ok_r0 & np.isfinite(E_r).all(axis=1)
            fin_n = ok_n0 & np.isfinite(E_n).all(axis=1)
            for ph in PHASES:
                m_r = fin_r & (ph_i[k_r] == ph)
                m_n = fin_n & (ph_i[k_n] == ph)
                if m_r.sum() < MIN_WIN or m_n.sum() < MIN_WIN:
                    continue
                rec = dict(session=s, subject=subj, roi=roiname, phase=ph,
                           n_ripple=int(m_r.sum()), n_nonripple=int(m_n.sum()),
                           width_s=width)
                rec["z_ripple_raw"] = float(_z(S_r[m_r].mean(axis=0)))
                rec["z_nonripple_raw"] = float(_z(S_n[m_n].mean(axis=0)))
                # count-matched: both window types down to the same n
                nmin = min(m_r.sum(), m_n.sum())
                rs = np.random.default_rng([SEED, s, abs(hash(roiname)) % 997])
                zr, zn = [], []
                for _ in range(N_MATCH):
                    zr.append(_z(S_r[rs.choice(np.flatnonzero(m_r), nmin,
                                               replace=False)].mean(axis=0)))
                    zn.append(_z(S_n[rs.choice(np.flatnonzero(m_n), nmin,
                                               replace=False)].mean(axis=0)))
                rec["n_matched"] = int(nmin)
                rec["z_ripple"] = float(np.nanmean(zr))
                rec["z_nonripple"] = float(np.nanmean(zn))
                rec["d_ripple_minus_non"] = float(np.nanmean(zr) -
                                                  np.nanmean(zn))
                rows.append(rec)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"method_check_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_session.csv"), index=False)

    print("\n=== DOES THE ESTIMATOR RECOVER CURRENT LOCATION OUTSIDE RIPPLES? ===")
    print("    same templates, same statistic, counts and width matched\n")
    out, ov = [], {}
    print(f"  {'ROI':12s} {'phase':8s} {'window':22s} {'n':>3s} {'z':>7s} "
          f"{'t':>6s} {'P':>10s}")
    for roiname in ROI_SETS:
        for ph in PHASES:
            g = R[(R.roi == roiname) & (R.phase == ph)]
            for c, nm in (("z_nonripple", "NON-ripple (matched)"),
                          ("z_ripple", "ripple (matched)"),
                          ("d_ripple_minus_non", "ripple minus non-ripple"),
                          ("z_nonripple_raw", "NON-ripple (all windows)")):
                v = g[c].dropna() if c in g else pd.Series(dtype=float)
                if len(v) < 5:
                    continue
                t, p = stats.ttest_1samp(v, 0)
                star = " *" if p < .05 else ""
                print(f"  {roiname:12s} {ph:8s} {nm:22s} {len(v):3d} "
                      f"{v.mean():+7.3f} {t:+6.2f} {p:10.4g}{star}")
                rec = dict(roi=roiname, phase=ph, window=nm,
                           n_sessions=int(len(v)), z=float(v.mean()),
                           t=float(t), p=float(p))
                out.append(rec); ov[f"{roiname}|{ph}|{c}"] = rec
            print()
    pd.DataFrame(out).to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(ov, open(os.path.join(d, "results_overview.json"), "w"), indent=2)
    json.dump(dict(seed=SEED, n_perm=N_PERM, n_match=N_MATCH,
                   roi_sets=ROI_SETS, phases=list(PHASES),
                   window="median ripple duration for both types",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
