#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What does the WHOLE location profile look like inside a ripple?

    python scripts/swr_ripple_profile.py

Everything so far scored `E[current] - mean(E[other 8])`. That contrast tests
the current location AGAINST the others, so **if a ripple replays a REMOTE
location the remote evidence enters the subtrahend and LOWERS the score.**
Remote replay would therefore look like a weaker within-ripple effect -- which
is exactly the pattern seen (HC_mid: flank +0.406 > ripple +0.304). The
estimator cannot distinguish "no location content" from "content that is not
about where the subject is standing".

So this drops the contrast and looks at the full nine-element profile, indexed
by STEP DISTANCE from the current location (0 = here, up to 4 on the 3x3 grid,
4-connected). Each window's profile is z-scored across its nine locations, so
the profile shape is scale-free and sums to zero -- a rise at distance>0 is
necessarily a fall at distance 0 and vice versa.

If ripples replay remote locations, the ripple profile should be FLATTER, or
elevated at larger distances, relative to its matched flank.

`HC_all` pools HC_anterior and HC_mid cells within each session.

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
import scripts.swr_place_templates as spt

SEED = 42
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_anterior": ["HC_anterior"], "HC_mid": ["HC_mid"],
            "mPFC": ["mPFC"], "mOFC": ["mOFC"]}
FLANK_GAP_S = 0.05
MAX_FLANKS = 2
MIN_RIPPLES = 30


def step_distance():
    """(9,9) 4-connected step distance; loc = col*3 + row + 1 (column-major)."""
    D = np.zeros((9, 9), int)
    for i in range(9):
        for j in range(9):
            ci, ri = i // 3, i % 3
            cj, rj = j // 3, j % 3
            D[i, j] = abs(ci - cj) + abs(ri - rj)
    return D


def zprofile(E):
    """Row-wise z across the nine locations; NaN where degenerate."""
    mu = E.mean(axis=1, keepdims=True)
    sd = E.std(axis=1, keepdims=True)
    out = np.full(E.shape, np.nan)
    ok = sd[:, 0] > 0
    out[ok] = (E[ok] - mu[ok]) / sd[ok]
    return out


def main():
    D9 = step_distance()
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s", "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        occ, r, t_rip, d_rip = spt.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]
        o = occ.sort_values("start_s").reset_index(drop=True)
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(o) - 1)
        ins = (j >= 0) & (t_rip <= b[jj])
        if ins.sum() < MIN_RIPPLES:
            continue
        t_rip, d_rip = t_rip[ins], d_rip[ins]
        loc = o["loc"].to_numpy()[jj][ins].astype(int)
        grid = o.cv_group.to_numpy()[jj][ins]
        half = d_rip / 2.0
        grids = np.unique(occ.cv_group.to_numpy())

        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        w_fl = half[owner]

        for name, members in ROI_SETS.items():
            cells = r[r.roi.isin(members)].cell.to_numpy()
            cells = [c for c in cells if c < len(spk[s]["spikes"])]
            if len(cells) < 2:
                continue
            _, loo = spt.build_templates(spk[s], cells, occ, grids)
            C_rip, C_fl = swc.zscore_cells([
                spt.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                spt.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
            Zr = zprofile(spt.score_windows(C_rip, t_rip, grid, loo))
            Zf = zprofile(spt.score_windows(C_fl, t_fl, grid[owner], loo))

            dr = D9[loc - 1]                 # (n_rip, 9) distance from current
            df = D9[loc[owner] - 1]
            for dist in range(5):
                mr = (dr == dist) & np.isfinite(Zr)
                mf = (df == dist) & np.isfinite(Zf)
                if mr.sum() < 20 or mf.sum() < 20:
                    continue
                rows.append(dict(session=s, subject=subj, roi=name, dist=dist,
                                 n_cells=len(cells),
                                 ripple=float(Zr[mr].mean()),
                                 flank=float(Zf[mf].mean()),
                                 diff=float(Zr[mr].mean() - Zf[mf].mean())))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_profile_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "profile_by_distance.csv"), index=False)

    print("\n=== Location-evidence profile by step distance from where the "
          "subject IS ===")
    print("    (within-window z, so the nine values sum to zero; "
          "a rise far away IS a fall nearby)\n")
    for name in ROI_SETS:
        g = R[R.roi == name]
        if not len(g):
            continue
        print(f"  {name}  (n = {g.session.nunique()} sessions)")
        print(f"    {'dist':>4s} {'ripple':>8s} {'flank':>8s} "
              f"{'ripple-flank':>13s} {'p':>9s}")
        for dist in range(5):
            v = g[g.dist == dist]
            if len(v) < 5:
                continue
            p = stats.ttest_1samp(v["diff"], 0)[1]
            print(f"    {dist:4d} {v.ripple.mean():+8.3f} {v.flank.mean():+8.3f} "
                  f"{v['diff'].mean():+13.3f} {p:9.3g}")
        print()
    json.dump(dict(seed=SEED, roi_sets=ROI_SETS, min_ripples=MIN_RIPPLES,
                   note="profile z-scored within window across the 9 locations; "
                        "flanks matched on location, occupancy and width",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
