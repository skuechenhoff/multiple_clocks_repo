#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How location information rises and falls around arriving at a new location.

    python scripts/swr_location_timecourse.py

The width sweep showed the location signal peaks at 125-250 ms windows. That
says how WIDE a window should be; it says nothing about WHEN the information is
there. This aligns to the moment the subject arrives at a new location and
tracks, in sliding 125 ms windows, the evidence for

    * the location just entered,
    * the location just left,
    * the mean of the seven irrelevant locations (the implicit floor).

A working place code should show evidence for the new location rising after
arrival while evidence for the old one decays -- and the crossing point is a
descriptive of how fast the representation turns over, which is exactly what
constrains how much a 60 ms ripple window can carry.

Per-session means, then across sessions; the shaded band is SEM across sessions.

@author: Svenja Kuchenhoff
"""

import os
import datetime

import numpy as np
import pandas as pd

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc

ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC"]
# roles a location can play at an arrival event. They are NOT mutually
# exclusive -- the task loops, so the location just left can also be the one
# coming next -- which is exactly why these go into one regression rather than
# being read off as separate means.
ROLE_COLS = ["b_current", "b_previous", "b_next", "intercept"]
WIN_S = 0.125
OFFSETS = np.arange(-1.0, 2.0 + 1e-9, 0.0625)
MIN_EVENTS = 100


def occupancy_by_lag(steps, sessions, deriv):
    """Where the subject actually IS at each lag, relative to the arrival square.

    ⚠ This is the control this figure needed from the start. The role labels
    `current` / `previous` / `next` are only true near lag 0: the median
    occupancy interval is 0.367 s, so by +0.5 s the subject is standing on the
    NEXT square 71% of the time and on the labelled `current` square only 54%.
    Beyond about +0.4 s, `b_next` outperforming `b_current` is not prospection,
    it is the subject having walked there. Saved beside the time course so the
    two are always read together.
    """
    rows = []
    for s in sessions:
        occ = swc.occupancy(steps, s).sort_values("start_s").reset_index(drop=True)
        if len(occ) < 50:
            continue
        a, b = occ.start_s.to_numpy(), occ.stop_s.to_numpy()
        L = occ["loc"].to_numpy()
        prev = np.concatenate([[np.nan], L[:-1]])
        nxt = np.concatenate([L[1:], [np.nan]])
        for t in OFFSETS:
            q = a + t
            j = np.searchsorted(a, q, side="right") - 1
            jj = np.clip(j, 0, len(a) - 1)
            ok = (j >= 0) & (q <= b[jj])
            at = np.where(ok, L[jj], np.nan)
            rows.append(dict(session=s, offset=t,
                             at_current=float(np.nanmean(at == L)),
                             at_previous=float(np.nanmean(at == prev)),
                             at_next=float(np.nanmean(at == nxt))))
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_content_timecourse_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "occupancy_by_lag.csv"),
                              index=False)
    print(f"  occupancy_by_lag.csv -> {d}")


def main():
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    rows = []
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, _ = swc.session_data(s, spk, roi, steps, rip)
        if len(occ) < MIN_EVENTS or not len(r):
            continue
        occ = occ.sort_values("start_s").reset_index(drop=True)
        # an "arrival" is an interval whose location differs from the previous
        prev = occ["loc"].shift(1)
        keep = (occ["loc"] != prev) & prev.notna()
        ev = occ[keep]
        if len(ev) < MIN_EVENTS:
            continue
        t0 = ev.start_s.to_numpy()
        new_loc = ev["loc"].to_numpy().astype(int)
        old_loc = prev[keep].to_numpy().astype(int)
        nxt = occ["loc"].shift(-1)[keep].to_numpy()
        nxt_loc = np.where(np.isfinite(nxt), np.nan_to_num(nxt, nan=0), 0).astype(int)
        grid = ev.cv_group.to_numpy()
        grids = np.unique(occ.cv_group.to_numpy())

        for roiname in ROIS:
            cells = r[r.roi == roiname].cell.to_numpy()
            cells = [c for c in cells if c < len(spk[s]["spikes"])]
            if not cells:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            # counts for every offset first, z-scored per cell with statistics
            # POOLED across offsets -- raw counts let loud cells carry the
            # pattern, which is why HC_anterior looked flat here while its
            # Stage 2 control was strong (CHANGELOG 2026-09-17 i)
            C_all = swc.zscore_cells(
                [swc.window_counts(spk[s], cells, t0 + off, t0 + off + WIN_S)
                 for off in OFFSETS])
            for off, C in zip(OFFSETS, C_all):
                E = swc.score_windows(C, t0, grid, loo)
                # scale-free within-window z: raw scores carry each session's
                # firing rate and cell count and must not be averaged across
                # sessions (CHANGELOG 2026-09-17 c, rule 2)
                sc_new = swc.target_z(E, new_loc)
                sc_old = swc.target_z(E, old_loc)
                ok = np.isfinite(sc_new) & np.isfinite(sc_old)
                if ok.sum() < MIN_EVENTS:
                    continue
                # --- regression over ALL NINE locations -------------------
                # DV: within-window z of each location. Predictors: whether
                # that location is the one just entered / just left / coming
                # next. Non-exclusive by design, so the coefficients are unique
                # contributions with the correlation between roles controlled.
                Ez = E[ok]
                mu = Ez.mean(axis=1, keepdims=True)
                sd = Ez.std(axis=1, keepdims=True)
                good = (sd[:, 0] > 0)
                Z = np.full_like(Ez, np.nan)
                Z[good] = (Ez[good] - mu[good]) / sd[good]
                nv = new_loc[ok]; ov = old_loc[ok]; xv = nxt_loc[ok]
                L = np.arange(1, 10)[None, :]
                X = np.stack([(L == nv[:, None]).ravel(),
                              (L == ov[:, None]).ravel(),
                              (L == xv[:, None]).ravel(),
                              np.ones(Z.size, bool)], axis=1).astype(float)
                y = Z.ravel()
                m = np.isfinite(y)
                beta = np.full(4, np.nan)
                if m.sum() > 50:
                    beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
                rows.append(dict(session=s, roi=roiname, offset=float(off),
                                 n_events=int(ok.sum()),
                                 new_loc=float(np.nanmean(sc_new[ok])),
                                 old_loc=float(np.nanmean(sc_old[ok])),
                                 **dict(zip(ROLE_COLS, beta))))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_content_timecourse_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "location_timecourse.csv"), index=False)
    print(f"\n-> {d}")
    for roiname in ROIS:
        g = R[R.roi == roiname]
        if not len(g):
            continue
        m = g.groupby("offset")[["new_loc", "old_loc"]].mean()
        pk = m.new_loc.idxmax()
        print(f"  {roiname:12s} peak(new) at {pk:+.3f} s = {m.new_loc.max():+.4f}, "
              f"sessions={g.session.nunique()}")


if __name__ == "__main__":
    main()
