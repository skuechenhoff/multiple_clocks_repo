#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is firing actually higher inside a ripple than outside? The raw-data check.

    python scripts/swr_spike_rate_in_ripples.py

The spikes-per-cell-per-ripple number (~0.2) is small enough to be alarming, but
it is a COUNT in a 60 ms window, not a rate. This asks the rate question
directly: spikes/s inside the ripple extent (t_peak +- duration/2) against
spikes/s in all other in-task time.

This is the classic ripple positive control -- hippocampal units must fire more
during ripples -- and it is reported per cell, session and ROI. Ripples are
detected on the hippocampal LFP, so HC is expected to rise; cortical ROIs are
the comparison.

@author: Svenja Kuchenhoff
"""

import os
import datetime

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc

ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]


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
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        half = d_rip / 2.0
        rip_s = float(half.sum() * 2)                  # total time inside ripples
        task_s = float((occ.stop_s - occ.start_s).sum())
        out_s = task_s - rip_s
        if out_s <= 0:
            continue
        for roiname in ROIS:
            cells = r[r.roi == roiname].cell.to_numpy()
            cells = [c for c in cells if c < len(spk[s]["spikes"])]
            for c in cells:
                st = spk[s]["spikes"][c]
                n_in = int(swc.count_in(st, t_rip - half, t_rip + half).sum())
                n_task = int(swc.count_in(st, occ.start_s.to_numpy(),
                                          occ.stop_s.to_numpy()).sum())
                n_out = max(n_task - n_in, 0)
                rows.append(dict(session=s, roi=roiname, cell=int(c),
                                 rate_in=n_in / rip_s, rate_out=n_out / out_s,
                                 n_in=n_in, n_out=n_out))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    R["ratio"] = R.rate_in / R.rate_out.replace(0, np.nan)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_spike_rate_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "per_cell_rate.csv"), index=False)

    print(f"\n{'ROI':12s} {'cells':>6s} {'in-ripple Hz':>13s} {'outside Hz':>11s} "
          f"{'ratio':>7s} {'% cells up':>11s} {'t (cells)':>10s} {'p':>10s} "
          f"{'p (session)':>12s}")
    for roiname in ROIS:
        g = R[R.roi == roiname].dropna(subset=["ratio"])
        if len(g) < 5:
            continue
        t, p = stats.ttest_rel(g.rate_in, g.rate_out)
        sess = g.groupby("session")[["rate_in", "rate_out"]].mean()
        _, ps = stats.ttest_rel(sess.rate_in, sess.rate_out)
        print(f"{roiname:12s} {len(g):6d} {g.rate_in.mean():13.3f} "
              f"{g.rate_out.mean():11.3f} {g.ratio.median():7.2f} "
              f"{100*np.mean(g.rate_in > g.rate_out):10.1f}% {t:10.2f} "
              f"{p:10.3g} {ps:12.3g}")
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
