#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptives for the ripple-content analysis -- the numbers behind Figure 1.

    python scripts/swr_content_descriptives.py

Computes and caches, per session and per ROI: ripple rate and duration, cell
counts, spikes per cell per ripple, co-active cells per ripple, location dwell
times and occupancy, and peri-ripple spike timing. Nothing is tested here; this
is the "what does the data look like" pass that tells us whether a cut is going
to be underpowered before we run it.

Everything is written to CSV next to the figures so plots can be re-made without
recomputing, and so every plotted number has a traceable source.

@author: Svenja Kuchenhoff
"""

import os
import datetime

import numpy as np
import pandas as pd

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import scripts.swr_place_templates as spt

ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC", "PCC"]
PERI_HALF_S = 0.5
PERI_BIN_S = 0.025


def out_dir():
    d = os.path.join(rrsa._derivatives(), "group", "swr",
                     f"ripple_content_descriptives_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)

    sess_rows, cell_rows, dwell_rows, coact_rows = [], [], [], []
    edges = np.arange(-PERI_HALF_S, PERI_HALF_S + PERI_BIN_S, PERI_BIN_S)
    # per-CELL normalised histograms: summing raw counts lets the highest-rate
    # cells dominate, which is exactly wrong for a spike-ripple alignment check
    peri = {r: [] for r in ROIS}

    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = spt.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(t_rip):
            continue
        task_s = float((occ.stop_s - occ.start_s).sum())
        sess_rows.append(dict(session=s, n_ripples=len(t_rip),
                              task_s=task_s,
                              ripple_rate_hz=len(t_rip) / task_s if task_s else np.nan,
                              median_duration_s=float(np.median(d_rip)),
                              n_cells=len(r),
                              n_grids=int(occ.grid_num.nunique())))
        dw = (occ.stop_s - occ.start_s).to_numpy()
        for L in range(1, 10):
            m = occ["loc"].to_numpy() == L
            dwell_rows.append(dict(session=s, loc=L, n_visits=int(m.sum()),
                                   total_s=float(dw[m].sum()),
                                   median_dwell_s=float(np.median(dw[m])) if m.any() else np.nan))

        for roiname in ROIS:
            cells = r[r.roi == roiname].cell.to_numpy()
            cells = [c for c in cells if c < len(spk[s]["spikes"])]
            if not cells:
                continue
            half = d_rip / 2.0
            M = np.zeros((len(cells), len(t_rip)))
            for j, c in enumerate(cells):
                st = spk[s]["spikes"][c]
                M[j] = swc.count_in(st, t_rip - half, t_rip + half)
                # peri-ripple raster, pooled -- one histogram, not one per ripple
                lo = np.searchsorted(st, t_rip - PERI_HALF_S)
                hi = np.searchsorted(st, t_rip + PERI_HALF_S)
                keep = hi > lo
                if keep.any():
                    idx = np.concatenate([np.arange(a, b)
                                          for a, b in zip(lo[keep], hi[keep])])
                    ref = np.repeat(t_rip[keep], (hi - lo)[keep])
                    h = np.histogram(st[idx] - ref, edges)[0].astype(float)
                    h = h / len(t_rip) / PERI_BIN_S            # -> Hz
                    b0 = h[(edges[:-1] >= -PERI_HALF_S) & (edges[:-1] < -0.25)].mean()
                    if b0 > 0:
                        peri[roiname].append(100 * (h - b0) / b0)
                rate = len(st) / task_s if task_s else np.nan
                cell_rows.append(dict(session=s, roi=roiname, cell=int(c),
                                      mean_rate_hz=rate,
                                      spikes_per_ripple=float(M[j].mean())))
            act = (M > 0).sum(0)
            for k in range(0, 6):
                coact_rows.append(dict(session=s, roi=roiname, n_active=k,
                                       n_ripples=int((act == k).sum()),
                                       frac=float((act == k).mean())))
        print(f"  s{s:02d}", flush=True)

    d = out_dir()
    pd.DataFrame(sess_rows).to_csv(os.path.join(d, "session_descriptives.csv"), index=False)
    pd.DataFrame(cell_rows).to_csv(os.path.join(d, "cell_descriptives.csv"), index=False)
    pd.DataFrame(dwell_rows).to_csv(os.path.join(d, "location_dwell.csv"), index=False)
    pd.DataFrame(coact_rows).to_csv(os.path.join(d, "coactivity.csv"), index=False)
    P = pd.DataFrame({"t_s": (edges[:-1] + edges[1:]) / 2})
    for r in ROIS:
        A = np.array(peri[r]) if peri[r] else np.zeros((0, len(edges) - 1))
        P[r] = A.mean(0) if len(A) else np.nan
        P[r + "_sem"] = A.std(0) / np.sqrt(len(A)) if len(A) else np.nan
        P[r + "_ncells"] = len(A)
    P.to_csv(os.path.join(d, "peri_ripple_rate.csv"), index=False)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
