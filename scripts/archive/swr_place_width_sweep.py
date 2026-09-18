#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic for Stage 2: how does the location signal scale with window width?

    python scripts/swr_place_width_sweep.py

The positive control at ripple width (~60 ms) is weak (HC_mid z = +0.44,
p = 0.024; HC_anterior z = +0.22, n.s.). That is either (a) a spike-budget
limit -- 60 ms simply does not contain enough spikes -- or (b) the estimator
does not work. The two are told apart by widening the window on the SAME
non-ripple navigation data: under (a) the signal must grow steeply with width,
under (b) it stays flat.

Inference is per-session z against that session's OWN label-permutation null,
then a t-test across sessions. The raw score cannot be averaged across sessions:
it scales with firing rate and cell count (|score| vs n_cells: r = 0.49).

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
N_PERM = 200
WIDTHS = [0.06, 0.125, 0.25, 0.5, 1.0, 2.0]
ROIS = ["HC_anterior", "HC_mid", "mPFC", "mOFC"]
N_WINDOWS = 1500


def main():
    rng = np.random.default_rng(SEED)
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
        occ, r, t_rip, _ = spt.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        grids = np.unique(occ.cv_group.to_numpy())
        tmpl = {}
        for roiname in ROIS:
            cells = r[r.roi == roiname].cell.to_numpy()
            if len(cells) >= 1:
                tmpl[roiname] = (cells,) + spt.build_templates(spk[s], cells,
                                                               occ, grids)
        if not tmpl:
            continue
        for width in WIDTHS:
            # EVERY eligible window, not a sample of 1500: the sampled version
            # gave HC_mid +0.30/+0.44/+0.53 across three runs that differed only
            # in the draw, and its apparent rise from 60->125 ms was that noise
            t0, loc = swc.tiled_windows(occ, width, avoid=t_rip, avoid_pad_s=1.0)
            if len(t0) < 200:
                continue
            o = occ.sort_values("start_s")
            j = np.searchsorted(o.start_s.to_numpy(), t0, side="right") - 1
            win_grid = o.cv_group.to_numpy()[np.clip(j, 0, len(o) - 1)]
            for roiname, (cells, full, loo) in tmpl.items():
                C_raw = spt.window_counts(spk[s], cells, t0, t0 + width)
                spk_per_cell = float(C_raw.mean())   # BEFORE z-scoring; the
                # z-scored mean is 0 by construction and says nothing
                C = swc.zscore_cells([C_raw])[0]
                E = spt.score_windows(C, t0, win_grid, loo)
                sc = swc.target_minus_others(E, loc)
                ok = np.isfinite(sc)
                if ok.sum() < 200:
                    continue
                obs = float(np.nanmean(sc[ok]))
                acc = float(np.nanmean(np.nanargmax(E[ok], 1) + 1 == loc[ok]))
                nul = np.empty(N_PERM)
                for p in range(N_PERM):
                    Ep = spt.score_windows(C, t0, win_grid, loo,
                                           perm=rng.permutation(9))
                    scp = swc.target_minus_others(Ep, loc)
                    nul[p] = np.nanmean(scp[np.isfinite(scp)])
                sd = nul.std()
                rows.append(dict(session=s, roi=roiname, width=width,
                                 n_cells=len(cells), n_win=int(ok.sum()),
                                 spikes_per_cell=spk_per_cell,
                                 score=obs, acc=acc,
                                 z=(obs - nul.mean()) / sd if sd > 0 else np.nan))
        print(f"  s{s:02d} done", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_content_widthsweep_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "width_sweep.csv"), index=False)

    print(f"\n{'ROI':12s} {'width':>7s} {'sess':>5s} {'spk/cell':>9s} "
          f"{'mean z':>8s} {'t':>7s} {'p':>9s} {'acc':>6s}")
    summ = []
    for roiname in ROIS:
        for width in WIDTHS:
            v = R[(R.roi == roiname) & (R.width == width)].dropna(subset=["z"])
            if len(v) < 5:
                continue
            t, p = stats.ttest_1samp(v.z, 0)
            print(f"{roiname:12s} {width:7.3f} {len(v):5d} "
                  f"{v.spikes_per_cell.mean():9.3f} {v.z.mean():+8.3f} "
                  f"{t:+7.2f} {p:9.3g} {v.acc.mean():6.3f}")
            summ.append(dict(roi=roiname, width=width, sessions=len(v),
                             spikes_per_cell=float(v.spikes_per_cell.mean()),
                             z=float(v.z.mean()), t=float(t), p=float(p),
                             acc=float(v.acc.mean())))
        print()
    pd.DataFrame(summ).to_csv(os.path.join(d, "width_sweep_summary.csv"),
                              index=False)
    json.dump(dict(seed=SEED, n_perm=N_PERM, widths=WIDTHS, rois=ROIS,
                   n_windows_requested=N_WINDOWS, avoid_pad_s=1.0,
                   inference="per-session z vs own label-permutation null, "
                             "t-test across sessions",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"-> {d}")


if __name__ == "__main__":
    main()
