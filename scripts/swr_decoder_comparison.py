#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Can location be decoded at all? Every signal we have, one common design.

    python scripts/swr_decoder_comparison.py

The I14 gate asked this of iEEG band power and got a null. This completes the
picture by running the SAME design on every signal in the dataset -- each band
separately, all bands together, and SPIKES -- so the answer is a comparison
rather than a single negative.

COMMON DESIGN. One row per VISIT (occupancy interval, median 0.367 s). Features
averaged across the visit. 9-way multinomial logistic regression, leave-one-
CONFIGURATION-out, balanced accuracy against a label-permutation null (which
puts chance at 1/9).

⚠ AND THE POINT OF THE FIGURE. Accuracy is not the only read-out. The spike
analyses in this project do not use argmax at all; they use the continuous
evidence score `E(L) = sum_c n_c z_c(L)`, which asks how far the true square
stands out from the other eight rather than whether it wins. That statistic
finds location in spikes (z = +0.64) where argmax does not, because neighbouring
squares have correlated templates and evidence that is clearly elevated can
still peak one square away. Both are computed here, on the same visits, so the
difference is visible instead of asserted.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
import scripts.swr_place_templates as spt
from scripts.swr_ieeg_location_decoder import (bundle, session_features,
                                               positive_control, decode,
                                               MIN_VISITS)

SEED = 42
BAND_SETS = {"HFB (high gamma)": ["hfb"], "theta": ["theta"], "beta": ["beta"],
             "ripple band": ["ripple"], "all bands": ["hfb", "theta", "beta"]}
ROI_SPIKES = {"HC_all": ["HC_anterior", "HC_mid"]}


def band_features(s, bands):
    f = os.path.join(bundle(), "hfb", f"s{s:02d}_hfb.npz")
    if not os.path.exists(f):
        return None, None
    z = np.load(f, allow_pickle=True)
    X = np.concatenate([np.asarray(z[b], dtype=np.float32) for b in bands],
                       axis=0).T
    return X, float(z["out_fs"])


def main():
    rng = np.random.default_rng(SEED)
    deriv = rrsa._derivatives()
    hp = pd.read_csv(os.path.join(bundle(), "hfb_pairs.csv"))
    hp = hp[~hp.excluded.astype(bool)]
    sessions = sorted(hp.session.unique().tolist())

    pc = positive_control(sessions)
    print(f"positive control: HPC HFB peak {pc['peak']:+.4f} at "
          f"{pc['lag_s']:+.3f} s -- "
          f"{'PASSED' if pc['peak'] > 4 * pc['sem'] and abs(pc['lag_s']) <= 0.02 else 'FAILED'}\n")

    steps = swl.load(sessions)
    rip = pd.read_csv(os.path.join(bundle(), "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s"])
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)

    rows = []
    for s in sessions:
        occ = swc.occupancy(steps, s)
        if len(occ) < MIN_VISITS:
            continue
        y = occ["loc"].to_numpy()
        g = occ.cv_group.to_numpy()

        # --- iEEG bands
        for name, bands in BAND_SETS.items():
            X, fs = band_features(s, bands)
            if X is None:
                continue
            a = np.round(occ.start_s.to_numpy() * fs).astype(int)
            b = np.round(occ.stop_s.to_numpy() * fs).astype(int)
            m = (a >= 0) & (b <= X.shape[0]) & (b > a)
            good = np.isfinite(X).all(axis=0)
            if good.sum() < 3 or m.sum() < MIN_VISITS:
                continue
            F = np.array([X[i:j, good].mean(axis=0)
                          for i, j in zip(a[m], b[m])])
            res = decode(F, y[m], g[m], rng)
            if res:
                acc, nmu, nsd, n = res
                rows.append(dict(session=s, signal=name, kind="iEEG",
                                 n_features=int(good.sum()), n_visits=n,
                                 acc=acc, null=nmu))

        # --- spikes, same visits, same CV
        if s not in spk:
            continue
        cells = [c for c in roi[(roi.session == s)
                                & roi.roi.isin(ROI_SPIKES["HC_all"])].cell
                 .to_numpy() if c < len(spk[s]["spikes"])]
        if len(cells) < 2:
            continue
        C = spt.window_counts(spk[s], cells, occ.start_s.to_numpy(),
                              occ.stop_s.to_numpy())
        dur = (occ.stop_s - occ.start_s).to_numpy()
        F = (C / np.where(dur > 0, dur, np.nan)).T          # firing rate
        ok = np.isfinite(F).all(axis=1)
        if ok.sum() < MIN_VISITS:
            continue
        res = decode(F[ok], y[ok], g[ok], rng)
        if res:
            acc, nmu, nsd, n = res
            rows.append(dict(session=s, signal="spikes (HC)", kind="spikes",
                             n_features=len(cells), n_visits=n,
                             acc=acc, null=nmu))
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"decoder_comparison_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "decoder_comparison.csv"), index=False)
    print("\n=== 9-way location decoding ACCURACY, one design, every signal ===")
    print(f"{'signal':20s} {'sess':>5s} {'feat':>6s} {'bal.acc':>8s} "
          f"{'null':>7s} {'p':>10s}")
    for name in list(BAND_SETS) + ["spikes (HC)"]:
        gg = R[R.signal == name]
        if len(gg) < 5:
            continue
        p = stats.ttest_1samp((gg.acc - gg["null"]).dropna(), 0)[1]
        print(f"{name:20s} {len(gg):5d} {gg.n_features.mean():6.1f} "
              f"{gg.acc.mean():8.4f} {gg['null'].mean():7.4f} {p:10.3g}")
    json.dump(dict(seed=SEED, band_sets=BAND_SETS,
                   unit="one occupancy interval (visit)",
                   cv="leave-one-configuration-out",
                   positive_control=pc,
                   note="accuracy only; the continuous evidence estimator is in "
                        "swr_ripple_content.py and finds location in spikes",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
