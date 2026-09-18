#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Can location be decoded from iEEG band power? The gate for sequence analysis.

    python scripts/swr_ieeg_location_decoder.py

WHY THIS EXISTS. SK asked whether ripples contain fast SEQUENCES -- squares
rolled out one after another -- which every window-averaging analysis in this
project is blind to. The standard method (He et al. 2026; Liu/Kurth-Nelson TDLM)
needs a decoder giving a reactivation probability PER STATE PER TIMEPOINT.
Spikes cannot supply it here: the median session has five hippocampal cells
firing 0.2 spikes per cell per ripple, and pseudo-populations cannot help
because pairing a different ripple per cell destroys exactly the within-event
order being tested. Contacts might: the bundle carries band power at 100 Hz for
1,485 bipolar derivations across 61 sessions.

**This script runs no sequence analysis.** It answers only the prerequisite:
does a 9-way location decoder beat chance at all? If it does not, there is
nothing to sequence.

⚠ THE POSITIVE CONTROL RUNS FIRST AND IS NOT OPTIONAL. An earlier version of
this analysis reported a confident null from a decoder that could not decode
ANYTHING -- not location, not the 4-way state, not even a button press against a
window 1.5 s later. A null from an unvalidated instrument says nothing about the
brain. `positive_control` reproduces this project's own established result --
HFB rises sharply at the ripple peak -- through the SAME loader and the SAME
time convention, and `main` refuses to report a decoding null unless it passes.

WHAT WENT WRONG THE FIRST TIME, so it is not repeated:

  per-timepoint features   the ripple-locked HFB response is about ONE SAMPLE
                           wide (+0.197 at 0 ms, +0.017 at +-100 ms). Averaging
                           features over a 400 ms window diluted it ~40x, and
                           sampling every 10 ms gave 24,000 rows whose effective
                           n was far smaller -- band power is 0.94 autocorrelated
                           at lag 1. Both decoders were asking band power to work
                           at a timescale where its signal is smeared or drowned.
  the unit                 is now the VISIT -- one occupancy interval, median
                           0.367 s, features averaged across it. That is the
                           same unit the spike analyses use, where location IS
                           decodable (z = +0.64), so a null here is meaningful.

TWO CROSS-VALIDATION LEVELS, because they answer different questions:

  config  leave one CONFIGURATION out (`grid_id`). The honest split for any
          content claim, and the one every spike analysis uses.
  run     leave one RUN out (`grid_num`), so the same configuration appears in
          training and test. A LEAK for content claims -- the decoder is allowed
          to know the route -- but the right question for a decoder used as the
          front end of a sequence analysis: can the signal identify a square at
          all, given the route? SK's proposal, and it does what she predicted:
          it removes the below-chance artefact of the config split.

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

SEED = 42
BANDS = ["hfb", "theta", "beta"]        # NOT "ripple" -- see note below
N_PERM = 50
MIN_VISITS = 200
MIN_TEST = 20
C_REG = 0.02
CV_LEVELS = ["config", "run"]
ROI_SETS = {"all_contacts": None,
            "HPC": ["HC_anterior", "HC_mid"],
            "mPFC": ["mPFC"], "mOFC": ["mOFC"], "Visual": ["Visual"]}
# The `ripple` band is excluded from the features on purpose: it is the band
# ripples are detected in, so including it would let a decoder read the ripple
# itself and make any later peri-ripple comparison circular.


def bundle():
    return os.path.join(rrsa._derivatives(), "group", "swr", "bundle_v2")


def session_features(s):
    """(n_samples, n_pairs * n_bands) band power, and the pair behind each column.

    Sample i is session time i / fs -- verified, not assumed: `positive_control`
    recovers the HFB peak at exactly lag 0.
    """
    f = os.path.join(bundle(), "hfb", f"s{s:02d}_hfb.npz")
    if not os.path.exists(f):
        return None, None, None
    z = np.load(f, allow_pickle=True)
    ids = np.asarray(z["pair_ids"], dtype=object)
    X = np.concatenate([np.asarray(z[b], dtype=np.float32) for b in BANDS],
                       axis=0).T
    return X, np.tile(ids, len(BANDS)), float(z["out_fs"])


def visit_features(X, occ, fs):
    """One row per occupancy interval: band power averaged across the visit."""
    a = np.round(occ.start_s.to_numpy() * fs).astype(int)
    b = np.round(occ.stop_s.to_numpy() * fs).astype(int)
    m = (a >= 0) & (b <= X.shape[0]) & (b > a)
    a, b = a[m], b[m]
    good = np.isfinite(X).all(axis=0)
    F = np.array([X[i:j, good].mean(axis=0) for i, j in zip(a, b)])
    return F, good, m


def decode(F, y, g, rng):
    """Balanced accuracy, leave-one-group-out, against a label-permutation null.

    Balanced accuracy is used because occupancy is not uniform across the nine
    squares (8.9% to 14.8%), which puts chance at 1/9 by construction -- and the
    permutation null confirms it lands there.
    """
    pred = np.full(len(y), np.nan)
    for gr in np.unique(g):
        te = g == gr
        tr = ~te
        if tr.sum() < 100 or te.sum() < MIN_TEST or len(np.unique(y[tr])) < 3:
            continue
        sc = StandardScaler().fit(F[tr])
        mdl = LogisticRegression(C=C_REG, max_iter=400,
                                 multi_class="multinomial")
        mdl.fit(sc.transform(F[tr]), y[tr])
        pred[te] = mdl.predict(sc.transform(F[te]))
    ok = np.isfinite(pred)
    if ok.sum() < 100:
        return None
    acc = balanced_accuracy_score(y[ok], pred[ok])
    nul = [balanced_accuracy_score(rng.permutation(y[ok]), pred[ok])
           for _ in range(N_PERM)]
    return acc, float(np.mean(nul)), float(np.std(nul)), int(ok.sum())


# ── the gate ───────────────────────────────────────────────────────────

def positive_control(sessions, n_check=14):
    """Does HFB rise at the ripple peak, through this loader? Known to be true.

    Reproduces the project's established peri-ripple HFB effect. Returns the
    hippocampal peak and its lag; `main` stops unless the peak is clearly
    positive AND lands within one sample of zero.
    """
    hp = pd.read_csv(os.path.join(bundle(), "hfb_pairs.csv"))
    hp = hp[~hp.excluded.astype(bool)]
    rip = pd.read_csv(os.path.join(bundle(), "ripples.csv"),
                      usecols=["session", "t_peak_s"])
    lags = np.arange(-1.0, 1.001, 0.02)
    prof = []
    for s in sessions[:n_check]:
        X, cols, fs = session_features(s)
        if X is None:
            continue
        n_pairs = len(cols) // len(BANDS)
        ids = cols[:n_pairs]
        hs = hp[hp.session == s]
        sel = np.isin(ids, list(hs[hs.pair_roi_atlas.isin(
            ["HC_anterior", "HC_mid"])].pair_id))
        t0 = rip[rip.session == s].t_peak_s.to_numpy(float)
        if sel.sum() < 1 or len(t0) < 200:
            continue
        H = X[:, :n_pairs][:, sel]
        p = np.array([H[np.clip(np.round((t0 + L) * fs).astype(int), 0,
                                len(H) - 1)].mean() for L in lags])
        prof.append(p - p[np.abs(lags + 0.75) <= 0.25].mean())
    if not prof:
        return None
    m = np.mean(prof, axis=0)
    k = int(np.argmax(np.abs(m)))
    return dict(n_sessions=len(prof), peak=float(m[k]), lag_s=float(lags[k]),
                sem=float(np.std(prof, axis=0)[k] / np.sqrt(len(prof))))


def main():
    rng = np.random.default_rng(SEED)
    deriv = rrsa._derivatives()
    hp = pd.read_csv(os.path.join(bundle(), "hfb_pairs.csv"))
    hp = hp[~hp.excluded.astype(bool)]
    sessions = sorted(hp.session.unique().tolist())

    print("=== POSITIVE CONTROL: does HFB rise at the ripple peak? ===")
    pc = positive_control(sessions)
    if pc is None:
        print("  could not run -- stopping"); return
    print(f"  HPC peak {pc['peak']:+.4f} at {pc['lag_s']:+.3f} s "
          f"(SEM {pc['sem']:.4f}, {pc['n_sessions']} sessions)")
    if not (pc["peak"] > 4 * pc["sem"] and abs(pc["lag_s"]) <= 0.02):
        print("  ⚠ FAILED. The loader or the time base is wrong; a decoding "
              "null would be meaningless. Stopping.")
        return
    print("  passed -- loader, time base and features are sound.\n")

    steps = swl.load(sessions)
    out_dir = os.path.join(deriv, "group", "swr",
                           f"ieeg_location_decoder_{datetime.date.today()}")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for s in sessions:
        X, cols, fs = session_features(s)
        if X is None:
            continue
        occ = swc.occupancy(steps, s)
        if len(occ) < MIN_VISITS:
            continue
        F, good, m = visit_features(X, occ, fs)
        if good.sum() < 3 or len(F) < MIN_VISITS:
            continue
        y = occ["loc"].to_numpy()[m]
        hs = hp[hp.session == s]
        ids_kept = cols[good]
        for name, members in ROI_SETS.items():
            if members is None:
                sub = np.ones(F.shape[1], bool)
            else:
                keep = set(hs[hs.pair_roi_atlas.isin(members)].pair_id)
                sub = np.isin(ids_kept, list(keep))
            if sub.sum() < 3:
                continue
            for lvl in CV_LEVELS:
                g = (occ.cv_group if lvl == "config" else occ.grid_num
                     ).to_numpy()[m]
                res = decode(F[:, sub], y, g, rng)
                if res is None:
                    continue
                acc, nmu, nsd, n = res
                rows.append(dict(session=s, roi=name, cv=lvl,
                                 n_features=int(sub.sum()), n_visits=n,
                                 acc=acc, null=nmu, null_sd=nsd,
                                 z=(acc - nmu) / nsd if nsd > 0 else np.nan))
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, "decoder_per_session.csv"), index=False)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    print("\n=== 9-way location decoding from iEEG band power, per VISIT ===")
    print(f"{'feature set':14s} {'cv':>7s} {'sess':>5s} {'chans':>6s} "
          f"{'bal.acc':>8s} {'null':>7s} {'p':>10s} {'above':>8s}")
    for name in ROI_SETS:
        for lvl in CV_LEVELS:
            g = R[(R.roi == name) & (R.cv == lvl)]
            if len(g) < 5:
                continue
            p = stats.ttest_1samp((g.acc - g["null"]).dropna(), 0)[1]
            print(f"{name:14s} {lvl:>7s} {len(g):5d} {g.n_features.mean():6.1f} "
                  f"{g.acc.mean():8.4f} {g['null'].mean():7.4f} {p:10.3g} "
                  f"{int((g.acc > g['null']).sum()):4d}/{len(g):<3d}")
    R.to_csv(os.path.join(out_dir, "decoder_per_session.csv"), index=False)
    json.dump(dict(seed=SEED, bands=BANDS, n_perm=N_PERM, C=C_REG,
                   unit="one occupancy interval (visit), median 0.367 s",
                   cv_levels=CV_LEVELS, roi_sets=ROI_SETS,
                   positive_control=pc,
                   excluded_band="ripple -- would make peri-ripple comparisons "
                                 "circular",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(out_dir, "settings.json"), "w"), indent=2)
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
