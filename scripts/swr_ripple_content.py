#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 -- what does a hippocampal ripple contain?

    python scripts/swr_ripple_content.py

The first test of ripple CONTENT in this project. Everything before it was
foundations: Stage 2 established that the estimator reads the current location
out of a 60 ms non-ripple window in HC_mid (z = +0.78, p = 0.00026) and NOT in
HC_anterior, mOFC or mPFC. Those three fail the control, so a ripple result in
them is uninterpretable under a plain place template and is reported for
completeness only.

TWO PRE-DECLARED QUESTIONS

  C0  Is there location information inside ripples at all?
      Ripple windows (t_peak +- duration/2) scored against the SAME
      label-permutation null used in Stage 2, so the number is directly
      comparable to the Stage 2 value on non-ripple windows.

  C1  Is it ripple-SPECIFIC?
      The same score in equal-width flank windows at -500 and +500 ms, which
      share the trial, the current location, the task phase and the stillness,
      and differ only in whether a ripple is happening. Statistic is
      ripple minus flank, through the identical estimator.

VECTORISATION. The null costs almost nothing, and it is worth seeing why.
`E = C.T @ T`, so permuting the template's location columns gives
`C.T @ T[:, perm] == E[:, perm]` -- the permuted evidence is just a column
reordering of the evidence we already computed. `target_z` divides by the row
mean and SD across the nine locations, and a permutation leaves both unchanged.
So every null draw reduces to one fancy-index into a precomputed array: no spike
counting, no matmul, no per-permutation loop over grids. Spike counts are
computed ONCE for the three window sets and nothing inside the null touches the
spike trains. Memory stays at a few (n_ripples x 9) arrays per session.

Window width is part of the estimator -- `E` scales with spike count -- so the
flanks use each ripple's OWN duration. PRIMARY STATISTIC = `target_minus_others`, THE SAME ONE STAGE 2 USED, with the
same per-session permutation null. An earlier version of this script used
`target_z` instead, which (a) made C0 not comparable to the Stage 2 number it
was supposed to be compared with, and (b) needs a non-degenerate window
(sd > 0), which discards ripples where few cells fired and then whole sessions
via MIN_RIPPLES. That was not neutral attrition: HC_mid fell 35 -> 20 sessions
and the DROPPED sessions had a HIGHER mean Stage-2 z (+0.920) than the kept ones
(+0.675), i.e. the rule selected against the effect. `target_minus_others` has
no such requirement -- a window with no spikes scores 0, which is unbiased --
so every session is retained. `target_z` is still reported as a secondary.

Unit of inference: SESSION (the project's convention for ephys). A subject-level
column is printed as a sensitivity check only.

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
N_PERM = 100
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "mOFC": ["mOFC"], "mPFC": ["mPFC"]}
ROIS = list(ROI_SETS)
# How much say each cell gets, by how reliable its place map is across
# configurations. `all` reproduces the original Stage 3 exactly.
#
# WHY. Every ripple-vs-flank null in this analysis line is UNDERPOWERED: the
# minimum detectable effect runs from 60% to 173% of the ambient location signal
# itself, so only an effect that roughly DOUBLES location coding could have been
# seen. A cell with no location tuning contributes noise to both windows and
# signal to neither. Thresholding fixes that by discarding it; WEIGHTING keeps it
# at reduced influence, which is strictly better because no session is lost when
# its few cells fall below a cut -- and it was session loss that cancelled most
# of the gain from thresholding (HC_mid 35 -> 21 sessions at rel >= 0.2).
# Reliability is estimated with the scored configuration held out, so it never
# sees the data it weights, and it uses all task time, so it is shared by ripple
# and flank alike and cannot manufacture a difference between them.
SCHEMES = ["all", "thresh_0.1", "thresh_0.2", "linear", "square"]
FLANK_GAP_S = 0.05        # minimum separation between a ripple and its flank
MAX_FLANKS = 2            # one before, one after, where the interval allows
MIN_RIPPLES = 30
# Flanks are matched to each ripple on LOCATION, OCCUPANCY and WIDTH by
# construction (`swc.matched_flanks`), replacing the fixed +-500 ms offsets that
# were location-matched only ~54% of the time and manufactured a positive
# ripple-minus-flank difference (CHANGELOG 2026-09-17 h).


def _eff_n(T):
    """Kish effective sample size of the cell weights behind a template.

    Reports what a weighting scheme actually costs in cells: with equal weights
    it is the cell count, and with one dominant cell it tends to 1.
    """
    w = np.nansum(np.abs(T), axis=1)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0


def main():
    rng = np.random.default_rng(SEED)
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
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = spt.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or len(t_rip) < MIN_RIPPLES:
            continue
        subj = rip[rip.session == s].subject_key.iloc[0]

        # the subject's location at each ripple, and the grid it belongs to.
        # `occ` already IS the interval table, so look the ripple up in it
        # directly -- `swl.location_at` takes the raw step table instead.
        o = occ.sort_values("start_s").reset_index(drop=True)
        st = o.start_s.to_numpy(); sp = o.stop_s.to_numpy()
        j = np.searchsorted(st, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(o) - 1)
        inside = (j >= 0) & (t_rip <= sp[jj])
        loc = np.where(inside, o["loc"].to_numpy()[jj], np.nan)
        grid = o.cv_group.to_numpy()[jj]
        ok0 = np.isfinite(loc)
        if ok0.sum() < MIN_RIPPLES:
            continue
        t_rip, d_rip, loc, grid = t_rip[ok0], d_rip[ok0], loc[ok0], grid[ok0]
        half = d_rip / 2.0

        grids = np.unique(occ.cv_group.to_numpy())

        # flanks inside the ripple's OWN occupancy interval: same location,
        # same dwell, same width, never overlapping another ripple
        t_fl, owner = swc.matched_flanks(t_rip, half, occ, other_t=t_rip,
                                         gap_s=FLANK_GAP_S, max_per=MAX_FLANKS)
        if not len(t_fl):
            continue
        has_flank = np.zeros(len(t_rip), bool)
        has_flank[np.unique(owner)] = True
        w_fl = half[owner]

        for roiname, members in ROI_SETS.items():
            all_cells = r[r.roi.isin(members)].cell.to_numpy()
            all_cells = [c for c in all_cells if c < len(spk[s]["spikes"])]
            if not all_cells:
                continue
            cells = all_cells
            if len(cells) < 2:
                continue
            for scheme in SCHEMES:
                loo = spt.weighted_loo(spk[s], cells, occ, grids, scheme)
                if not any(np.nansum(np.abs(T)) > 0 for T in loo.values()):
                    continue      # no cell in this ROI carries any location map

                C_rip, C_fl = swc.zscore_cells([
                    spt.window_counts(spk[s], cells, t_rip - half, t_rip + half),
                    spt.window_counts(spk[s], cells, t_fl - w_fl, t_fl + w_fl)])
                E_rip = spt.score_windows(C_rip, t_rip, grid, loo)
                E_fl = spt.score_windows(C_fl, t_fl, grid[owner], loo)

                ok_r = np.isfinite(E_rip).all(axis=1) & has_flank
                ok_f = np.isfinite(E_fl).all(axis=1)
                if ok_r.sum() < MIN_RIPPLES or ok_f.sum() < MIN_RIPPLES:
                    continue
                li_r = loc.astype(int) - 1
                li_f = loc.astype(int)[owner] - 1
                tot_r, tot_f = E_rip.sum(axis=1), E_fl.sum(axis=1)

                def stat(perm):
                    hr = E_rip[np.arange(len(E_rip)), perm[li_r]]
                    sr = hr - (tot_r - hr) / 8.0
                    hf = E_fl[np.arange(len(E_fl)), perm[li_f]]
                    sf = hf - (tot_f - hf) / 8.0
                    # each ripple's flanks averaged, then paired with that ripple
                    fl_mean = np.full(len(t_rip), np.nan)
                    np.add.at(fl_mean, owner[ok_f], 0)      # ensure entries exist
                    acc = np.zeros(len(t_rip)); cnt = np.zeros(len(t_rip))
                    np.add.at(acc, owner[ok_f], sf[ok_f])
                    np.add.at(cnt, owner[ok_f], 1.0)
                    good = ok_r & (cnt > 0)
                    fl_mean[good] = acc[good] / cnt[good]
                    return (float(np.nanmean(sr[good])),
                            float(np.nanmean(fl_mean[good])),
                            float(np.nanmean(sr[good] - fl_mean[good])),
                            int(good.sum()))

                ident = np.arange(9)
                obs = stat(ident)
                if obs[3] < MIN_RIPPLES:
                    continue
                rng_p = np.random.default_rng(
                    [SEED, s, ROIS.index(roiname), SCHEMES.index(scheme)])
                nul = np.array([stat(rng_p.permutation(9))[:3]
                                for _ in range(N_PERM)], float)
                out = dict(session=s, subject=subj, roi=roiname,
                           scheme=scheme, n_cells=len(cells),
                           n_ripples=obs[3], n_flanks=int(ok_f.sum()),
                           eff_cells=float(np.mean([_eff_n(T)
                                                    for T in loo.values()])))
                for k, name in enumerate(("ripple", "flank", "diff")):
                    sd = np.nanstd(nul[:, k])
                    out[f"{name}"] = obs[k]
                    out[f"z_{name}"] = ((obs[k] - np.nanmean(nul[:, k])) / sd
                                        if sd > 0 else np.nan)
                rows.append(out)
        print(f"  s{s:02d}", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_content_stage3_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "stage3_per_session.csv"), index=False)

    print("\n=== C0: location content INSIDE ripples (vs permutation null) ===")
    print("=== C1: ripple minus flanks matched on LOCATION, OCCUPANCY, WIDTH ===")
    print("    swept over how location-tuned a cell must be to be included.")
    print("    MDE = the smallest C1 this many sessions could detect at 80% "
          "power;\n    compare it with C0, the ambient signal it would have to "
          "stand out from.\n")
    print(f"{'ROI':12s} {'weights':>10s} {'sess':>5s} {'eff.n':>6s} | "
          f"{'C0':>7s} {'p':>8s} | {'C1':>7s} {'p':>8s} {'MDE':>6s} {'%C0':>5s}")
    summ = []
    for roiname in ROIS:
        for scheme in SCHEMES:
            g = R[(R.roi == roiname) & (R.scheme == scheme)].dropna(
                subset=["z_ripple"])
            if len(g) < 5:
                continue
            p0 = stats.ttest_1samp(g.z_ripple, 0)[1]
            gd = g.dropna(subset=["z_diff"])
            if len(gd) < 5:
                continue
            p1 = stats.ttest_1samp(gd.z_diff, 0)[1]
            sem = gd.z_diff.std(ddof=1) / np.sqrt(len(gd))
            mde = (stats.t.ppf(0.975, len(gd) - 1)
                   + stats.t.ppf(0.80, len(gd) - 1)) * sem
            amb = g.z_ripple.mean()
            print(f"{roiname:12s} {scheme:>10s} {len(g):5d} "
                  f"{g.eff_cells.mean():6.1f} | {amb:+7.3f} {p0:8.3g} | "
                  f"{gd.z_diff.mean():+7.3f} {p1:8.3g} {mde:6.3f} "
                  f"{100 * mde / amb if amb > 0 else np.nan:5.0f}")
            summ.append(dict(roi=roiname, scheme=scheme, sessions=len(g),
                             mean_cells=float(g.n_cells.mean()),
                             eff_cells=float(g.eff_cells.mean()),
                             ripples=int(g.n_ripples.sum()),
                             z_ripple=float(amb), p_ripple=float(p0),
                             z_flank=float(g.z_flank.mean()),
                             z_diff=float(gd.z_diff.mean()), p_diff=float(p1),
                             mde_80=float(mde)))
        print()
    pd.DataFrame(summ).to_csv(os.path.join(d, "stage3_summary.csv"), index=False)
    json.dump(dict(seed=SEED, n_perm=N_PERM, rois=ROIS,
                   flank="matched on location, occupancy and width (swc.matched_flanks)",
                   window="ripple t_peak +- duration/2; flanks the same width",
                   score="swc.target_minus_others",
                   schemes=SCHEMES,
                   scheme_rationale="each cell's template is scaled by its "
                                    "split-half place-map reliability, estimated "
                                    "with the scored configuration held out; "
                                    "weighting keeps every session, thresholding "
                                    "loses the ones whose few cells fall below "
                                    "the cut",
                   inference="per-session z vs own label-permutation null, "
                             "t across sessions; subject-level as sensitivity",
                   note="HC_anterior/mOFC/mPFC FAIL the Stage 2 current-location "
                        "control; their rows are for completeness only",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


if __name__ == "__main__":
    main()
