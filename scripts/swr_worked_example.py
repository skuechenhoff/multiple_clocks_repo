#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A worked numerical example of the ripple-content estimator, on a toy session.

    python scripts/swr_worked_example.py

Everything here runs through the SAME functions as the real analysis
(`mc.analyse.swr_content`), so the printed numbers are the estimator's actual
behaviour rather than a re-derivation that could drift from it.

Toy session: 3 cells, 9 locations, 2 configurations, 36 visits, 6 ripples.
  cell 0 fires for location 5, cell 1 for location 2, cell 2 is location-blind.
Six ripples: three at location 5, two at location 2, across both configurations.

@author: Svenja Kuchenhoff
"""

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.swr_content as swc

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=140)


def toy_session():
    """Each of the 9 locations visited twice per configuration, 1 s per visit.

    All nine must appear in BOTH configurations, otherwise leaving one out
    makes template entries NaN and every evidence value is undefined -- which
    is exactly the failure mode STEP 5 is about.
    """
    locs = ([1, 2, 3, 4, 5, 6, 7, 8, 9] * 2) + ([1, 2, 3, 4, 5, 6, 7, 8, 9] * 2)
    grid = ["A"] * 18 + ["B"] * 18
    occ = pd.DataFrame(dict(
        start_s=np.arange(36, dtype=float),
        stop_s=np.arange(36, dtype=float) + 1.0,
        loc=locs, cv_group=grid, step_idx=np.arange(36)))
    # cell 0 is a place cell for location 5, cell 1 for location 2, cell 2
    # fires the same everywhere (location-blind).
    rates = [[60 if L == 5 else 10 for L in locs],
             [50 if L == 2 else 10 for L in locs],
             [30] * 36]
    spikes = []
    for r in rates:
        t = []
        for i, rate in enumerate(r):
            t.append(np.sort(np.random.uniform(i, i + 1,
                                               np.random.poisson(rate))))
        spikes.append(np.concatenate(t))
    return occ, {"spikes": spikes}


def main():
    occ, spk = toy_session()
    cells = [0, 1, 2]
    grids = np.array(["A", "B"])

    print("=" * 78)
    print("STEP 1 -- the place-field template, per cell")
    print("=" * 78)
    print("\nRaw firing rate per location (Hz), all data:")
    for c in cells:
        print(f"  cell {c}: {swc.place_map(spk['spikes'][c], occ)}")
    print("\n  Cell 0 is ~60 Hz at location 5 and ~10 Hz elsewhere; cell 2 is")
    print("  flat. A location never visited would show NaN.")

    print("\nTemplate = the same thing z-scored ACROSS locations:")
    full, loo = swc.build_templates(spk, cells, occ, grids)
    for c in cells:
        T = full[c]
        print(f"  cell {c}: {T}   sum = {np.nansum(T):+.3f}")
    print("\n  Each row sums to zero over the locations it has. That is what")
    print("  makes a location-blind cell (cell 2) contribute ~nothing.")

    print("\nLeave-one-configuration-out: the template used to score a ripple")
    print("that happened during configuration A is built WITHOUT any A data:")
    print(f"  cell 0, all data      : {full[0]}")
    print(f"  cell 0, A held out    : {loo['A'][0]}")
    print("  Here every location appears in both configurations, so both")
    print("  templates are complete. If a location appeared ONLY in the")
    print("  held-out configuration its entry would be NaN -- that is where")
    print("  'non-finite evidence' comes from. See STEP 5.")

    print("\n" + "=" * 78)
    print("STEP 2 -- spike counts in each ripple window, z-scored per cell")
    print("=" * 78)
    # 6 ripples, 60 ms each, sitting inside visits 0,1,3,6,7,9
    #        visit:   4     1     13     22     19     31
    t_rip = np.array([4.5, 1.5, 13.5, 22.5, 19.5, 31.5])
    half = np.full(6, 0.050)          # 100 ms windows
    rip_loc = np.array([5, 2, 5, 5, 2, 5])
    rip_grid = np.array(["A", "A", "A", "B", "B", "B"])

    raw = swc.window_counts(spk, cells, t_rip - half, t_rip + half)
    print("\nRaw counts (cells x ripples), 100 ms windows:")
    print(raw)
    C, = swc.zscore_cells([raw])
    print("\nSame counts z-scored WITHIN each cell across all windows:")
    print(C)
    print("\n  Why: E(L) = sum_c n_c * z_c(L) weights a cell by its raw count,")
    print("  so a 10 Hz cell would shout over a 1 Hz cell. z-scoring per cell")
    print("  gives every cell the same vote.")

    print("\n" + "=" * 78)
    print("STEP 3 -- the dot product: evidence for each of the 9 locations")
    print("=" * 78)
    E = swc.score_windows(C, t_rip, rip_grid, loo)
    print("\nE (ripples x locations) = population count vector . template column")
    print(pd.DataFrame(E, columns=[f"L{i+1}" for i in range(9)],
                       index=[f"ripple {i} (at L{rip_loc[i]}, cfg {rip_grid[i]})"
                              for i in range(6)]).round(3).to_string())

    print("\n" + "=" * 78)
    print("STEP 4 -- the statistic: occupied location minus the other eight")
    print("=" * 78)
    li = rip_loc - 1
    ar = np.arange(len(E))
    h = E[ar, li]
    tot = E.sum(axis=1)
    stat = h - (tot - h) / 8.0
    print("\n  ripple   E[occupied]   mean E[other 8]   statistic")
    for i in range(6):
        print(f"    {i}       {h[i]:+8.3f}       {(tot[i]-h[i])/8:+8.3f}      "
              f"{stat[i]:+8.3f}")
    print(f"\n  session mean statistic = {np.nanmean(stat):+.4f}")
    print("\n  Note each row of E sums to ~0 (the templates do), so the")
    print("  statistic is just (9/8) x E[occupied]:")
    print(f"    (9/8) * E[occupied] = {9/8*h}")
    print(f"    statistic           = {stat}")

    print("\n" + "=" * 78)
    print("STEP 5 -- what 'finite evidence' means")
    print("=" * 78)
    print("\nE row-by-row, is every entry finite?")
    for i in range(6):
        print(f"  ripple {i} (cfg {rip_grid[i]}): "
              f"{'all finite' if np.isfinite(E[i]).all() else 'HAS NaN'}"
              f"   NaN at L{list(np.flatnonzero(~np.isfinite(E[i])) + 1)}")
    print("\n  Now the same session with location 7 occurring ONLY in cfg A,")
    print("  so holding A out leaves location 7 undefined:")
    occ2 = occ.copy()
    occ2.loc[(occ2.cv_group == "B") & (occ2["loc"] == 7), "loc"] = 3
    _, loo2 = swc.build_templates(spk, cells, occ2, grids)
    print(f"    cell 0 template, A held out: {loo2['A'][0]}")
    E2 = swc.score_windows(C, t_rip, rip_grid, loo2)
    for i in (0, 3):
        bad = list(np.flatnonzero(~np.isfinite(E2[i])) + 1)
        print(f"    ripple {i} (cfg {rip_grid[i]}): "
              f"{'all finite' if not bad else f'NaN at L{bad}'}")
    print("    -> cfg-A ripples are now unusable and drop out; cfg-B ones are")
    print("       unaffected, because their template holds B out, not A.")

    print("\n  It is NOT about spike count -- a ripple with zero spikes still")
    print("  gives a finite (uninformative) E. It is about the TEMPLATE: if a")
    print("  location was never visited outside the held-out configuration,")
    print("  its template entry is NaN and evidence for it is undefined.")

    print("\n" + "=" * 78)
    print("STEP 6 -- the null: permute the nine TEMPLATE LABELS")
    print("=" * 78)
    perms = [np.random.permutation(9) for _ in range(5)]
    print("\n  A permutation is a relabelling of the template columns, e.g.")
    print(f"    {perms[0]}  means 'pretend column {perms[0][0]+1} is location 1'")
    print("\n  Because E is already computed, the null needs NO recomputation:")
    print("  permuting template columns just REORDERS E's columns.")
    print("\n  observed statistic and 5 null statistics:")
    P = np.vstack([np.arange(9)] + perms)
    H = np.take_along_axis(E, P[:, li].T, axis=1)
    S = H - (E.sum(axis=1)[:, None] - H) / 8.0
    m = np.nanmean(S, axis=0)
    print(f"    observed = {m[0]:+.4f}")
    for k in range(1, 6):
        print(f"    null {k}   = {m[k]:+.4f}")
    print("\n  With the real 1000 permutations:")
    P = np.vstack([np.arange(9)] + [np.random.permutation(9) for _ in range(1000)])
    H = np.take_along_axis(E, P[:, li].T, axis=1)
    S = H - (E.sum(axis=1)[:, None] - H) / 8.0
    m = np.nanmean(S, axis=0)
    z = (m[0] - m[1:].mean()) / m[1:].std()
    print(f"    observed        = {m[0]:+.4f}")
    print(f"    null mean       = {m[1:].mean():+.4f}")
    print(f"    null SD         = {m[1:].std():.4f}")
    print(f"    z for this session = ({m[0]:+.4f} - {m[1:].mean():+.4f})"
          f" / {m[1:].std():.4f} = {z:+.3f}")
    print("\n  ONE z per session. Those z values are what the t-test across")
    print("  sessions is run on. There is no permutation at the group level.")

    print("\n" + "=" * 78)
    print("STEP 7 -- why the null SD shrinks with ripple count (the sqrt(n))")
    print("=" * 78)
    print("\n  The observed value is a MEAN over ripples, so its null narrows")
    print("  as more ripples are averaged. Simulated, same per-ripple effect:")
    rng = np.random.default_rng(0)
    for n in (20, 80, 320, 1280):
        nulls = rng.normal(0, 1, (2000, n)).mean(axis=1)
        print(f"    n = {n:5d} ripples -> null SD = {nulls.std():.4f}"
              f"   (1/sqrt(n) = {1/np.sqrt(n):.4f})")
    print("\n  z = (obs - null mean) / null SD, so HALVING the null SD DOUBLES")
    print("  z for the identical per-ripple effect. That is why a phase with")
    print("  twice as many ripples gets a bigger z for free, and why the two")
    print("  phases must be compared at the same n.")

    print("=" * 78)
    print("STEP 8 -- count matching, demonstrated")
    print("=" * 78)
    print("""
  Setup: simulated sessions with two phases carrying the IDENTICAL per-ripple
  effect (mean +0.20 per ripple, SD 1). The only difference between phases is
  how many ripples each has -- encoding 300, once-known 700 -- which is the
  structural imbalance in the real data.
""")
    rng = np.random.default_rng(1)
    eff, n_enc, n_exe = 0.20, 300, 700
    nperm, n_draw, n_sess = 200, 20, 100   # reduced for speed; real run is 1000

    def z_of_sample(x):
        """z exactly as the pipeline builds it: observed vs its OWN null."""
        null = rng.normal(0, 1, (nperm, len(x))).mean(axis=1)
        return (x.mean() - null.mean()) / null.std()

    unm, mat, sd_draw = [], [], []
    for _ in range(n_sess):
        enc = rng.normal(eff, 1, n_enc)
        exe = rng.normal(eff, 1, n_exe)
        unm.append(z_of_sample(enc) - z_of_sample(exe))
        n = min(n_enc, n_exe)
        d = np.array([z_of_sample(rng.choice(enc, n, replace=False))
                      - z_of_sample(rng.choice(exe, n, replace=False))
                      for _ in range(n_draw)])
        mat.append(d.mean())
        sd_draw.append(d.std())
    unm, mat = np.array(unm), np.array(mat)
    sd_draw = float(np.mean(sd_draw))
    print(f"  Averaged over {n_sess} simulated sessions, where the TRUE")
    print(f"  per-ripple effect is +{eff} in BOTH phases:\n")
    print(f"    UNMATCHED   mean z difference = {unm.mean():+.3f}"
          f"   (t = {stats.ttest_1samp(unm, 0)[0]:+.1f},"
          f" P = {stats.ttest_1samp(unm, 0)[1]:.1e})")
    print(f"    MATCHED     mean z difference = {mat.mean():+.3f}"
          f"   (t = {stats.ttest_1samp(mat, 0)[0]:+.1f},"
          f" P = {stats.ttest_1samp(mat, 0)[1]:.2f})")
    print(f"\n  Unmatched, the phase with more ripples wins by "
          f"{abs(unm.mean()):.1f} z for free, and it is hugely 'significant'.")
    print("  Matched, the difference is zero, which is the truth. Nothing")
    print("  about the signal changed -- only n.\n")
    print(f"  WITHIN-session SD of the contrast across draws = {sd_draw:.3f}")
    print(f"  Monte-Carlo error of a session's contrast at   20 draws: "
          f"{sd_draw / np.sqrt(20):.3f}")
    print(f"                                             at 1000 draws: "
          f"{sd_draw / np.sqrt(1000):.3f}")
    print(f"  (between-session s.e.m. here = {stats.sem(mat):.3f} -- at 20")
    print("   draws the Monte-Carlo noise is comparable to it, at 1000 it is")
    print("   negligible. Same argument as in the real data.)")
    print("  -- which is why the reported analysis uses 1000 draws.")
    print("""
  In words, per session and ROI:
    n         = min(n_encoding, n_once_known)
    draw 1    = a random n from each phase -> z_enc, z_known -> difference
    ...
    draw 1000 = another random n from each                  -> difference
    session contrast = mean of those 1000 differences

  "the full estimator, including its permutation null, was recomputed on each
  subsample" means each subsample's z comes from THAT subsample's own 1000
  permutation values (the `z_of_sample` call above) -- not from the full
  phase's null with a re-averaged observed part. Both phases then carry the
  same sqrt(n) term and it cancels in the difference.

  One contrast per session -> t-test across sessions against zero. That is
  the reported P value.
""")

    print("=" * 78)
    print("STEP 9 -- 'z-scored across all offsets pooled'")
    print("=" * 78)
    print("""
  For the TIME COURSE the window slides to 41 offsets (-500..+500 ms). At each
  offset we count spikes in a window of the ripple's own duration.

  If each offset's counts were z-scored separately, every offset would be
  forced to mean 0 and SD 1 -- and any real difference in firing between
  offsets would be normalised away before the estimator ever saw it.

  Pooling means: stack all 41 offsets' count matrices side by side, compute
  ONE mean and SD per cell over that whole stack, and use it everywhere. A
  cell that fires more at offset 0 than at offset -400 ms keeps that
  difference. So a change across the time course is a change in signal.
""")
    print("=" * 78)
    print("STEP 10 -- 'the same ripple set at every offset'")
    print("=" * 78)
    print("""
  A ripple is eligible only if its E is finite at EVERY one of the 41 offsets
  (`ok_all` in the code). Why it matters: if ripple 17 dropped out at offset
  +200 ms but not at offset 0, the two offsets would be averaging different
  ripples, and the curve could move between offsets purely because its
  membership changed. Requiring finiteness everywhere makes the curve a
  statement about the signal, at fixed membership.
""")


if __name__ == "__main__":
    main()
