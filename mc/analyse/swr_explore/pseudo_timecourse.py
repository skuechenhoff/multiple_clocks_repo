#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What is in a ripple, resolved in TIME, at the pseudo-population level.

    python scripts/swr_pseudo_timecourse.py

Every test in this line so far -- C0, C1, C2, the 9-location profile, I11, I12,
I13 -- averages over the whole ripple window and asks whether a square is
represented. All of them say: the current square is, ripples are not special
about it, and nothing prospective shows up. But a window average is blind to
anything that HAPPENS during a ripple. If hippocampus sweeps from where you are
to where you are going, the average over the sweep is elevated everywhere and
therefore looks like nothing.

This asks the time-resolved question, and it can only be asked of a
pseudo-population: a single session has ~5 cells and 0.2 spikes per cell per
ripple, which cannot support a decode in 50 ms bins. Pooled across the sessions
that ran the same configuration there are up to 316.

THE ESTIMATOR IS THE DETERMINISTIC LIMIT OF THE PSEUDO-TRIAL

`swr_pseudo_population.py` draws one ripple per cell and stacks them. Averaging
many such draws converges on the cell's mean, so this script goes straight
there: for a configuration, an offset from the ripple peak and a square,

    mu_c(L, t) = that cell's mean z-scored count in a 50 ms window at offset t,
                 over its own session's ripples that happened at square L
    E(L', t)   = sum_c mu_c(L, t) * z_c(L')

with each cell's template estimated with its own session's runs of the tested
configuration held out. No sampling, so no draw noise and no seed to report.
The read-out is the within-row z of the true square -- the same `target_z` used
everywhere else, and validated against the sampled version (diagonal +0.144 in
both).

THREE GROUPINGS, because "what is in the ripple" has three candidate answers:

  current    ripples grouped by the square the subject is standing on
  goal       grouped by the square they are trying to REACH (`looking_for_loc`)
  next_step  grouped by the square they step to next

Each is contaminated by the others to the extent the roles correlate, which is
why the window-level version of this question is run as a regression in
`swr_ripple_content_roles.py`. Here they are separate curves, and the
interesting thing is their SHAPE in time, not their level.

⚠ WHAT THIS CAN AND CANNOT SEE. Each cell's bin comes from a different ripple,
so only structure that is LOCKED TO THE RIPPLE survives the averaging. A
stereotyped sweep -- current square first, goal later, on every ripple -- would
show. Classical rodent replay, which starts at a random place and runs in a
random direction on each event, would average away. This is a test of ripple-
locked broadcasting, which is the hypothesis this project actually holds; it is
not a test of random-start replay, and a null here does not exclude one.

Null: the nine template labels are permuted, through the same scoring path.
Unit of inference: the CONFIGURATION. They share sessions, so the t across them
is anticonservative and the sign count is reported beside it.

@author: Svenja Kuchenhoff
"""

import os
import json
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.ripple_rsa as rrsa
import mc.analyse.swr_location as swl
import mc.analyse.swr_content as swc
from mc.analyse.swr_explore.pseudo_population import config_key

SEED = 42
N_PERM = 200
WIN_S = 0.050
OFFSETS = np.round(np.arange(-0.50, 0.5001, 0.025), 4)
MIN_SESSIONS = 10
MIN_EVENTS = 3
GROUPINGS = ("current", "goal", "next_step")
ROI_SETS = {"HC_all": ["HC_anterior", "HC_mid"],
            "HC_mid": ["HC_mid"], "HC_anterior": ["HC_anterior"],
            "mOFC": ["mOFC"], "mPFC": ["mPFC"]}
LOC = np.arange(1, 10)


def collect(sessions, spk, roi, steps, rip, keys):
    """pool[roi][config] = list of per-cell records.

    A record holds the cell's leave-this-configuration-out template and, per
    grouping and square, the mean z-scored count at every offset.
    """
    pool = {r: defaultdict(list) for r in ROI_SETS}
    for s in sessions:
        if s not in spk:
            continue
        occ, r, t_rip, d_rip = swc.session_data(s, spk, roi, steps, rip)
        if not len(occ) or not len(r) or not len(t_rip):
            continue
        st = steps[steps.session == s].sort_values("t_s").reset_index(drop=True)
        o = occ.sort_values("start_s")
        row_of = o.index.to_numpy()
        a, b = o.start_s.to_numpy(), o.stop_s.to_numpy()
        j = np.searchsorted(a, t_rip, side="right") - 1
        jj = np.clip(j, 0, len(o) - 1)
        k = row_of[jj]
        inside = (j >= 0) & (t_rip <= b[jj])
        if inside.sum() < 30:
            continue
        t_rip, k = t_rip[inside], k[inside]
        si = occ.step_idx.to_numpy()[k]
        grid = occ.cv_group.to_numpy()[k]
        lab = {"current": occ["loc"].to_numpy()[k],
               "goal": st.looking_for_loc.to_numpy(float)[si],
               "next_step": st.next_loc.to_numpy(float)[si]}
        grids = np.unique(occ.cv_group.to_numpy())
        cfg = np.array([keys.get((s, g), "") for g in grid])

        for roiname, members in ROI_SETS.items():
            cells = r[r.roi.isin(members)].cell.to_numpy()
            cells = [c for c in cells if c < len(spk[s]["spikes"])]
            if not cells:
                continue
            _, loo = swc.build_templates(spk[s], cells, occ, grids)
            # every offset z-scored on ONE pooled scale per cell, so a
            # difference between offsets is a real difference and not a
            # by-product of normalising each offset separately
            C = np.stack([swc.window_counts(spk[s], cells,
                                            t_rip + off - WIN_S / 2,
                                            t_rip + off + WIN_S / 2)
                          for off in OFFSETS])              # (n_off, n_c, n_rip)
            mu = C.mean(axis=(0, 2), keepdims=True)
            sd = C.std(axis=(0, 2), keepdims=True)
            C = (C - mu) / np.where(sd > 0, sd, np.nan)

            for g in grids:
                key = keys.get((s, g), "")
                T = loo.get(g)
                m0 = grid == g
                if not key or T is None or m0.sum() < MIN_EVENTS:
                    continue
                for ci in range(len(cells)):
                    if not np.isfinite(T[ci]).all() or not np.isfinite(C[:, ci]).any():
                        continue
                    prof = {}
                    for gr in GROUPINGS:
                        P = np.full((9, len(OFFSETS)), np.nan)
                        for L in LOC:
                            m = m0 & (lab[gr] == L)
                            if m.sum() >= MIN_EVENTS:
                                P[L - 1] = np.nanmean(C[:, ci, m], axis=1)
                        if np.isfinite(P).any():
                            prof[gr] = P
                    if prof:
                        pool[roiname][key].append(
                            dict(session=s, template=T[ci], prof=prof))
        print(f"  s{s:02d}", flush=True)
    return pool


def curve(recs, gr, perms):
    """(n_offsets,) target_z through time, plus the null, for one configuration.

    A cell with no data for a square contributes 0 there -- the neutral value
    for a z-scored count, which scores exactly zero evidence, so it abstains.
    """
    T = np.array([rc["template"] for rc in recs])              # (n_c, 9)
    have = [i for i, rc in enumerate(recs) if gr in rc["prof"]]
    if len(have) < 5:
        return None
    M = np.zeros((9, len(OFFSETS), len(recs)))
    seen = np.zeros((9, len(recs)), bool)
    for i in have:
        P = recs[i]["prof"][gr]
        ok = np.isfinite(P).all(axis=1)
        M[ok, :, i] = P[ok]
        seen[ok, i] = True
    keep = seen.any(axis=1)
    if keep.sum() < 4:
        return None

    out = np.full((len(OFFSETS), 1 + len(perms)), np.nan)
    for oi in range(len(OFFSETS)):
        E = M[:, oi, :] @ T                                    # (9 true, 9 cand)
        mu = E.mean(axis=1, keepdims=True)
        sd = E.std(axis=1, keepdims=True)
        Z = (E - mu) / np.where(sd > 0, sd, np.nan)
        rows = np.flatnonzero(keep)
        for pi, pm in enumerate([np.arange(9)] + list(perms)):
            out[oi, pi] = np.nanmean(Z[rows, pm[rows]])
    return out


def main():
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(9) for _ in range(N_PERM)]
    deriv = rrsa._derivatives()
    rip = pd.read_csv(os.path.join(deriv, "group", "swr", "bundle_v2",
                                   "ripples.csv"),
                      usecols=["session", "t_peak_s", "duration_s", "subject_key"])
    sessions = sorted(rip.session.unique().tolist())
    spk = rrsa.load_spike_times(sessions=sessions, verbose=False)
    roi = rrsa.cell_roi_table(sessions=sessions)
    steps = swl.load(sessions)
    keys = config_key(steps)

    print("collecting ...")
    pool = collect(sessions, spk, roi, steps, rip, keys)

    rows = []
    for roiname in ROI_SETS:
        for key, recs in pool[roiname].items():
            if len({rc["session"] for rc in recs}) < MIN_SESSIONS:
                continue
            for gr in GROUPINGS:
                c = curve(recs, gr, perms)
                if c is None:
                    continue
                nul = c[:, 1:]
                sd = np.nanstd(nul, axis=1)
                for oi, off in enumerate(OFFSETS):
                    rows.append(dict(
                        roi=roiname, config=key, grouping=gr, offset_s=off,
                        n_cells=len(recs), target_z=c[oi, 0],
                        null_mean=float(np.nanmean(nul[oi])),
                        z=(c[oi, 0] - np.nanmean(nul[oi])) / sd[oi]
                        if sd[oi] > 0 else np.nan))
        print(f"  {roiname} done", flush=True)

    R = pd.DataFrame(rows)
    d = os.path.join(deriv, "group", "swr",
                     f"ripple_pseudo_timecourse_{datetime.date.today()}")
    os.makedirs(d, exist_ok=True)
    R.to_csv(os.path.join(d, "timecourse.csv"), index=False)
    report(R, d)
    json.dump(dict(seed=SEED, n_perm=N_PERM, win_s=WIN_S,
                   offsets_s=OFFSETS.tolist(), groupings=list(GROUPINGS),
                   min_sessions=MIN_SESSIONS, roi_sets=ROI_SETS,
                   estimator="deterministic limit of the pseudo-trial: per-cell "
                             "mean z-count per square per offset, scored "
                             "against leave-this-configuration-out templates",
                   caveat="only ripple-LOCKED structure survives pseudo-"
                          "population averaging; random-start replay would not",
                   created=datetime.datetime.now().isoformat(timespec="seconds")),
              open(os.path.join(d, "settings.json"), "w"), indent=2)
    print(f"\n-> {d}")


def report(R, d):
    print("\n=== location signal through the peri-ripple second ===")
    print("    target_z at the ripple peak, against the surrounding baseline\n")
    base = (np.abs(R.offset_s) >= 0.3)
    summ = []
    for roiname in ROI_SETS:
        for gr in GROUPINGS:
            g = R[(R.roi == roiname) & (R.grouping == gr)]
            if g.config.nunique() < 5:
                continue
            pk = g[np.abs(g.offset_s) <= 0.0501].groupby("config").target_z.mean()
            bl = g[base.loc[g.index]].groupby("config").target_z.mean()
            both = pd.concat([pk.rename("peak"), bl.rename("base")], axis=1).dropna()
            if len(both) < 5:
                continue
            t, p = stats.ttest_rel(both.peak, both.base)
            zz = g[np.abs(g.offset_s) <= 0.0501].groupby("config").z.mean()
            pz = stats.ttest_1samp(zz.dropna(), 0)[1]
            print(f"  {roiname:12s} {gr:10s} n = {len(both):2d} configs | "
                  f"peak {both.peak.mean():+.3f} base {both.base.mean():+.3f} "
                  f"diff {(both.peak - both.base).mean():+.3f} p = {p:.3g} | "
                  f"peak z vs null {zz.mean():+.2f} p = {pz:.3g} | "
                  f"{int((both.peak > both.base).sum())}/{len(both)} up")
            summ.append(dict(roi=roiname, grouping=gr, n_configs=len(both),
                             peak=float(both.peak.mean()),
                             base=float(both.base.mean()),
                             diff=float((both.peak - both.base).mean()),
                             p=float(p), peak_z=float(zz.mean()), p_z=float(pz)))
        print()
    pd.DataFrame(summ).to_csv(os.path.join(d, "timecourse_summary.csv"),
                              index=False)


if __name__ == "__main__":
    main()
