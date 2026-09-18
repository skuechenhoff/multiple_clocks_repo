#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What do hippocampal ripples CONTAIN? Shared machinery for the content analysis.

THE ESTIMATOR
-------------
Each cell gets a location template `z_c(L)`, L = 1..9: its mean firing rate at
each location, z-scored ACROSS locations so that `sum_L z_c(L) = 0`. Any window
of spiking then scores every location:

    E(L) = sum_c  n_c(window) * z_c(L)

Because every template sums to zero across locations, firing that is unrelated
to location contributes EXACTLY ZERO in expectation to every E(L). The null is
built into the estimator rather than bolted on, which is what makes a session
with four cells still contribute an unbiased score.

The score is then read relative to what the location MEANS on that trial (the
current location, a reward location, an irrelevant one), so sessions are
commensurable without ever building a pseudo-population.

WHY LEAVE-ONE-GRID-OUT
----------------------
A template built from all the data includes the very windows being tested. The
contamination is small (a 60 ms window inside a per-location average of many
minutes) but it is free to remove: templates for grid g are built from every
grid EXCEPT g. That covers the ripple windows and their flanks in one move, so
no ripple-proximity exclusion is needed.

WINDOW WIDTH IS PART OF THE ESTIMATOR
-------------------------------------
`E` scales with the number of spikes, so a control window must have the SAME
width as the test window or the comparison is meaningless. This is the
count-vs-window-length trap that `SWR_SUMMARY.md` §5 records twice.

@author: Svenja Kuchenhoff
"""

import numpy as np
import pandas as pd

LOCATIONS = np.arange(1, 10)
MAX_STEP_S = 10.0      # an "occupancy interval" longer than this is a task gap


def occupancy(steps, session):
    """Location occupancy intervals for one session.

    Returns start_s, stop_s, loc, grid_num, grid_id, cv_group. The subject sits
    at `loc` from the move's timestamp until the next move, capped at the end of
    its own trial so that between-trial gaps are not credited to a location.

    ⚠ `cv_group` is `grid_id`, NOT `grid_num`, and every cross-validation must
    split on it. A configuration is run in several blocks -- median 3 runs per
    `grid_id`, up to 5 -- so holding out a single `grid_num` leaves other runs of
    the SAME configuration in the training set. The template then already
    encodes that configuration's reward locations, including ones the subject
    has not yet discovered at ripple time. Leaving out the whole set is the only
    honest split.
    """
    s = steps[steps.session == session].sort_values("t_s").reset_index(drop=True)
    if not len(s):
        return pd.DataFrame(columns=["start_s", "stop_s", "loc", "grid_num"])
    start = s.t_s.to_numpy(float)
    nxt = np.append(start[1:], np.inf)
    stop = np.minimum(nxt, s.trial_end_s.to_numpy(float))
    out = pd.DataFrame({"start_s": start, "stop_s": stop,
                        "loc": s["loc"].to_numpy(float),
                        "grid_num": s.grid_num.to_numpy(float),
                        "grid_id": s.grid_id.to_numpy(float),
                        "cv_group": s.grid_id.to_numpy(float),
                        # row in the time-sorted step table this interval came
                        # from, so callers can recover per-step fields
                        # (`is_uncover`, `correct_uncover`, `first_trial`, ...)
                        # after the filtering below drops rows
                        "step_idx": np.arange(len(s))})
    out = out[(out.stop_s > out.start_s) & np.isfinite(out.stop_s)]
    return out[(out.stop_s - out.start_s) <= MAX_STEP_S].reset_index(drop=True)


def count_in(spike_times, start, stop):
    """Spikes of one cell inside each [start, stop) interval."""
    st = np.asarray(spike_times, float)
    return (np.searchsorted(st, np.asarray(stop, float))
            - np.searchsorted(st, np.asarray(start, float)))


def place_map(spike_times, occ, exclude_grid=None):
    """(9,) firing rate per location for one cell, or NaN where unvisited.

    `exclude_grid` is a `cv_group` (= grid_id) value and drops EVERY run of that
    configuration -- see `occupancy` for why grid_num is the wrong unit.
    """
    o = occ if exclude_grid is None else occ[occ.cv_group != exclude_grid]
    if not len(o):
        return np.full(9, np.nan)
    n = count_in(spike_times, o.start_s.to_numpy(), o.stop_s.to_numpy())
    dur = (o.stop_s - o.start_s).to_numpy()
    loc = o["loc"].to_numpy()
    out = np.full(9, np.nan)
    for i, L in enumerate(LOCATIONS):
        m = loc == L
        if m.any() and dur[m].sum() > 0:
            out[i] = n[m].sum() / dur[m].sum()
    return out


def zscore_map(m):
    """Template: z-scored across locations so it sums to zero. NaN if flat."""
    m = np.asarray(m, float)
    ok = np.isfinite(m)
    if ok.sum() < 3:
        return np.full(9, np.nan)
    mu, sd = np.nanmean(m), np.nanstd(m)
    if not np.isfinite(sd) or sd == 0:
        return np.full(9, np.nan)
    z = np.full(9, np.nan)
    z[ok] = (m[ok] - mu) / sd
    z[ok] -= z[ok].mean()          # enforce sum-to-zero on the observed entries
    return z


def matched_flanks(t_rip, half, occ, other_t=None, gap_s=0.05, max_per=2):
    """Flank windows matched to each ripple on LOCATION, OCCUPANCY and WIDTH.

    A fixed +-500 ms flank is NOT location-matched -- measured, it sits at the
    same location as the ripple only ~53.7% of the time, and when it does not it
    is scored for the location the subject held at the ripple, i.e. the wrong
    one. Its evidence is then depressed for a reason having nothing to do with
    ripples, which manufactures a positive ripple-minus-flank difference
    (CHANGELOG 2026-09-17 h).

    Here each flank is placed INSIDE the ripple's own occupancy interval, so it
    shares the location, the dwell and the task phase by construction, and has
    the same width. Up to `max_per` flanks per ripple (one before, one after
    where the interval allows), centred in the free space, separated from the
    ripple by `gap_s`, and never overlapping another ripple in `other_t`.

    Returns (centres, owner) where `centres[k]` belongs to ripple `owner[k]`.
    Ripples with no room contribute nothing and simply drop out.
    """
    t_rip = np.asarray(t_rip, float)
    half = np.broadcast_to(np.asarray(half, float), t_rip.shape)
    o = occ.sort_values("start_s")
    a, b = o.start_s.to_numpy(float), o.stop_s.to_numpy(float)
    j = np.searchsorted(a, t_rip, side="right") - 1
    jj = np.clip(j, 0, len(a) - 1)
    inside = (j >= 0) & (t_rip <= b[jj])
    other = np.sort(np.asarray(other_t, float)) if other_t is not None \
        else np.empty(0)

    centres, owner = [], []
    for i in np.flatnonzero(inside):
        w, lo, hi = half[i], a[jj[i]], b[jj[i]]
        # free space before and after the ripple, inside its own interval
        for seg_lo, seg_hi in ((lo, t_rip[i] - w - gap_s),
                               (t_rip[i] + w + gap_s, hi)):
            if seg_hi - seg_lo < 2 * w:
                continue
            c = 0.5 * (seg_lo + seg_hi)          # centred in the free space
            if other.size:
                k = np.searchsorted(other, c)
                near = False
                for kk_ in (k - 1, k):
                    if 0 <= kk_ < other.size and abs(other[kk_] - c) < 2 * w + gap_s:
                        near = True
                if near:
                    continue
            centres.append(c)
            owner.append(i)
            if sum(1 for x in owner if x == i) >= max_per:
                break
    return np.asarray(centres, float), np.asarray(owner, int)


def zscore_cells(count_sets):
    """z-score each cell's counts so every cell carries the pattern equally.

    `E(L) = sum_c n_c * z_c(L)` weights each cell by its RAW spike count, so a
    10 Hz cell contributes ten times more to the population read-out than a
    1 Hz one and a few loud cells dominate. z-scoring per cell is the standard
    fix (and is what the project's single-spike RSA already does).

    Measured effect on the Stage 2 positive control at 60 ms: HC_mid +0.513 ->
    +0.678, and **HC_anterior +0.108 (p = 0.48) -> +0.517 (p = 0.0011)** -- a
    region that looked location-blind was simply being swamped.

    `count_sets` is a list of (n_cells, n_windows) arrays sharing the same cells
    (e.g. ripple and its flanks). Statistics are pooled ACROSS the sets, so a
    genuine firing difference between them -- ripples really do carry ~4% more
    spikes -- is preserved rather than normalised away.

    The null property survives: sum_L z_c(L) = 0 holds per cell however the
    counts are scaled, so the estimator stays unbiased.
    """
    pooled = np.concatenate([np.asarray(c, float) for c in count_sets], axis=1)
    mu = pooled.mean(axis=1, keepdims=True)
    sd = pooled.std(axis=1, keepdims=True)
    ok = sd[:, 0] > 0
    out = []
    for c in count_sets:
        z = np.zeros_like(np.asarray(c, float))
        z[ok] = (np.asarray(c, float)[ok] - mu[ok]) / sd[ok]
        out.append(z)
    return out


def evidence(counts, templates):
    """E(L) for one window. `counts` (n_cells,), `templates` (n_cells, 9)."""
    counts = np.asarray(counts, float)[:, None]
    T = np.asarray(templates, float)
    ok = np.isfinite(T).all(1) & np.isfinite(counts[:, 0])
    if not ok.any():
        return np.full(9, np.nan)
    return (counts[ok] * T[ok]).sum(0)


def evidence_matrix(counts, templates):
    """E for many windows at once. `counts` (n_cells, n_windows) -> (n_windows, 9)."""
    C = np.asarray(counts, float)
    T = np.asarray(templates, float)
    ok = np.isfinite(T).all(1)
    if not ok.any():
        return np.full((C.shape[1], 9), np.nan)
    return C[ok].T @ T[ok]


def target_minus_others(E, target_loc):
    """Score for the location of interest minus the mean of the other eight.

    The quantity the tests are built on. Zero in expectation whenever firing is
    location-independent, for any number of cells and any firing rate.
    """
    E = np.atleast_2d(np.asarray(E, float))
    idx = np.asarray(target_loc, int) - 1
    ok = (idx >= 0) & (idx < 9) & np.isfinite(E).all(1)
    out = np.full(E.shape[0], np.nan)
    if ok.any():
        hit = E[ok, idx[ok]]
        tot = E[ok].sum(1)
        out[ok] = hit - (tot - hit) / 8.0
    return out


def tiled_windows(occ, width_s, avoid=None, avoid_pad_s=1.0):
    """EVERY non-overlapping window of `width_s` inside task time.

    `random_windows` draws a subsample, and the resulting estimate moves with
    the seed: HC_mid at 60 ms came out at +0.30, +0.44 and +0.53 across three
    runs that differed only in the draw (CHANGELOG 2026-09-17 c/d). Tiling is
    deterministic and uses ~10x more windows, so the estimate stops depending on
    a random choice that has nothing to do with the data.
    """
    o = occ[(occ.stop_s - occ.start_s) >= width_s]
    if not len(o):
        return np.empty(0), np.empty(0)
    starts, locs = [], []
    for st, sp, L in zip(o.start_s.to_numpy(), o.stop_s.to_numpy(),
                         o["loc"].to_numpy()):
        k = int((sp - st) // width_s)
        if k > 0:
            starts.append(st + np.arange(k) * width_s)
            locs.append(np.full(k, L))
    if not starts:
        return np.empty(0), np.empty(0)
    t0 = np.concatenate(starts)
    loc = np.concatenate(locs)
    if avoid is not None and len(avoid):
        a = np.sort(np.asarray(avoid, float))
        j = np.searchsorted(a, t0)
        near = np.zeros(len(t0), bool)
        for jj in (j - 1, j):
            v = np.clip(jj, 0, len(a) - 1)
            near |= np.abs(a[v] - t0) < (avoid_pad_s + width_s)
        t0, loc = t0[~near], loc[~near]
    return t0, loc


def target_z(E, target_loc):
    """Scale-free version of `target_minus_others`: z of the target location
    within the window's own 9 scores.

    `target_minus_others` inherits the window's scale, so raw scores cannot be
    averaged across sessions -- they carry each session's firing rate and cell
    count (|score| vs n_cells: r = 0.49). Dividing by the SD across the 9
    locations removes that at the WINDOW level, so session means become
    comparable without a per-session permutation null. Windows whose nine scores
    are degenerate (sd = 0, e.g. no spikes at all) return NaN.
    """
    E = np.atleast_2d(np.asarray(E, float))
    idx = np.asarray(target_loc, int) - 1
    out = np.full(E.shape[0], np.nan)
    ok = (idx >= 0) & (idx < 9) & np.isfinite(E).all(1)
    if not ok.any():
        return out
    sub = E[ok]
    sd = sub.std(axis=1)
    good = sd > 0
    vals = np.full(sub.shape[0], np.nan)
    vals[good] = (sub[good, idx[ok][good]] - sub[good].mean(axis=1)) / sd[good]
    out[ok] = vals
    return out


def random_windows(occ, width_s, n, rng, avoid=None, avoid_pad_s=1.0):
    """Uniformly sampled windows inside task time, optionally avoiding events.

    Used for the positive control: the same estimator on ordinary navigation, at
    the SAME window width as a ripple, so the two are directly comparable.
    """
    o = occ[(occ.stop_s - occ.start_s) > width_s]
    if not len(o):
        return np.empty(0), np.empty(0)
    dur = (o.stop_s - o.start_s - width_s).to_numpy()
    p = dur / dur.sum()
    k = rng.choice(len(o), size=n, p=p)
    t0 = o.start_s.to_numpy()[k] + rng.random(n) * dur[k]
    loc = o["loc"].to_numpy()[k]
    if avoid is not None and len(avoid):
        a = np.sort(np.asarray(avoid, float))
        j = np.searchsorted(a, t0)
        near = np.zeros(n, bool)
        for jj in (j - 1, j):
            v = np.clip(jj, 0, len(a) - 1)
            near |= np.abs(a[v] - t0) < (avoid_pad_s + width_s)
        t0, loc = t0[~near], loc[~near]
    return t0, loc
