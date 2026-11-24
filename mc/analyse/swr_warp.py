#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The bridge between the two time bases in this project.

The per-lag encoding pipeline lives in TIME-WARPED space: each of the four
subgoal intervals is linearly resampled to 90 bins, so one reward loop is 360
bins and 1 bin = 1 degree (`save_humanABCDneurons_normalised.m`). Ripples live
in REAL time, as seconds. A ripple therefore has no theta until one is computed,
which is what this module does.

    theta(t) = (state - 1) * 90 + 90 * (t - t_state_start)
                                     / (t_state_end - t_state_start)

using exactly the boundary set the MATLAB warp uses:
`[trial_start, t_A, t_B, t_C, t_D]`.

⚠ The warp assumes time runs linearly inside each subgoal interval. That
assumption is the accuracy limit on any fine lag distinction: 30 degrees is one
third of a subgoal interval (median 1.85 s), so ~0.6 s of real time. Check
`warp_diagnostics` before trusting 30 vs 60 degrees.

@author: Svenja Kuchenhoff
"""

import numpy as np
import pandas as pd

N_STATE = 4
BINS_PER_STATE = 90
N_BINS = N_STATE * BINS_PER_STATE      # 360


def trial_boundaries(steps_s):
    """One row per (grid_num, trial_num_in_grid): [start, t_A, t_B, t_C, t_D].

    Rows with a missing or non-monotonic boundary set are dropped -- they
    cannot be warped and silently keeping them would put ripples at wrong
    angles.
    """
    st = steps_s.sort_values("t_s")
    g = st.groupby(["grid_num", "trial_num_in_grid"], dropna=True)
    rows = []
    for (gn, tn), d in g:
        b = [d.t_s.min(), d.t_A.iloc[0], d.t_B.iloc[0],
             d.t_C.iloc[0], d.t_D.iloc[0]]
        if not np.all(np.isfinite(b)) or not np.all(np.diff(b) > 0):
            continue
        rows.append(dict(grid_num=gn, trial=tn, t0=b[0], tA=b[1], tB=b[2],
                         tC=b[3], tD=b[4], grid_id=d.grid_id.iloc[0]))
    return pd.DataFrame(rows)


def theta_of(t, bnd):
    """Warped angle (0-360) for real times `t`, or NaN outside any trial.

    `bnd` is the output of `trial_boundaries`. Also returns the row index of
    the trial each time fell in, so callers can recover that trial's layout.
    """
    t = np.asarray(t, float)
    theta = np.full(len(t), np.nan)
    which = np.full(len(t), -1, int)
    if not len(bnd):
        return theta, which
    edges = bnd[["t0", "tA", "tB", "tC", "tD"]].to_numpy(float)
    for i in range(len(bnd)):
        e = edges[i]
        m = (t >= e[0]) & (t <= e[-1])
        if not m.any():
            continue
        tt = t[m]
        # which subgoal interval, then the fraction through it
        k = np.clip(np.searchsorted(e, tt, side="right") - 1, 0, N_STATE - 1)
        frac = (tt - e[k]) / (e[k + 1] - e[k])
        theta[m] = k * BINS_PER_STATE + frac * BINS_PER_STATE
        which[m] = i
    return theta, which


def location_at_theta(theta, which, bnd, steps_s, lag_deg):
    """The square occupied at (theta + lag) within the SAME trial.

    Returns -1 where undefined. This is what a lag-shifted template has to be
    scored against: the template says "this cell fires when the location
    `lag_deg` ahead is L", so the target label is the location actually there.
    """
    st = steps_s.sort_values("t_s")
    out = np.full(len(theta), -1, int)
    edges = bnd[["t0", "tA", "tB", "tC", "tD"]].to_numpy(float)
    t_all = st.t_s.to_numpy(float)
    loc_all = st["loc"].to_numpy(float)
    for i in range(len(bnd)):
        m = (which == i) & np.isfinite(theta)
        if not m.any():
            continue
        e = edges[i]
        th = (theta[m] + lag_deg) % N_BINS        # circular, as np.roll is
        k = np.clip((th // BINS_PER_STATE).astype(int), 0, N_STATE - 1)
        frac = (th - k * BINS_PER_STATE) / BINS_PER_STATE
        t_target = e[k] + frac * (e[k + 1] - e[k])
        # the square held at that moment = last step at or before it
        j = np.searchsorted(t_all, t_target, side="right") - 1
        ok = j >= 0
        v = np.full(len(th), -1, int)
        v[ok] = loc_all[j[ok]].astype(int)
        out[m] = v
    return out


def warp_diagnostics(steps_s, bnd):
    """How badly does 'time is linear within a subgoal' fail?

    Compares each step's true position in its subgoal interval against the
    position the warp assigns it. Large error means fine lag bins are not
    trustworthy.
    """
    st = steps_s.sort_values("t_s")
    th, which = theta_of(st.t_s.to_numpy(float), bnd)
    m = np.isfinite(th)
    if not m.any():
        return {}
    # steps are the events; within a subgoal they should tile it roughly evenly
    # if behaviour were uniform. The spread of step-to-step theta increments is
    # the practical resolution limit.
    d = np.diff(th[m])
    d = d[(d > 0) & (d < BINS_PER_STATE)]
    return dict(n_steps=int(m.sum()),
                median_step_deg=float(np.median(d)) if len(d) else np.nan,
                p90_step_deg=float(np.percentile(d, 90)) if len(d) else np.nan)


def subgoal_intervals(steps_s):
    """(start, stop) of every subgoal interval: trial start->A->B->C->D.

    The unit that holds the PLAN constant, which is what a plan-loading test
    has to match on. Median 1.85 s, versus 0.45 s for one occupancy interval --
    which is why a pre/post window fits here and does not fit inside a visit.
    """
    bnd = trial_boundaries(steps_s)
    if not len(bnd):
        return np.empty((0, 2))
    e = bnd[["t0", "tA", "tB", "tC", "tD"]].to_numpy(float)
    return np.column_stack([e[:, :-1].ravel(), e[:, 1:].ravel()])


def prepost_windows(t_rip, half, steps_s, other_t=None, win_s=0.25,
                    gap_s=0.05):
    """Symmetric pre/post windows around each ripple, inside ONE subgoal.

    Returns (pre_lo, pre_hi, post_lo, post_hi, ok). `ok` is False where either
    side does not fit inside the ripple's own subgoal interval, or where a
    window would overlap another detected ripple -- in which case that ripple
    contributes nothing rather than contributing an unmatched single side.

    Both sides are required so the test is PAIRED within ripple: an unpaired
    post-window would let sampling differences between the two sides masquerade
    as a step.
    """
    t_rip = np.asarray(t_rip, float)
    half = np.broadcast_to(np.asarray(half, float), t_rip.shape)
    seg = subgoal_intervals(steps_s)
    other = np.sort(np.asarray(other_t, float)) if other_t is not None \
        else np.empty(0)

    pre_lo = t_rip - half - gap_s - win_s
    pre_hi = t_rip - half - gap_s
    post_lo = t_rip + half + gap_s
    post_hi = t_rip + half + gap_s + win_s
    ok = np.zeros(len(t_rip), bool)
    if not len(seg):
        return pre_lo, pre_hi, post_lo, post_hi, ok
    lo = np.searchsorted(seg[:, 0], t_rip, side="right") - 1
    lo = np.clip(lo, 0, len(seg) - 1)
    inside = (t_rip >= seg[lo, 0]) & (t_rip <= seg[lo, 1])
    fits = inside & (pre_lo >= seg[lo, 0]) & (post_hi <= seg[lo, 1])
    ok = fits.copy()
    if len(other):
        for i in np.flatnonzero(ok):
            near = other[(other > pre_lo[i] - .2) & (other < post_hi[i] + .2)]
            near = near[np.abs(near - t_rip[i]) > 1e-9]     # not itself
            if np.any((near > pre_lo[i]) & (near < pre_hi[i])) or \
               np.any((near > post_lo[i]) & (near < post_hi[i])):
                ok[i] = False
    return pre_lo, pre_hi, post_lo, post_hi, ok
