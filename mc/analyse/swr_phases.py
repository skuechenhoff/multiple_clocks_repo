#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The task phase definitions, and the helpers every phase analysis shares.

THE AGREED DEFINITION (fixed 2026-09-27). With `fs` = a grid's first fully
correct (error-free) repeat and `r0` its first repeat:

    encoding   rep <  fs     every traversal before the first correct loop
    execution  rep >= fs     the first correct loop and everything after

This is `swr_windows.add_phase3` collapsed to two. It is NOT
`ripples.stage_of_repeat`, which puts the first fully correct repeat on the
ENCODING side -- the two disagree about 1432 repeats, 39% of encoding. Both
labellings are provided here (`stage` is the inclusive one the bundle already
carries; `add_exclusive_stage` adds the agreed one as `stage_exc`) so that the
difference between them stays testable rather than hidden.

This module exists so that the reported analysis
(`scripts/swr_figure_phase_panels.py`) does not import the exploratory sweep
(`scripts/exploratory/swr_robustness_grid.py`). Both import from here.

@author: Svenja Kuchenhoff
"""

import numpy as np
import pandas as pd

import mc.analyse.ripples as rip

VALENCES = ('correct', 'error')
BASE_A = (-1.6, -1.1)      # Sakon Eq. 2 as originally implemented
BASE_B = (-1.5, -1.0)      # the agreed baseline window


# ── The eight phase sets, as queries on the press table. `stage` is the
# INCLUSIVE labelling the bundle already carries; `stage_exc` is added by
# `add_exclusive_stage` below. Deduplicated on the query string, so `first`
# and `later` are rated once and shared by both boundaries.
PHASE_Q = {
    'first':     "stage == 'first uncovers'",
    'later':     "stage != 'first uncovers'",
    'mid_inc':   "stage == 'while learning'",
    'known_inc': "stage == 'once known'",
    'learn_inc': "stage != 'once known'",
    'mid_exc':   "stage_exc == 'while learning'",
    'known_exc': "stage_exc == 'once known'",
    'learn_exc': "stage_exc != 'once known'",
}
PHASE_DEF = {
    'first':     'rep == r0            (the first traversal)',
    'later':     'rep >  r0            (everything after the first traversal)',
    'mid_inc':   'r0 < rep <= fs       (while learning, first solve INCLUDED)',
    'known_inc': 'rep >  fs            (once known, first solve EXCLUDED)',
    'learn_inc': 'rep <= fs            (first + while learning, solve INCLUDED)',
    'mid_exc':   'r0 < rep <  fs       (while learning, first solve EXCLUDED)',
    'known_exc': 'rep >= fs            (once known, first solve INCLUDED)',
    'learn_exc': 'rep <  fs            (first + while learning, solve EXCLUDED)',
}
PHASE_SHORT = {'first': 'first', 'later': 'later', 'mid_inc': 'learning',
               'known_inc': 'known', 'mid_exc': 'learning', 'known_exc': 'known',
               'learn_inc': 'learning', 'learn_exc': 'learning'}

BOUNDARIES = {
    'inclusive': 'first fully correct repeat -> LEARNING  (rip.stage_of_repeat)',
    'exclusive': 'first fully correct repeat -> EXECUTION (swr_windows.add_phase3)',
}
GROUPINGS = {
    ('inclusive', 'three_phase'):       ['first', 'mid_inc', 'known_inc'],
    ('inclusive', 'first_vs_later'):    ['first', 'later'],
    ('inclusive', 'learning_vs_known'): ['learn_inc', 'known_inc'],
    ('exclusive', 'three_phase'):       ['first', 'mid_exc', 'known_exc'],
    ('exclusive', 'first_vs_later'):    ['first', 'later'],
    ('exclusive', 'learning_vs_known'): ['learn_exc', 'known_exc'],
}

def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running = np.empty(p.size), 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def base(profiles, centres, win):
    """Sakon Eq. 2 at an explicit window -- `fin.baseline` hard-codes A."""
    units, X = rip.baseline_subtract(profiles, centres, baseline=win)
    return {u: X[i] for i, u in enumerate(units)}


# ══ 1) The exclusive stage labelling ══════════════════════════════════

def add_exclusive_stage(data, presses):
    """Add `stage_exc`: `stage` with the first fully correct repeat moved.

    `rip.stage_of_repeat` puts that repeat in `while learning` (rep <= fs);
    this labels it `once known` (rep >= fs), which is what
    `swr_windows.add_phase3` does. Built from the bundle's behaviour table and
    merged onto the presses the same way `fin.build_press_table` merged
    `stage` -- nearest uncovering within 30 ms -- so the two columns are
    attached to exactly the same press rows.
    """
    beh = data['behaviour']
    frames = []
    for session in rip.sessions_in(data):
        tab = rip.uncover_table(data, session)
        if not len(tab) or 'rep_overall' not in tab:
            continue
        b = beh[beh.session == session]
        lab = {}
        for grid, g in b.groupby('grid_no'):
            g = g.sort_values('rep_overall')
            reps = g.rep_overall.to_numpy(int)
            solved = reps[g.correct.to_numpy(int) == 1]
            fs = int(solved[0]) if solved.size else np.inf
            for rep in reps:
                lab[(int(grid), int(rep))] = (
                    'first uncovers' if rep == reps[0]
                    else 'while learning' if rep < fs else 'once known')
        t = tab[['t_s', 'grid_no', 'rep_overall']].copy()
        t['stage_exc'] = [lab.get((int(gr), int(rp)))
                          for gr, rp in zip(t.grid_no, t.rep_overall)]
        t['session'] = int(session)
        frames.append(t[['session', 't_s', 'stage_exc']])
    if not frames:
        presses['stage_exc'] = np.nan
        return presses
    exc = pd.concat(frames, ignore_index=True)
    out = []
    for session, p in presses.groupby('session'):
        e = exc[exc.session == int(session)].sort_values('t_s')
        p = p.sort_values('t_s')
        if len(e):
            p = pd.merge_asof(p, e[['t_s', 'stage_exc']], on='t_s',
                              direction='nearest', tolerance=0.03)
        else:
            p = p.assign(stage_exc=np.nan)
        p.loc[p.kind == 'move', 'stage_exc'] = np.nan
        out.append(p)
    out = pd.concat(out, ignore_index=True)
    n_moved = int((out.stage.notna() & out.stage_exc.notna()
                   & (out.stage != out.stage_exc)).sum())
    print(f"    exclusive labelling: {n_moved} uncover presses change phase")
    return out
