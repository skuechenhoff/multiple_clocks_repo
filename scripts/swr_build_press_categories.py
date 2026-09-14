#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Every button press in the dataset, with its KEY and the stillness around it.

`swr_probes.press_times` collapses the four arrow keys into "movement", which
is enough for the stillness probes but not for a matched-control design: a
control press has to come from a category the hypothesis says nothing about,
and "an arrow key" is four categories, not one. This writes them out
separately, once, so nothing downstream has to touch the raw 25 ms button
series again (that read is the slow part -- it is I/O bound and takes ~15 min
for 61 sessions).

Columns
    session, t_s, key            LeftArrow / RightArrow / UpArrow / DownArrow /
                                 Return
    kind                         move / uncover
    still_next_s, still_prev_s   gap to the next / previous press of ANY kind
    grid_no, rep_overall         from the behaviour table, for uncover presses
    valence, stage               from `rip.uncover_table`, for uncover presses

    python scripts/swr_build_press_categories.py
    python scripts/swr_build_press_categories.py --out=<path.csv>

@author: Svenja Kuchenhoff
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import mc.analyse.swr_behaviour as swb

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


def presses_with_keys(session, beh, data_root=None):
    """Every key press in a session as (time, key), from the 25 ms series.

    Same transition logic as `swr_probes.press_times` -- a press is a bin where
    the held key changes -- but the key itself is kept instead of being mapped
    to move/uncover.
    """
    rows = []
    for grid, g in beh.groupby('grid_no'):
        btn = swb._grid_series(session, grid, 'buttons', data_root)
        if btn is None:
            continue
        onset = float(g.new_grid_onset.iloc[0])
        b = btn.astype(str)
        trans = np.flatnonzero(b[1:] != b[:-1]) + 1
        for t in trans:
            key = b[t]
            if key in swb.MOVE_KEYS or key == swb.UNCOVER_KEY:
                rows.append((onset + t * swb.BIN_S, key, int(grid)))
    if not rows:
        return pd.DataFrame(columns=['t_s', 'key', 'grid_no'])
    return pd.DataFrame(rows, columns=['t_s', 'key', 'grid_no']) \
             .sort_values('t_s').reset_index(drop=True)


def build(bundle=None, out=None):
    root = swr_io.get_data_root()
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                              'bundle')
    if out is None:
        out = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                           'press_categories.csv')
    data = rip.load_bundle(bundle)
    beh = data['behaviour']
    print(f"\n  bundle: {bundle}\n  writing -> {out}")

    frames = []
    for i, session in enumerate(rip.sessions_in(data), 1):
        p = presses_with_keys(session, beh[beh.session == session])
        if not len(p):
            continue
        t = p.t_s.to_numpy(float)
        i_next = np.searchsorted(t, t, 'right')
        i_prev = np.searchsorted(t, t, 'left') - 1
        nxt = np.where(i_next < t.size, t[np.clip(i_next, 0, t.size - 1)],
                       np.nan) - t
        prv = t - np.where(i_prev >= 0, t[np.clip(i_prev, 0, None)], np.nan)
        p = p.assign(session=int(session), kind=np.where(
            p.key == swb.UNCOVER_KEY, 'uncover', 'move'),
            still_next_s=nxt, still_prev_s=prv)

        # carry the task labels onto the uncover presses, matched on time
        tab = rip.uncover_table(data, session)
        if len(tab):
            lab = tab[['t_s', 'valence', 'stage', 'reward', 'rep_overall']] \
                .sort_values('t_s')
            p = pd.merge_asof(p.sort_values('t_s'), lab, on='t_s',
                              direction='nearest', tolerance=0.03)
            p.loc[p.kind == 'move', ['valence', 'stage', 'reward',
                                     'rep_overall']] = np.nan
        frames.append(p)
        print(f"    [{i:2d}/61] session {session}: {len(p)} presses "
              f"({(p.kind == 'uncover').sum()} uncover)")

    allp = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allp.to_csv(out, index=False)
    print(f"\n  {len(allp)} presses, {allp.session.nunique()} sessions")
    print(allp.key.value_counts().to_string())
    print(f"\n  uncover presses labelled: "
          f"{allp[allp.kind == 'uncover'].valence.notna().sum()} of "
          f"{(allp.kind == 'uncover').sum()}")
    print(f"  written {datetime.now():%Y-%m-%d %H:%M}")
    return out


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(build)
    else:
        build()
