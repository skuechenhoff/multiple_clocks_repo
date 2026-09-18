#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time location timeline, straight from `abcd_passed.mat` -> `trial_vars`.

WHY THIS EXISTS
---------------
Everything that asks what a ripple CONTAINS needs to know where the subject was
at the moment the ripple happened, on the session clock. Two existing sources
look like they answer that and do not:

  * `all_location_snippets.csv` is built on a 360-bin NORMALISED trial,
    averaged across correct repeats of a grid. It is time-warped, so it cannot
    be aligned to a 60 ms ripple. Measured: only 7.2% of ripples fall inside a
    snippet if you try. It is a fine source of location TEMPLATES and a useless
    source of timing.
  * Integrating arrow presses from `press_categories.csv` and re-anchoring at
    every uncover reproduces the true location only 62.9% of the time.

`trial_vars` carries the ground truth and needs no reconstruction at all:
`start_location` / `end_location` per move, with `grid_onset_timestamp` giving
the time of each move.

⚠ CLOCKS. `grid_onset_timestamp`, `state_change_times`, `end_trial_timestamp`
and `visit_begin_timestamp_c` are trigger-derived, i.e. the same clock as the
spikes and the ripples. `button_pressed_timestamp` is a Matlab clock and the
mat file itself renames it `DONOTUSE_button_pressed_timestamp`. Use the
trigger ones. `visit_begin_timestamp_c_bp` is the button-press variant of the
compressed visit times and differs by ~7 ms; it is not used here.

SESSION NUMBERING. `abcd_data` is (63, 1) and the project's session s01..s63 is
the mat index + 1. The `session_num` FIELD is 1 for every session -- it means
"which recording file within this session", not the global index. Do not use it.

@author: Svenja Kuchenhoff
"""

import os

import numpy as np
import pandas as pd

import mc.analyse.swr_io as swr_io

N_SESSIONS = 63

# one row per move; the subject sits at `loc` from `t_s` until the next row's t_s
STEP_COLS = ["session", "grid_num", "grid_id", "sequence_num", "trial_idx",
             "trial_num_in_grid", "first_trial", "trial_correct",
             "step", "t_s", "loc", "next_loc",
             "is_uncover", "correct_uncover", "state", "looking_for_loc",
             "rew_A", "rew_B", "rew_C", "rew_D",
             "t_A", "t_B", "t_C", "t_D", "trial_end_s"]


def cache_dir(data_root=None):
    return os.path.join(swr_io.derivatives_dir(data_root), "group", "swr",
                        "location_timeline")


def _flat(f, ref):
    return np.array(f[ref]).ravel()


def _step(a, n):
    """Per-step field forced to length n -- some trials carry short arrays."""
    a = np.asarray(a, float).ravel()
    if a.size == n:
        return a
    out = np.full(n, np.nan)
    out[:min(n, a.size)] = a[:min(n, a.size)]
    return out


def _scalar(f, ref):
    v = _flat(f, ref)
    return float(v[0]) if v.size else np.nan


def extract_session(f, sess_idx):
    """One session's step table. `sess_idx` is 0-based; session = sess_idx + 1."""
    g = f["abcd_passed"]["abcd_data"]
    tv = f[g["trial_vars"][sess_idx, 0]]
    n_trials = tv["start_location"].shape[0]

    rows = []
    for tr in range(n_trials):
        loc_a = _flat(f, tv["start_location"][tr, 0])
        loc_b = _flat(f, tv["end_location"][tr, 0])
        t = _flat(f, tv["grid_onset_timestamp"][tr, 0])
        if loc_a.size == 0 or t.size != loc_a.size:
            continue                      # malformed trial; skipped and counted
        unc = _flat(f, tv["pressed_to_uncover"][tr, 0])
        cor = _flat(f, tv["correct_uncover"][tr, 0])
        st = _flat(f, tv["start_state"][tr, 0])
        lf = _flat(f, tv["looking_for_location"][tr, 0])
        seq = _flat(f, tv["sequence_locations"][tr, 0])
        sct = _flat(f, tv["state_change_times"][tr, 0])
        seq = np.pad(seq.astype(float), (0, max(0, 4 - seq.size)),
                     constant_values=np.nan)[:4]
        sct = np.pad(sct.astype(float), (0, max(0, 4 - sct.size)),
                     constant_values=np.nan)[:4]

        d = dict(
            session=sess_idx + 1,
            grid_num=_scalar(f, tv["grid_num"][tr, 0]),
            grid_id=_scalar(f, tv["grid_id"][tr, 0]),
            sequence_num=_scalar(f, tv["sequence_num"][tr, 0]),
            trial_idx=tr,
            trial_num_in_grid=_scalar(f, tv["trial_num_in_grid"][tr, 0]),
            first_trial=_scalar(f, tv["first_trial"][tr, 0]),
            trial_correct=_scalar(f, tv["trial_correct"][tr, 0]),
            trial_end_s=_scalar(f, tv["end_trial_timestamp"][tr, 0]),
        )
        n = loc_a.size
        rows.append(pd.DataFrame({
            **{k: np.repeat(v, n) for k, v in d.items()},
            "step": np.arange(n),
            "t_s": t.astype(float),
            "loc": loc_a.astype(float),
            "next_loc": _step(loc_b, n),
            "is_uncover": _step(unc, n),
            "correct_uncover": _step(cor, n),
            "state": _step(st, n),
            "looking_for_loc": _step(lf, n),
            "rew_A": seq[0], "rew_B": seq[1], "rew_C": seq[2], "rew_D": seq[3],
            "t_A": sct[0], "t_B": sct[1], "t_C": sct[2], "t_D": sct[3],
        }))
    if not rows:
        return pd.DataFrame(columns=STEP_COLS)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["t_s", "step"]).reset_index(drop=True)[STEP_COLS]


def build(sessions=None, data_root=None, mat_path=None, overwrite=False,
          verbose=True):
    """Extract and cache one step table per session. Returns {session: path}."""
    import h5py
    sessions = list(range(1, N_SESSIONS + 1)) if sessions is None else sessions
    out_dir = cache_dir(data_root)
    os.makedirs(out_dir, exist_ok=True)
    mat = mat_path or os.path.join(swr_io.derivatives_dir(data_root),
                                   "abcd_passed.mat")

    paths, need = {}, []
    for s in sessions:
        p = os.path.join(out_dir, f"steps_s{s:02d}.csv")
        paths[s] = p
        if overwrite or not os.path.isfile(p):
            need.append(s)
    if need:
        if verbose:
            print(f"  reading {len(need)} session(s) from {os.path.basename(mat)}")
        with h5py.File(mat, "r") as f:
            for s in need:
                df = extract_session(f, s - 1)
                df.to_csv(paths[s], index=False)
                if verbose:
                    print(f"    s{s:02d}: {len(df):6d} steps, "
                          f"{df.t_s.min():.0f}-{df.t_s.max():.0f} s"
                          if len(df) else f"    s{s:02d}: EMPTY")
    return paths


def load(sessions=None, data_root=None):
    """The cached step tables, concatenated. Builds anything missing."""
    paths = build(sessions, data_root=data_root, verbose=False)
    out = [pd.read_csv(p) for p in paths.values() if os.path.isfile(p)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def location_at(steps, t, session=None):
    """Location at each time in `t`, from one session's step table.

    The subject sits at `loc` from `t_s` until the next step begins, so this is
    a right-continuous step function. Times before the first step or after the
    last trial's end return NaN.
    """
    s = steps if session is None else steps[steps.session == session]
    s = s.sort_values("t_s")
    ts = s.t_s.to_numpy(float)
    loc = s.loc[:, "loc"].to_numpy(float)
    end = np.nanmax(s.trial_end_s.to_numpy(float)) if len(s) else np.nan
    t = np.asarray(t, float)
    j = np.searchsorted(ts, t, side="right") - 1
    out = np.full(t.shape, np.nan)
    ok = (j >= 0) & (t <= end)
    out[ok] = loc[j[ok]]
    return out
