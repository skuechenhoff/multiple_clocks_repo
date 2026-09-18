#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioural summary for fMRI + ephys datasets — methods-section ready.

Per dataset we report:
    1) Per-loop time (A→D within a single attempt) per subject, per repeat
       index, and across subjects.
    2) Learning-curve slope (linear regression of loop time on repeat
       index per subject, plus group t vs. 0).
    3) fMRI-only fine-grained timing: median step time, dwell time, and
       button RT per subject.
    4) fMRI-only forw vs. backw comparison (paired t per subject).
    5) Ephys-only completion rate + count of incorrect attempts.
    6) Completeness counts: number of unique reward-location configs and
       correct repeats per config.

"Session" is defined by the rewarded-location tuple (loc_A, loc_B, loc_C,
loc_D), so two halves with the same 4 rewards count as the same session.

Outputs land in
    data/behaviour_summary/<timestamp>/
with a single combined JSON, per-subject CSVs, and overview plots.

@author: Svenja Kuchenhoff
"""

import os
import json
import glob
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats


# ── Settings ──────────────────────────────────────────────────────────
DATA_ROOT       = '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
FMRI_BEH_GLOB   = os.path.join(
    DATA_ROOT, 'derivatives', 'sub-*', 'beh', 'sub-*_beh_fmri_clean.csv')
FMRI_EXCLUDE    = {'sub-21', 'sub-29'}    # mirrors clean_fmri_behaviour.py
EPHYS_DERIV     = os.path.join(DATA_ROOT, 'ephys_humans', 'derivatives')
EPHYS_BEH_COLS  = ['rep_correct', 't_A', 't_B', 't_C', 't_D',
                   'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall',
                   'new_grid_onset', 'session_no', 'grid_no', 'correct']
# Minimal timing exclusion.  This single 314.58 s correct-repeat-8 loop
# appears to be a recording/behavioural interruption, so we retain the rest
# of s23 and exclude only this exact attempt.  ``rep_correct`` is zero-based
# in the source table, hence 7 = displayed repeat 8.
EPHYS_EXCLUDE_ATTEMPTS = [
    {'subject': 's23', 'session_no': 1, 'grid_no': 3, 'rep_correct': 7},
]
# Per-attempt uncover counts, written by scripts/extract_uncovers_ephys.py
# out of the raw session struct.  ``all_trial_times_*.csv`` only carries the
# binary trial_correct flag; this table adds how many wrong fields were
# uncovered, which is the direct read-out of "did they remember the sequence".
EPHYS_UNCOVERS_CSV = os.path.join(
    EPHYS_DERIV, 'group', 'ephys_uncovers_per_attempt.csv')

# ── Task stages ──────────────────────────────────────────────────────
# The three stages are defined by the information available to the subject,
# never by how well they did, so that stage-wise error and speed measures are
# not circular with the stage definition itself:
#   explore : the first attempt on a grid.  All nine fields start covered, so
#             this is the only attempt during which reward locations are
#             genuinely unknown.  It ends when the fourth reward (D) is
#             uncovered for the first time.
#   learn   : every further attempt up to and including the first error-free
#             loop.  All four rewards have been seen, but the subject has not
#             yet shown that the sequence is retrievable.
#   execute : every attempt after the first error-free loop.  Whether these
#             are in fact near-errorless is then an empirical result rather
#             than part of the definition.
STAGE_ORDER = ('explore', 'learn', 'execute')
# Project-wide phase ramp (pastel pink -> bordeaux), light = early.
STAGE_COLORS = {
    'explore': '#FCDDE3',
    'learn':   '#D7657F',
    'execute': '#5C1027',
}
STAGE_LABELS = {'explore': 'expl', 'learn': 'learn', 'execute': 'exec'}

# Sample-trajectory figures.  The task layouts below are the two layouts
# requested for the first-draft figure.  The acquisition differs between the
# modalities, so the first is available in the cell data and the second in
# fMRI.
PLOT_SAMPLE_TRAJECTORIES = True
TRAJECTORY_PREFERRED_LAYOUTS = ((3, 7, 9, 5), (5, 9, 4, 3))
TRAJECTORY_N_STABLE = 5
TRAJECTORY_N_RANDOM = 3
TRAJECTORY_RANDOM_SEED = 20260815
# Create both requested visual encodings for the same eight selected people.
# In both, a transition's frequency is the fraction of repeats that include
# it at least once; the two versions differ only in the visual encoding.
TRAJECTORY_LINE_STYLES = ('opacity', 'thickness')

# The override keeps normal output unchanged, while allowing a safe temporary
# output location for checks without writing into the data directory.
OUT_BASE        = os.environ.get(
    'BEHAVIOUR_SUMMARY_OUT_BASE', os.path.join(DATA_ROOT, 'behaviour_summary'))
RUN_TAG         = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR         = os.path.join(OUT_BASE, RUN_TAG)
PLOT_DIR        = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Writing to {OUT_DIR}")


# ── Helpers ───────────────────────────────────────────────────────────
def _describe(x):
    """{mean, sd, median, n} for an array; NaNs ignored."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'mean': None, 'sd': None, 'median': None, 'n': 0}
    return {
        'mean':   float(np.mean(x)),
        'sd':     float(np.std(x, ddof=1)) if x.size > 1 else None,
        'median': float(np.median(x)),
        'n':      int(x.size),
    }


# ── Grid geometry and routes ─────────────────────────────────────────
_GRID_XY = {
    1: (0, 2), 2: (1, 2), 3: (2, 2),
    4: (0, 1), 5: (1, 1), 6: (2, 1),
    7: (0, 0), 8: (1, 0), 9: (2, 0),
}


def _collapse_locations(values):
    """Convert sampled locations into a route without dwell duplicates."""
    route = []
    for value in values:
        if pd.isna(value):
            continue
        try:
            loc = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= loc <= 9 and (not route or loc != route[-1]):
            route.append(loc)
    return tuple(route)


def _read_location_trace(path):
    """Load one 25-ms location trace.

    The traces are stored as a single very wide row, which ``read_csv``
    parses column by column and is therefore ~50x slower than reading the
    line directly.
    """
    with open(path) as handle:
        text = handle.read()
    return np.fromstring(text.replace('\n', ','), sep=',')


def _grid_distance(start, stop):
    """Minimum number of steps between two locations of the 3 x 3 grid.

    Movements are to the four neighbouring squares only — no diagonals and
    no wrap-around — so the step-minimum equals the Manhattan distance.
    """
    return (abs(_GRID_XY[start][0] - _GRID_XY[stop][0])
            + abs(_GRID_XY[start][1] - _GRID_XY[stop][1]))


def _route_metrics(route):
    """{n_steps, min_steps, is_shortest} for one reward-to-reward route."""
    n_steps = len(route) - 1
    min_steps = _grid_distance(route[0], route[-1])
    return {'n_steps': int(n_steps), 'min_steps': int(min_steps),
            'extra_steps': int(n_steps - min_steps),
            'is_shortest': bool(n_steps == min_steps)}


# ── fMRI side ─────────────────────────────────────────────────────────
def fmri_loop_table(df, sub):
    """One row per (task_half, instruction, loc-tuple, repeat)."""
    rows = []
    grouper = ['task_half', 'instruction', 'task_config_seq', 'repeat']
    for keys, g in df.groupby(grouper, sort=False):
        if not (g['state'] == 'D').any():
            continue
        # Loop time: arrival at D minus arrival at start (first row of this
        # (cfg, repeat)). first row's t_curr_loc is the A-onset.
        t_start = g['t_curr_loc'].dropna().iloc[0] if g['t_curr_loc'].notna().any() else np.nan
        t_d     = g.loc[g['state'] == 'D', 't_curr_rew'].dropna()
        if t_d.empty or not np.isfinite(t_start):
            continue
        loop_t = float(t_d.iloc[-1] - t_start)
        # fine-grained timing.  `button_rts` in the cleaned CSV stores
        # absolute within-run times, not reaction times, so we skip it
        # and rely on dwell + step.
        step_t   = float(g['t_move_to_next_loc'].mean(skipna=True))
        dwell_t  = float(g.loc[g['state'].notna(), 't_dwell_curr_loc']
                         .mean(skipna=True))
        # Floor / ceiling-speed reference.  `length_step` and
        # `reward_delay` are the enforced per-step ISI and reward
        # waiting time set by the experiment (3x3_fMRI_part1.py: jitter()
        # → secs_per_step → wait()).  We sum them over every row of the
        # loop EXCEPT the D-arrival row — its length_step is the forced
        # ISI into the next loop and its reward_delay is the post-D dwell,
        # both of which happen after loop end (t_curr_rew of D).
        d_arrival_idx = g.index[(g['state'] == 'D')
                                & g['t_curr_rew'].notna()]
        if len(d_arrival_idx):
            within_loop = g.drop(d_arrival_idx[-1])
        else:
            within_loop = g
        floor_t  = float(within_loop['t_move_to_next_loc'].fillna(0).sum()
                         + within_loop['reward_delay'].fillna(0).sum())
        rows.append({
            'subject':         sub,
            'task_half':       int(keys[0]),
            'instruction':     keys[1],
            'task_config_seq': keys[2],
            'repeat':          int(keys[3]),
            'loop_time':       loop_t,
            'floor_loop_time': floor_t,
            'slack':           loop_t - floor_t,
            'step_time_mean':  step_t,
            'dwell_time_mean': dwell_t,
        })
    return pd.DataFrame(rows)


def fmri_shortest_path_table(df, sub):
    """One row per walk between two consecutive rewards.

    A walk is "shortest" when the number of steps taken equals the grid
    distance between the two rewards.  The walk towards A starts at the D
    reward of the preceding repeat; the very first A of a configuration is
    skipped, because the location the subject starts from is not stored.
    """
    rows = []
    for keys, block in df.groupby(['task_half', 'instruction',
                                   'task_config_seq'], sort=False):
        block = block.sort_values('t_curr_loc')
        prev_state, prev_reward = None, None
        for (repeat, state), seg in block.groupby(['repeat', 'state'],
                                                  sort=False):
            reward = seg['curr_rew'].dropna()
            route = _collapse_locations(seg['curr_loc'])
            if reward.empty or not route:
                continue
            if prev_reward is not None:
                full_route = _collapse_locations((prev_reward, *route))
                rows.append({
                    'subject':         sub,
                    'task_half':       int(keys[0]),
                    'instruction':     keys[1],
                    'task_config_seq': keys[2],
                    'repeat':          int(repeat),
                    'segment':         f'{prev_state}-{state}',
                    **_route_metrics(full_route),
                })
            prev_state, prev_reward = state, int(reward.iloc[0])
    return pd.DataFrame(rows)


def fmri_summarise():
    paths = sorted(glob.glob(FMRI_BEH_GLOB))
    per_loop_all = []
    per_path_all = []
    per_subject = []
    for path in paths:
        sub = os.path.basename(path).split('_')[0]
        if sub in FMRI_EXCLUDE:
            continue
        df = pd.read_csv(path)
        loop_df = fmri_loop_table(df, sub)
        if loop_df.empty:
            print(f"  {sub}: no usable loops — skipped.")
            continue
        per_loop_all.append(loop_df)
        path_df = fmri_shortest_path_table(df, sub)
        per_path_all.append(path_df)

        # Per-subject collapse.
        subj_loop = loop_df['loop_time'].to_numpy()
        # Learning slope: regress loop time on repeat index, pooled across
        # halves / configs / instructions.
        slope, intercept, r, p, se = stats.linregress(
            loop_df['repeat'].to_numpy(dtype=float),
            subj_loop)
        # Forw / backw paired contrast averaged per repeat within subject.
        per_rep_dir = (
            loop_df.groupby(['repeat', 'instruction'])['loop_time']
                   .mean().unstack('instruction'))
        if {'forw', 'backw'}.issubset(per_rep_dir.columns):
            paired = per_rep_dir.dropna()
            if len(paired) >= 2:
                t_fb, p_fb = stats.ttest_rel(paired['forw'], paired['backw'])
                forw_minus_backw_mean = float(
                    paired['forw'].mean() - paired['backw'].mean())
            else:
                t_fb = p_fb = np.nan
                forw_minus_backw_mean = np.nan
        else:
            t_fb = p_fb = np.nan
            forw_minus_backw_mean = np.nan

        per_subject.append({
            'subject':            sub,
            'n_loops':            int(len(loop_df)),
            'n_reward_to_reward_paths': int(len(path_df)),
            'shortest_path_percent': float(
                100.0 * path_df['is_shortest'].mean()) if len(path_df) else None,
            'n_unique_configs':   int(loop_df['task_config_seq'].nunique()),
            'n_repeats_max':      int(loop_df['repeat'].max()) + 1,
            'loop_time_mean':     float(np.nanmean(subj_loop)),
            'loop_time_sd':       float(np.nanstd(subj_loop, ddof=1))
                                  if subj_loop.size > 1 else None,
            'floor_loop_time_mean': float(np.nanmean(
                loop_df['floor_loop_time'])),
            'slack_mean':         float(np.nanmean(loop_df['slack'])),
            'slack_sd':           float(np.nanstd(loop_df['slack'], ddof=1))
                                  if len(loop_df) > 1 else None,
            'loop_time_by_repeat': {
                int(r_): float(np.nanmean(
                    loop_df.loc[loop_df['repeat'] == r_, 'loop_time']))
                for r_ in sorted(loop_df['repeat'].unique())
            },
            'learning_slope':     float(slope),
            'learning_slope_p':   float(p),
            'step_time_median':   float(np.nanmedian(loop_df['step_time_mean'])),
            'dwell_time_median':  float(np.nanmedian(loop_df['dwell_time_mean'])),
            'forw_minus_backw_loop_mean': forw_minus_backw_mean,
            'forw_vs_backw_paired_t':     None if np.isnan(t_fb) else float(t_fb),
            'forw_vs_backw_paired_p':     None if np.isnan(p_fb) else float(p_fb),
        })

    if not per_subject:
        print("No fMRI subjects passed filter.")
        return None, None, None

    all_loops = pd.concat(per_loop_all, ignore_index=True)
    subj_df   = pd.DataFrame(per_subject)
    subj_df.to_csv(os.path.join(OUT_DIR, 'fmri_per_subject.csv'), index=False)
    all_loops.to_csv(os.path.join(OUT_DIR, 'fmri_loops.csv'), index=False)
    all_paths = pd.concat(per_path_all, ignore_index=True)
    all_paths.to_csv(os.path.join(OUT_DIR, 'fmri_shortest_paths.csv'),
                     index=False)

    # Group-level summary.
    means = subj_df['loop_time_mean'].to_numpy()
    slopes = subj_df['learning_slope'].to_numpy()
    t_slope, p_slope = stats.ttest_1samp(slopes, 0.0)
    # paired t on forw - backw across subjects
    fb = subj_df['forw_minus_backw_loop_mean'].dropna().to_numpy()
    if fb.size >= 2:
        t_fb_grp, p_fb_grp = stats.ttest_1samp(fb, 0.0)
    else:
        t_fb_grp = p_fb_grp = np.nan

    # By repeat across subjects
    loop_by_rep = (
        all_loops.groupby(['subject', 'repeat'])['loop_time'].mean()
                 .unstack('repeat'))
    by_rep = {
        int(r_): _describe(loop_by_rep[r_].to_numpy())
        for r_ in sorted(loop_by_rep.columns)
    }
    # Equal weight per subject, as everywhere else in this script.
    shortest_by_repeat = (
        all_paths.groupby(['subject', 'repeat'])['is_shortest'].mean()
                 .unstack('repeat'))

    group = {
        'n_subjects':            int(len(subj_df)),
        'excluded':              sorted(FMRI_EXCLUDE),
        'loop_time_across_subj': _describe(means),
        'floor_loop_time_across_subj': _describe(
            subj_df['floor_loop_time_mean'].to_numpy()),
        'slack_across_subj':     _describe(
            subj_df['slack_mean'].to_numpy()),
        'loop_time_by_repeat':   by_rep,
        'learning_slope_group':  {
            **_describe(slopes),
            't': float(t_slope), 'p': float(p_slope),
        },
        'step_time_median':      _describe(
            subj_df['step_time_median'].to_numpy()),
        'dwell_time_median':     _describe(
            subj_df['dwell_time_median'].to_numpy()),
        'forw_vs_backw_subj_diff': {
            **_describe(fb),
            't_vs_0': None if np.isnan(t_fb_grp) else float(t_fb_grp),
            'p_vs_0': None if np.isnan(p_fb_grp) else float(p_fb_grp),
        },
        'completeness_n_configs': _describe(
            subj_df['n_unique_configs'].to_numpy()),
        'shortest_path_percent': _describe(
            subj_df['shortest_path_percent'].to_numpy()),
        'shortest_path_percent_pooled': float(
            100.0 * all_paths['is_shortest'].mean()),
        'shortest_path_percent_by_repeat': {
            int(r_): _describe(100.0 * shortest_by_repeat[r_].to_numpy())
            for r_ in sorted(shortest_by_repeat.columns)
        },
    }
    return subj_df, all_loops, group


# ── Ephys side ────────────────────────────────────────────────────────
def ephys_loop_table(df, sub):
    """One row per attempt; loop time = t_D − t_A. Adds session-by-locs."""
    df = df.copy()
    df['loc_tuple'] = df[['loc_A', 'loc_B', 'loc_C', 'loc_D']].astype(int).agg(
        lambda r: f"{r[0]}-{r[1]}-{r[2]}-{r[3]}", axis=1)
    df['subject'] = sub
    df['loop_time'] = df['t_D'] - df['t_A']
    return df


def add_stage_labels(attempts):
    """Label every attempt with the task stage it belongs to.

    See the STAGE_ORDER comment above for the definition.  The boundary
    between ``learn`` and ``execute`` is the first error-free loop of that
    grid, which is a milestone in what the subject has demonstrated, not a
    threshold placed on the measures we later compare between stages.

    Grids without any error-free loop (7 of 1489) simply have no ``execute``
    stage; they are flagged so they can be reported.
    """
    attempts = attempts.sort_values(
        ['subject', 'session_no', 'grid_no', 'rep_overall']).copy()
    stage = np.full(len(attempts), 'learn', dtype=object)
    solved = np.zeros(len(attempts), dtype=bool)
    position = 0
    for _, grid in attempts.groupby(['subject', 'session_no', 'grid_no'],
                                    sort=False):
        n = len(grid)
        block = slice(position, position + n)
        correct = grid['correct'].to_numpy()
        first_correct = np.flatnonzero(correct == 1)
        labels = np.full(n, 'learn', dtype=object)
        labels[grid['rep_overall'].to_numpy() == 1] = 'explore'
        if first_correct.size:
            labels[first_correct[0] + 1:] = 'execute'
            solved[block] = True
        stage[block] = labels
        position += n
    attempts['stage'] = stage
    attempts['grid_solved'] = solved
    return attempts


def add_attempt_timing(attempts):
    """Add the per-attempt durations used by the stage descriptives.

    ``new_grid_onset`` is the start of the attempt: for the first attempt of
    a grid it is when the covered grid appeared, and for every later attempt
    it equals the previous attempt's ``t_D``.  Hence
        attempt_duration = t_D - new_grid_onset   (whole attempt)
        loop_time        = t_D - t_A              (A->D, as used elsewhere)
        time_to_A        = t_A - new_grid_onset   (getting to / finding A)
    """
    attempts = attempts.copy()
    attempts['attempt_duration'] = attempts['t_D'] - attempts['new_grid_onset']
    attempts['time_to_A'] = attempts['t_A'] - attempts['new_grid_onset']
    return attempts


def add_run_index(attempts):
    """Count how often this subject has already seen this reward layout.

    Layouts are repeated within a session (between 1 and 4 runs each), so the
    first attempt of a *repeated* layout is not a first exposure.  Run 1 is
    the only genuinely novel exploration.
    """
    attempts = attempts.copy()
    grids = (attempts.groupby(['subject', 'session_no', 'grid_no'],
                              sort=False)['loc_tuple'].first()
             .reset_index())
    grids['run_index'] = (
        grids.sort_values(['subject', 'session_no', 'grid_no'])
             .groupby(['subject', 'session_no', 'loc_tuple']).cumcount() + 1)
    return attempts.merge(
        grids[['subject', 'session_no', 'grid_no', 'run_index']],
        on=['subject', 'session_no', 'grid_no'], how='left')


def attach_uncovers(attempts):
    """Merge in per-attempt uncover counts, if they have been extracted.

    Returns the table unchanged (with NaN count columns) when the derivative
    is missing, so the rest of the summary still runs.
    """
    count_columns = ['n_uncovers', 'n_correct_uncovers',
                     'n_incorrect_uncovers', 'n_steps_total', 'grid_id']
    if not os.path.isfile(EPHYS_UNCOVERS_CSV):
        print(f"  no uncover table at {EPHYS_UNCOVERS_CSV} — run "
              f"scripts/extract_uncovers_ephys.py for mistake counts.")
        for column in count_columns:
            attempts[column] = np.nan
        return attempts
    uncovers = pd.read_csv(EPHYS_UNCOVERS_CSV, dtype={'subject': str})
    keys = ['subject', 'session_no', 'grid_no', 'rep_overall']
    merged = attempts.merge(uncovers[keys + count_columns], on=keys,
                            how='left')
    n_missing = int(merged['n_uncovers'].isna().sum())
    if n_missing:
        print(f"  {n_missing} attempts without uncover counts.")
    return merged


def exclude_ephys_attempts(tbl):
    """Drop only explicitly registered interrupted ephys attempts."""
    keep = np.ones(len(tbl), dtype=bool)
    for attempt in EPHYS_EXCLUDE_ATTEMPTS:
        if attempt['subject'] not in set(tbl['subject']):
            continue
        mask = np.ones(len(tbl), dtype=bool)
        for column, value in attempt.items():
            mask &= tbl[column].eq(value).to_numpy()
        n_matches = int(mask.sum())
        if n_matches != 1:
            raise ValueError(
                f"Expected exactly one excluded ephys attempt for {attempt}; "
                f"found {n_matches}.")
        keep &= ~mask
        print(f"  excluded interrupted attempt: {attempt}")
    return tbl.loc[keep].copy()


def ephys_shortest_path_table(raw_attempts, kept_index, folder, sub_number):
    """One row per walk between two consecutive rewards, for every attempt.

    Routes come from the 25-ms location traces.  ``timings_rewards`` holds,
    per attempt and in the original unfiltered attempt order, the trace
    sample at which the attempt started (the preceding D) and at which A, B,
    C and D were reached.

    Incorrect attempts are included here, with ``correct`` and
    ``prev_correct`` kept per row so the correct-repeat-only subset used for
    the ABCD analyses can be selected afterwards with
    ``correct_walk_mask``.  Walking to a wrong field and back shows up as
    extra steps, so these rows are what makes path efficiency comparable
    across the three task stages.
    """
    rows = []
    for grid_value, raw_grid in raw_attempts.groupby('grid_no', sort=False):
        grid_no = int(grid_value)
        raw_grid = raw_grid.sort_index()
        timing_path = os.path.join(
            folder, f'timings_rewards_grid{grid_no}_sub{sub_number}.csv')
        locations_path = os.path.join(
            folder, f'locations_per_25ms_grid{grid_no}_sub{sub_number}.csv')
        if not (os.path.isfile(timing_path)
                and os.path.isfile(locations_path)):
            continue
        timings = pd.read_csv(timing_path, header=None).to_numpy()
        locations = _read_location_trace(locations_path)
        if len(raw_grid) != len(timings):
            continue
        for position, (source_index, attempt) in enumerate(
                raw_grid.iterrows()):
            if source_index not in kept_index:
                continue
            endpoints = timings[position]
            if not np.all(np.isfinite(endpoints)):
                continue
            # Whether the preceding attempt was a retained, completed repeat
            # that this attempt continues from without a gap.  Only then is
            # the walk into A a genuine D->A walk.
            previous_index = raw_grid.index[position - 1] if position else None
            prev_correct = bool(
                position > 0
                and previous_index in kept_index
                and raw_grid.iloc[position - 1]['correct'] == 1
                and timings[position - 1][-1] == endpoints[0])
            for column, (start_state, stop_state) in enumerate(
                    zip('DABC', 'ABCD')):
                if start_state == 'D' and not prev_correct:
                    continue
                start = int(endpoints[column])
                stop = int(endpoints[column + 1])
                if start < 0 or stop < start or stop >= len(locations):
                    continue
                route = _collapse_locations(locations[start:stop + 1])
                if len(route) < 2:
                    continue
                rows.append({
                    'subject':      attempt['subject'],
                    'session_no':   int(attempt['session_no']),
                    'grid_no':      grid_no,
                    'rep_overall':  int(attempt['rep_overall']),
                    'rep_correct':  int(attempt['rep_correct']),
                    'correct':      int(attempt['correct']),
                    'prev_correct': prev_correct,
                    'segment':      f'{start_state}-{stop_state}',
                    **_route_metrics(route),
                })
    return pd.DataFrame(rows)


def correct_walk_mask(paths):
    """The correct-repeat-only subset of the reward-to-reward walks.

    This is the set used for the ABCD-code analyses: walks inside an
    error-free loop, within the ten design-defined repeats.  D->A walks are
    already restricted to those following a completed repeat when the table
    is built, so no extra condition is needed for them here.
    """
    return (paths['correct'].eq(1)
            & paths['rep_correct'].between(0, 9)).to_numpy()


def _loop_time_slope(attempts, repeat_column):
    """Slope of loop time on a repeat index, in seconds per repeat.

    Ten attempts across the dataset have no ``t_D`` because the recording
    stopped mid-attempt, and hence no loop time.  They are dropped here:
    passing them to ``linregress`` returns NaN for the whole session, which
    silently cost two sessions their correct-repeat slope.
    """
    x = attempts[repeat_column].to_numpy(dtype=float)
    y = attempts['loop_time'].to_numpy(dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 2 or np.unique(x[usable]).size < 2:
        return np.nan, np.nan
    result = stats.linregress(x[usable], y[usable])
    return float(result.slope), float(result.pvalue)


def repeated_measures_anova(values):
    """One-way repeated-measures ANOVA over the columns of ``values``.

    Rows are sessions, columns are conditions (here: the three stages).  Only
    sessions contributing every condition enter, which is what makes the test
    repeated-measures rather than between-groups.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.all(np.isfinite(values), axis=1)]
    n, k = values.shape
    if n < 2 or k < 2:
        return None
    grand = values.mean()
    ss_condition = n * np.sum((values.mean(axis=0) - grand) ** 2)
    ss_subject = k * np.sum((values.mean(axis=1) - grand) ** 2)
    ss_error = np.sum((values - grand) ** 2) - ss_condition - ss_subject
    df_condition, df_error = k - 1, (k - 1) * (n - 1)
    f = (ss_condition / df_condition) / (ss_error / df_error)
    return {
        'F': float(f),
        'df1': int(df_condition),
        'df2': int(df_error),
        'p': float(stats.f.sf(f, df_condition, df_error)),
        'partial_eta_sq': float(ss_condition / (ss_condition + ss_error)),
        'n_sessions': int(n),
    }


def _holm(p_values):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = p_values.size
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_contrasts(table, pairs):
    """Paired t-tests between stage columns, Holm-corrected across ``pairs``.

    Reports Cohen's dz (the paired effect size) alongside t and p, on the
    sessions that contribute both stages of a given pair.
    """
    results, raw_p = [], []
    for first, second in pairs:
        both = table[[first, second]].dropna()
        if len(both) < 2:
            continue
        difference = (both[first] - both[second]).to_numpy(dtype=float)
        t, p = stats.ttest_rel(both[first], both[second])
        results.append({
            'contrast': f'{first} - {second}',
            'n_sessions': int(len(both)),
            'mean_difference': float(difference.mean()),
            'cohens_dz': float(difference.mean() / difference.std(ddof=1))
                         if difference.std(ddof=1) > 0 else None,
            't': float(t),
            'p': float(p),
        })
        raw_p.append(p)
    for result, adjusted in zip(results, _holm(raw_p)):
        result['p_holm'] = float(adjusted)
    return results


def ephys_stage_table(attempts, paths):
    """One row per (session, stage) with the stage's behavioural profile."""
    rows = []
    walks = paths.copy()
    for (subject, stage), block in attempts.groupby(['subject', 'stage'],
                                                    sort=False):
        stage_walks = walks[walks['subject'].eq(subject)
                            & walks['stage'].eq(stage)]
        n_attempts = int(len(block))
        rows.append({
            'subject': subject,
            'stage': stage,
            'n_grids': int(block.groupby(['session_no', 'grid_no']).ngroups),
            'n_attempts': n_attempts,
            # Time the stage occupies.  Per-grid durations sum to the whole
            # grid (attempt starts are contiguous: each attempt begins where
            # the previous one ended), so the three stages partition the time
            # on task and ``total_duration`` adds up to it across stages.
            'duration_per_grid': float(
                block.groupby(['session_no', 'grid_no'])['attempt_duration']
                     .sum().mean()),
            'total_duration': float(block['attempt_duration'].sum()),
            'attempts_per_grid': float(
                block.groupby(['session_no', 'grid_no']).size().mean()),
            'attempt_duration_mean': float(block['attempt_duration'].mean()),
            'loop_time_mean': float(block['loop_time'].mean()),
            'time_to_A_mean': float(block['time_to_A'].mean()),
            'error_fraction': float((block['correct'] == 0).mean()),
            'incorrect_uncovers_per_attempt': float(
                block['n_incorrect_uncovers'].mean()),
            'uncovers_per_attempt': float(block['n_uncovers'].mean()),
            'shortest_path_percent': float(
                100.0 * stage_walks['is_shortest'].mean())
                if len(stage_walks) else np.nan,
            'extra_steps_per_walk': float(stage_walks['extra_steps'].mean())
                if len(stage_walks) else np.nan,
            'n_walks': int(len(stage_walks)),
        })
    table = pd.DataFrame(rows)
    # Share of the session's total time on task that this stage takes up.
    table['duration_percent'] = (
        100.0 * table['total_duration']
        / table.groupby('subject')['total_duration'].transform('sum'))
    table['stage'] = pd.Categorical(table['stage'], STAGE_ORDER, ordered=True)
    return table.sort_values(['subject', 'stage']).reset_index(drop=True)


# Measures on which the three stages are compared.  Each is a different kind
# of evidence that the stages are distinct: how long they take, how wrong the
# subject is, and how directed the movement is.
#
# Two of them carry a caveat for the explore stage, and it matters for how
# they can be written up.  A subject cannot complete an error-free loop on
# their first visit to a fully covered grid, so ``error_fraction`` is ~1 in
# explore by construction; and an uncover of a not-yet-known field is
# information-gathering rather than a memory failure, so the explore stage's
# uncover count measures search, not mistakes.  The stage contrast that is
# free of both problems is learn vs. execute: in both, all four rewards have
# already been seen, so a wrong uncover there really is a memory failure.
STAGE_TEST_MEASURES = {
    'duration_per_grid': {
        'label': 'time spent in stage, per grid [s]'},
    'duration_percent': {
        'label': '% of time on task spent in stage'},
    'attempt_duration_mean': {
        'label': 'attempt duration [s]'},
    'loop_time_mean': {
        'label': 'A->D loop time [s]'},
    'time_to_A_mean': {
        'label': 'time to reach A [s]'},
    'error_fraction': {
        'label': 'fraction of attempts with an error',
        'caveat': "~1 in explore by construction: the first attempt on a "
                  "covered grid cannot be error-free. Only the learn vs. "
                  "execute contrast is informative."},
    'incorrect_uncovers_per_attempt': {
        'label': '# incorrect uncovers per attempt',
        'caveat': "In explore these are search uncovers of fields whose "
                  "content is not yet known, not memory failures. They are "
                  "memory failures in learn and execute, where all four "
                  "rewards have already been seen."},
    'shortest_path_percent': {
        'label': '% shortest reward-to-reward walks'},
    'extra_steps_per_walk': {
        'label': 'extra steps per walk'},
}


def ephys_stage_statistics(stage_table):
    """Test whether the three stages differ, measure by measure.

    A one-way repeated-measures ANOVA over sessions asks whether the stages
    differ at all; the two adjacent contrasts (explore vs. learn, learn vs.
    execute) then localise where.  Holm correction is applied across the two
    contrasts of each measure.
    """
    results = {}
    for measure, spec in STAGE_TEST_MEASURES.items():
        wide = (stage_table.pivot(index='subject', columns='stage',
                                  values=measure)
                .reindex(columns=list(STAGE_ORDER)))
        results[measure] = {
            **spec,
            'stage_means': {
                stage: _describe(wide[stage].to_numpy(dtype=float))
                for stage in STAGE_ORDER
            },
            'rm_anova': repeated_measures_anova(wide.to_numpy(dtype=float)),
            'contrasts': paired_contrasts(
                wide, [('explore', 'learn'), ('learn', 'execute'),
                       ('explore', 'execute')]),
        }
    return results


def ephys_learning_table(attempts):
    """Per-session learning speed and speed-up, one row per session.

    ``attempts_to_criterion`` is how many attempts a grid needed before its
    first error-free loop, and ``time_to_criterion`` how long that took from
    grid onset.  The speed-up compares the first and the last of the ten
    design-defined correct repeats.
    """
    rows = []
    for subject, block in attempts.groupby('subject', sort=False):
        per_grid = []
        for _, grid in block.groupby(['session_no', 'grid_no'], sort=False):
            grid = grid.sort_values('rep_overall')
            correct = np.flatnonzero(grid['correct'].to_numpy() == 1)
            if not correct.size:
                continue
            first = grid.iloc[correct[0]]
            per_grid.append({
                'attempts_to_criterion': float(first['rep_overall']),
                'time_to_criterion': float(
                    first['t_D'] - grid.iloc[0]['new_grid_onset']),
                'explore_duration': float(grid.iloc[0]['attempt_duration']),
                'incorrect_uncovers_to_criterion': float(
                    grid.iloc[:correct[0] + 1]['n_incorrect_uncovers'].sum()),
            })
        if not per_grid:
            continue
        per_grid = pd.DataFrame(per_grid)
        correct_block = block[block['correct'].eq(1)
                              & block['rep_correct'].between(0, 9)]
        by_repeat = correct_block.groupby('rep_correct')['loop_time'].mean()
        first_loop = by_repeat.get(0, np.nan)
        last_loop = by_repeat.get(9, np.nan)
        rows.append({
            'subject': subject,
            'n_grids_solved': int(len(per_grid)),
            **{column: float(per_grid[column].mean())
               for column in per_grid.columns},
            'loop_time_first_repeat': float(first_loop),
            'loop_time_last_repeat': float(last_loop),
            'speedup_seconds': float(first_loop - last_loop),
            'speedup_percent': float(
                100.0 * (1.0 - last_loop / first_loop)),
        })
    return pd.DataFrame(rows)


def ephys_error_fraction_by_repeat(attempts):
    """Pooled incorrect-attempt fraction at each of the 10 task repeats.

    ``rep_correct`` identifies how many correct repeats had already been
    completed when an attempt occurred.  Hence source value 0 is displayed as
    repeat 1 and includes both the eventual first correct attempt and any
    errors made before it.  This is intentionally pooled across everybody:
    it answers the requested fraction of *all attempts* that were errors at
    each repeat, rather than giving each participant equal weight.
    """
    usable = attempts[attempts['rep_correct'].between(0, 9)].copy()
    summary = (usable.groupby('rep_correct')['correct']
               .agg(n_attempts='size', n_correct='sum')
               .reset_index())
    summary['n_errors'] = (
        summary['n_attempts'] - summary['n_correct']).astype(int)
    summary['error_fraction'] = (
        summary['n_errors'] / summary['n_attempts'])
    summary['repeat_display'] = summary['rep_correct'].astype(int) + 1
    return summary


def ephys_summarise():
    sub_dirs = sorted([
        d for d in os.listdir(EPHYS_DERIV)
        if d.startswith('s') and os.path.isdir(
            os.path.join(EPHYS_DERIV, d, 'cells_and_beh'))
    ])
    per_session = []
    all_attempts = []
    all_paths = []
    for d in sub_dirs:
        sub = d[1:]
        path = os.path.join(EPHYS_DERIV, d, 'cells_and_beh',
                            f'all_trial_times_{sub}.csv')
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, header=None)
        if df.shape[1] != len(EPHYS_BEH_COLS):
            print(f"  s{sub}: unexpected n_cols={df.shape[1]} — skipped.")
            continue
        df.columns = EPHYS_BEH_COLS
        raw_tbl = ephys_loop_table(df, sub)
        tbl = exclude_ephys_attempts(raw_tbl)
        all_attempts.append(tbl)
        path_df = ephys_shortest_path_table(
            raw_tbl, set(tbl.index), os.path.dirname(path), sub)
        all_paths.append(path_df)

        correct = tbl[tbl['correct'] == 1]
        if correct.empty:
            print(f"  s{sub}: no correct trials — skipped.")
            continue

        # Level 1 — every attempt, on the rep_overall axis.  Level 2 — only
        # error-free repeats, on the rep_correct axis; that is the subset the
        # ABCD-code analyses use.  Both slopes describe the same sessions, so
        # they are computed with the same call on different rows.
        slope_all, p_all = _loop_time_slope(tbl, 'rep_overall')
        slope, p = _loop_time_slope(correct, 'rep_correct')

        correct_walks = path_df[correct_walk_mask(path_df)] \
            if len(path_df) else path_df

        n_attempts = int(len(tbl))
        n_correct  = int(len(correct))
        n_incorr   = int((tbl['correct'] == 0).sum())
        # "configurations solved" = unique loc-tuples with ≥1 correct trial.
        n_configs_solved = int(
            correct.groupby('loc_tuple').ngroups)
        per_session.append({
            'subject':                  sub,
            'n_attempts':               n_attempts,
            'n_reward_to_reward_paths': int(len(correct_walks)),
            'n_reward_to_reward_paths_all': int(len(path_df)),
            'shortest_path_percent':    float(
                100.0 * correct_walks['is_shortest'].mean())
                if len(correct_walks) else None,
            'shortest_path_percent_all': float(
                100.0 * path_df['is_shortest'].mean()) if len(path_df) else None,
            'extra_steps_per_walk_all': float(path_df['extra_steps'].mean())
                if len(path_df) else None,
            'loop_time_mean_all':       float(tbl['loop_time'].mean()),
            'loop_time_sd_all':         float(tbl['loop_time'].std(ddof=1))
                                        if len(tbl) > 1 else None,
            'attempt_duration_mean_all': float(
                (tbl['t_D'] - tbl['new_grid_onset']).mean()),
            'learning_slope_all':       float(slope_all),
            'learning_slope_all_p':     float(p_all),
            'n_correct':                n_correct,
            'completion_rate':          float(n_correct / n_attempts),
            'n_incorrect_attempts':     n_incorr,
            'incorrect_proportion':     float(n_incorr / n_attempts),
            'n_unique_loc_configs':     int(tbl['loc_tuple'].nunique()),
            'n_configs_solved':         n_configs_solved,
            'incorrect_per_config':     float(
                tbl[tbl['correct'] == 0].groupby('loc_tuple').size().mean()
                if (tbl['correct'] == 0).any() else 0.0),
            'loop_time_mean':         float(correct['loop_time'].mean()),
            'loop_time_sd':           float(correct['loop_time'].std(ddof=1))
                                      if len(correct) > 1 else None,
            'loop_time_by_repeat': {
                int(r_): float(np.nanmean(
                    correct.loc[correct['rep_correct'] == r_, 'loop_time']))
                for r_ in sorted(correct['rep_correct'].unique())
            },
            'learning_slope':         float(slope),
            'learning_slope_p':       float(p),
        })

    if not per_session:
        print("No ephys sessions parsed.")
        return None, None, None, None, None, None

    # Enrich the attempt table: stage, layout-run index, and the per-attempt
    # uncover counts.  Everything below can then work off this one table.
    all_df = pd.concat(all_attempts, ignore_index=True)
    all_df = add_stage_labels(add_attempt_timing(all_df))
    all_df = attach_uncovers(add_run_index(all_df))
    sess_df = pd.DataFrame(per_session)

    all_path_df = pd.concat(all_paths, ignore_index=True)
    # Carry the stage onto the walks, so path efficiency can be compared
    # between stages exactly as the attempt-level measures are.
    all_path_df = all_path_df.merge(
        all_df[['subject', 'session_no', 'grid_no', 'rep_overall', 'stage',
                'run_index']],
        on=['subject', 'session_no', 'grid_no', 'rep_overall'], how='left')
    correct_path_df = all_path_df[correct_walk_mask(all_path_df)]

    sess_df.to_csv(os.path.join(OUT_DIR, 'ephys_per_session.csv'), index=False)
    all_df.to_csv(os.path.join(OUT_DIR, 'ephys_attempts.csv'), index=False)
    all_path_df.to_csv(os.path.join(OUT_DIR, 'ephys_shortest_paths.csv'),
                       index=False)

    # Stage-wise profile and the tests that the stages differ at all.
    stage_df = ephys_stage_table(all_df, all_path_df)
    stage_df.to_csv(os.path.join(OUT_DIR, 'ephys_per_stage.csv'), index=False)
    stage_stats = ephys_stage_statistics(stage_df)
    learning_df = ephys_learning_table(all_df)
    learning_df.to_csv(os.path.join(OUT_DIR, 'ephys_learning.csv'),
                       index=False)

    # Equal weight per session, as everywhere else in this script.
    shortest_by_repeat = (
        correct_path_df.groupby(['subject', 'rep_correct'])['is_shortest']
                       .mean().unstack('rep_correct'))
    shortest_by_attempt = (
        all_path_df[all_path_df['rep_overall'].between(1, 12)]
        .groupby(['subject', 'rep_overall'])['is_shortest'].mean()
        .unstack('rep_overall'))

    # Group summary.  Drop any NaN slopes (sessions with <2 correct reps).
    slopes = sess_df['learning_slope'].to_numpy(dtype=float)
    slopes_ok = slopes[np.isfinite(slopes)]
    t_slope, p_slope = stats.ttest_1samp(slopes_ok, 0.0)

    # Loop time by rep_correct across sessions; cap at 9 (per task design).
    corr_all = all_df[(all_df['correct'] == 1)
                      & (all_df['rep_correct'].between(0, 9))]
    by_rep_session_mean = (
        corr_all.groupby(['subject', 'rep_correct'])['loop_time'].mean()
                .unstack('rep_correct'))
    by_rep = {
        int(r_): _describe(by_rep_session_mean[r_].to_numpy())
        for r_ in sorted(by_rep_session_mean.columns)
    }
    error_by_repeat = ephys_error_fraction_by_repeat(all_df)

    # Level 1 — all attempts on the rep_overall axis.  Attempts 1-12 cover
    # the design (10 correct repeats plus the usual couple of errors); later
    # attempts exist but only in the minority of grids that needed them.
    all_by_attempt = all_df[all_df['rep_overall'].between(1, 12)]
    by_attempt_session_mean = (
        all_by_attempt.groupby(['subject', 'rep_overall'])['loop_time'].mean()
                      .unstack('rep_overall'))
    uncovers_by_attempt_session_mean = (
        all_by_attempt.groupby(['subject',
                                'rep_overall'])['n_incorrect_uncovers'].mean()
                      .unstack('rep_overall'))

    group = {
        'n_sessions':           int(len(sess_df)),
        'excluded_attempts':    EPHYS_EXCLUDE_ATTEMPTS,
        'n_attempts_total':     int(len(all_df)),
        'n_grids_total':        int(all_df.groupby(
            ['subject', 'session_no', 'grid_no']).ngroups),
        'loop_time_across_subj': _describe(
            sess_df['loop_time_mean'].to_numpy()),
        'all_attempts': {
            'loop_time_across_subj': _describe(
                sess_df['loop_time_mean_all'].to_numpy()),
            'attempt_duration_across_subj': _describe(
                sess_df['attempt_duration_mean_all'].to_numpy()),
            'loop_time_by_rep_overall': {
                int(r_): _describe(by_attempt_session_mean[r_].to_numpy())
                for r_ in sorted(by_attempt_session_mean.columns)
            },
            'incorrect_uncovers_by_rep_overall': {
                int(r_): _describe(
                    uncovers_by_attempt_session_mean[r_].to_numpy())
                for r_ in sorted(uncovers_by_attempt_session_mean.columns)
            },
            'learning_slope_group': {
                **_describe(sess_df['learning_slope_all'].to_numpy()),
                **dict(zip(('t', 'p'), (float(v) for v in stats.ttest_1samp(
                    sess_df['learning_slope_all'].dropna().to_numpy(), 0.0)))),
            },
            'shortest_path_percent': _describe(
                sess_df['shortest_path_percent_all'].to_numpy(dtype=float)),
            'shortest_path_percent_pooled': float(
                100.0 * all_path_df['is_shortest'].mean()),
            'shortest_path_percent_by_rep_overall': {
                int(r_): _describe(100.0 * shortest_by_attempt[r_].to_numpy())
                for r_ in sorted(shortest_by_attempt.columns)
            },
            'extra_steps_per_walk': _describe(
                sess_df['extra_steps_per_walk_all'].to_numpy(dtype=float)),
        },
        'stages': {
            'definition': (
                "explore = first attempt of a grid (all fields covered, "
                "reward locations unknown); learn = further attempts up to "
                "and including the first error-free loop; execute = attempts "
                "after the first error-free loop.  Defined by the information "
                "the subject has, not by performance."),
            'per_stage': {
                stage: {
                    measure: _describe(
                        stage_df.loc[stage_df['stage'].eq(stage),
                                     measure].to_numpy(dtype=float))
                    for measure in STAGE_TEST_MEASURES
                }
                for stage in STAGE_ORDER
            },
            'n_sessions_per_stage': {
                stage: int(stage_df['stage'].eq(stage).sum())
                for stage in STAGE_ORDER
            },
            'tests': stage_stats,
            'n_grids_without_error_free_loop': int(
                (~all_df.groupby(['subject', 'session_no',
                                  'grid_no'])['grid_solved'].first()).sum()),
        },
        'learning': {
            measure: _describe(learning_df[measure].to_numpy(dtype=float))
            for measure in ('attempts_to_criterion', 'time_to_criterion',
                            'explore_duration',
                            'incorrect_uncovers_to_criterion',
                            'loop_time_first_repeat', 'loop_time_last_repeat',
                            'speedup_seconds', 'speedup_percent')
        },
        'loop_time_by_rep_correct': by_rep,
        'pooled_error_fraction_by_rep_correct': {
            int(row.rep_correct): {
                'repeat_display': int(row.repeat_display),
                'n_attempts': int(row.n_attempts),
                'n_errors': int(row.n_errors),
                'error_fraction': float(row.error_fraction),
            }
            for row in error_by_repeat.itertuples(index=False)
        },
        'learning_slope_group': {
            **_describe(slopes_ok),
            't': float(t_slope) if np.isfinite(t_slope) else None,
            'p': float(p_slope) if np.isfinite(p_slope) else None,
        },
        'completion_rate':      _describe(
            sess_df['completion_rate'].to_numpy()),
        'n_incorrect_attempts': _describe(
            sess_df['n_incorrect_attempts'].to_numpy()),
        'incorrect_proportion': _describe(
            sess_df['incorrect_proportion'].to_numpy()),
        'incorrect_per_config': _describe(
            sess_df['incorrect_per_config'].to_numpy()),
        'completeness_n_configs': _describe(
            sess_df['n_unique_loc_configs'].to_numpy()),
        'n_configs_solved':     _describe(
            sess_df['n_configs_solved'].to_numpy()),
        'shortest_path_percent': _describe(
            sess_df['shortest_path_percent'].to_numpy(dtype=float)),
        'shortest_path_percent_pooled': float(
            100.0 * correct_path_df['is_shortest'].mean()),
        'shortest_path_percent_by_rep_correct': {
            int(r_): _describe(100.0 * shortest_by_repeat[r_].to_numpy())
            for r_ in sorted(shortest_by_repeat.columns)
        },
        'uncover_counts': {
            'available': bool(all_df['n_uncovers'].notna().any()),
            'incorrect_uncovers_per_attempt_across_subj': _describe(
                all_df.groupby('subject')['n_incorrect_uncovers']
                      .mean().to_numpy()),
            'total_incorrect_uncovers': (
                None if all_df['n_incorrect_uncovers'].isna().all()
                else int(all_df['n_incorrect_uncovers'].sum())),
            'note': (
                "From scripts/extract_uncovers_ephys.py.  12 of 18228 "
                "attempts are flagged trial_correct=1 by the task although "
                "the move record contains an incorrect uncover; those 12 "
                "also incremented the correct-repeat counter, which is what "
                "produces the rep_correct==10 overflow below.  The flag is "
                "left untouched here — the discrepant attempts are listed in "
                "derivatives/group/ephys_uncover_flag_discrepancies.csv."),
        },
        'rep_correct_overflow_note': (
            "rep_correct nominally runs 0-9 (10 correct repeats/grid). "
            "12 trials across 9 sessions have rep_correct == 10 "
            "(s05, s07, s11, s13, s18, s20, s24, s26, s27). "
            "Capped at 9 in all summary tables."),
    }
    speedup = learning_df['speedup_percent'].dropna().to_numpy()
    if speedup.size >= 2:
        t_speed, p_speed = stats.ttest_1samp(speedup, 0.0)
        group['learning']['speedup_percent_vs_0'] = {
            't': float(t_speed), 'p': float(p_speed),
            'n_sessions': int(speedup.size)}
    return sess_df, all_df, group, stage_df, learning_df, all_path_df


# ── Sample trajectory figures ────────────────────────────────────────
# Same location palette as the task schematic: dark blue/teal at locations
# 1, 4, 7; pale blue/green at 3, 6, 9.  The grid itself therefore carries
# the location identity, without printing a number in every square.
_LOCATION_COLOURS = {
    1: '#0a607a', 2: '#7eb1c4', 3: '#b6d4e0',
    4: '#175e62', 5: '#5b9b8d', 6: '#c8e0d0',
    7: '#0e3d3a', 8: '#3d8b7d', 9: '#a7d9b2',
}
# Colour identities for the ordered A–D reward locations.  These are drawn
# as large square outlines, matching the task-configuration schematic.
_STATE_OUTLINE_COLOURS = {
    'A': '#F15A29',  # orange
    'B': '#F7931E',  # yellow-orange
    'C': '#C7C6E2',  # light purple
    'D': '#6B60AA',  # dark purple
}


def _fmri_reward_layout(group):
    """Read the ordered A–D rewarded locations from a clean fMRI group."""
    layout = []
    for state in 'ABCD':
        values = group.loc[group['state'].eq(state), 'curr_rew'].dropna()
        if values.empty:
            return None
        layout.append(int(values.iloc[0]))
    return tuple(layout)


def _make_trajectory_candidate(modality, subject, layout, paths, **metadata):
    """Attach reproducible route-consistency metadata to one task grid."""
    if not paths:
        return None
    paths = sorted(paths, key=lambda item: item[0])
    route_counts = Counter(route for _, route in paths)
    # Explicit sorting makes ties deterministic rather than dependent on how
    # the source CSV happened to be ordered.
    modal_route, modal_count = sorted(
        route_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return {
        'modality': modality,
        'subject': subject,
        'layout': tuple(int(v) for v in layout),
        'paths': paths,
        'n_repeats': len(paths),
        'modal_route': modal_route,
        'modal_route_count': int(modal_count),
        'modal_route_fraction': float(modal_count / len(paths)),
        'n_unique_routes': int(len(route_counts)),
        'preferred_layout': tuple(layout) in TRAJECTORY_PREFERRED_LAYOUTS,
        **metadata,
    }


def fmri_trajectory_candidates():
    """Return one route candidate per fMRI subject/configuration."""
    candidates = []
    for path in sorted(glob.glob(FMRI_BEH_GLOB)):
        subject = os.path.basename(path).split('_')[0]
        if subject in FMRI_EXCLUDE:
            continue
        df = pd.read_csv(path)
        grouper = ['task_half', 'instruction', 'task_config_seq']
        for keys, group in df.groupby(grouper, sort=False):
            layout = _fmri_reward_layout(group)
            if layout is None:
                continue
            paths = []
            for repeat, repeat_group in group.groupby('repeat', sort=True):
                route = _collapse_locations(repeat_group['curr_loc'])
                if route:
                    paths.append((int(repeat), route))
            candidate = _make_trajectory_candidate(
                'fMRI', subject, layout, paths,
                task_half=int(keys[0]), instruction=str(keys[1]),
                task_config_seq=str(keys[2]))
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def ephys_trajectory_candidates():
    """Return one route candidate per cell-data subject/grid.

    The location traces are sampled every 25 ms.  ``timings_rewards`` maps
    each behavioural attempt onto its start and final sample, so the two raw
    files are joined through their shared, original attempt order.  Only the
    ten design-defined correct repeats (0–9) enter these figures.
    """
    sub_dirs = sorted([
        d for d in os.listdir(EPHYS_DERIV)
        if d.startswith('s') and os.path.isdir(
            os.path.join(EPHYS_DERIV, d, 'cells_and_beh'))
    ])
    candidates = []
    skipped = []
    for directory in sub_dirs:
        sub_number = directory[1:]
        subject = f's{sub_number}'
        folder = os.path.join(EPHYS_DERIV, directory, 'cells_and_beh')
        behaviour_path = os.path.join(
            folder, f'all_trial_times_{sub_number}.csv')
        if not os.path.isfile(behaviour_path):
            continue
        behaviour = pd.read_csv(behaviour_path, header=None)
        if behaviour.shape[1] != len(EPHYS_BEH_COLS):
            skipped.append(f'{subject}: unexpected behaviour columns')
            continue
        behaviour.columns = EPHYS_BEH_COLS
        raw_attempts = ephys_loop_table(behaviour, subject)
        attempts = exclude_ephys_attempts(raw_attempts)

        for grid_value, raw_grid in raw_attempts.groupby('grid_no', sort=False):
            grid_no = int(grid_value)
            raw_grid = raw_grid.sort_index()
            layout = tuple(
                raw_grid.iloc[0][['loc_A', 'loc_B', 'loc_C', 'loc_D']]
                .astype(int))
            # The 25-ms location arrays are large.  Inspect the requested
            # layouts only, after the tiny behavioural table has identified
            # them, rather than loading every grid from every participant.
            if layout not in TRAJECTORY_PREFERRED_LAYOUTS:
                continue
            timing_path = os.path.join(
                folder, f'timings_rewards_grid{grid_no}_sub{sub_number}.csv')
            locations_path = os.path.join(
                folder, f'locations_per_25ms_grid{grid_no}_sub{sub_number}.csv')
            if not (os.path.isfile(timing_path)
                    and os.path.isfile(locations_path)):
                skipped.append(f'{subject}, grid {grid_no}: trace file missing')
                continue
            timings = pd.read_csv(timing_path, header=None).to_numpy()
            locations = _read_location_trace(locations_path)
            if len(raw_grid) != len(timings):
                skipped.append(
                    f'{subject}, grid {grid_no}: {len(raw_grid)} attempts but '
                    f'{len(timings)} timing rows')
                continue

            # Index in the unfiltered grid table = row in timings_rewards.
            timing_row = {
                source_index: position
                for position, source_index in enumerate(raw_grid.index)
            }
            good_attempts = attempts[
                attempts['grid_no'].eq(grid_no)
                & attempts['correct'].eq(1)
                & attempts['rep_correct'].between(0, 9)
            ].sort_index()
            paths = []
            for source_index, attempt in good_attempts.iterrows():
                endpoints = timings[timing_row[source_index]]
                if not (np.isfinite(endpoints[0])
                        and np.isfinite(endpoints[-1])):
                    skipped.append(
                        f'{subject}, grid {grid_no}: missing trace boundaries')
                    continue
                start, stop = int(endpoints[0]), int(endpoints[-1])
                if start < 0 or stop < start or stop >= len(locations):
                    skipped.append(
                        f'{subject}, grid {grid_no}: invalid trace boundaries')
                    continue
                route = _collapse_locations(locations[start:stop + 1])
                if route:
                    paths.append((int(attempt['rep_correct']), route))
            candidate = _make_trajectory_candidate(
                'cells', subject, layout, paths,
                grid_no=grid_no,
                session_no=int(raw_grid.iloc[0]['session_no']))
            if candidate is not None:
                candidates.append(candidate)
    if skipped:
        print(f'  trajectory traces skipped ({len(skipped)}): {skipped[0]}')
    return candidates


def _trajectory_sort_key(candidate):
    """Preferred layout first, then increasingly reliable whole routes."""
    return (
        0 if candidate['preferred_layout'] else 1,
        -candidate['modal_route_fraction'],
        -candidate['n_repeats'],
        candidate['n_unique_routes'],
        str(candidate['subject']),
    )


def select_trajectory_examples(candidates, expected_repeats):
    """Pick five highly consistent and three seeded-random subjects.

    Each subject appears at most once.  Complete examples are used whenever
    possible; the fallback is deliberately retained for incomplete datasets
    and is documented in the selection CSV.
    """
    n_needed = TRAJECTORY_N_STABLE + TRAJECTORY_N_RANDOM
    complete = [c for c in candidates
                if c['n_repeats'] >= expected_repeats]
    pool = complete if len({c['subject'] for c in complete}) >= n_needed \
        else list(candidates)

    selected = []
    used_subjects = set()
    for candidate in sorted(pool, key=_trajectory_sort_key):
        if candidate['subject'] in used_subjects:
            continue
        example = dict(candidate)
        example['selection_type'] = 'stable'
        example['selection_rank'] = len(selected) + 1
        selected.append(example)
        used_subjects.add(candidate['subject'])
        if len(selected) == TRAJECTORY_N_STABLE:
            break

    # Retain one best target-layout grid per remaining person, then sample
    # people without replacement.  The seed makes the three comparison plots
    # reproducible across reruns, while leaving them independent of route
    # consistency.
    by_subject = {}
    for candidate in sorted(pool, key=_trajectory_sort_key):
        if candidate['subject'] not in used_subjects:
            by_subject.setdefault(candidate['subject'], candidate)
    random_pool = list(by_subject.values())
    preferred_pool = [c for c in random_pool if c['preferred_layout']]
    if len(preferred_pool) >= TRAJECTORY_N_RANDOM:
        random_pool = preferred_pool
    rng = np.random.default_rng(TRAJECTORY_RANDOM_SEED)
    n_random = min(TRAJECTORY_N_RANDOM, len(random_pool))
    if n_random:
        sampled_indices = rng.choice(len(random_pool), size=n_random,
                                     replace=False)
        for index in sampled_indices:
            example = dict(random_pool[int(index)])
            example['selection_type'] = 'random'
            example['selection_rank'] = len(selected) + 1
            selected.append(example)
    return selected


def _route_text(route):
    return '-'.join(str(location) for location in route)


def _trajectory_edge_counts(candidate):
    """Number of repeats using each directed transition, at most once/repeat."""
    edge_counts = Counter()
    for _, route in candidate['paths']:
        edge_counts.update(set(zip(route[:-1], route[1:])))
    return edge_counts


def _plot_trajectory_grid(ax, candidate, line_style):
    """Draw a coloured location grid and transition-frequency path overlay."""
    if line_style not in TRAJECTORY_LINE_STYLES:
        raise ValueError(f'Unknown trajectory line style: {line_style!r}')

    # A solid, square version of the task's location colour map.  No location
    # numbers are printed here: its position and colour identify each square.
    for location, (x, y) in _GRID_XY.items():
        ax.add_patch(Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor=_LOCATION_COLOURS[location], edgecolor='black',
            linewidth=0.9, zorder=0))

    edge_counts = _trajectory_edge_counts(candidate)
    n_repeats = candidate['n_repeats']
    # Plot weak/rare edges first so the dominant route is never obscured.
    for (start, stop), count in sorted(edge_counts.items(),
                                       key=lambda item: item[1]):
        if start not in _GRID_XY or stop not in _GRID_XY:
            continue
        fraction = count / n_repeats
        if line_style == 'opacity':
            # Same broad path for every transition; low-frequency detours
            # recede through transparency alone.  The full 0–1 range makes
            # each 1/5 or 1/10 increment visibly distinguishable.
            width = 7.2
            alpha = 0.02 + 0.98 * fraction
        else:  # thickness
            # A deliberately wide dynamic range makes every repeat level
            # (1–5 or 1–10) visible.  A small accompanying opacity cue keeps
            # the rarest, very thin detours from competing with the main path.
            width = 0.25 + 12.0 * fraction
            alpha = 0.12 + 0.88 * fraction
        start_xy, stop_xy = _GRID_XY[start], _GRID_XY[stop]
        ax.plot([start_xy[0], stop_xy[0]], [start_xy[1], stop_xy[1]],
                color='black', linewidth=width, alpha=alpha,
                solid_capstyle='round', solid_joinstyle='round', zorder=2)

    # Frame the four reward locations using their A–D identity colours.  The
    # thinner frames sit beneath the path, keeping the task configuration
    # visible without obscuring the route itself.
    for state, location in zip('ABCD', candidate['layout']):
        x, y = _GRID_XY[location]
        ax.add_patch(Rectangle(
            (x - 0.57, y - 0.57), 1.14, 1.14,
            fill=False, edgecolor=_STATE_OUTLINE_COLOURS[state],
            linewidth=2.5, joinstyle='miter', clip_on=False, zorder=1))

    label = (f"{candidate['subject']}  ·  {candidate['selection_type']}\n"
             f"{candidate['modal_route_count']}/{n_repeats} same route")
    ax.set_title(label, fontsize=9, fontname='Arial', pad=3)
    ax.text(1, -0.89, f"A→D: {_route_text(candidate['layout'])}",
            ha='center', va='top', fontsize=8, fontname='Arial')
    ax.set(xlim=(-0.64, 2.64), ylim=(-1.00, 2.64), aspect='equal')
    ax.axis('off')


def plot_sample_trajectories(candidates, modality, expected_repeats):
    """Save opacity and thickness versions of the eight selected examples."""
    selected = select_trajectory_examples(candidates, expected_repeats)
    if not selected:
        print(f'  no usable {modality} trajectory candidates.')
        return

    style_labels = {
        'opacity': 'opacity encodes transition frequency',
        'thickness': 'thickness + opacity encode transition frequency',
    }
    for line_style in TRAJECTORY_LINE_STYLES:
        # The coloured reward frames deliberately extend just beyond each
        # square, so give the two panel rows extra vertical breathing room.
        fig, axes = plt.subplots(2, 4, figsize=(7.35, 5.80))
        # Fixed margins are more reliable here than constrained_layout: each
        # square grid deliberately has a title above and an A–D label below.
        fig.subplots_adjust(left=0.035, right=0.99, bottom=0.060, top=0.86,
                            wspace=0.36, hspace=0.44)
        for axis, candidate in zip(axes.flat, selected):
            _plot_trajectory_grid(axis, candidate, line_style)
        for axis in axes.flat[len(selected):]:
            axis.axis('off')
        fig.suptitle(
            f'{modality} sample trajectories — {style_labels[line_style]}',
            fontname='Arial', fontsize=11, fontweight='bold', y=0.965)
        stem = os.path.join(
            PLOT_DIR,
            f'{modality.lower()}_sample_trajectories_{line_style}')
        for extension in ('.pdf', '.png'):
            fig.savefig(f'{stem}{extension}', dpi=300)
        plt.close(fig)

    records = []
    for candidate in selected:
        records.append({
            'modality': candidate['modality'],
            'selection_type': candidate['selection_type'],
            'selection_rank': candidate['selection_rank'],
            'subject': candidate['subject'],
            'reward_layout_A_to_D': _route_text(candidate['layout']),
            'n_correct_repeats': candidate['n_repeats'],
            'modal_route_count': candidate['modal_route_count'],
            'modal_route_fraction': candidate['modal_route_fraction'],
            'n_unique_routes': candidate['n_unique_routes'],
            'modal_route': _route_text(candidate['modal_route']),
            'routes_by_repeat': ' | '.join(
                f'{repeat + 1}:{_route_text(route)}'
                for repeat, route in candidate['paths']),
            'task_half': candidate.get('task_half'),
            'instruction': candidate.get('instruction'),
            'task_config_seq': candidate.get('task_config_seq'),
            'session_no': candidate.get('session_no'),
            'grid_no': candidate.get('grid_no'),
            'preferred_layout': candidate['preferred_layout'],
        })
    output = os.path.join(PLOT_DIR, f'{modality.lower()}_sample_trajectories_selection.csv')
    pd.DataFrame(records).to_csv(output, index=False)
    print(f'  wrote {len(selected)} {modality} trajectory examples: {output}')


# ── Plots ────────────────────────────────────────────────────────────
# Compact, A4-ready loop-time panels.  The axes are intentionally shared in
# seconds: for fMRI this lets the observed time and the enforced floor be
# compared directly.  A second y-axis for their difference would duplicate
# that information with a changing transformation and be difficult to read
# at this small panel size.
PANEL_WIDTH_CM = 4.0
PANEL_HEIGHT_CM = 2.0
# A common span makes the cell and fMRI panels visually comparable.  5.5 s
# accommodates the fMRI observed mean ± SEM and floor without clipping it.
LOOP_PANEL_Y_SPAN_SECONDS = 5.5
FONT_TICK = 9
FONT_AXIS = 9
FONT_TITLE = 11
_ACTUAL_COLOR = 'black'
_INDIVIDUAL_COLOR = '0.72'
_SEM_COLOR = '0.72'
_FLOOR_COLOR = '0.35'


def _repeat_subject_means(loops_df, x_col, value_col, raw_repeats):
    """Return equal-weight subject trajectories for the requested repeats."""
    table = (loops_df[loops_df[x_col].isin(raw_repeats)]
             .groupby(['subject', x_col])[value_col].mean()
             .unstack(x_col)
             .reindex(columns=raw_repeats))
    return table.to_numpy(dtype=float)


def _mean_and_sem(values):
    """Column-wise mean and SEM, retaining NaN where no subject contributes."""
    n = np.isfinite(values).sum(axis=0)
    mean = np.nanmean(values, axis=0)
    sem = np.full(values.shape[1], np.nan, dtype=float)
    has_sem = n >= 2
    if np.any(has_sem):
        sem[has_sem] = (np.nanstd(values[:, has_sem], axis=0, ddof=1)
                        / np.sqrt(n[has_sem]))
    return mean, sem


def _plot_compact_loop_panel(actual_values, x_values, x_tick_values,
                             x_tick_labels, save_stem, style,
                             floor_values=None):
    """Save a 4 × 2 cm individual-trace or SEM loop-time panel.

    ``actual_values`` and optional ``floor_values`` are subjects × repeats,
    already averaged within subject.  Thus every plotted group value gives
    each participant equal weight regardless of their number of loops.
    """
    if style not in {'individuals', 'sem'}:
        raise ValueError(f'Unknown loop-panel style: {style!r}')
    fig, ax = plt.subplots(
        figsize=(PANEL_WIDTH_CM / 2.54, PANEL_HEIGHT_CM / 2.54))
    # Leave sufficient room for 9 pt labels while retaining a usable panel.
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.34, top=0.96)

    actual_mean, actual_sem = _mean_and_sem(actual_values)
    if style == 'individuals':
        for trace in actual_values:
            ax.plot(x_values, trace, color=_INDIVIDUAL_COLOR, lw=0.45,
                    alpha=0.85, zorder=1)
    else:
        ax.fill_between(x_values, actual_mean - actual_sem,
                        actual_mean + actual_sem, color=_SEM_COLOR,
                        alpha=0.60, linewidth=0, zorder=1)

    # Fat central group mean.
    ax.plot(x_values, actual_mean, color=_ACTUAL_COLOR, lw=1.8,
            marker='o', ms=1.8, zorder=3)

    if floor_values is not None:
        # Show the average enforced delay as one flat reference line.  The
        # per-repeat floor varies slightly because of the jittered timing,
        # but connecting those values makes it look like an intentional
        # decrease in the delay across repeats.
        floor_mean = float(np.nanmean(floor_values))
        ax.axhline(floor_mean, color=_FLOOR_COLOR, lw=1.25,
                   ls='--', zorder=2)

    # Keep every compact panel on exactly the same y-range.  Centre the span
    # on its group-level references (mean ± SEM and, for fMRI, floor), rather
    # than on individual trajectories, so a single unusual participant does
    # not determine the publication-panel scale.
    reference = [actual_mean - actual_sem, actual_mean + actual_sem]
    if floor_values is not None:
        reference.append(np.asarray([floor_mean]))
    reference = np.concatenate(reference)
    reference = reference[np.isfinite(reference)]
    if reference.size:
        centre = (reference.min() + reference.max()) / 2.0
        half_span = LOOP_PANEL_Y_SPAN_SECONDS / 2.0
        ax.set_ylim(centre - half_span, centre + half_span)

    ax.set_xticks(x_tick_values)
    ax.set_xticklabels(x_tick_labels, fontname='Arial', fontsize=FONT_TICK)
    ax.set_xlabel('correct repeat', fontname='Arial', fontsize=FONT_AXIS,
                  labelpad=1)
    ax.set_ylabel('ABCD time [s]', fontname='Arial', fontsize=FONT_AXIS,
                  labelpad=1)
    ax.tick_params(axis='both', labelsize=FONT_TICK, length=2, pad=1)
    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.5)

    for ext in ('.pdf', '.png'):
        fig.savefig(f'{save_stem}{ext}', dpi=300)
    plt.close(fig)


def _plot_compact_error_fraction_hist(error_fractions, save_stem):
    """4 × 2 cm histogram of participant-level ephys error fractions."""
    fractions = np.asarray(error_fractions, dtype=float)
    fractions = fractions[np.isfinite(fractions)]
    if fractions.size == 0:
        return
    fig, ax = plt.subplots(
        figsize=(PANEL_WIDTH_CM / 2.54, PANEL_HEIGHT_CM / 2.54))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.34, top=0.96)
    # The current distribution is contained in 0–0.4.  Retain that readable
    # publication range if future data have a slightly larger maximum.
    upper = max(0.4, np.ceil(fractions.max() * 10) / 10)
    bins = np.linspace(0, upper, 7)
    ax.hist(fractions, bins=bins, color='0.20', edgecolor='white',
            linewidth=0.4)
    ax.set_xlim(0, upper)
    ax.set_xlabel('error fraction', fontname='Arial', fontsize=FONT_AXIS,
                  labelpad=1)
    ax.set_ylabel('# people', fontname='Arial', fontsize=FONT_AXIS,
                  labelpad=1)
    ax.tick_params(axis='both', labelsize=FONT_TICK, length=2, pad=1)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontname('Arial')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.5)
    for ext in ('.pdf', '.png'):
        fig.savefig(f'{save_stem}{ext}', dpi=300)
    plt.close(fig)


def _plot_compact_pooled_error_by_repeat(error_by_repeat, save_stem):
    """4 × 2 cm pooled ephys error fraction at every task repeat."""
    if error_by_repeat.empty:
        return
    fig, ax = plt.subplots(
        figsize=(PANEL_WIDTH_CM / 2.54, PANEL_HEIGHT_CM / 2.54))
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.34, top=0.96)
    x = error_by_repeat['repeat_display'].to_numpy(dtype=float)
    values = error_by_repeat['error_fraction'].to_numpy(dtype=float)
    ax.bar(x, values, width=0.70, color='0.20', edgecolor='black',
           linewidth=0.35)
    upper = max(0.1, np.ceil(values.max() * 10) / 10)
    ax.set_ylim(0, upper)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xticklabels(['2', '4', '6', '8', '10'], fontname='Arial',
                       fontsize=FONT_TICK)
    ax.set_xlabel('correct repeat', fontname='Arial', fontsize=FONT_AXIS,
                  labelpad=1)
    ax.set_ylabel('error fraction', fontname='Arial', fontsize=FONT_AXIS,
                  labelpad=1)
    ax.tick_params(axis='both', labelsize=FONT_TICK, length=2, pad=1)
    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.5)
    for ext in ('.pdf', '.png'):
        fig.savefig(f'{save_stem}{ext}', dpi=300)
    plt.close(fig)


def _plot_compact_ephys_time_with_errors(actual_values, x_values,
                                         error_by_repeat, save_stem):
    """Test panel: mean correct-loop time with pooled error-fraction bars."""
    fig, ax_time = plt.subplots(
        figsize=(PANEL_WIDTH_CM / 2.54, PANEL_HEIGHT_CM / 2.54))
    # Extra right margin accommodates the requested second y-axis while
    # retaining the same 4 × 2 cm figure footprint.
    fig.subplots_adjust(left=0.28, right=0.74, bottom=0.34, top=0.96)
    ax_error = ax_time.twinx()

    error_x = error_by_repeat['repeat_display'].to_numpy(dtype=float)
    error_values = error_by_repeat['error_fraction'].to_numpy(dtype=float)
    ax_error.bar(error_x, error_values, width=0.70, color='0.30', alpha=0.50,
                 edgecolor='none', zorder=0)
    error_upper = max(0.1, np.ceil(error_values.max() * 10) / 10)
    ax_error.set_ylim(0, error_upper)
    ax_error.set_ylabel('error fraction', fontname='Arial', fontsize=FONT_AXIS,
                        labelpad=1)
    ax_error.tick_params(axis='y', labelsize=FONT_TICK, length=2, pad=1)
    for label in ax_error.get_yticklabels():
        label.set_fontname('Arial')
    ax_error.spines['top'].set_visible(False)
    ax_error.spines['right'].set_linewidth(0.5)

    # Draw the time series above the transparent bars.
    ax_error.set_zorder(0)
    ax_time.set_zorder(1)
    ax_time.patch.set_visible(False)
    actual_mean, actual_sem = _mean_and_sem(actual_values)
    ax_time.fill_between(x_values, actual_mean - actual_sem,
                         actual_mean + actual_sem, color=_SEM_COLOR,
                         alpha=0.60, linewidth=0, zorder=2)
    ax_time.plot(x_values, actual_mean, color=_ACTUAL_COLOR, lw=1.8,
                 marker='o', ms=1.8, zorder=3)

    reference = np.concatenate([actual_mean - actual_sem,
                                actual_mean + actual_sem])
    reference = reference[np.isfinite(reference)]
    if reference.size:
        centre = (reference.min() + reference.max()) / 2.0
        half_span = LOOP_PANEL_Y_SPAN_SECONDS / 2.0
        ax_time.set_ylim(centre - half_span, centre + half_span)
    ax_time.set_xticks([2, 4, 6, 8, 10])
    ax_time.set_xticklabels(['2', '4', '6', '8', '10'], fontname='Arial',
                             fontsize=FONT_TICK)
    ax_time.set_xlabel('correct repeat', fontname='Arial', fontsize=FONT_AXIS,
                       labelpad=1)
    ax_time.set_ylabel('ABCD time [s]', fontname='Arial', fontsize=FONT_AXIS,
                       labelpad=1)
    ax_time.tick_params(axis='both', labelsize=FONT_TICK, length=2, pad=1)
    for label in [*ax_time.get_xticklabels(), *ax_time.get_yticklabels()]:
        label.set_fontname('Arial')
    ax_time.spines[['top', 'right']].set_visible(False)
    ax_time.spines[['left', 'bottom']].set_linewidth(0.5)
    for ext in ('.pdf', '.png'):
        fig.savefig(f'{save_stem}{ext}', dpi=300)
    plt.close(fig)


# ── Square stage panels (2.5 x 2.5 cm) ───────────────────────────────
# These are the new stage figures.  At 2.5 cm a panel holds three bars or one
# short curve and nothing else, so every one of them carries a single claim.
# Fonts stay at the project minimum of Arial 9 pt.
SQUARE_CM = 2.5
FONT_SQUARE = 9


def _new_square_axes(left=0.46, bottom=0.34):
    """A 2.5 x 2.5 cm panel.

    The default margins are what a two-line rotated y-label plus tick labels
    need at Arial 9 pt; at this size they take a large share of the panel,
    which is the price of keeping the font at the project minimum.
    """
    fig, ax = plt.subplots(figsize=(SQUARE_CM / 2.54, SQUARE_CM / 2.54))
    fig.subplots_adjust(left=left, right=0.97, bottom=bottom, top=0.95)
    return fig, ax


def _finish_square_axes(ax, xlabel, ylabel):
    if xlabel:
        ax.set_xlabel(xlabel, fontname='Arial', fontsize=FONT_SQUARE,
                      labelpad=1)
    ax.set_ylabel(ylabel, fontname='Arial', fontsize=FONT_SQUARE, labelpad=1)
    ax.tick_params(axis='both', labelsize=FONT_SQUARE, length=2, pad=1)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontname('Arial')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.5)


def _save_square(fig, save_stem):
    for extension in ('.pdf', '.png'):
        fig.savefig(f'{save_stem}{extension}', dpi=300)
    plt.close(fig)


def _plot_square_stage_bars(stage_table, measure, ylabel, save_stem,
                            left=0.46):
    """One bar per stage, mean +- SEM over sessions, with session dots.

    Plots exactly the per-session values in ``stage_table`` — the same
    numbers that enter the repeated-measures tests.  ``left`` needs raising
    when the y tick labels run to three digits, which at this panel size
    costs as much width as the axis label itself.
    """
    fig, ax = _new_square_axes(left=left)
    rng = np.random.default_rng(TRAJECTORY_RANDOM_SEED)
    for position, stage in enumerate(STAGE_ORDER):
        values = stage_table.loc[stage_table['stage'].eq(stage),
                                 measure].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 \
            else 0.0
        ax.bar(position, mean, width=0.72, color=STAGE_COLORS[stage],
               edgecolor='black', linewidth=0.4, zorder=1)
        # Individual sessions, so the reader sees the spread behind the bar.
        ax.scatter(position + rng.uniform(-0.16, 0.16, values.size), values,
                   s=0.9, color='0.35', alpha=0.55, linewidths=0, zorder=2)
        ax.errorbar(position, mean, yerr=sem, color='black', lw=0.7,
                    capsize=1.4, capthick=0.7, zorder=3)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_LABELS[stage] for stage in STAGE_ORDER],
                       fontname='Arial', fontsize=FONT_SQUARE, rotation=45,
                       ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.65, len(STAGE_ORDER) - 0.35)
    _finish_square_axes(ax, None, ylabel)
    _save_square(fig, save_stem)


def _plot_square_stage_stack(stage_table, save_stem):
    """One stacked bar: how the time on task divides between the stages.

    Segments are the mean over sessions of each stage's share, so the bar
    reaches 100 % by construction and shows the composition at a glance.
    """
    fig, ax = _new_square_axes(left=0.52, bottom=0.12)
    bottom = 0.0
    for stage in STAGE_ORDER:
        share = float(stage_table.loc[stage_table['stage'].eq(stage),
                                      'duration_percent'].mean())
        ax.bar(0, share, bottom=bottom, width=0.85,
               color=STAGE_COLORS[stage], edgecolor='black', linewidth=0.4)
        # Label inside the segment, in whichever ink stays readable on it.
        if share >= 12:
            ax.text(0, bottom + share / 2, f'{share:.0f}%', ha='center',
                    va='center', fontname='Arial', fontsize=FONT_SQUARE,
                    color='white' if stage == 'execute' else 'black')
        bottom += share
    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.spines['bottom'].set_visible(False)
    _finish_square_axes(ax, None, '% of time\non task')
    _save_square(fig, save_stem)


def _plot_square_by_attempt(by_attempt, ylabel, save_stem, colour='black'):
    """Mean +- SEM across sessions at each attempt index within a grid.

    ``by_attempt`` is sessions x attempts, already averaged within session,
    so every session carries the same weight.
    """
    values = np.asarray(by_attempt, dtype=float)
    x = np.arange(1, values.shape[1] + 1)
    mean, sem = _mean_and_sem(values)
    fig, ax = _new_square_axes(bottom=0.36)
    ax.fill_between(x, mean - sem, mean + sem, color=colour, alpha=0.25,
                    linewidth=0)
    ax.plot(x, mean, color=colour, lw=1.3, marker='o', ms=1.4)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_xticklabels(['1', '4', '8', '12'], fontname='Arial',
                       fontsize=FONT_SQUARE)
    _finish_square_axes(ax, 'attempt', ylabel)
    _save_square(fig, save_stem)


def _plot_square_hist(values, xlabel, save_stem, bins=None, colour='0.20'):
    """Session-level histogram in the same 2.5 cm format."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    # Two-line x-labels need the extra bottom margin.
    fig, ax = _new_square_axes(left=0.38, bottom=0.46)
    ax.hist(values, bins=12 if bins is None else bins, color=colour,
            edgecolor='white', linewidth=0.3)
    _finish_square_axes(ax, xlabel, '# sessions')
    _save_square(fig, save_stem)


def _plot_square_stage_legend(save_stem):
    """Stand-alone key for the stage colours, to sit next to the panels."""
    fig, ax = plt.subplots(figsize=(SQUARE_CM / 2.54, SQUARE_CM / 2.54))
    ax.axis('off')
    handles = [Rectangle((0, 0), 1, 1, facecolor=STAGE_COLORS[stage],
                         edgecolor='black', linewidth=0.4)
               for stage in STAGE_ORDER]
    ax.legend(handles, list(STAGE_ORDER), frameon=False, loc='center',
              fontsize=FONT_SQUARE, handlelength=1.0, handleheight=1.0,
              labelspacing=0.5, borderpad=0.0,
              prop={'family': 'Arial', 'size': FONT_SQUARE})
    _save_square(fig, save_stem)


def plot_ephys_stage_panels(stage_df, learning_df, attempts):
    """All 2.5 x 2.5 cm stage panels for the cell dataset."""
    panels = [
        ('duration_per_grid', 'time per\ngrid [s]',
         'ephys_stage_time_spent', 0.54),
        ('duration_percent', '% of time\non task',
         'ephys_stage_time_share', 0.46),
        ('attempt_duration_mean', 'attempt\ntime [s]',
         'ephys_stage_attempt_duration', 0.46),
        ('loop_time_mean', 'A\u2192D [s]',
         'ephys_stage_loop_time', 0.46),
        ('incorrect_uncovers_per_attempt', 'wrong\nuncovers',
         'ephys_stage_incorrect_uncovers', 0.46),
        ('error_fraction', 'error\nfraction',
         'ephys_stage_error_fraction', 0.46),
        ('shortest_path_percent', 'shortest\nwalks [%]',
         'ephys_stage_shortest_path', 0.50),
        ('extra_steps_per_walk', 'extra\nsteps',
         'ephys_stage_extra_steps', 0.46),
    ]
    for measure, ylabel, stem, left in panels:
        _plot_square_stage_bars(
            stage_df, measure, ylabel, os.path.join(PLOT_DIR, stem),
            left=left)
    _plot_square_stage_stack(
        stage_df, os.path.join(PLOT_DIR, 'ephys_stage_time_stacked'))
    _plot_square_stage_legend(os.path.join(PLOT_DIR, 'ephys_stage_legend'))

    # Level 1 — every attempt, on the within-grid attempt axis.
    within_grid = attempts[attempts['rep_overall'].between(1, 12)]
    _plot_square_by_attempt(
        _repeat_subject_means(within_grid, 'rep_overall', 'loop_time',
                              list(range(1, 13))),
        'A\u2192D [s]',
        os.path.join(PLOT_DIR, 'ephys_loop_time_by_attempt_all'))
    if within_grid['n_incorrect_uncovers'].notna().any():
        _plot_square_by_attempt(
            _repeat_subject_means(within_grid, 'rep_overall',
                                  'n_incorrect_uncovers',
                                  list(range(1, 13))),
            'wrong\nuncovers',
            os.path.join(PLOT_DIR, 'ephys_incorrect_uncovers_by_attempt'),
            colour=STAGE_COLORS['execute'])

    _plot_square_hist(
        learning_df['attempts_to_criterion'].to_numpy(dtype=float),
        'attempts to\n1st correct',
        os.path.join(PLOT_DIR, 'ephys_attempts_to_criterion'))
    _plot_square_hist(
        learning_df['speedup_percent'].to_numpy(dtype=float),
        'speed-up [%]',
        os.path.join(PLOT_DIR, 'ephys_speedup_percent'))
    _plot_square_hist(
        learning_df['explore_duration'].to_numpy(dtype=float),
        'explore\ntime [s]',
        os.path.join(PLOT_DIR, 'ephys_explore_duration'),
        colour=STAGE_COLORS['learn'])


def _plot_loop_by_repeat(by_rep, title, save_path,
                          rep_label='repeat',
                          floor_mean=None, floor_sd=None):
    reps = sorted(by_rep.keys())
    means = [by_rep[r]['mean'] for r in reps]
    sds   = [by_rep[r]['sd'] if by_rep[r]['sd'] is not None else 0
             for r in reps]
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    if floor_mean is not None:
        ax.axhline(floor_mean, color=_FLOOR_COLOR, ls='--', lw=1.4,
                   label=_FLOOR_LABEL)
        if floor_sd is not None:
            ax.axhspan(floor_mean - floor_sd, floor_mean + floor_sd,
                       color=_FLOOR_COLOR, alpha=0.12)
    ax.errorbar(reps, means, yerr=sds, marker='o',
                color='#2B4159', ecolor='#7191A9', capsize=3, lw=2,
                label='observed loop time (mean ± s.d.)')
    ax.set_xlabel(rep_label)
    ax.set_ylabel('ABCD-loop time [s]')
    ax.set_title(title, fontsize=11)
    if floor_mean is not None:
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_per_subj_loop(loops_df, x_col, title, save_path,
                          floor_series_col=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    for sub, g in loops_df.groupby('subject'):
        per_rep = g.groupby(x_col)['loop_time'].mean().sort_index()
        ax.plot(per_rep.index, per_rep.values, color='0.7', alpha=0.5, lw=0.7)
    group_mean = loops_df.groupby(x_col)['loop_time'].mean().sort_index()
    ax.plot(group_mean.index, group_mean.values, color='#9A383C', lw=2.5,
            label='observed (group mean)')
    if floor_series_col and floor_series_col in loops_df.columns:
        floor_mean = (loops_df.groupby(x_col)[floor_series_col]
                              .mean().sort_index())
        ax.plot(floor_mean.index, floor_mean.values, color=_FLOOR_COLOR,
                ls='--', lw=2.0, label='enforced floor (group mean)')
    ax.set_xlabel(x_col)
    ax.set_ylabel('ABCD-loop time [s]')
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_hist(values, title, xlabel, save_path, floor_values=None):
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    # If a floor distribution is supplied, share a common bin range so the
    # two histograms are directly comparable.
    if floor_values is not None:
        floor = np.asarray(floor_values, dtype=float)
        floor = floor[np.isfinite(floor)]
        lo = float(np.nanmin(np.concatenate([vals, floor]))) - 0.5
        hi = float(np.nanmax(np.concatenate([vals, floor]))) + 0.5
        bins = np.linspace(lo, hi, 14)
        ax.hist(vals, bins=bins, color='#BC7E6A', edgecolor='white',
                label='observed')
        ax.hist(floor, bins=bins, color=_FLOOR_COLOR, edgecolor='white',
                alpha=0.55, label='enforced floor')
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.hist(vals, bins=12, color='#BC7E6A', edgecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('# subjects')
    ax.set_title(title, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def print_stage_report(group):
    """Print the stage comparison in the form the methods section needs."""
    stages = group['stages']
    print("\n--- Task stages (cells, n = "
          f"{group['n_sessions']} sessions, {group['n_grids_total']} grids) ---")
    print(stages['definition'])
    print(f"grids without any error-free loop: "
          f"{stages['n_grids_without_error_free_loop']}")
    for measure, result in stages['tests'].items():
        anova = result['rm_anova']
        means = ', '.join(
            f"{stage} {result['stage_means'][stage]['mean']:.2f}"
            for stage in STAGE_ORDER
            if result['stage_means'][stage]['mean'] is not None)
        print(f"\n{result['label']}: {means}")
        if 'caveat' in result:
            print(f"  CAVEAT: {result['caveat']}")
        if anova is not None:
            print(f"  RM-ANOVA F({anova['df1']},{anova['df2']}) = "
                  f"{anova['F']:.2f}, p = {anova['p']:.3g}, "
                  f"partial eta^2 = {anova['partial_eta_sq']:.2f} "
                  f"(n = {anova['n_sessions']})")
        for contrast in result['contrasts']:
            dz = contrast['cohens_dz']
            print(f"  {contrast['contrast']}: "
                  f"diff = {contrast['mean_difference']:.2f}, "
                  f"t({contrast['n_sessions'] - 1}) = {contrast['t']:.2f}, "
                  f"p_holm = {contrast['p_holm']:.3g}"
                  + (f", dz = {dz:.2f}" if dz is not None else ""))
    learning = group['learning']
    print("\n--- Learning (cells) ---")
    for measure in ('explore_duration', 'attempts_to_criterion',
                    'time_to_criterion', 'incorrect_uncovers_to_criterion',
                    'loop_time_first_repeat', 'loop_time_last_repeat',
                    'speedup_seconds', 'speedup_percent'):
        values = learning[measure]
        sd = values['sd']
        print(f"  {measure}: {values['mean']:.2f}"
              + (f" ± {sd:.2f}" if sd is not None else "")
              + f" (median {values['median']:.2f}, n = {values['n']})")
    if 'speedup_percent_vs_0' in learning:
        test = learning['speedup_percent_vs_0']
        print(f"  speed-up vs. 0: t({test['n_sessions'] - 1}) = "
              f"{test['t']:.2f}, p = {test['p']:.3g}")


# ── Run ───────────────────────────────────────────────────────────────
print("\n=== fMRI ===")
fmri_subj, fmri_loops, fmri_group = fmri_summarise()
print("\n=== Ephys ===")
(eph_sess, eph_attempts, eph_group,
 eph_stages, eph_learning, eph_paths) = ephys_summarise()

# Compact loop-time panels.  These are the matched cell/fMRI figures intended
# for a DIN A4 layout: each modality gets an individual-trajectory version and
# a mean ± SEM version.
if fmri_group is not None:
    # The fMRI task has five repeats (stored 0–4); display them as 1–5.
    fmri_repeats_raw = list(range(5))
    fmri_x = np.arange(1, 6)
    fmri_actual = _repeat_subject_means(
        fmri_loops, 'repeat', 'loop_time', fmri_repeats_raw)
    fmri_floor = _repeat_subject_means(
        fmri_loops, 'repeat', 'floor_loop_time', fmri_repeats_raw)
    _plot_compact_loop_panel(
        fmri_actual, fmri_x, fmri_x, [str(x) for x in fmri_x],
        os.path.join(PLOT_DIR, 'fmri_loop_time_individuals'),
        style='individuals', floor_values=fmri_floor)
    _plot_compact_loop_panel(
        fmri_actual, fmri_x, fmri_x, [str(x) for x in fmri_x],
        os.path.join(PLOT_DIR, 'fmri_loop_time_sem'),
        style='sem', floor_values=fmri_floor)
    _plot_hist(
        fmri_subj['loop_time_mean'].to_numpy(),
        'fMRI — subject-mean loop time',
        'loop time [s]',
        os.path.join(PLOT_DIR, 'fmri_subj_mean_hist.png'),
        floor_values=fmri_subj['floor_loop_time_mean'].to_numpy())
    _plot_hist(
        fmri_subj['shortest_path_percent'].to_numpy(dtype=float),
        'fMRI — shortest paths between consecutive rewards',
        '% of reward-to-reward paths that were shortest',
        os.path.join(PLOT_DIR, 'fmri_shortest_path_percent_hist.png'))
    _plot_hist(
        fmri_subj['slack_mean'].to_numpy(),
        'fMRI — per-subject voluntary slack (observed − floor)',
        'slack [s]',
        os.path.join(PLOT_DIR, 'fmri_slack_hist.png'))
    # Observed loop time vs. enforced floor — slack = headroom to speed up.
    fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
    ax.scatter(fmri_subj['floor_loop_time_mean'],
               fmri_subj['loop_time_mean'],
               color='#2B4159', s=24)
    lo = min(fmri_subj['floor_loop_time_mean'].min(),
             fmri_subj['loop_time_mean'].min()) * 0.95
    hi = max(fmri_subj['floor_loop_time_mean'].max(),
             fmri_subj['loop_time_mean'].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='y = x (no slack)')
    ax.set_xlabel('enforced floor [s] (sum length_step + reward_delay)')
    ax.set_ylabel('observed loop time [s]')
    ax.set_title('fMRI — observed vs. enforced floor per subject',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(PLOT_DIR, 'fmri_floor_vs_observed.png'),
                dpi=200)
    plt.close(fig)
    # forw vs backw subj-level scatter
    fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
    forw = (fmri_loops[fmri_loops['instruction'] == 'forw']
            .groupby('subject')['loop_time'].mean())
    backw = (fmri_loops[fmri_loops['instruction'] == 'backw']
             .groupby('subject')['loop_time'].mean())
    common = forw.index.intersection(backw.index)
    ax.scatter(forw.loc[common], backw.loc[common], color='#2B4159')
    lim_lo = min(forw.loc[common].min(), backw.loc[common].min()) * 0.95
    lim_hi = max(forw.loc[common].max(), backw.loc[common].max()) * 1.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=0.8)
    ax.set_xlabel('forw mean loop time [s]')
    ax.set_ylabel('backw mean loop time [s]')
    ax.set_title('fMRI — forw vs. backw per subject', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(PLOT_DIR, 'fmri_forw_vs_backw.png'), dpi=200)
    plt.close(fig)

if eph_group is not None:
    # Keep exactly the ten design-defined correct repeats.  The data store
    # them as rep_correct=0–9, but the figure displays human-readable 1–10.
    correct = eph_attempts[(eph_attempts['correct'] == 1)
                           & (eph_attempts['rep_correct'].between(0, 9))].copy()
    ephys_repeats_raw = list(range(10))
    ephys_x = np.arange(1, 11)
    ephys_actual = _repeat_subject_means(
        correct, 'rep_correct', 'loop_time', ephys_repeats_raw)
    _plot_compact_loop_panel(
        ephys_actual, ephys_x, [2, 4, 6, 8, 10],
        ['2', '4', '6', '8', '10'],
        os.path.join(PLOT_DIR, 'ephys_loop_time_individuals'),
        style='individuals')
    _plot_compact_loop_panel(
        ephys_actual, ephys_x, [2, 4, 6, 8, 10],
        ['2', '4', '6', '8', '10'],
        os.path.join(PLOT_DIR, 'ephys_loop_time_sem'),
        style='sem')
    # Error summary panels.  Histogram: one overall error fraction per
    # participant.  By-repeat bars: pooled fraction of all attempts that were
    # incorrect at that repeat (including errors made before the first correct
    # completion of the repeat).
    ephys_errors_by_repeat = ephys_error_fraction_by_repeat(eph_attempts)
    ephys_errors_by_repeat.to_csv(
        os.path.join(PLOT_DIR, 'ephys_error_fraction_by_repeat.csv'),
        index=False)
    _plot_compact_error_fraction_hist(
        eph_sess['incorrect_proportion'].to_numpy(),
        os.path.join(PLOT_DIR, 'ephys_error_fraction_histogram'))
    _plot_compact_pooled_error_by_repeat(
        ephys_errors_by_repeat,
        os.path.join(PLOT_DIR, 'ephys_error_fraction_by_repeat'))
    _plot_compact_ephys_time_with_errors(
        ephys_actual, ephys_x, ephys_errors_by_repeat,
        os.path.join(PLOT_DIR, 'ephys_loop_time_sem_with_error_fraction'))
    _plot_hist(
        eph_sess['shortest_path_percent'].to_numpy(dtype=float),
        'Cells — shortest paths between consecutive rewards',
        '% of reward-to-reward paths that were shortest',
        os.path.join(PLOT_DIR, 'ephys_shortest_path_percent_hist.png'))
    _plot_hist(
        eph_sess['completion_rate'].to_numpy(),
        'Ephys — session completion rate',
        'fraction correct',
        os.path.join(PLOT_DIR, 'ephys_completion_hist.png'))
    _plot_hist(
        eph_sess['incorrect_proportion'].to_numpy(),
        'Ephys — proportion of incorrect attempts per session',
        'fraction incorrect',
        os.path.join(PLOT_DIR, 'ephys_incorrect_proportion_hist.png'))
    _plot_hist(
        eph_sess['n_incorrect_attempts'].to_numpy(),
        'Ephys — # incorrect attempts per session',
        '# incorrect attempts',
        os.path.join(PLOT_DIR, 'ephys_incorrect_hist.png'))
    _plot_hist(
        eph_sess['n_configs_solved'].to_numpy(),
        'Ephys — # reward configurations solved per session',
        '# configurations solved (≥1 correct trial)',
        os.path.join(PLOT_DIR, 'ephys_configs_solved_hist.png'))

    # Level 1 (all attempts) and the stage comparison.  These are the new
    # 2.5 x 2.5 cm panels.
    plot_ephys_stage_panels(eph_stages, eph_learning, eph_attempts)
    print_stage_report(eph_group)


if PLOT_SAMPLE_TRAJECTORIES:
    print("\n=== Sample trajectories ===")
    if fmri_group is not None:
        plot_sample_trajectories(
            fmri_trajectory_candidates(), 'fMRI', expected_repeats=5)
    if eph_group is not None:
        plot_sample_trajectories(
            ephys_trajectory_candidates(), 'cells', expected_repeats=10)


# ── Final combined JSON ──────────────────────────────────────────────
combined = {
    'meta': {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'out_dir':   OUT_DIR,
        'definitions': {
            'loop_time': "fMRI: t_curr_rew(D) − t_curr_loc of first row of "
                         "that (cfg, repeat). Ephys: t_D − t_A per attempt.",
            'session':   "Defined by the rewarded-location tuple "
                         "(loc_A, loc_B, loc_C, loc_D).",
            'shortest_path_percent':
                "Percentage of walks between two consecutive rewards "
                "(D->A, A->B, B->C, C->D) that took the minimum possible "
                "number of steps, i.e. the Manhattan distance on the "
                "4-connected 3x3 grid. fMRI: steps from the cleaned "
                "behavioural table; cells: steps from the 25-ms location "
                "traces, correct repeats (rep_correct 0-9) only. The D->A "
                "walk is only counted when the preceding repeat was itself "
                "completed, and the first A of a task configuration is "
                "excluded because its starting location is not a reward. "
                "Group values weight each subject/session equally; "
                "'_pooled' weights each walk equally.",
            'fmri_exclude': sorted(FMRI_EXCLUDE),
            'ephys_scope':  "All sessions with cells_and_beh/all_trial_times_*.csv (n=63).",
            'ephys_loop_time_repeat_axis': "rep_correct (0-9), correct trials only.",
            'ephys_two_levels': (
                "Every cell-data measure is reported at two levels. Level 1 "
                "('all_attempts' / rep_overall axis) uses every attempt, "
                "including those containing an incorrect uncover; it is what "
                "describes the behaviour. Level 2 (the top-level entries / "
                "rep_correct axis) keeps only error-free repeats, which is "
                "the subset the ABCD-code analyses run on."),
            'ephys_stages': (
                "explore = first attempt of a grid; learn = further attempts "
                "up to and including the first error-free loop; execute = "
                "attempts after it. Defined by the information available to "
                "the subject, so stage-wise error and timing measures are "
                "not circular with the definition. Compared with a one-way "
                "repeated-measures ANOVA over sessions plus Holm-corrected "
                "paired contrasts."),
            'attempt_duration': (
                "t_D - new_grid_onset. new_grid_onset is the start of the "
                "attempt: grid onset for the first attempt of a grid, and "
                "the previous attempt's t_D thereafter."),
            'incorrect_uncovers': (
                "Moves with pressed_to_uncover == 1 and correct_uncover == 0, "
                "from scripts/extract_uncovers_ephys.py."),
            'sample_trajectory_figures': {
                'enabled': PLOT_SAMPLE_TRAJECTORIES,
                'preferred_layouts_A_to_D': [
                    list(layout) for layout in TRAJECTORY_PREFERRED_LAYOUTS],
                'n_stable_examples': TRAJECTORY_N_STABLE,
                'n_random_examples': TRAJECTORY_N_RANDOM,
                'random_seed': TRAJECTORY_RANDOM_SEED,
            },
        },
    },
    'fmri':  fmri_group,
    'ephys': eph_group,
}
out_json = os.path.join(OUT_DIR, 'behaviour_summary.json')
with open(out_json, 'w') as f:
    json.dump(combined, f, indent=2,
              default=lambda o: float(o) if hasattr(o, 'item') else str(o))
print(f"\nWrote {out_json}")
print(f"All outputs under: {OUT_DIR}")
