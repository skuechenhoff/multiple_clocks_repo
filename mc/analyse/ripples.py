#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared functions for the hippocampal ripple analyses.

Everything both ripple scripts need lives here, so a change to how a condition
is defined or a test is run happens in one place:

    scripts/swr_ripple_tests.py   the conditions we settled on, tested and plotted
    scripts/swr_explore.py        scratch probes, nothing claimed

Sections:
    1) Loading            the bundle written by the cluster
    2) Task conditions    stages, valence, and which reward was being sought
    3) Rates              exposure-corrected ripple rate in windows and over time
    4) Tests              sliding window vs the trial's own baseline, with a
                          cluster permutation over window positions
    5) Plots

Design parameters are Sakon & Kahana (2022, PNAS 119:e2201657119) and He et al.
(2026, Nat Neurosci 29:1711), not tuned on this dataset.

@author: Svenja Kuchenhoff
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import stats

import mc.analyse.swr_io as swr_io


# ── Settings ──────────────────────────────────────────────────────────
BIN_S           = 0.100      # peri-event histogram bin (Sakon)
BASELINE_WIN    = (-1.6, -1.1)   # the same window shifted 1 s earlier (Sakon Eq. 2)
DEDUP_S         = 2.0        # events closer than this share ripples; drop the later
HALF_S          = 2.0        # peri-event window half-width
SLIDE_WIDTHS_S  = (0.3, 0.5) # sliding window widths to report
MIN_CLEAN_FRAC  = 0.5        # a window must be at least half artifact-free
MIN_EVENTS      = 20         # below this a condition is not estimated
N_SIGN_FLIPS    = 1000       # permutations for the cluster test
CLUSTER_ALPHA   = 0.05       # cluster-forming threshold, two-sided

STAGES = ('first uncovers', 'while learning', 'once known')
VALENCE = ('correct', 'error')
REWARDS = ('A', 'B', 'C', 'D')

# Feedback valence sets the hue, stage sets the lightness, so a crossed label
# like "error, while learning" is never drawn the same colour as its partner.
VALENCE_COLOUR = {'correct': '#0E3D3A', 'error': '#B03A5B'}
STAGE_LIGHTEN  = {'first uncovers': 0.0, 'while learning': 0.35, 'once known': 0.65}

# When the four rewards are the thing being compared -- every row of `full` --
# they take the project's fixed A-D ramp (CLAUDE.md), not a valence hue.
REWARD_COLOUR = {'A': '#F15A29', 'B': '#F7931E', 'C': '#C7C6E2', 'D': '#6B60AA'}

# When the three stages of ONE reward are compared -- the `stage` test -- the
# stage is the variable, so it gets its own scale: what the participant is
# doing at that point, first seeing it / planning the route / executing a known
# route.
STAGE_COLOUR = {'first uncovers': '#6E1410',      # first seeing  - dark red
                'while learning': '#1D4B54',      # planning      - dark blue
                'once known':     '#667F6C'}      # executing     - sage


# ── 1) Loading ────────────────────────────────────────────────────────

def load_bundle(bundle_dir):
    """Read the bundle the cluster wrote. Everything downstream starts here."""
    out = {}
    for name in ('ripples', 'intervals', 'channel_qc', 'behaviour', 'uncover',
                 'pairs'):
        path = os.path.join(bundle_dir, f'{name}.csv')
        out[name] = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()
    return out


def sessions_in(bundle):
    return sorted(bundle['ripples'].session.unique())


def derivations(bundle, session):
    """(pair_id, ripple times, artifact-free intervals) for each good derivation."""
    rip = bundle['ripples']
    iv = bundle['intervals']
    qc = bundle['channel_qc']
    qc_s = qc[qc.session == session].set_index('pair_id')
    out = []
    for pair_id, e in rip[rip.session == session].groupby('pair_id'):
        if pair_id in qc_s.index and bool(qc_s.loc[pair_id, 'excluded']):
            continue
        intervals = iv[(iv.session == session) & (iv.pair_id == pair_id)]
        intervals = intervals[['start_s', 'stop_s']].to_numpy()
        if not len(intervals):
            continue
        out.append((pair_id, e.t_peak_s.to_numpy(float), intervals))
    return out


def subject_of(bundle, session):
    r = bundle['ripples']
    hit = r[r.session == session]
    return str(hit.subject_key.iloc[0]) if len(hit) else f's{session:02d}'


# ── 2) Task conditions ────────────────────────────────────────────────

def stage_of_repeat(beh_one_grid):
    """Learning stage of every repeat of one grid.

        first uncovers   the first traversal
        while learning   up to AND INCLUDING the first fully correct repeat
        once known       every repeat after that

    The boundary is deliberate: the first error-free repeat is the one on which
    the route is first demonstrated, not yet merely relied on.
    """
    g = beh_one_grid.sort_values('rep_overall')
    reps = g.rep_overall.to_numpy(int)
    solved = reps[g.correct.to_numpy(int) == 1]
    first_solved = int(solved[0]) if solved.size else np.inf
    out = {}
    for rep in reps:
        if rep == reps[0]:
            out[int(rep)] = 'first uncovers'
        elif rep <= first_solved:
            out[int(rep)] = 'while learning'
        else:
            out[int(rep)] = 'once known'
    return out


def uncover_table(bundle, session):
    """Every uncovering attempt, labelled with everything a condition needs.

    Columns: t_s, valence (correct/error), stage, reward, grid_no, rep_overall.

    `reward` is the reward the subject was SEEKING. A correct uncovering names
    its own reward; an error inherits the one being searched for, counted as the
    (k+1)th after k rewards have already been collected in that repeat.
    """
    beh = bundle['behaviour']
    beh = beh[beh.session == session]
    unc = bundle['uncover']
    unc = unc[unc.session == session]
    if not len(beh) or not len(unc):
        return pd.DataFrame()

    stages, reward_times = {}, {}
    for grid, g in beh.groupby('grid_no'):
        stages.update({(int(grid), r): s for r, s in stage_of_repeat(g).items()})
        for _, row in g.iterrows():
            reward_times[(int(grid), int(row.rep_overall))] = [
                float(row[f't_{x}']) for x in REWARDS]

    rows = []
    for e in unc.itertuples():
        key = (int(e.grid_no), int(e.rep_overall))
        if key not in stages or key not in reward_times:
            continue
        collected = sum(1 for t in reward_times[key]
                        if np.isfinite(t) and t < e.t_s)
        reward = REWARDS[min(collected, 3)]
        if int(e.correct) == 1 and isinstance(e.state, str):
            reward = e.state
        rows.append({'t_s': float(e.t_s), 'grid_no': int(e.grid_no),
                     'rep_overall': int(e.rep_overall),
                     'valence': 'correct' if int(e.correct) == 1 else 'error',
                     'stage': stages[key], 'reward': reward})
    return pd.DataFrame(rows)


def dedup(times, min_gap_s=DEDUP_S):
    """Drop events within `min_gap_s` of the previous one (Sakon).

    With a +-2 s analysis window, two events closer than that share ripples,
    and those ripples would then be counted as independent observations.
    """
    t = np.sort(np.asarray(times, float))
    t = t[np.isfinite(t)]
    if not t.size:
        return t
    return t[np.concatenate([[True], np.diff(t) >= min_gap_s])]


# ── 3) Rates ──────────────────────────────────────────────────────────

def clean_seconds(intervals, starts, stops):
    """Artifact-free seconds inside each [start, stop]."""
    iv = np.asarray(intervals, float).reshape(-1, 2)
    if not len(iv):
        return np.zeros(len(starts))
    iv = iv[np.argsort(iv[:, 0])]
    dur = np.diff(iv, axis=1).ravel()
    xs = np.empty(2 * len(iv))
    ys = np.empty(2 * len(iv))
    xs[0::2], xs[1::2] = iv[:, 0], iv[:, 1]
    cum = np.concatenate([[0.0], np.cumsum(dur)])
    ys[0::2], ys[1::2] = cum[:-1], cum[1:]
    return np.interp(stops, xs, ys) - np.interp(starts, xs, ys)


def rate_in_window(event_times, ripple_times, intervals, window):
    """Exposure-corrected ripple rate in `window` around each event.

    NaN where the window is more than half artifact, rather than an unstable
    rate from a sliver of clean time.
    """
    ev = np.asarray(event_times, float)
    starts, stops = ev + window[0], ev + window[1]
    t = np.sort(np.asarray(ripple_times, float))
    n = (np.searchsorted(t, stops, side='right')
         - np.searchsorted(t, starts, side='left')).astype(float)
    exposure = clean_seconds(intervals, starts, stops)
    too_dirty = exposure < MIN_CLEAN_FRAC * (window[1] - window[0])
    rate = np.where(too_dirty, np.nan,
                    n / np.where(exposure > 0, exposure, np.nan))
    return rate, n, exposure


def peri_event_rate(event_times, ripple_times, intervals, half_s=HALF_S,
                    bin_s=BIN_S):
    """Ripple rate in bins around each event. Returns (bin centres, rate)."""
    edges = np.arange(-half_s, half_s + bin_s / 2, bin_s)
    centres = edges[:-1] + bin_s / 2
    ev = np.asarray(event_times, float)
    t = np.sort(np.asarray(ripple_times, float))
    out = np.full((len(ev), len(centres)), np.nan)
    for k in range(len(centres)):
        starts, stops = ev + edges[k], ev + edges[k + 1]
        n = (np.searchsorted(t, stops, side='right')
             - np.searchsorted(t, starts, side='left')).astype(float)
        exposure = clean_seconds(intervals, starts, stops)
        # `exposure > 0` is not a sufficient guard. A bin overlapping a clean
        # interval by microseconds passes it, and n/exposure then returns a
        # rate of 1e8 Hz which dominates every average it enters -- visible as
        # a 1e8 y-axis on `full`, correct/once known. MIN_CLEAN_FRAC already
        # states the intended rule ("a window must be at least half
        # artifact-free") and `window_test` enforces it; this did not.
        out[:, k] = np.where(exposure >= MIN_CLEAN_FRAC * bin_s,
                             n / np.maximum(exposure, 1e-12), np.nan)
    return centres, out


def rate_by_unit(bundle, events_per_session, unit='subject',
                 min_events=MIN_EVENTS, half_s=HALF_S, bin_s=BIN_S):
    """Peri-event rate per analysis unit, averaged over its derivations.

    `unit` is the thing the group test treats as independent:

      'derivation' one value per bipolar pair per session -- the electrode
                 level. Most units, least independence: pairs on the same probe
                 in the same session see overlapping tissue.
      'session'  one value per session, pairs pooled first. The convention used
                 elsewhere in this project -- a cell recorded on a second day
                 through the same electrode counts as a separate cell.
      'subject'  one value per patient, sessions pooled first. Conservative:
                 16 of 41 patients gave 2-3 sessions, sharing electrodes,
                 anatomy and physiology.

    All three are reported rather than one being chosen silently. They differ
    in what they treat as exchangeable, not in the data that goes in.

    `min_events` is the per-session floor AFTER dedup. A session below it
    contributes nothing to that condition.

    Returns (bin centres, {unit_key: profile}, counts).
    """
    per_unit, centres = {}, None
    counts = {'n_sessions': 0, 'n_derivations': 0,
              'n_events_raw': 0, 'n_events_used': 0, 'n_sessions_dropped': 0}
    for session, raw in events_per_session.items():
        t = dedup(raw)
        counts['n_events_raw'] += len(raw)
        if t.size < min_events:
            counts['n_sessions_dropped'] += 1
            continue
        counts['n_sessions'] += 1
        counts['n_events_used'] += int(t.size)
        for pair_id, ripples, intervals in derivations(bundle, session):
            counts['n_derivations'] += 1
            centres, profile = peri_event_rate(t, ripples, intervals,
                                               half_s=half_s, bin_s=bin_s)
            if unit == 'subject':
                key = subject_of(bundle, session)
            elif unit == 'derivation':
                key = (int(session), str(pair_id))
            else:
                key = int(session)
            per_unit.setdefault(key, []).append(np.nanmean(profile, axis=0))
    per_unit = {k: np.nanmean(np.vstack(v), axis=0) for k, v in per_unit.items()}
    counts['n_subjects'] = len({subject_of(bundle, s)
                                for s in events_per_session
                                if dedup(events_per_session[s]).size >= min_events})
    counts['n_units'] = len(per_unit)
    counts['unit'] = unit
    counts['min_events'] = min_events
    return centres, per_unit, counts


def rate_by_subject(bundle, events_per_session, half_s=HALF_S, bin_s=BIN_S):
    """Backwards-compatible wrapper: `rate_by_unit` with unit='subject'."""
    return rate_by_unit(bundle, events_per_session, unit='subject',
                        half_s=half_s, bin_s=bin_s)


# ── 4) Tests ──────────────────────────────────────────────────────────

def baseline_subtract(profiles, centres, baseline=BASELINE_WIN):
    """Subtract each subject's OWN baseline (Sakon Eq. 2).

    This is what makes conditions with different overall rates comparable: a
    transient is measured against the same trial's floor, so a between-condition
    baseline difference cancels instead of masquerading as an effect.
    """
    subjects = sorted(profiles)
    X = np.vstack([profiles[s] for s in subjects])
    in_base = (centres >= baseline[0]) & (centres < baseline[1])
    return subjects, X - np.nanmean(X[:, in_base], axis=1, keepdims=True)


def _clusters(t_values, threshold):
    """Runs of consecutive positions exceeding the threshold, either sign."""
    over = np.abs(t_values) > threshold
    out, i = [], 0
    while i < len(over):
        if over[i]:
            j = i
            while j + 1 < len(over) and over[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def _smooth_and_t(X, centres, width_s, bin_s):
    """Moving-average each subject's course, then a one-sample t per position."""
    k = max(int(round(width_s / bin_s)), 1)
    if X.shape[1] < k or X.shape[0] < 3:
        return None, None, None
    kernel = np.ones(k) / k
    smoothed = np.vstack([np.convolve(row, kernel, mode='valid') for row in X])
    lo = k // 2
    times = np.asarray(centres, float)[lo:lo + smoothed.shape[1]]
    t = np.asarray(stats.ttest_1samp(smoothed, 0.0, nan_policy='omit').statistic,
                   float)
    return smoothed, times, t


def sliding_window_test(profiles_by_condition, centres, width_s=0.3,
                        bin_s=BIN_S, n_perm=N_SIGN_FLIPS, seed=42,
                        alpha=CLUSTER_ALPHA, correct_over='time'):
    """Test every window position in every condition, then correct.

    No window is chosen: a window of `width_s` is a moving average of
    width_s/bin_s bins, so every position is evaluated and the multiple
    comparisons are handled by a cluster-mass permutation. Subjects' signs are
    flipped at random, which is the exact null for a within-subject contrast.

    correct_over
        'time'                 family-wise across window POSITIONS, separately
                               for each condition. A p of 0.02 means a cluster
                               this large appears anywhere in the time course 2%
                               of the time -- but running six conditions then
                               gives six such tests, uncorrected between them.
        'time_and_conditions'  family-wise across positions AND conditions. The
                               same sign-flip is applied to a subject in every
                               condition, which preserves the dependence between
                               conditions that share subjects, and the null takes
                               the maximum cluster mass over the whole family.
                               This is the honest p when several conditions are
                               inspected together, and it is stricter.

    `profiles_by_condition` is {label: {subject: time course}}, already
    baseline-subtracted. Returns {label: result dict} with the same fields as
    before, plus the correction actually applied.
    """
    prepared = {}
    for label, profiles in profiles_by_condition.items():
        subjects = sorted(profiles)
        X = np.vstack([profiles[s] for s in subjects])
        smoothed, times, t_obs = _smooth_and_t(X, centres, width_s, bin_s)
        if smoothed is None:
            continue
        prepared[label] = {'subjects': subjects, 'smoothed': smoothed,
                           'times': times, 't': t_obs,
                           'threshold': float(stats.t.ppf(1 - alpha / 2,
                                                          smoothed.shape[0] - 1))}
    if not prepared:
        return {}

    for d in prepared.values():
        d['found'] = _clusters(d['t'], d['threshold'])
        d['mass'] = [float(np.nansum(np.abs(d['t'][a:b]))) for a, b in d['found']]

    rng = np.random.default_rng(seed)
    all_subjects = sorted({s for d in prepared.values() for s in d['subjects']})
    shared = correct_over == 'time_and_conditions'
    nulls = {label: np.zeros(n_perm) for label in prepared}
    pooled = np.zeros(n_perm)

    for i in range(n_perm):
        # one flip per subject, reused across conditions when correcting over
        # the family, so conditions sharing subjects stay correlated in the null
        flip_of = {s: rng.choice([-1.0, 1.0]) for s in all_subjects}
        biggest = 0.0
        for label, d in prepared.items():
            if shared:
                signs = np.array([[flip_of[s]] for s in d['subjects']])
            else:
                signs = rng.choice([-1.0, 1.0], size=(len(d['subjects']), 1))
            t_i = np.asarray(stats.ttest_1samp(d['smoothed'] * signs, 0.0,
                                               nan_policy='omit').statistic,
                             float)
            masses = [np.nansum(np.abs(t_i[a:b]))
                      for a, b in _clusters(t_i, d['threshold'])]
            m = max(masses, default=0.0)
            nulls[label][i] = m
            biggest = max(biggest, m)
        pooled[i] = biggest

    out = {}
    for label, d in prepared.items():
        null = pooled if shared else nulls[label]
        clusters = []
        for (a, b), mass in zip(d['found'], d['mass']):
            p = float((1 + np.sum(null >= mass)) / (1 + n_perm))
            peak = a + int(np.nanargmax(np.abs(d['t'][a:b])))
            clusters.append({'start_s': float(d['times'][a]),
                             'stop_s': float(d['times'][b - 1]),
                             'peak_s': float(d['times'][peak]),
                             'peak_t': float(d['t'][peak]), 'mass': mass,
                             'p': p,
                             'direction': 'increase' if d['t'][peak] > 0
                                          else 'decrease'})
        out[label] = {'times': d['times'], 't': d['t'],
                      'threshold': d['threshold'], 'clusters': clusters,
                      'null_mass': null, 'width_s': width_s,
                      'n_subjects': int(d['smoothed'].shape[0]),
                      'corrected_over': ('window positions and conditions'
                                         if shared else 'window positions'),
                      'n_conditions_in_family': len(prepared) if shared else 1}
    return out


def window_test(profiles, centres, window, baseline=BASELINE_WIN,
                n_perm=10000, seed=42):
    """One named window against the trial's own baseline, per subject.

    Reported alongside the sliding test because a named window is easier to
    quote, not because it is the primary result -- it involves a choice the
    sliding version does not.
    """
    subjects, X = baseline_subtract(profiles, centres, baseline)
    in_win = (centres >= window[0]) & (centres < window[1])
    values = np.nanmean(X[:, in_win], axis=1)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return None
    t, p = stats.ttest_1samp(values, 0.0)
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, values.size))
            * values).mean(axis=1)
    p_perm = float((1 + np.sum(np.abs(null) >= abs(values.mean())))
                   / (1 + n_perm))
    return {'window_s': list(window), 'n_subjects': int(values.size),
            'mean_hz': float(values.mean()), 't': float(t), 'p': float(p),
            'p_perm': p_perm}


# ── 5) Plots ──────────────────────────────────────────────────────────

def condition_colour(label, index=0, scheme=None):
    """Colour by whatever the panel is actually contrasting.

    `scheme='reward'` -- the four rewards are the comparison (every row of
    `full`), so use the project's fixed A-D orange-to-purple ramp.
    `scheme='stage'`  -- the three stages of one reward are the comparison
    (the `stage` test), so use the stage scale.
    Otherwise valence sets the hue and stage sets the lightness, which is what
    a crossed label like "error, while learning" needs.
    """
    low = str(label).lower()
    if scheme == 'reward':
        for r, c in REWARD_COLOUR.items():
            if re.search(rf'\b{r}\b', str(label)):
                return c
    if scheme == 'stage':
        for s, c in STAGE_COLOUR.items():
            if s in low:
                return c
    valence = next((v for v in VALENCE if v in low), None)
    if valence is None:
        return plt.get_cmap('tab10')(index % 10)
    lighten = next((f for s, f in STAGE_LIGHTEN.items() if s in low), 0.0)
    base = np.array(mcolors.to_rgb(VALENCE_COLOUR[valence]))
    return tuple(base + (1.0 - base) * lighten)


# ── Figure geometry ───────────────────────────────────────────────────
# One row is 16 cm x 4 cm on the page, which is the width of a two-column
# manuscript figure and a height that stacks without becoming a full page.
# Everything is set in points at that size: no post-hoc scaling, so the font
# that comes out is the font asked for.
ROW_W_CM, ROW_H_CM = 16.0, 4.0
CM = 1 / 2.54
FS = 9                                        # Arial 9 pt, per the house style
LW_RATE = 1.2                                 # peri-event traces: thin enough
LW = 2.0                                      # that the SEM band stays visible


# The data rows are 4 cm each. The legend and the title get their own strips
# on top of that rather than eating into the panels -- at 4 cm there is not
# enough height to share.
LEGEND_CM, TITLE_CM = 1.2, 0.6


def _row_axes(n_rows, extra_cm=0.0):
    fig, axes = plt.subplots(n_rows, 3, squeeze=False,
                             figsize=(ROW_W_CM * CM,
                                      (ROW_H_CM * n_rows + extra_cm) * CM),
                             gridspec_kw=dict(width_ratios=[1.35, 0.85, 1.25]))
    for ax in axes.ravel():
        ax.tick_params(labelsize=FS - 1, length=2.5, width=0.8)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
    return fig, axes


def _mean_sem(profiles):
    X = np.vstack([profiles[s] for s in sorted(profiles)])
    return (np.nanmean(X, axis=0),
            np.nanstd(X, axis=0) / max(np.sqrt(X.shape[0]), 1), X.shape[0])


def _n_label(label, n_units, counts=None):
    """Legend text: the TEST unit first, then the other two levels.

    `n_units` is whatever `unit` was -- derivations, sessions or subjects. It
    was previously printed as "subj" regardless, so a session-level run read
    "56 subj, 56 sess" when there are only 41 patients in the whole dataset.
    """
    c = (counts or {}).get(label, {})
    unit = c.get('unit', 'subject')
    short = {'derivation': 'deriv', 'session': 'sess', 'subject': 'subj'}[unit]
    # Short: four conditions with the full three-level breakdown each pushed
    # the figure to 26 cm wide. The breakdown goes in one footnote instead.
    return f'{label} (n={n_units} {short})'


def _win_vs_baseline(profiles, centres, win, baseline):
    """Paired t across subjects, window vs baseline. Returns (t, p, stars)."""
    mb = (centres >= baseline[0]) & (centres <= baseline[1])
    mw = (centres >= win[0]) & (centres <= win[1])
    b, w = [], []
    for s in sorted(profiles):
        vb, vw = np.nanmean(profiles[s][mb]), np.nanmean(profiles[s][mw])
        if np.isfinite(vb) and np.isfinite(vw):
            b.append(vb); w.append(vw)
    if len(b) < 3:
        return np.nan, np.nan, ''
    tt, pp = stats.ttest_rel(w, b)
    star = '***' if pp < 0.001 else '**' if pp < 0.01 else '*' if pp < 0.05 else ''
    return float(tt), float(pp), star


def _win_mean(profiles, centres, win):
    """Per-subject mean rate inside a window -> (mean, sem, n)."""
    m = (centres >= win[0]) & (centres <= win[1])
    v = np.array([np.nanmean(profiles[s][m]) for s in sorted(profiles)], float)
    v = v[np.isfinite(v)]
    return (float(np.mean(v)) if v.size else np.nan,
            float(np.std(v) / max(np.sqrt(v.size), 1)) if v.size else np.nan,
            v.size)


def plot_rows(rows, out_png, baseline=BASELINE_WIN, width_s=None,
              suptitle=None, scheme=None, counts=None, share_y=False):
    """Stacked rows of (title, profiles, sliding), 16 cm wide, 4 cm per row.

    Per row, three panels:

    left    peri-event rate, mean +- SEM across subjects. The BASELINE window is
            shaded grey and the widest surviving cluster, if any, is shaded in
            the condition's colour.
    middle  the same rates as ABSOLUTE Hz in the baseline window and in the
            test window, side by side. This replaces the permutation-null
            histogram: the null says how surprising a cluster is, but not what
            is being compared to what, and with events 1.25 s apart (median)
            the baseline is the part of this analysis that needs looking at.
    right   t at every window position with the surviving clusters shaded, and
            the sliding width stated on the panel.
    """
    extra = LEGEND_CM + (TITLE_CM if suptitle else 0.0)
    fig, axes = _row_axes(len(rows), extra_cm=extra)
    legend_labels = {}
    for r, (row_title, profiles_by_condition, sliding_by_condition) in enumerate(rows):
        # ---- left: peri-event rate ---------------------------------------
        ax = axes[r][0]
        ax.axvspan(*baseline, color='0.88', lw=0, zorder=0)
        for i, (label, profiles) in enumerate(profiles_by_condition.items()):
            mean, sem, n = _mean_sem(profiles)
            c = condition_colour(label, i, scheme)
            ax.plot(centres_of(sliding_by_condition, profiles), mean, color=c,
                    lw=LW_RATE, label=_n_label(label, n, counts),
                    solid_capstyle='round', zorder=3)
            ax.fill_between(centres_of(sliding_by_condition, profiles),
                            mean - sem, mean + sem, color=c, alpha=0.22,
                            lw=0, zorder=2)
        ax.axvline(0, color='0.35', lw=1.0)
        ax.set_ylabel('Ripple rate (Hz)', fontsize=FS)
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in legend_labels:
                legend_labels[lab] = h
        ax.set_xlabel('Time from event (s)', fontsize=FS, labelpad=1)
        if r == 0:
            ax.set_title('Peri-event rate', fontsize=FS, pad=3)
        # Rows are named down the left margin so the three panel titles can
        # stay on the top row only. A single-row figure has nothing to
        # distinguish, and the suptitle already says what it is.
        if len(rows) > 1:
            ax.annotate(row_title, xy=(-0.34, 0.5), xycoords='axes fraction',
                        rotation=90, va='center', ha='center', fontsize=FS,
                        fontweight='bold')

        # ---- middle: what is compared to what ----------------------------
        ax = axes[r][1]
        cen = centres_of(sliding_by_condition,
                         next(iter(profiles_by_condition.values())))
        win = _test_window(sliding_by_condition, width_s)
        for i, (label, profiles) in enumerate(profiles_by_condition.items()):
            c = condition_colour(label, i, scheme)
            bm, bs, _ = _win_mean(profiles, cen, baseline)
            wm, ws, _ = _win_mean(profiles, cen, win)
            x = np.array([0, 1]) + (i - (len(profiles_by_condition) - 1) / 2) * 0.13
            ax.errorbar(x, [bm, wm], yerr=[bs, ws], color=c, lw=LW,
                        marker='o', ms=3.5, capsize=2, elinewidth=1.0)
            # Paired across subjects, window vs baseline. This is the
            # comparison the panel draws, so it carries its own test rather
            # than borrowing the cluster p from the sliding search.
            _, pv, star = _win_vs_baseline(profiles, cen, win, baseline)
            if star:
                # At the top of the frame above the condition's own point.
                # Placed at the data point they collided as soon as four
                # conditions had similar means.
                ax.annotate(star, xy=(x[1], 0.97), xycoords=('data', 'axes fraction'),
                            ha='center', va='top', fontsize=FS, color=c)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.14 * (hi - lo))       # headroom for the stars
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['base', 'test'], fontsize=FS)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel('Ripple rate (Hz)', fontsize=FS)
        ax.set_xlabel('base %.1f..%.1f s   test %.2f..%.2f s'
                      % (*baseline, *win), fontsize=FS - 3, color='0.35',
                      labelpad=2)
        if r == 0:
            ax.set_title('Baseline vs window', fontsize=FS, pad=3)

        # ---- right: the sliding test --------------------------------------
        ax = axes[r][2]
        sig = []
        for i, (label, sliding) in enumerate(sliding_by_condition.items()):
            if sliding is None:
                continue
            c = condition_colour(label, i, scheme)
            ax.plot(sliding['times'], sliding['t'], color=c, lw=1.6)
            for cl in sliding['clusters']:
                if cl['p'] < 0.05:
                    ax.axvspan(cl['start_s'], cl['stop_s'], color=c,
                               alpha=0.16, lw=0)
                    sig.append((cl, c))
            for sign in (1, -1):
                ax.axhline(sign * sliding['threshold'], color='0.6', lw=0.7,
                           ls=':')
        # Headroom sized to the number of labels ACTUALLY drawn, then stack them
        # from the top of the frame down. Staggering by condition index instead
        # put a lone label from the 4th condition at 0.67 of the panel height,
        # which is on top of the curves.
        if sig:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.15 * len(sig) * (hi - lo))
        for j, (cl, c) in enumerate(sig):
            ax.annotate(f"p={cl['p']:.3f}",
                        xy=(cl['peak_s'], 0.97 - 0.10 * j),
                        xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=FS - 3, color=c)
        ax.axhline(0, color='0.45', lw=0.8)
        ax.axvline(0, color='0.35', lw=1.0)
        ax.set_ylabel('t vs own baseline', fontsize=FS)
        ax.set_xlabel('Sliding window centre (s)', fontsize=FS, labelpad=1)
        if width_s is not None:
            # State the window width on the panel and draw it to scale, so the
            # smoothing implied by the test is visible rather than inferred.
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.16 * (hi - lo), hi)     # room for the scale bar
            lo, hi = ax.get_ylim()
            y = lo + 0.06 * (hi - lo)
            ax.plot([-1.9, -1.9 + width_s], [y] * 2, color='0.25', lw=3.0,
                    solid_capstyle='butt')
            ax.annotate(f'{width_s:g} s', xy=(-1.9 + width_s / 2, y),
                        xytext=(0, 4), textcoords='offset points', ha='center',
                        fontsize=FS - 3, color='0.25')
        if r == 0:
            n_lab = sum(len(s['clusters']) for s in sliding_by_condition.values()
                        if s is not None)
            ax.set_title('Sliding test', fontsize=FS, pad=3)

    if share_y and len(rows) > 1:
        # One scale per COLUMN, across rows. Panels within a column then
        # compare directly by eye; the cost is that a small effect in a row
        # with a large one is flattened, which is why both versions are saved.
        for col in range(3):
            lims = [axes[r][col].get_ylim() for r in range(len(rows))]
            lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
            for r in range(len(rows)):
                axes[r][col].set_ylim(lo, hi)

    total_cm = ROW_H_CM * len(rows) + extra
    fig.tight_layout(pad=0.4, h_pad=1.4, w_pad=1.3,
                     rect=[0, LEGEND_CM / total_cm,
                           1, 1 - (TITLE_CM / total_cm if suptitle else 0)])
    if legend_labels:
        ncol = min(len(legend_labels), 3 if len(legend_labels) > 4 else 4)
        fig.legend(legend_labels.values(), legend_labels.keys(),
                   loc='lower center', bbox_to_anchor=(0.5, 0.055),
                   ncol=ncol, fontsize=FS - 2, frameon=False,
                   handlelength=1.3, handletextpad=0.4, columnspacing=1.4,
                   borderaxespad=0.0)
    if suptitle:
        fig.suptitle(suptitle, fontsize=FS, y=0.998, va='top')
    if counts:
        # One line naming all three levels, so the legend can stay short while
        # the sample is still stated on the figure.
        c0 = next(iter(counts.values()))
        fig.text(0.5, 0.005,
                 f"test unit: {c0.get('unit', '?')}   |   "
                 f"{c0.get('n_sessions', '?')} sessions, "
                 f"{c0.get('n_subjects', '?')} subjects, "
                 f"{c0.get('n_derivations', '?')} derivations   |   "
                 f"min {c0.get('min_events', '?')} events per session",
                 ha='center', va='bottom', fontsize=FS - 3, color='0.4')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.splitext(out_png)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return out_png


def centres_of(sliding_by_condition, profiles):
    """Bin centres, taken from the profiles themselves."""
    any_p = next(iter(profiles.values())) if isinstance(profiles, dict) else profiles
    n = len(any_p)
    return np.linspace(-HALF_S + BIN_S / 2, HALF_S - BIN_S / 2, n)


A_PRIORI_WIN = (0.0, 0.5)      # the pre-specified post-event window


def _test_window(sliding_by_condition, width_s):
    """The window the middle panel tests: the PRE-SPECIFIED one, always.

    This used to be the widest surviving cluster, which made the panel
    circular. For `reward` that window was D's cluster (0.35-0.65 s), so every
    other reward was tested in a window chosen by D's own data -- and A came
    out starred at p = 0.0098 while surviving nothing in the corrected sliding
    test. A fixed a-priori window cannot select itself.

    The surviving cluster is still shown, shaded on the peri-event panel, so
    nothing is hidden -- it is just no longer what defines the test.
    """
    return A_PRIORI_WIN


def plot_condition(centres, profiles_by_condition, sliding_by_condition,
                   title, out_png, baseline=BASELINE_WIN):
    """Three panels: the rate over time, the sliding test, and the null.

    Left    peri-event rate, mean +- SEM across subjects, baseline window shaded
    Middle  the t course at every window position, surviving clusters shaded
    Right   the permutation null of cluster mass, with the observed clusters
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2),
                             gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    fig.subplots_adjust(wspace=0.28, top=0.80)

    ax = axes[0]
    ax.axvspan(*baseline, color='0.88', lw=0, zorder=0)
    for i, (label, profiles) in enumerate(profiles_by_condition.items()):
        X = np.vstack([profiles[s] for s in sorted(profiles)])
        mean = np.nanmean(X, axis=0)
        sem = np.nanstd(X, axis=0) / max(np.sqrt(X.shape[0]), 1)
        colour = condition_colour(label, i)
        ax.plot(centres, mean, color=colour, lw=1.5,
                label=f'{label} (n={X.shape[0]})')
        ax.fill_between(centres, mean - sem, mean + sem, color=colour,
                        alpha=0.20, lw=0)
    ax.axvline(0, color='0.35', lw=1.1)
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Ripple rate (Hz)')
    ax.set_title('Peri-event rate\ngrey = baseline window', fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    for i, (label, sliding) in enumerate(sliding_by_condition.items()):
        if sliding is None:
            continue
        colour = condition_colour(label, i)
        ax.plot(sliding['times'], sliding['t'], color=colour, lw=1.5)
        for cluster in sliding['clusters']:
            if cluster['p'] < 0.05:
                ax.axvspan(cluster['start_s'], cluster['stop_s'], color=colour,
                           alpha=0.16, lw=0)
                ax.annotate(f"p = {cluster['p']:.3f}",
                            xy=(cluster['peak_s'], cluster['peak_t']),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=8, color=colour)
        for sign in (1, -1):
            ax.axhline(sign * sliding['threshold'], color='0.6', lw=0.8, ls=':')
    ax.axhline(0, color='0.45', lw=0.9)
    ax.axvline(0, color='0.35', lw=1.1)
    ax.set_xlabel('Centre of the sliding window (s)')
    ax.set_ylabel('t vs own baseline')
    ax.set_title('Every window position tested\ndotted = cluster threshold',
                 fontsize=10)

    ax = axes[2]
    for i, (label, sliding) in enumerate(sliding_by_condition.items()):
        if sliding is None:
            continue
        colour = condition_colour(label, i)
        mass = sliding['null_mass']
        positive = mass[mass > 0]
        if positive.size:
            ax.hist(positive, bins=40, color=colour, alpha=0.40, lw=0)
        for cluster in sliding['clusters']:
            ax.axvline(cluster['mass'], color=colour, lw=1.6)
    ax.set_yscale('log')
    ax.set_xlabel('Max cluster mass, sign-flipped')
    ax.set_ylabel('Permutations (log)')
    ax.set_title('Permutation null\nlines = observed clusters', fontsize=10)

    fig.suptitle(title, fontsize=12, y=0.98)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_png
