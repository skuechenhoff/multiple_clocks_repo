#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL ripple analysis: uncover presses against stillness-matched control presses.

One command, one figure, one statistics table. Everything the manuscript needs
about the peri-event ripple result comes out of this file.

THE PROBLEM THIS SOLVES
    Ripple rate rises with how long the subject then sits still, and the event
    classes we care about are not balanced on stillness: after a correct
    uncovering on the first traversal the subject waits a median 1.4 s before
    pressing anything again, later in a grid 0.4 s. Comparing those classes
    directly, or each against its own pre-event baseline, measures that
    behavioural difference as much as any neural one.

THE DESIGN
    For every uncover press of interest, find an ARROW-KEY press in the same
    session followed by as near as possible the same amount of stillness, and
    use it as the comparison. Arrow keys are button presses (motor act matched)
    that uncover nothing (no information arrives) and carry no hypothesis in
    this project. Matching is 1:1, nearest neighbour on the gap to the next
    press, without replacement, inside a caliper, seeded.

    Two readings of every contrast:
      raw              no baseline window anywhere; the matched press IS the
                       reference. THE PRIMARY READING.
      vs own baseline  each side against its own -1.6..-1.1 s window first
                       (Sakon Eq. 2), then differenced. Reported because it is
                       the field's convention, but it is inflated here: control
                       presses sit in already-quiet stretches, so their
                       baselines are ripple-richer and their (window-baseline)
                       is depressed.

WHAT IT TESTS
    1  control validity     do the four arrow keys behave alike?
    2  matching balance     did the matching equalise stillness?
    3  selection            do the events that found no partner differ?
    4  per-cell effects     each uncover class vs its matched control
    5  between-cell         stage, valence, and the valence x stage interaction

RUNNING IT ON NEW DATA
    Nothing is hard-coded to a bundle. Point --bundle at a new one and the
    press-category cache rebuilds itself if it is missing or older than the
    bundle. That rebuild reads the raw 25 ms button series and is the only slow
    step (~15 min, I/O bound); everything after it is minutes.

        python scripts/swr_final_ripple_analysis.py
        python scripts/swr_final_ripple_analysis.py --bundle=<dir> --rebuild=True

OUTPUTS, in <out_dir>/
    ripple_final_figure.png / .pdf   8 panels, each 3.5 x 3.5 cm
    ripple_statistics.csv            one row per test, every number
    ripple_statistics.json           the same plus settings and provenance
    press_categories.csv             the cache (written beside the bundle)
    logs/                            the console record

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import mc.analyse.swr_behaviour as swb

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


# ══ Settings ══════════════════════════════════════════════════════════
SEED = 42
UNIT = 'session'            # the level the group test treats as independent
MIN_EVENTS = 10             # per unit per cell, after de-duplication
CALIPER_S = 0.15            # |target stillness - control stillness| ceiling...
REL_CALIPER = 0.10          # ...or this fraction of the target, whichever looser
PRIMARY_WINDOW = (0.0, 0.5)
NAMED_WINDOWS = {'pre (-0.6..-0.1)': (-0.6, -0.1),
                 'post (0..0.5)': (0.0, 0.5),
                 'post (0.5..1.0)': (0.5, 1.0)}
CONTROL_KEYS = ('LeftArrow', 'RightArrow', 'UpArrow', 'DownArrow')
NEUTRALITY_GAP = (1.0, 2.5)   # gap range the arrow-key comparison uses

# The uncover classes. Order matters: it is the order of every table and of the
# x axis of every per-cell panel.
TARGETS = {
    'correct, first uncovers': "valence == 'correct' and stage == 'first uncovers'",
    'correct, while learning': "valence == 'correct' and stage == 'while learning'",
    'correct, once known':     "valence == 'correct' and stage == 'once known'",
    'correct, later':          "valence == 'correct' and stage != 'first uncovers'",
    'error, first uncovers':   "valence == 'error' and stage == 'first uncovers'",
    'error, later':            "valence == 'error' and stage != 'first uncovers'",
    'first D (F5)':            ("valence == 'correct' and reward == 'D' "
                                "and stage == 'first uncovers'"),
}
HEADLINE = 'correct, first uncovers'

# Between-cell contrasts. Each side is already control-adjusted, so a
# difference between two of them is stillness-adjusted on both sides.
BETWEEN = {
    'stage | correct (first - later)':
        {'correct, first uncovers': 1.0, 'correct, later': -1.0},
    'stage | error (first - later)':
        {'error, first uncovers': 1.0, 'error, later': -1.0},
    'valence | first (correct - error)':
        {'correct, first uncovers': 1.0, 'error, first uncovers': -1.0},
    'valence | later (correct - error)':
        {'correct, later': 1.0, 'error, later': -1.0},
    'INTERACTION valence x stage':
        {'correct, first uncovers': 1.0, 'error, first uncovers': -1.0,
         'correct, later': -1.0, 'error, later': 1.0},
}

# ── Figure geometry ───────────────────────────────────────────────────
# Panels are 3.5 x 3.5 cm as specified. At that size the house 9 pt would fill
# a third of the panel with text, so type steps down one notch; raise these
# three numbers together if the panels ever grow.
PANEL_CM = 3.5
FS_TITLE, FS_LABEL, FS_TICK = 8, 7.5, 7
CM = 1 / 2.54

# Seven categories on a 3.5 cm axis: single-line names rotated 45 deg are the
# only spelling that neither overlaps nor becomes cryptic.
SHORT = {'correct, first uncovers': 'corr first', 'correct, while learning': 'corr learn',
         'correct, once known': 'corr known', 'correct, later': 'corr later',
         'error, first uncovers': 'err first', 'error, later': 'err later',
         'first D (F5)': 'first D'}


def _cat_ticks(ax, names, fontsize=None):
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=fontsize or FS_TICK - 1, rotation=45,
                       ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.6, len(names) - 0.4)
COLOUR = {'correct, first uncovers': rip.STAGE_COLOUR['first uncovers'],
          'correct, while learning': rip.STAGE_COLOUR['while learning'],
          'correct, once known': rip.STAGE_COLOUR['once known'],
          'correct, later': '#667F6C',
          'error, first uncovers': rip.VALENCE_COLOUR['error'],
          'error, later': '#D7657F',
          'first D (F5)': '#6B60AA'}
CONTROL_COLOUR = '#5a5a5a'


def _rc():
    return {'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': FS_TICK, 'axes.titlesize': FS_TITLE,
            'axes.labelsize': FS_LABEL, 'xtick.labelsize': FS_TICK,
            'ytick.labelsize': FS_TICK, 'axes.spines.top': False,
            'axes.spines.right': False, 'axes.linewidth': 0.7,
            'lines.linewidth': 1.1}


# ══ 1) The press table ════════════════════════════════════════════════

def build_press_table(data, out_csv):
    """Every key press with its key, its stillness and its task label.

    A press is a bin in which the held key changes, the same definition
    `swr_probes.press_times` uses -- the key itself is kept here instead of
    being collapsed to move/uncover, because the control has to come from a
    category that carries no hypothesis, and "an arrow key" is four categories.

    `still_next_s` is the gap to the next press of ANY kind: it says how long
    the subject then sits still, which is the quantity everything downstream
    matches on.
    """
    beh = data['behaviour']
    frames = []
    for i, session in enumerate(rip.sessions_in(data), 1):
        b = beh[beh.session == session]
        rows = []
        for grid, g in b.groupby('grid_no'):
            btn = swb._grid_series(session, grid, 'buttons')
            if btn is None:
                continue
            onset = float(g.new_grid_onset.iloc[0])
            keys = btn.astype(str)
            trans = np.flatnonzero(keys[1:] != keys[:-1]) + 1
            rows += [(onset + t * swb.BIN_S, keys[t], int(grid)) for t in trans
                     if keys[t] in swb.MOVE_KEYS or keys[t] == swb.UNCOVER_KEY]
        if not rows:
            continue
        p = pd.DataFrame(rows, columns=['t_s', 'key', 'grid_no']) \
              .sort_values('t_s').reset_index(drop=True)
        t = p.t_s.to_numpy(float)
        nxt = np.r_[np.diff(t), np.nan]
        p = p.assign(session=int(session), still_next_s=nxt,
                     kind=np.where(p.key == swb.UNCOVER_KEY, 'uncover', 'move'))
        tab = rip.uncover_table(data, session)
        if len(tab):
            p = pd.merge_asof(
                p, tab[['t_s', 'valence', 'stage', 'reward']].sort_values('t_s'),
                on='t_s', direction='nearest', tolerance=0.03)
            p.loc[p.kind == 'move', ['valence', 'stage', 'reward']] = np.nan
        frames.append(p)
        print(f"    [{i:3d}] session {session}: {len(p)} presses")
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def press_table(data, bundle, out_csv, rebuild=False):
    """Load the cache, rebuilding it if it is missing, stale or forced."""
    stamp = os.path.join(bundle, 'swr_bundle.pkl')
    stale = (os.path.exists(out_csv) and os.path.exists(stamp)
             and os.path.getmtime(out_csv) < os.path.getmtime(stamp))
    if rebuild or stale or not os.path.exists(out_csv):
        why = ('forced' if rebuild else 'older than the bundle' if stale
               else 'not found')
        print(f"  building press categories ({why}) -- reads the raw button "
              f"series, ~15 min")
        return build_press_table(data, out_csv)
    print(f"  press categories: reusing {os.path.basename(out_csv)}")
    return pd.read_csv(out_csv)


# ══ 2) Matching ═══════════════════════════════════════════════════════

def match_session(t_tgt, s_tgt, t_ctl, s_ctl, caliper_s, rel_caliper, rng):
    """1:1 nearest neighbour on stillness, without replacement.

    Targets are served in seeded random order so the ones handled first do not
    systematically take the best controls, and each control is used once so no
    single long pause can stand in for fifty targets. A target with nothing
    inside its caliper is returned as unmatched, not discarded silently.
    """
    free = np.ones(t_ctl.size, bool)
    keep_t, keep_c, missed = [], [], []
    for i in rng.permutation(t_tgt.size):
        s = s_tgt[i]
        if not np.isfinite(s):
            missed.append(t_tgt[i])
            continue
        d = np.abs(s_ctl - s)
        d[~free] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > max(caliper_s, rel_caliper * s):
            missed.append(t_tgt[i])
            continue
        free[j] = False
        keep_t.append(t_tgt[i])
        keep_c.append(t_ctl[j])
    return (np.sort(np.array(keep_t)), np.sort(np.array(keep_c)),
            np.sort(np.array(missed)))


def matched_sets(presses, query, caliper_s, rel_caliper, seed):
    """{session: times} for targets, their controls and the unmatched targets.

    Both sides are de-duplicated at `rip.DEDUP_S` BEFORE matching, so the
    pairing survives the de-duplication `rate_by_unit` applies later rather
    than being broken by it.
    """
    rng = np.random.default_rng(seed)
    tgt_out, ctl_out, miss_out, rows = {}, {}, {}, []
    for session, g in presses.groupby('session'):
        tgt = g.query(query)
        ctl = g[g.key.isin(CONTROL_KEYS)]
        tgt = tgt[np.isfinite(tgt.still_next_s)].sort_values('t_s')
        ctl = ctl[np.isfinite(ctl.still_next_s)].sort_values('t_s')
        if not len(tgt) or not len(ctl):
            continue
        tgt = tgt[np.isin(tgt.t_s, rip.dedup(tgt.t_s.to_numpy(float)))]
        ctl = ctl[np.isin(ctl.t_s, rip.dedup(ctl.t_s.to_numpy(float)))]
        if not len(tgt) or not len(ctl):
            continue
        t_m, c_m, t_u = match_session(
            tgt.t_s.to_numpy(float), tgt.still_next_s.to_numpy(float),
            ctl.t_s.to_numpy(float), ctl.still_next_s.to_numpy(float),
            caliper_s, rel_caliper, rng)
        if not t_m.size:
            continue
        tgt_out[int(session)], ctl_out[int(session)] = t_m, c_m
        if t_u.size:
            miss_out[int(session)] = t_u
        st, sc = tgt.set_index('t_s').still_next_s, ctl.set_index('t_s').still_next_s
        rows.append({'session': int(session), 'n_matched': int(t_m.size),
                     'n_unmatched': int(t_u.size),
                     'still_target_s': float(st.loc[t_m].median()),
                     'still_control_s': float(sc.loc[c_m].median()),
                     'still_unmatched_s': (float(st.loc[t_u].median())
                                           if t_u.size else np.nan)})
    return tgt_out, ctl_out, miss_out, pd.DataFrame(rows)


# ══ 3) Statistics ═════════════════════════════════════════════════════

def rates(data, events, unit=UNIT, min_events=MIN_EVENTS):
    """Peri-event rate per unit -- the pipeline's own routine."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return rip.rate_by_unit(data, events, unit=unit, min_events=min_events)


def baseline(profiles, centres):
    units, X = rip.baseline_subtract(profiles, centres)
    return {u: X[i] for i, u in enumerate(units)}


def window_stats(profiles, centres, window, n_perm=10000, seed=SEED):
    """Mean, t, parametric p, sign-flip p and a 95% CI for one window.

    The sign-flip null is the exact null for a within-unit contrast under
    symmetry, and is the same null the cluster test uses -- one statistic,
    computed one way, for the data and for every permutation.
    """
    units = sorted(profiles)
    X = np.vstack([profiles[u] for u in units])
    m = (centres >= window[0]) & (centres < window[1])
    v = np.nanmean(X[:, m], axis=1)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None
    t, p = stats.ttest_1samp(v, 0.0)
    sem = float(v.std(ddof=1) / np.sqrt(v.size))
    crit = float(stats.t.ppf(0.975, v.size - 1))
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, v.size)) * v).mean(axis=1)
    return {'window_s': list(window), 'n_units': int(v.size),
            'mean_hz': float(v.mean()), 'sem_hz': sem,
            'ci_low_hz': float(v.mean() - crit * sem),
            'ci_high_hz': float(v.mean() + crit * sem),
            't': float(t), 'df': int(v.size - 1), 'p': float(p),
            'p_perm': float((1 + np.sum(np.abs(null) >= abs(v.mean())))
                            / (1 + n_perm)),
            'cohens_d': float(v.mean() / v.std(ddof=1)) if v.std(ddof=1) else np.nan}


def cluster_test(profiles, centres, label, n_perm, seed=SEED):
    """Sliding-window cluster permutation -- no window is chosen anywhere."""
    units = sorted(profiles)
    X = np.vstack([profiles[u] for u in units])
    res = rip.sliding_window_test(
        {label: {u: X[i] for i, u in enumerate(units)}}, centres,
        width_s=rip.SLIDE_WIDTHS_S[0], n_perm=n_perm, seed=seed).get(label)
    if res is None:
        return None, []
    return res, [c for c in res['clusters'] if c['p'] < 0.05]


def difference(a, b):
    """a - b on the units both have."""
    shared = sorted(set(a) & set(b))
    return {u: a[u] - b[u] for u in shared}


# ══ 4) The analysis ═══════════════════════════════════════════════════

def analyse(data, presses, n_perm, caliper_s, rel_caliper, unit, min_events):
    out = {'control_neutrality': {}, 'cells': {}, 'between': {},
           'selection': {}, 'descriptives': {}}
    rows = []

    # -- descriptives ---------------------------------------------------
    qc = data['channel_qc']
    qc = qc[~qc.excluded.fillna(False)] if 'excluded' in qc else qc
    out['descriptives'] = {
        'n_sessions': int(data['ripples'].session.nunique()),
        'n_subjects': int(data['ripples'].subject_key.nunique()),
        'n_derivations': int(len(qc)),
        'n_ripples': int(len(data['ripples'])),
        'clean_hours': float(qc.clean_s.sum() / 3600) if 'clean_s' in qc else None,
        'n_presses': int(len(presses)),
        'presses_by_key': presses.key.value_counts().to_dict()}
    print(f"\n  {out['descriptives']['n_sessions']} sessions | "
          f"{out['descriptives']['n_subjects']} subjects | "
          f"{out['descriptives']['n_derivations']} derivations | "
          f"{out['descriptives']['n_ripples']} ripples | "
          f"{len(presses)} presses")

    # -- 1) is the control neutral? -------------------------------------
    print(f"\n  1) control validity: rate after each arrow key "
          f"(gaps {NEUTRALITY_GAP[0]:g}-{NEUTRALITY_GAP[1]:g} s)")
    sel = presses[(presses.still_next_s >= NEUTRALITY_GAP[0])
                  & (presses.still_next_s <= NEUTRALITY_GAP[1])]
    for key in CONTROL_KEYS:
        g = sel[sel.key == key]
        if not len(g):
            continue
        c, per_unit, cnt = rates(data, {int(s): gg.t_s.to_numpy(float)
                                        for s, gg in g.groupby('session')},
                                 unit, min_events)
        if c is None or len(per_unit) < 3:
            continue
        got = window_stats(per_unit, c, PRIMARY_WINDOW)
        got.update({'n_events': int(cnt['n_events_used'])})
        out['control_neutrality'][key] = got
        rows.append(dict(analysis='control_neutrality', contrast=key,
                         reading='raw', **got))
        print(f"     {key:12s} n={got['n_units']:3d} "
              f"events={cnt['n_events_used']:6d} "
              f"rate={got['mean_hz']:.4f} ± {got['sem_hz']:.4f} Hz")
    vals = [v['mean_hz'] for v in out['control_neutrality'].values()]
    if len(vals) > 1:
        out['control_neutrality']['spread_hz'] = float(max(vals) - min(vals))
        print(f"     spread across keys: {max(vals) - min(vals):.4f} Hz")

    # -- 2-4) per cell ---------------------------------------------------
    print(f"\n  2-4) per cell: uncover minus its stillness-matched arrow key")
    raw_diffs = {}
    for label, query in TARGETS.items():
        tgt, ctl, miss, bal = matched_sets(presses, query, caliper_s,
                                           rel_caliper, SEED)
        if not tgt:
            print(f"\n     {label}: nothing matched")
            continue
        c_t, raw_t, cnt_t = rates(data, tgt, unit, min_events)
        c_c, raw_c, cnt_c = rates(data, ctl, unit, min_events)
        if c_t is None or c_c is None or not raw_t or not raw_c:
            print(f"\n     {label}: not computable")
            continue
        centres = c_t
        n_m, n_u = int(bal.n_matched.sum()), int(bal.n_unmatched.sum())
        entry = {'n_matched_events': n_m, 'n_unmatched_events': n_u,
                 'match_rate': n_m / max(n_m + n_u, 1),
                 'n_sessions': int(len(bal)),
                 'still_target_s': float(bal.still_target_s.median()),
                 'still_control_s': float(bal.still_control_s.median()),
                 'still_unmatched_s': float(bal.still_unmatched_s.median()),
                 'n_events_target': int(cnt_t['n_events_used']),
                 'n_events_control': int(cnt_c['n_events_used']),
                 'readings': {}}
        print(f"\n     {label}")
        print(f"       matched {n_m}/{n_m + n_u} "
              f"({100 * entry['match_rate']:.0f}%), {len(bal)} sessions | "
              f"stillness {entry['still_target_s']:.3f} vs "
              f"{entry['still_control_s']:.3f} s")

        for reading in ('raw', 'vs own baseline'):
            a, b = (raw_t, raw_c) if reading == 'raw' else (
                baseline(raw_t, centres), baseline(raw_c, centres))
            diff = difference(a, b)
            if len(diff) < 3:
                continue
            rec = {'n_units': len(diff), 'windows': {}}
            for wname, win in NAMED_WINDOWS.items():
                got = window_stats(diff, centres, win)
                if got:
                    rec['windows'][wname] = got
                    rows.append(dict(analysis='cell', contrast=label,
                                     reading=reading, window_name=wname,
                                     n_events=n_m, **got))
            sres, keep = cluster_test(diff, centres, label, n_perm)
            rec['clusters'] = keep
            if sres is not None:
                rec['t_curve'] = {'times_s': np.asarray(sres['times']).tolist(),
                                  't': np.asarray(sres['t']).tolist(),
                                  'threshold': float(sres['threshold'])}
                if label == HEADLINE and reading == 'raw':
                    rec['sliding_result'] = sres
            rec['profile'] = _profile(diff, centres)
            entry['readings'][reading] = rec
            if reading == 'raw':
                raw_diffs[label] = diff
                raw_centres = centres
                if label == HEADLINE:
                    shared = sorted(set(raw_t) & set(raw_c))
                    out['headline'] = {
                        'centres_s': centres.tolist(),
                        'target': dict(_profile(raw_t, centres),
                                       per_unit=[raw_t[u].tolist()
                                                 for u in shared]),
                        'control': dict(_profile(raw_c, centres),
                                        per_unit=[raw_c[u].tolist()
                                                  for u in shared])}
            w = rec['windows'].get('post (0..0.5)')
            star = ' *' if w and w['p_perm'] < 0.05 else ''
            print(f"       {reading:16s} {w['mean_hz']:+.4f} Hz "
                  f"[{w['ci_low_hz']:+.4f}, {w['ci_high_hz']:+.4f}] "
                  f"t({w['df']})={w['t']:+.2f} p={w['p_perm']:.4f}{star}")
            print(f"       {'':16s} cluster: " +
                  ('; '.join(f"{c['start_s']:+.2f}..{c['stop_s']:+.2f} s "
                             f"p={c['p']:.4f}" for c in keep) if keep else 'none'))

        # -- selection: do the unmatched behave differently? -------------
        if miss:
            c_u, raw_u, cnt_u = rates(data, miss, unit, min_events)
            if c_u is not None and len(raw_u) >= 3:
                bm = window_stats(baseline(raw_t, centres), centres,
                                  PRIMARY_WINDOW)
                bu = window_stats(baseline(raw_u, c_u), c_u, PRIMARY_WINDOW)
                shared = sorted(set(raw_t) & set(raw_u))
                paired = None
                if len(shared) >= 3:
                    d = difference(baseline({u: raw_t[u] for u in shared}, centres),
                                   baseline({u: raw_u[u] for u in shared}, c_u))
                    paired = window_stats(d, centres, PRIMARY_WINDOW)
                sel_rec = {'matched': bm, 'unmatched': bu, 'paired_diff': paired,
                           'still_matched_s': entry['still_target_s'],
                           'still_unmatched_s': entry['still_unmatched_s'],
                           'n_events_unmatched': int(cnt_u['n_events_used'])}
                out['selection'][label] = sel_rec
                rows.append(dict(analysis='selection', contrast=label,
                                 reading='vs own baseline',
                                 window_name='matched subset', **bm))
                rows.append(dict(analysis='selection', contrast=label,
                                 reading='vs own baseline',
                                 window_name='unmatched subset', **bu))
                if paired:
                    rows.append(dict(analysis='selection', contrast=label,
                                     reading='vs own baseline',
                                     window_name='matched - unmatched',
                                     **paired))
                print(f"       selection: matched {bm['mean_hz']:+.4f} "
                      f"(still {entry['still_target_s']:.2f} s) vs unmatched "
                      f"{bu['mean_hz']:+.4f} "
                      f"(still {entry['still_unmatched_s']:.2f} s)"
                      + (f", Δ={paired['mean_hz']:+.4f} p={paired['p_perm']:.3f}"
                         if paired else ''))
        entry['balance'] = bal.to_dict('records')
        out['cells'][label] = entry

    # -- 5) between cells ------------------------------------------------
    print(f"\n  5) between cells (each side already control-adjusted, raw)")
    for name, weights in BETWEEN.items():
        if any(k not in raw_diffs for k in weights):
            continue
        d = rip.contrast_profiles(raw_diffs, weights)
        if len(d) < 3:
            continue
        got = window_stats(d, raw_centres, PRIMARY_WINDOW)
        res, keep = cluster_test(d, raw_centres, name, n_perm)
        out['between'][name] = {'window': got, 'clusters': keep,
                                'profile': _profile(d, raw_centres)}
        rows.append(dict(analysis='between_cells', contrast=name,
                         reading='raw', window_name='post (0..0.5)', **got))
        star = ' *' if got['p_perm'] < 0.05 else ''
        print(f"     {name:36s} {got['mean_hz']:+.4f} Hz "
              f"[{got['ci_low_hz']:+.4f}, {got['ci_high_hz']:+.4f}] "
              f"t({got['df']})={got['t']:+.2f} p={got['p_perm']:.4f}{star}")
        print(f"     {'':36s} cluster: " +
              ('; '.join(f"{c['start_s']:+.2f}..{c['stop_s']:+.2f} s "
                         f"p={c['p']:.4f}" for c in keep) if keep else 'none'))

    out['_raw_diffs'] = (raw_diffs, raw_centres) if raw_diffs else None
    return out, pd.DataFrame(rows)


def _profile(profiles, centres):
    units = sorted(profiles)
    X = np.vstack([profiles[u] for u in units])
    return {'centres_s': centres.tolist(),
            'mean_hz': np.nanmean(X, axis=0).tolist(),
            'sem_hz': (np.nanstd(X, axis=0) / np.sqrt(X.shape[0])).tolist(),
            'n_units': len(units)}


def stillness_distributions(presses, edges=(0, 0.5, 1.0, 1.5, 2.5, np.inf)):
    """Share of events per stillness bin, for the two stages and the control.

    The panel that motivates the whole design: the three distributions are not
    the same, so an unmatched comparison is a comparison of these histograms as
    much as of anything neural.
    """
    labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2.5', '≥2.5']
    p = presses.assign(bin=pd.cut(presses.still_next_s, list(edges),
                                  labels=labels, right=False))
    out = {}
    for name, sel in (
            ('first uncovers', (p.kind == 'uncover') & (p.valence == 'correct')
             & (p.stage == 'first uncovers')),
            ('later uncovers', (p.kind == 'uncover') & (p.valence == 'correct')
             & (p.stage != 'first uncovers')),
            ('arrow keys', p.kind == 'move')):
        v = p[sel]['bin'].value_counts(normalize=True)
        out[name] = {l: float(100 * v.get(l, 0.0)) for l in labels}
    return labels, out


# ══ 5) Figure ═════════════════════════════════════════════════════════

FEEDBACK_CELLS = {
    f'{v}, {st}': f"valence == '{v}' and stage == '{st}'"
    for v in ('correct', 'error') for st in rip.STAGES}
SMOOTH_BINS = 5      # Sakon & Kahana's triangle smooth, DISPLAY ONLY


def feedback_stage_rows(data, presses, unit, min_events, n_perm):
    """The six valence x stage cells: raw profiles, counts and sliding tests.

    This reproduces the `feedback_stage` panel -- each cell against its own
    baseline -- so the final figure can carry it as its first two rows and the
    stillness-matched contrast can sit underneath as the third.
    """
    raw, counts, centres = {}, {}, None
    for label, query in FEEDBACK_CELLS.items():
        per_sess = {}
        for session, g in presses.query(query).groupby('session'):
            per_sess[int(session)] = g.t_s.to_numpy(float)
        if not per_sess:
            continue
        c, per_unit, cnt = rates(data, per_sess, unit, min_events)
        if c is None or len(per_unit) < 3:
            continue
        centres = c
        raw[label], counts[label] = per_unit, cnt
    if centres is None:
        return None, {}, {}, {}
    baselined = {l: baseline(p, centres) for l, p in raw.items()}
    sliding = rip.sliding_window_test(baselined, centres,
                                      width_s=rip.SLIDE_WIDTHS_S[0],
                                      n_perm=n_perm, seed=SEED)
    return centres, raw, counts, sliding


def figure_main(out_png, centres, fb_raw, fb_counts, fb_sliding, res, n_perm):
    """Three rows, in the house 16 cm x 4 cm format.

    Rows 1-2 are the conventional analysis -- every valence x stage cell against
    its own baseline. Row 3 is the same headline cell against its
    stillness-matched control press, so the reader can see in one figure that
    the effect does not depend on which reference is used.
    """
    head = res['headline']
    c = np.asarray(head['centres_s'])
    rec = res['cells'][HEADLINE]['readings']['raw']
    tgt_lab = 'correct, first uncovers'          # same colour as row 1
    ctl_lab = 'matched arrow press'
    row3_prof = {tgt_lab: {i: np.asarray(v) for i, v in
                           enumerate(head['target']['per_unit'])},
                 ctl_lab: {i: np.asarray(v) for i, v in
                           enumerate(head['control']['per_unit'])}}
    d = res['descriptives']
    row3_counts = {ctl_lab: {'unit': UNIT,
                             'n_units': head['control']['n_units'],
                             'n_sessions': d['n_sessions'],
                             'n_subjects': d['n_subjects'],
                             'n_derivations': d['n_derivations'],
                             'min_events': MIN_EVENTS}}
    row3_sliding = {tgt_lab: rec.get('sliding_result'), ctl_lab: None}

    rows = [('positive feedback',
             {l: fb_raw[l] for l in FEEDBACK_CELLS if l.startswith('correct')
              and l in fb_raw},
             {l: fb_sliding.get(l) for l in FEEDBACK_CELLS
              if l.startswith('correct') and l in fb_raw}),
            ('negative feedback',
             {l: fb_raw[l] for l in FEEDBACK_CELLS if l.startswith('error')
              and l in fb_raw},
             {l: fb_sliding.get(l) for l in FEEDBACK_CELLS
              if l.startswith('error') and l in fb_raw}),
            ('stillness-matched', row3_prof, row3_sliding)]
    # `tgt_lab` is deliberately the same label as row 1, so the two share a
    # legend entry and a colour. fb_counts must therefore win for that key --
    # the row-3 dict only supplies the control press.
    counts = {**row3_counts, **fb_counts}
    # The control press is neither a valence nor a stage, so it gets an
    # explicit neutral grey rather than falling through to a tab10 hue (which
    # came out orange -- reserved for state A in this project).
    return rip.plot_rows(
        rows, out_png, width_s=rip.SLIDE_WIDTHS_S[0], counts=counts,
        smooth_bins=SMOOTH_BINS, legend_cm=2.4,
        colours={ctl_lab: CONTROL_COLOUR},
        suptitle='Ripples at uncovering: own baseline (rows 1-2) and a '
                 'stillness-matched control press (row 3)')


def figure_methods(out_png, res, still_labels, still_dists):
    """The controls, in the same 16 cm house format as the main figure."""
    labels = [l for l in TARGETS if l in res['cells']]
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
          'axes.spines.top': False, 'axes.spines.right': False}
    FS = rip.FS
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(rip.ROW_W_CM * CM,
                                                2 * rip.ROW_H_CM * CM * 1.25))
        for ax in axes.ravel():
            ax.tick_params(labelsize=FS - 1, length=2.5, width=0.8)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)

        def ticks(ax, names, rot=30):
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=FS - 1, rotation=rot,
                               ha='right' if rot else 'center',
                               rotation_mode='anchor' if rot else None)
            ax.set_xlim(-0.6, len(names) - 0.4)

        # (a) the imbalance ---------------------------------------------
        ax = axes[0][0]
        w = 0.27
        for k, (name, colour) in enumerate((
                ('first uncovers', COLOUR['correct, first uncovers']),
                ('later uncovers', COLOUR['correct, later']),
                ('arrow keys', CONTROL_COLOUR))):
            ax.bar(np.arange(len(still_labels)) + (k - 1) * w,
                   [still_dists[name][l] for l in still_labels], width=w,
                   color=colour, lw=0, label=name)
        ticks(ax, still_labels, rot=0)
        ax.set_xlabel('Stillness after press (s)', fontsize=FS)
        ax.set_ylabel('Events (%)', fontsize=FS)
        ax.set_title('a  Stillness is unbalanced', fontsize=FS, loc='left',
                     fontweight='bold')
        ax.legend(fontsize=FS - 1, frameon=False, loc='upper right',
                  handlelength=1.1, handletextpad=0.5)

        # (b) matching balance -------------------------------------------
        ax = axes[0][1]
        for i, l in enumerate(labels):
            cc = res['cells'][l]
            ax.plot([i - 0.17], [cc['still_target_s']], 'o', ms=6,
                    color=COLOUR[l])
            ax.plot([i + 0.17], [cc['still_control_s']], 's', ms=6,
                    color=CONTROL_COLOUR)
            ax.plot([i - 0.17, i + 0.17],
                    [cc['still_target_s'], cc['still_control_s']],
                    color='0.75', lw=1.2, zorder=0)
        ticks(ax, [SHORT[l] for l in labels])
        ax.set_ylabel('Stillness (s, median)', fontsize=FS)
        ax.set_title('b  ● target   ■ its matched control', fontsize=FS,
                     loc='left', fontweight='bold')

        # (c) control neutrality -----------------------------------------
        ax = axes[1][0]
        keys = [k for k in CONTROL_KEYS if k in res['control_neutrality']]
        for i, k in enumerate(keys):
            d = res['control_neutrality'][k]
            ax.errorbar([i], [d['mean_hz']], yerr=[d['sem_hz']],
                        color=CONTROL_COLOUR, marker='s', ms=6, capsize=3,
                        elinewidth=1.6, lw=0)
        ticks(ax, [k.replace('Arrow', '') for k in keys], rot=0)
        ax.set_ylabel('Ripple rate (Hz)', fontsize=FS)
        ax.set_xlabel('control key', fontsize=FS)
        ax.set_title('c  Is the control neutral?', fontsize=FS, loc='left',
                     fontweight='bold')

        # (d) selection ---------------------------------------------------
        ax = axes[1][1]
        ax.axhline(0, color='0.5', lw=0.9)
        sel = [l for l in labels if l in res['selection']]
        for i, l in enumerate(sel):
            sc = res['selection'][l]
            ax.errorbar([i - 0.15], [sc['matched']['mean_hz']],
                        yerr=[sc['matched']['sem_hz']], color=COLOUR[l],
                        marker='o', ms=6, capsize=3, elinewidth=1.6, lw=0)
            ax.errorbar([i + 0.15], [sc['unmatched']['mean_hz']],
                        yerr=[sc['unmatched']['sem_hz']], color=COLOUR[l],
                        marker='o', ms=6, capsize=3, elinewidth=1.6, lw=0,
                        alpha=0.40)
        ticks(ax, [SHORT[l] for l in sel])
        ax.set_ylabel('Δ rate 0–0.5 s (Hz)', fontsize=FS)
        ax.set_title('d  ● matched   ○ unmatched', fontsize=FS, loc='left',
                     fontweight='bold')

        fig.tight_layout(pad=0.5, h_pad=2.0, w_pad=2.0)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.splitext(out_png)[0] + '.pdf', bbox_inches='tight')
        plt.close(fig)
    return out_png


# ══ Main ══════════════════════════════════════════════════════════════

def run(bundle=None, out_dir=None, presses_csv=None, rebuild=False,
        unit=UNIT, min_events=MIN_EVENTS, n_perm=rip.N_SIGN_FLIPS,
        caliper_s=CALIPER_S, rel_caliper=REL_CALIPER, pad_s=None):
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr')
    bundle = bundle or os.path.join(group, 'bundle')
    presses_csv = presses_csv or os.path.join(group, 'press_categories.csv')
    out_dir = out_dir or os.path.join(
        group, f"ripple_final_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_final_ripple_analysis')
    np.random.seed(SEED)

    print(f"\n  bundle : {bundle}\n  presses: {presses_csv}\n  out    : {out_dir}")
    print(f"  unit = {unit}, min {min_events} events, caliper = "
          f"max({caliper_s:g} s, {100 * rel_caliper:.0f}%), {n_perm} sign-flips, "
          f"seed {SEED}")

    data = rip.load_bundle(bundle)
    if pad_s is not None:
        # Re-impose the artifact pad at analysis time (methods SS3.4c). Both
        # halves: events filtered on dist_to_artifact_s AND exposure rebuilt.
        import mc.analyse.swr_bundle as swb
        n0 = len(data['ripples'])
        data = swb.repad_bundle(data, float(pad_s))
        print(f"  re-padded to {pad_s} s: {n0} -> {len(data['ripples'])} ripples")
    presses = press_table(data, bundle, presses_csv, rebuild=rebuild)

    res, table = analyse(data, presses, n_perm, caliper_s, rel_caliper, unit,
                         min_events)
    res.pop('_raw_diffs', None)

    table.to_csv(os.path.join(out_dir, 'ripple_statistics.csv'), index=False)
    payload = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'script': os.path.basename(__file__),
        'settings': {'bundle': bundle, 'presses_csv': presses_csv,
                     'unit': unit, 'min_events': min_events,
                     'caliper_s': caliper_s, 'rel_caliper': rel_caliper,
                     'n_sign_flips': n_perm, 'seed': SEED,
                     'primary_window_s': list(PRIMARY_WINDOW),
                     'named_windows_s': {k: list(v)
                                         for k, v in NAMED_WINDOWS.items()},
                     'baseline_window_s': list(rip.BASELINE_WIN),
                     'bin_s': rip.BIN_S, 'half_s': rip.HALF_S,
                     'dedup_s': rip.DEDUP_S,
                     'sliding_width_s': rip.SLIDE_WIDTHS_S[0],
                     'cluster_alpha': rip.CLUSTER_ALPHA,
                     'min_clean_frac': rip.MIN_CLEAN_FRAC,
                     'control_keys': list(CONTROL_KEYS)},
        **res}
    with open(os.path.join(out_dir, 'ripple_statistics.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)

    still_labels, still_dists = stillness_distributions(presses)
    res['stillness_distributions'] = still_dists
    print("\n  feedback_stage rows (each cell vs its own baseline)")
    fb_c, fb_raw, fb_counts, fb_sliding = feedback_stage_rows(
        data, presses[presses.kind == 'uncover'], unit, min_events, n_perm)
    for l, r in fb_sliding.items():
        keep = [k for k in r['clusters'] if k['p'] < 0.05]
        print(f"    {l:26s} " + ('; '.join(
            f"{k['direction']} {k['start_s']:+.2f}..{k['stop_s']:+.2f} "
            f"p={k['p']:.4f}" for k in keep) if keep else 'none'))
    figure_main(os.path.join(out_dir, 'ripple_main_figure.png'), fb_c, fb_raw,
                fb_counts, fb_sliding, res, n_perm)
    figure_methods(os.path.join(out_dir, 'ripple_methods_figure.png'), res,
                   still_labels, still_dists)
    print(f"\n  wrote ripple_main_figure.png/.pdf, "
          f"ripple_methods_figure.png/.pdf, ripple_statistics.csv, "
          f"ripple_statistics.json")
    print(f"  saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
