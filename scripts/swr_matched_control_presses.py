#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Each uncover press against its own stillness-matched control press.

SK's design, and it is the right one. The earlier controls compared an event
class against a POOL (all movement presses, or a coarse stillness bin). This
matches event by event: for every uncover press of interest, find a press of a
category the hypothesis says nothing about -- an arrow key -- in the same
session, followed by as near as possible the same amount of stillness, and use
that press as the comparison.

Why this beats binning. A 1.5-2.5 s bin still lets a target sitting at 1.6 s be
compared with a control at 2.4 s, and the two stages are not uniformly spread
inside a bin, so residual imbalance survives stratification. Nearest-neighbour
matching on the actual gap removes that, and the achieved balance is reported
rather than assumed.

Why arrow keys are the right control. They are button presses, so the motor act
matches; they uncover nothing, so no information arrives; and the hypotheses in
this project are all about uncover presses, so no arrow key is a
hypothesis-carrying event. Splitting them by direction (Left/Right/Up/Down)
also lets the control itself be checked: if the four directions disagree, the
control is not neutral.

Two readings of each contrast, because SK wanted both:

  vs own baseline   each side measured against its own -1.6..-1.1 s window
                    first (Sakon Eq. 2), then differenced. Keeps the
                    within-trial logic she liked.
  raw              no baseline anywhere; the matched press IS the reference.
                   Immune to the pre-event stillness imbalance that the
                   baseline version inherits.

Matching is seeded and every drop is counted: a target with no control inside
the caliper is reported, not silently discarded.

    python scripts/swr_matched_control_presses.py
    python scripts/swr_matched_control_presses.py --caliper_s=0.1

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
from swr_valence_stage_interaction import _rc, _axes

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


QUESTION = ('Does an uncover press carry more ripples than an arrow-key press '
            'followed by the same amount of stillness?')

MOVE_KEYS = ('LeftArrow', 'RightArrow', 'UpArrow', 'DownArrow')
PRIMARY_WINDOW = (0.0, 0.5)
NAMED_WINDOWS = {'pre  (-0.6..-0.1)': (-0.6, -0.1),
                 'post (0..0.5)': (0.0, 0.5),
                 'post (0.5..1.0)': (0.5, 1.0)}

# Targets: the hypothesis-carrying uncover presses. `first D` is F5, the one
# claim in this project that was stated before any analysis and the one that
# has not yet been through a stillness control.
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

CALIPER_S = 0.15        # |target stillness - control stillness| must be under this
REL_CALIPER = 0.10      # ...or within this fraction of the target, whichever is looser
SEED = 42

# Compact tick labels: six full names on a 3 cm axis overlap into mush.
SHORT = {'correct, first uncovers': 'corr\nfirst',
         'correct, while learning': 'corr\nlearn',
         'correct, once known': 'corr\nknown',
         'correct, later': 'corr\nlater',
         'error, first uncovers': 'err\nfirst',
         'error, later': 'err\nlater',
         'first D (F5)': 'first D\n(F5)'}

TARGET_COLOUR = {
    'correct, first uncovers': rip.STAGE_COLOUR['first uncovers'],
    'correct, while learning': rip.STAGE_COLOUR['while learning'],
    'correct, once known': rip.STAGE_COLOUR['once known'],
    'correct, later': '#667F6C',
    'error, first uncovers': rip.VALENCE_COLOUR['error'],
    'error, later': '#D7657F',
    'first D (F5)': '#6B60AA',
}


# ── 1) Matching ───────────────────────────────────────────────────────

def match_one_session(target_t, target_still, control_t, control_still,
                      caliper_s, rel_caliper, rng):
    """Nearest-neighbour match on stillness, WITHOUT replacement.

    Targets are served in a random order (seeded) so the ones processed first
    do not systematically get the best controls; each control is used at most
    once, so no single long pause can stand in for fifty targets. A target with
    nothing inside its caliper is dropped and counted.

    Returns (matched target times, matched control times, UNMATCHED target
    times). The unmatched ones are handed back rather than counted away, so the
    question "are the events that found no partner different?" can be asked
    instead of assumed away.
    """
    order = rng.permutation(len(target_t))
    free = np.ones(len(control_t), bool)
    t_out, c_out, missed = [], [], []
    for i in order:
        s = target_still[i]
        if not np.isfinite(s):
            missed.append(target_t[i])
            continue
        tol = max(caliper_s, rel_caliper * s)
        d = np.abs(control_still - s)
        d[~free] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > tol:
            missed.append(target_t[i])
            continue
        free[j] = False
        t_out.append(target_t[i])
        c_out.append(control_t[j])
    return np.array(t_out), np.array(c_out), np.array(missed)


def matched_sets(presses, query, caliper_s=CALIPER_S, rel_caliper=REL_CALIPER,
                 seed=SEED, control_keys=MOVE_KEYS):
    """{session: target times}, {session: control times}, plus a balance table.

    Both sides are de-duplicated at `rip.DEDUP_S` BEFORE matching, so the
    pairing survives the de-duplication that `rate_by_unit` applies later
    instead of being broken by it.
    """
    rng = np.random.default_rng(seed)
    targets, controls, dropped, rows = {}, {}, {}, []
    for session, g in presses.groupby('session'):
        tgt = g.query(query) if query else g.iloc[0:0]
        ctl = g[g.key.isin(control_keys)]
        if not len(tgt) or not len(ctl):
            continue
        tgt = tgt[np.isfinite(tgt.still_next_s)].sort_values('t_s')
        ctl = ctl[np.isfinite(ctl.still_next_s)].sort_values('t_s')
        keep_t = np.isin(tgt.t_s.to_numpy(float), rip.dedup(tgt.t_s.to_numpy(float)))
        keep_c = np.isin(ctl.t_s.to_numpy(float), rip.dedup(ctl.t_s.to_numpy(float)))
        tgt, ctl = tgt[keep_t], ctl[keep_c]
        if not len(tgt) or not len(ctl):
            continue
        t_m, c_m, miss_t = match_one_session(
            tgt.t_s.to_numpy(float), tgt.still_next_s.to_numpy(float),
            ctl.t_s.to_numpy(float), ctl.still_next_s.to_numpy(float),
            caliper_s, rel_caliper, rng)
        if not t_m.size:
            continue
        targets[int(session)] = np.sort(t_m)
        controls[int(session)] = np.sort(c_m)
        if miss_t.size:
            dropped[int(session)] = np.sort(miss_t)
        s_t = tgt.set_index('t_s').still_next_s
        s_c = ctl.set_index('t_s').still_next_s
        rows.append({'session': int(session), 'n_target': int(len(tgt)),
                     'n_matched': int(t_m.size), 'n_unmatched': int(miss_t.size),
                     'target_still_median': float(s_t.loc[t_m].median()),
                     'control_still_median': float(s_c.loc[c_m].median()),
                     'unmatched_still_median': (float(s_t.loc[miss_t].median())
                                                if miss_t.size else np.nan)})
    return targets, controls, dropped, pd.DataFrame(rows)


# ── 2) Testing ────────────────────────────────────────────────────────

def contrast(data, targets, controls, unit, min_events, baselined):
    """target - matched control, either baselined per side or raw.

    Both readings go through the pipeline's own routines; `baselined=True` is
    Sakon Eq. 2 applied to each side before differencing, `False` leaves the
    matched press as the only reference.
    """
    c_t, raw_t, cnt_t = rip.rate_by_unit(data, targets, unit=unit,
                                         min_events=min_events)
    c_c, raw_c, cnt_c = rip.rate_by_unit(data, controls, unit=unit,
                                         min_events=min_events)
    if c_t is None or c_c is None or not raw_t or not raw_c:
        return None, None, None, None
    centres = c_t
    if baselined:
        for d in (raw_t, raw_c):
            units, X = rip.baseline_subtract(d, centres)
            for i, u in enumerate(units):
                d[u] = X[i]
    diff = rip.contrast_profiles({'t': raw_t, 'c': raw_c}, {'t': 1.0, 'c': -1.0})
    if len(diff) < 3:
        return centres, None, cnt_t, cnt_c
    return centres, diff, cnt_t, cnt_c


def test_all(data, presses, unit, min_events, n_perm, caliper_s, rel_caliper,
             seed):
    out, raw_diffs, matched_sets_by_label = {}, {}, {}
    for label, query in TARGETS.items():
        targets, controls, dropped, balance = matched_sets(
            presses, query, caliper_s=caliper_s, rel_caliper=rel_caliper,
            seed=seed)
        if not targets:
            print(f"\n  {label}: nothing matched")
            continue
        n_t = int(balance.n_matched.sum())
        n_u = int(balance.n_unmatched.sum())
        print(f"\n  {label}")
        print(f"    matched {n_t} of {n_t + n_u} events "
              f"({100 * n_t / max(n_t + n_u, 1):.0f}%), "
              f"{len(balance)} sessions")
        print(f"    stillness median: target "
              f"{balance.target_still_median.median():.3f} s vs control "
              f"{balance.control_still_median.median():.3f} s")
        entry = {'n_matched': n_t, 'n_unmatched': n_u,
                 'n_sessions': len(balance),
                 'target_still_median_s': float(balance.target_still_median.median()),
                 'control_still_median_s': float(balance.control_still_median.median()),
                 'readings': {}}
        for name, baselined in (('vs own baseline', True), ('raw', False)):
            centres, diff, cnt_t, cnt_c = contrast(data, targets, controls,
                                                   unit, min_events, baselined)
            if diff is None:
                print(f"    {name:16s} not computable")
                continue
            rec = {'n_units': len(diff), 'windows': {},
                   'counts_target': cnt_t, 'counts_control': cnt_c}
            for wname, win in NAMED_WINDOWS.items():
                got = (rip.window_test(diff, centres, win) if baselined
                       else _raw_window_test(diff, centres, win))
                if got is None:
                    continue
                rec['windows'][wname] = got
                if win == PRIMARY_WINDOW:
                    star = ' *' if got['p_perm'] < 0.05 else ''
                    print(f"    {name:16s} n={got['n_subjects']:3d} "
                          f"mean={got['mean_hz']:+.4f} t={got['t']:+6.2f} "
                          f"p_perm={got['p_perm']:.4g}{star}")
            X = np.vstack([diff[u] for u in sorted(diff)])
            if baselined:
                units, X = rip.baseline_subtract(diff, centres)
            sl = rip.sliding_window_test(
                {label: {u: X[i] for i, u in enumerate(sorted(diff))}},
                centres, width_s=rip.SLIDE_WIDTHS_S[0], n_perm=n_perm,
                seed=seed)
            res = sl.get(label)
            keep = [c for c in res['clusters'] if c['p'] < 0.05] if res else []
            rec['sliding'] = keep
            rec['profile'] = {'centres': centres.tolist(),
                              'mean': np.nanmean(X, axis=0).tolist(),
                              'sem': (np.nanstd(X, axis=0)
                                      / np.sqrt(X.shape[0])).tolist()}
            print(f"    {name:16s} sliding: " +
                  ('; '.join(f"{c['direction']} {c['start_s']:+.2f}.."
                             f"{c['stop_s']:+.2f} p={c['p']:.4f}"
                             for c in keep) if keep else 'none'))
            entry['readings'][name] = rec
            if not baselined:
                # keep the per-session control-adjusted profiles: the 2 x 2
                # interaction is a difference BETWEEN these, each already
                # matched to its own control
                raw_diffs[label] = (centres, diff)
        entry['balance'] = balance.to_dict('records')
        out[label] = entry
        matched_sets_by_label[label] = (targets, dropped, balance)
    return out, raw_diffs, matched_sets_by_label


def between_targets(raw_diffs, n_perm, seed=SEED):
    """Contrasts BETWEEN control-adjusted targets: stage, valence, interaction.

    Each target is already `uncover - its own stillness-matched arrow key`, so
    differencing two of them is a stillness-adjusted comparison of two event
    classes. The 2 x 2 uses the pooled `correct, later` and `error, later` so
    both stages have one cell per valence.
    """
    if not raw_diffs:
        return {}
    centres = next(iter(raw_diffs.values()))[0]
    prof = {k: v[1] for k, v in raw_diffs.items()}
    CF, CL = 'correct, first uncovers', 'correct, later'
    EF, EL = 'error, first uncovers', 'error, later'
    defs = {
        'stage within correct (first - later)': {CF: 1.0, CL: -1.0},
        'stage within error (first - later)': {EF: 1.0, EL: -1.0},
        'valence at first (correct - error)': {CF: 1.0, EF: -1.0},
        'valence later (correct - error)': {CL: 1.0, EL: -1.0},
        'INTERACTION valence x stage': {CF: 1.0, EF: -1.0, CL: -1.0, EL: 1.0},
    }
    out = {}
    for name, w in defs.items():
        if any(k not in prof for k in w):
            continue
        d = rip.contrast_profiles(prof, w)
        if len(d) < 3:
            continue
        got = _raw_window_test(d, centres, PRIMARY_WINDOW)
        X = np.vstack([d[u] for u in sorted(d)])
        sl = rip.sliding_window_test(
            {name: {u: X[i] for i, u in enumerate(sorted(d))}}, centres,
            width_s=rip.SLIDE_WIDTHS_S[0], n_perm=n_perm, seed=seed)
        res = sl.get(name)
        keep = [c for c in res['clusters'] if c['p'] < 0.05] if res else []
        out[name] = {'window': got, 'sliding': keep, 'n_units': len(d),
                     'profile': {'centres': centres.tolist(),
                                 'mean': np.nanmean(X, axis=0).tolist(),
                                 'sem': (np.nanstd(X, axis=0)
                                         / np.sqrt(X.shape[0])).tolist()}}
        star = ' *' if got and got['p_perm'] < 0.05 else ''
        print(f"    {name:38s} n={got['n_subjects']:3d} "
              f"mean={got['mean_hz']:+.4f} t={got['t']:+6.2f} "
              f"p_perm={got['p_perm']:.4g}{star}")
        print(f"    {'':38s} sliding: " +
              ('; '.join(f"{c['direction']} {c['start_s']:+.2f}.."
                         f"{c['stop_s']:+.2f} p={c['p']:.4f}" for c in keep)
               if keep else 'none'))
    return out


def matched_vs_unmatched(data, matched_sets_by_label, unit, min_events):
    """Do the events that found no partner behave differently from those that did?

    The unmatched ones have no control by construction, so the only comparison
    available is each subset against ITS OWN baseline. That is enough to answer
    the question that matters: is the matched subset representative, or did
    matching quietly select the events with the smaller effect?
    """
    rows = []
    for label, (matched, dropped, balance) in matched_sets_by_label.items():
        got = {}
        for name, evs in (('matched', matched), ('unmatched', dropped)):
            if not evs:
                continue
            c, per_unit, cnt = rip.rate_by_unit(data, evs, unit=unit,
                                                min_events=min_events)
            if c is None or len(per_unit) < 3:
                continue
            r = rip.window_test(per_unit, c, PRIMARY_WINDOW)
            if r is None:
                continue
            got[name] = (per_unit, c, r, cnt)
        if not got:
            continue
        still_m = balance.target_still_median.median()
        still_u = balance.unmatched_still_median.median()
        line = {'target': label,
                'still_matched_s': float(still_m),
                'still_unmatched_s': float(still_u)}
        for name, (_, _, r, cnt) in got.items():
            line[f'n_{name}'] = r['n_subjects']
            line[f'events_{name}'] = cnt['n_events_used']
            line[f'mean_{name}_hz'] = r['mean_hz']
            line[f't_{name}'] = r['t']
            line[f'p_{name}'] = r['p_perm']
        if 'matched' in got and 'unmatched' in got:
            (pm, cm, _, _), (pu, _, _, _) = got['matched'], got['unmatched']
            shared = sorted(set(pm) & set(pu))
            if len(shared) >= 3:
                m = (cm >= PRIMARY_WINDOW[0]) & (cm < PRIMARY_WINDOW[1])
                mb = (cm >= rip.BASELINE_WIN[0]) & (cm < rip.BASELINE_WIN[1])
                a = np.array([np.nanmean(pm[u][m]) - np.nanmean(pm[u][mb])
                              for u in shared])
                b = np.array([np.nanmean(pu[u][m]) - np.nanmean(pu[u][mb])
                              for u in shared])
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 3:
                    t, p = stats.ttest_rel(a[ok], b[ok])
                    line['n_paired'] = int(ok.sum())
                    line['diff_hz'] = float((a[ok] - b[ok]).mean())
                    line['diff_t'] = float(t)
                    line['diff_p'] = float(p)
        rows.append(line)
        print(f"    {label:26s} matched {line.get('mean_matched_hz', np.nan):+.4f} "
              f"(still {still_m:.2f} s, n={line.get('n_matched', 0)})  "
              f"unmatched {line.get('mean_unmatched_hz', np.nan):+.4f} "
              f"(still {still_u:.2f} s, n={line.get('n_unmatched', 0)})"
              + (f"  Δ={line['diff_hz']:+.4f}, t={line['diff_t']:+.2f}, "
                 f"p={line['diff_p']:.3f}" if 'diff_hz' in line else ''))
    return pd.DataFrame(rows)


def _raw_window_test(profiles, centres, window, n_perm=10000, seed=SEED):
    """`rip.window_test` without the baseline subtraction it always applies.

    The raw reading needs the window mean of the contrast itself; the pipeline
    function would subtract a baseline that this design has deliberately
    removed. Same one-sample sign-flip null, same output fields.
    """
    units = sorted(profiles)
    X = np.vstack([profiles[u] for u in units])
    m = (centres >= window[0]) & (centres < window[1])
    v = np.nanmean(X[:, m], axis=1)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None
    t, p = stats.ttest_1samp(v, 0.0)
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, v.size)) * v).mean(axis=1)
    return {'window_s': list(window), 'n_subjects': int(v.size),
            'mean_hz': float(v.mean()), 't': float(t), 'p': float(p),
            'p_perm': float((1 + np.sum(np.abs(null) >= abs(v.mean())))
                            / (1 + n_perm))}


# ── 3) Control sanity: do the four arrow keys agree? ──────────────────

def control_by_direction(data, presses, unit, min_events):
    """Rate after each arrow key separately, for gaps of 1-2.5 s.

    If the four directions give different rates, "an arrow key" is not one
    control category and the matching has to respect direction too.
    """
    out = {}
    sel = presses[(presses.still_next_s >= 1.0) & (presses.still_next_s <= 2.5)]
    for key in MOVE_KEYS:
        g = sel[sel.key == key]
        if not len(g):
            continue
        per_sess = {int(s): gg.t_s.to_numpy(float)
                    for s, gg in g.groupby('session')}
        c, per_unit, cnt = rip.rate_by_unit(data, per_sess, unit=unit,
                                            min_events=min_events)
        if c is None or len(per_unit) < 3:
            continue
        m = (c >= PRIMARY_WINDOW[0]) & (c < PRIMARY_WINDOW[1])
        v = np.array([np.nanmean(per_unit[u][m]) for u in sorted(per_unit)])
        v = v[np.isfinite(v)]
        out[key] = {'n_units': int(v.size), 'n_events': cnt['n_events_used'],
                    'mean_hz': float(v.mean()),
                    'sem_hz': float(v.std() / max(np.sqrt(v.size), 1))}
        print(f"    {key:12s} n={v.size:3d} events={cnt['n_events_used']:6d} "
              f"rate={v.mean():.4f} ± {out[key]['sem_hz']:.4f} Hz")
    return out


# ── 4) Figure ─────────────────────────────────────────────────────────

def figure(out_png, results, by_direction, suptitle, footnote):
    legend_cm, title_cm = 2.0, 0.7
    labels = [l for l in TARGETS if l in results]
    with plt.rc_context(_rc()):
        fig, axes = _axes(2, extra_cm=legend_cm + title_cm + 0.6)

        for row, reading in enumerate(('vs own baseline', 'raw')):
            # ---- left: the contrast time courses ---------------------------
            ax = axes[row][0]
            ax.axhline(0, color='0.45', lw=0.8)
            if reading == 'vs own baseline':
                ax.axvspan(*rip.BASELINE_WIN, color='0.88', lw=0, zorder=0)
            for label in labels:
                rec = results[label]['readings'].get(reading)
                if not rec:
                    continue
                p = rec['profile']
                c = np.asarray(p['centres'])
                mean = np.asarray(p['mean'])
                sem = np.asarray(p['sem'])
                ax.plot(c, mean, color=TARGET_COLOUR[label], lw=rip.LW_RATE,
                        zorder=3, label=f"{label} (n={rec['n_units']})")
                ax.fill_between(c, mean - sem, mean + sem,
                                color=TARGET_COLOUR[label], alpha=0.16, lw=0,
                                zorder=2)
            ax.axvline(0, color='0.35', lw=1.0)
            ax.set_xlabel('Time from press (s)', fontsize=rip.FS, labelpad=1)
            ax.set_ylabel('Uncover − matched arrow key\n(Hz)',
                          fontsize=rip.FS - 1)
            ax.set_title(f'Contrast, {reading}', fontsize=rip.FS, pad=3)

            # ---- middle: the primary window, per target --------------------
            ax = axes[row][1]
            ax.axhline(0, color='0.45', lw=0.8)
            for i, label in enumerate(labels):
                rec = results[label]['readings'].get(reading)
                if not rec:
                    continue
                got = rec['windows'].get('post (0..0.5)')
                if not got:
                    continue
                sem = abs(got['mean_hz'] / got['t']) if got['t'] else np.nan
                ax.errorbar([i], [got['mean_hz']], yerr=[sem],
                            color=TARGET_COLOUR[label], marker='o', ms=5,
                            capsize=3, elinewidth=1.4, lw=0)
                if got['p_perm'] < 0.05:
                    ax.annotate('*', xy=(i, 0.96),
                                xycoords=('data', 'axes fraction'),
                                ha='center', va='top', fontsize=rip.FS,
                                color=TARGET_COLOUR[label])
                ax.annotate(f"{got['n_subjects']}", xy=(i, 0.02),
                            xycoords=('data', 'axes fraction'), ha='center',
                            va='bottom', fontsize=rip.FS - 3, color='0.45')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([SHORT.get(l, l) for l in labels],
                               fontsize=rip.FS - 3)
            ax.set_xlim(-0.5, len(labels) - 0.5)
            ax.set_ylabel('Δ rate %.1f–%.1f s (Hz)' % PRIMARY_WINDOW,
                          fontsize=rip.FS - 1)
            ax.set_title(f'0–0.5 s, {reading}', fontsize=rip.FS, pad=3)

            # ---- right: matching balance / control sanity ------------------
            ax = axes[row][2]
            if row == 0:
                for i, label in enumerate(labels):
                    r = results[label]
                    ax.plot([i - 0.16], [r['target_still_median_s']], marker='o',
                            ms=5, color=TARGET_COLOUR[label])
                    ax.plot([i + 0.16], [r['control_still_median_s']],
                            marker='s', ms=5, color='0.45')
                    ax.plot([i - 0.16, i + 0.16],
                            [r['target_still_median_s'],
                             r['control_still_median_s']],
                            color='0.7', lw=0.8, zorder=0)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels([SHORT.get(l, l) for l in labels],
                                   fontsize=rip.FS - 3)
                ax.set_xlim(-0.5, len(labels) - 0.5)
                ax.set_ylabel('Stillness (s, median)\n● target  ■ control',
                              fontsize=rip.FS - 1)
                ax.set_title('Matching balance', fontsize=rip.FS, pad=3)
            else:
                keys = [k for k in MOVE_KEYS if k in by_direction]
                for i, k in enumerate(keys):
                    d = by_direction[k]
                    ax.errorbar([i], [d['mean_hz']], yerr=[d['sem_hz']],
                                color='#252525', marker='s', ms=5, capsize=3,
                                elinewidth=1.4, lw=0)
                    ax.annotate(f"{d['n_units']}", xy=(i, 0.02),
                                xycoords=('data', 'axes fraction'),
                                ha='center', va='bottom',
                                fontsize=rip.FS - 3, color='0.45')
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels([k.replace('Arrow', '') for k in keys],
                                   fontsize=rip.FS - 2)
                ax.set_xlim(-0.5, len(keys) - 0.5)
                ax.set_ylabel('Ripple rate (Hz)\n0–0.5 s, gaps 1–2.5 s',
                              fontsize=rip.FS - 1)
                ax.set_title('Is the control neutral?', fontsize=rip.FS, pad=3)

        handles, lab = axes[0][0].get_legend_handles_labels()
        total = rip.ROW_H_CM * 2 + legend_cm + title_cm + 0.6
        fig.tight_layout(pad=0.4, h_pad=2.0, w_pad=1.8,
                         rect=[0, legend_cm / total, 1, 1 - title_cm / total])
        fig.legend(handles, lab, loc='lower center',
                   bbox_to_anchor=(0.5, 0.03), ncol=3, fontsize=rip.FS - 2,
                   frameon=False, handlelength=1.4, handletextpad=0.4,
                   columnspacing=1.4, borderaxespad=0.0)
        fig.suptitle(suptitle, fontsize=rip.FS, y=0.998, va='top')
        fig.text(0.5, 0.002, footnote, ha='center', va='bottom',
                 fontsize=rip.FS - 3, color='0.4')
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.splitext(out_png)[0] + '.pdf', bbox_inches='tight')
        plt.close(fig)
    return out_png


# ── 5) Main ───────────────────────────────────────────────────────────

def run(bundle=None, out_dir=None, presses=None, unit='session', min_events=10,
        n_perm=rip.N_SIGN_FLIPS, caliper_s=CALIPER_S,
        rel_caliper=REL_CALIPER):
    root = swr_io.get_data_root()
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                              'bundle')
    if presses is None:
        presses = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                               'press_categories.csv')
    if out_dir is None:
        out_dir = os.path.join(
            swr_io.derivatives_dir(root), 'group', 'swr',
            f"matched_control_presses_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_matched_control_presses')
    np.random.seed(SEED)

    if not os.path.exists(presses):
        print(f"  missing {presses}\n  run scripts/swr_build_press_categories.py "
              f"first")
        return None
    data = rip.load_bundle(bundle)
    P = pd.read_csv(presses)
    print(f"\n  bundle: {bundle}\n  presses: {presses}")
    print(f"  {QUESTION}")
    print(f"  {len(P)} presses, {P.session.nunique()} sessions | "
          f"unit = {unit}, min {min_events} events | "
          f"caliper = max({caliper_s:g} s, {100 * rel_caliper:.0f}%)")
    print("\n  press counts by key")
    print(P.key.value_counts().to_string())

    print("\n  control sanity: rate after each arrow key (gaps 1-2.5 s)")
    by_direction = control_by_direction(data, P, unit, min_events)

    print("\n  matched contrasts")
    results, raw_diffs, sets_by_label = test_all(
        data, P, unit, min_events, n_perm, caliper_s, rel_caliper, SEED)

    print("\n  between-target contrasts (each side control-adjusted, raw)")
    contrasts = between_targets(raw_diffs, n_perm, SEED)

    print("\n  matched vs unmatched events, each against its OWN baseline")
    mvu = matched_vs_unmatched(data, sets_by_label, unit, min_events)
    mvu.to_csv(os.path.join(out_dir, 'matched_vs_unmatched.csv'), index=False)

    rows = []
    for label, entry in results.items():
        for reading, rec in entry['readings'].items():
            for wname, got in rec['windows'].items():
                rows.append(dict(target=label, reading=reading, window=wname,
                                 n_matched=entry['n_matched'],
                                 n_unmatched=entry['n_unmatched'],
                                 target_still_s=entry['target_still_median_s'],
                                 control_still_s=entry['control_still_median_s'],
                                 **{k: v for k, v in got.items()
                                    if k != 'window_s'}))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'matched_contrasts.csv'),
                              index=False)

    figure(os.path.join(out_dir, 'matched_control_presses.png'), results,
           by_direction,
           'Each uncover press vs a stillness-matched arrow-key press',
           f'Control = arrow key in the same session, matched 1:1 on time to '
           f'the next press (caliper max({caliper_s:g} s, '
           f'{100 * rel_caliper:.0f}%), without replacement, seed {SEED}). '
           f'unit = {unit}, min {min_events} events per {unit}.')

    payload = {'created': datetime.now().isoformat(timespec='seconds'),
               'question': QUESTION, 'bundle': bundle, 'presses': presses,
               'unit': unit, 'min_events': min_events, 'n_sign_flips': n_perm,
               'caliper_s': caliper_s, 'rel_caliper': rel_caliper,
               'seed': SEED, 'control_keys': list(MOVE_KEYS),
               'control_by_direction': by_direction,
               'targets': {k: {kk: vv for kk, vv in v.items()
                               if kk != 'balance'}
                           for k, v in results.items()},
               'between_target_contrasts': contrasts,
               'matched_vs_unmatched': mvu.to_dict('records')}
    with open(os.path.join(out_dir, 'matched_control_result.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
