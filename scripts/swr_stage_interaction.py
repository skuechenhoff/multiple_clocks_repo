#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore vs execute: does the ripple response to an uncovering depend on stage?

Rows 1-2 of `ripple_main_figure.png` show each valence x stage cell against its
own baseline, and the two error cells look crossed. A crossing of two base->test
slopes is an INTERACTION, and `ripple_statistics.json` never tests it. Its
`between` block carries `stage | error (first - later)`, but that pools
`while learning` with `once known` -- the wrong side of the task for this
question, because `while learning` is still exploration.

This script pools the other way, which is the split the task actually has:

    exploring   = first uncovers + while learning   (the rewards are not yet known)
    once known  = the grid is known and being executed

Primary reading is the stillness-matched one, as everywhere else in this
pipeline: each cell is the uncovering minus its own stillness-matched arrow
press, so a difference between two cells is stillness-adjusted on both sides.
The own-baseline reading (Sakon Eq. 2, what the figure draws) is reported
alongside, because that is the quantity the eye is reading off the figure.

Every contrast is a linear combination of the same per-session profiles the
rest of the pipeline uses, pushed through the same `window_stats` and the same
sliding-window cluster test against the same sign-flip null. Nothing here
re-implements a test. The three contrasts are one family per reading and carry
a Holm adjustment across it.

    python scripts/swr_stage_interaction.py --pad_s=0.25

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import scripts.swr_final_ripple_analysis as fin


SEED = fin.SEED
PRIMARY_WINDOW = fin.PRIMARY_WINDOW
NAMED_WINDOWS = fin.NAMED_WINDOWS

# The pooling. `exploring` keeps the two stages in which the rewards are still
# being discovered; `once known` is the executed grid.
CELLS = {
    'correct, exploring':  "valence == 'correct' and stage != 'once known'",
    'correct, once known': "valence == 'correct' and stage == 'once known'",
    'error, exploring':    "valence == 'error' and stage != 'once known'",
    'error, once known':   "valence == 'error' and stage == 'once known'",
}

CONTRASTS = {
    'error: exploring - once known':
        {'error, exploring': 1.0, 'error, once known': -1.0},
    'correct: exploring - once known':
        {'correct, exploring': 1.0, 'correct, once known': -1.0},
    'INTERACTION (correct - error) x (exploring - once known)':
        {'correct, exploring': 1.0, 'correct, once known': -1.0,
         'error, exploring': -1.0, 'error, once known': 1.0},
}

# The three-stage scheme: every pairwise comparison between stages, inside one
# valence. This is what a claim of the form "significant here, not there" needs
# -- the difference of the two changes, not two separate tests against zero.
CELLS_3 = {f'{v}, {s}': f"valence == '{v}' and stage == '{s}'"
           for v in ('correct', 'error') for s in rip.STAGES}
CONTRASTS_3 = {
    f'{v}: {a} - {b}': {f'{v}, {a}': 1.0, f'{v}, {b}': -1.0}
    for v in ('correct', 'error')
    for a, b in (('first uncovers', 'while learning'),
                 ('first uncovers', 'once known'),
                 ('while learning', 'once known'))}

SCHEMES = {'explore_known': (CELLS, CONTRASTS),
           'three_stage': (CELLS_3, CONTRASTS_3)}

READINGS = ('stillness-matched', 'own baseline')


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the order given."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running = np.empty(p.size), 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def matched_cells(data, presses, unit, min_events, caliper_s, rel_caliper):
    """Per-cell uncovering-minus-matched-arrow-press profiles, per session.

    Both sides are kept as well as their difference: the main-figure layout
    draws the target and its control as two traces, not one contrast.
    """
    prof, meta, centres, sides = {}, {}, None, {}
    for label, query in CELLS.items():
        tgt, ctl, _, bal = fin.matched_sets(presses, query, caliper_s,
                                            rel_caliper, SEED)
        if not tgt:
            continue
        c_t, raw_t, cnt_t = fin.rates(data, tgt, unit, min_events)
        c_c, raw_c, cnt_c = fin.rates(data, ctl, unit, min_events)
        if c_t is None or c_c is None:
            continue
        d = fin.difference(raw_t, raw_c)
        if len(d) < 3:
            continue
        centres, prof[label] = c_t, d
        sides[label] = {'target': raw_t, 'control': raw_c, 'cnt_control': cnt_c}
        n_m, n_u = int(bal.n_matched.sum()), int(bal.n_unmatched.sum())
        meta[label] = {'n_units': len(d), 'n_matched_events': n_m,
                       'n_unmatched_events': n_u,
                       'match_rate': n_m / max(n_m + n_u, 1),
                       'n_events_target': int(cnt_t['n_events_used']),
                       'n_events_control': int(cnt_c['n_events_used']),
                       'still_target_s': float(bal.still_target_s.median()),
                       'still_control_s': float(bal.still_control_s.median())}
    return centres, prof, meta, sides


def own_baseline_cells(data, presses, unit, min_events):
    """Per-cell baseline-subtracted profiles -- the quantity the figure draws."""
    raw, meta, centres, counts = {}, {}, None, {}
    for label, query in CELLS.items():
        per_sess = {int(s): g.t_s.to_numpy(float)
                    for s, g in presses.query(query).groupby('session')}
        if not per_sess:
            continue
        c, per_unit, cnt = fin.rates(data, per_sess, unit, min_events)
        if c is None or len(per_unit) < 3:
            continue
        centres, raw[label], counts[label] = c, per_unit, cnt
        meta[label] = {'n_units': len(per_unit),
                       'n_events': int(cnt['n_events_used'])}
    prof = {l: fin.baseline(p, centres) for l, p in raw.items()}
    for label in meta:
        for name, win in (('base', rip.BASELINE_WIN), ('test', PRIMARY_WINDOW)):
            got = fin.window_stats(raw[label], centres, win)
            meta[label][name] = {'window_s': list(win),
                                 'mean_hz': got['mean_hz'],
                                 'sem_hz': got['sem_hz']}
    return centres, prof, meta, raw, counts


def baseline_window_block(raw, centres, n_perm):
    """The same contrasts on the BASELINE window, with nothing subtracted.

    A 2 x 2 with epoch (baseline, window) as a factor has an epoch x phase
    interaction that is algebraically the paired t on (window - baseline)
    differenced across phase -- i.e. exactly the contrast already reported, not
    a new test. What is NOT tested anywhere is whether the baselines themselves
    differ across cells. If they do, a baseline-subtracted interaction can be
    inherited from the baseline rather than earned by the response.
    """
    out = {'cells': {}, 'contrasts': {}}
    for label, p in raw.items():
        out['cells'][label] = fin.window_stats(p, centres, rip.BASELINE_WIN)
    for name, weights in CONTRASTS.items():
        if any(k not in raw for k in weights):
            continue
        d = rip.contrast_profiles(raw, weights)
        if len(d) < 3:
            continue
        out['contrasts'][name] = {
            'baseline_window': fin.window_stats(d, centres, rip.BASELINE_WIN),
            'test_window_unsubtracted': fin.window_stats(d, centres,
                                                         PRIMARY_WINDOW)}
    # The unsubtracted test window is a second family of the same size, so it
    # carries its own Holm adjustment -- otherwise it is just the subtracted
    # test re-run until one of them passes.
    names = list(out['contrasts'])
    for key in ('baseline_window', 'test_window_unsubtracted'):
        adj = holm([out['contrasts'][n][key]['p_perm'] for n in names])
        for n, a in zip(names, adj):
            out['contrasts'][n][key]['p_perm_holm'] = float(a)
            out['contrasts'][n][key]['family_size'] = len(names)
    return out


def common_cells(prof, centres):
    """The 2 x 2 on the sessions EVERY cell has.

    The interaction is paired across the four cells, so it is tested only on
    their intersection. Plotting each cell on its own session set would draw a
    crossing that no test ever saw; this draws the tested quantity.
    """
    shared = set.intersection(*[set(p) for p in prof.values()])
    if len(shared) < 3:
        return {}
    out = {'n_units': len(shared), 'sessions': sorted(int(s) for s in shared),
           'cells': {}}
    for label, p in prof.items():
        got = fin.window_stats({u: p[u] for u in shared}, centres,
                               PRIMARY_WINDOW)
        out['cells'][label] = got
    # the valence difference inside each stage, same sessions
    for stage in ('exploring', 'once known'):
        a, b = f'correct, {stage}', f'error, {stage}'
        if a in prof and b in prof:
            d = {u: prof[a][u] - prof[b][u] for u in shared}
            out.setdefault('valence_within_stage', {})[stage] = {
                'window': fin.window_stats(d, centres, PRIMARY_WINDOW),
                'profile': fin._profile(d, centres)}
    return out


def test_block(prof, centres, n_perm, tag):
    """Per-cell tests, the three contrasts, and Holm over that family."""
    cells_out = {}
    for label, p in prof.items():
        rec = {'windows': {}}
        for wname, win in NAMED_WINDOWS.items():
            got = fin.window_stats(p, centres, win)
            if got:
                rec['windows'][wname] = got
        _, keep = fin.cluster_test(p, centres, f'{tag}|{label}', n_perm)
        rec['clusters'] = keep
        rec['profile'] = fin._profile(p, centres)
        cells_out[label] = rec

    recs = {}
    for name, weights in CONTRASTS.items():
        if any(k not in prof for k in weights):
            continue
        d = rip.contrast_profiles(prof, weights)
        if len(d) < 3:
            continue
        rec = {'weights': weights, 'n_units': len(d), 'windows': {}}
        for wname, win in NAMED_WINDOWS.items():
            got = fin.window_stats(d, centres, win)
            if got:
                rec['windows'][wname] = got
        _, keep = fin.cluster_test(d, centres, f'{tag}|{name}', n_perm)
        rec['clusters'] = keep
        rec['profile'] = fin._profile(d, centres)
        recs[name] = rec

    names = list(recs)
    if names:
        adj = holm([recs[n]['windows']['post (0..0.5)']['p_perm'] for n in names])
        for n, a in zip(names, adj):
            recs[n]['windows']['post (0..0.5)']['p_perm_holm'] = float(a)
            recs[n]['windows']['post (0..0.5)']['family_size'] = len(names)
    return cells_out, recs


def _report_common(common):
    if not common:
        return
    print(f"    2x2 on the {common['n_units']} sessions every cell has:")
    for label, w in common['cells'].items():
        print(f"      {label:22s} {w['mean_hz']:+.4f} Hz "
              f"[{w['ci_low_hz']:+.4f},{w['ci_high_hz']:+.4f}] "
              f"t({w['df']})={w['t']:+.2f} p={w['p_perm']:.4f}")
    for stage, rec in common.get('valence_within_stage', {}).items():
        w = rec['window']
        print(f"      correct - error | {stage:12s} {w['mean_hz']:+.4f} Hz "
              f"t({w['df']})={w['t']:+.2f} p={w['p_perm']:.4f}")


def _report(reading, cells_out, recs, meta):
    print(f"\n  ── {reading} ──")
    for label, rec in cells_out.items():
        w = rec['windows']['post (0..0.5)']
        extra = ''
        if 'still_target_s' in meta.get(label, {}):
            extra = (f" | still {meta[label]['still_target_s']:.2f} vs "
                     f"{meta[label]['still_control_s']:.2f} s, "
                     f"{meta[label]['n_matched_events']} events")
        elif 'base' in meta.get(label, {}):
            extra = (f" | base {meta[label]['base']['mean_hz']:.4f} -> test "
                     f"{meta[label]['test']['mean_hz']:.4f}, "
                     f"{meta[label]['n_events']} events")
        print(f"    {label:22s} n={w['n_units']:3d} {w['mean_hz']:+.4f} Hz "
              f"t({w['df']})={w['t']:+.2f} p={w['p_perm']:.4f}{extra}")
    for name, rec in recs.items():
        w = rec['windows']['post (0..0.5)']
        keep = rec['clusters']
        print(f"    {name:56s} n={w['n_units']:3d} {w['mean_hz']:+.4f} Hz "
              f"[{w['ci_low_hz']:+.4f},{w['ci_high_hz']:+.4f}] "
              f"t({w['df']})={w['t']:+.2f} p={w['p_perm']:.4f} "
              f"(Holm {w['p_perm_holm']:.4f}) d={w['cohens_d']:+.2f}")
        print(f"    {'':56s} cluster: " + ('; '.join(
            f"{c['direction']} {c['start_s']:+.2f}..{c['stop_s']:+.2f} s "
            f"p={c['p']:.4f}" for c in keep) if keep else 'none'))


def run(bundle=None, out_dir=None, presses_csv=None, unit=fin.UNIT,
        min_events=fin.MIN_EVENTS, n_perm=rip.N_SIGN_FLIPS,
        caliper_s=fin.CALIPER_S, rel_caliper=fin.REL_CALIPER, pad_s=0.25,
        scheme='explore_known'):
    global CELLS, CONTRASTS
    CELLS, CONTRASTS = SCHEMES[scheme]
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr')
    bundle = bundle or os.path.join(group, 'bundle_v2')
    presses_csv = presses_csv or os.path.join(group, 'press_categories.csv')
    suffix = '' if scheme == 'explore_known' else f'_{scheme}'
    out_dir = out_dir or os.path.join(
        group, f"ripple_stage_interaction{suffix}_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_stage_interaction')
    np.random.seed(SEED)

    print(f"\n  bundle : {bundle}\n  presses: {presses_csv}\n  out    : {out_dir}")
    print(f"  scheme: {scheme} -> cells {list(CELLS)}")
    data = rip.load_bundle(bundle)
    if pad_s is not None:
        import mc.analyse.swr_bundle as swb
        n0 = len(data['ripples'])
        data = swb.repad_bundle(data, float(pad_s))
        print(f"  re-padded to {pad_s} s: {n0} -> {len(data['ripples'])} ripples")
    presses = fin.press_table(data, bundle, presses_csv)

    blocks = {}
    c_m, prof_m, meta_m, sides_m = matched_cells(data, presses, unit,
                                                 min_events, caliper_s,
                                                 rel_caliper)
    cells_m, recs_m = test_block(prof_m, c_m, n_perm, 'matched')
    _report('stillness-matched (uncover − matched arrow press)',
            cells_m, recs_m, meta_m)
    blocks_common_m = common_cells(prof_m, c_m) if scheme == 'explore_known' else {}
    _report_common(blocks_common_m)
    blocks['stillness-matched'] = {'cells': cells_m, 'contrasts': recs_m,
                                   'meta': meta_m,
                                   'common': blocks_common_m}

    unc = presses[presses.kind == 'uncover']
    c_b, prof_b, meta_b, raw_b, counts_b = own_baseline_cells(
        data, unc, unit, min_events)
    cells_b, recs_b = test_block(prof_b, c_b, n_perm, 'ownbase')
    _report('own baseline (Sakon Eq. 2, what the figure draws)',
            cells_b, recs_b, meta_b)
    blocks_common_b = common_cells(prof_b, c_b) if scheme == 'explore_known' else {}
    _report_common(blocks_common_b)
    blocks['own baseline'] = {'cells': cells_b, 'contrasts': recs_b,
                              'meta': meta_b,
                              'common': blocks_common_b}

    # Do the baselines themselves differ? Nothing else in the pipeline asks.
    bw = baseline_window_block(raw_b, c_b, n_perm)
    print(f"\n  ── baseline window {rip.BASELINE_WIN} alone, nothing subtracted ──")
    for label, w in bw['cells'].items():
        print(f"    {label:22s} n={w['n_units']:3d} rate={w['mean_hz']:.4f} "
              f"± {w['sem_hz']:.4f} Hz")
    for name, rec in bw['contrasts'].items():
        b, t = rec['baseline_window'], rec['test_window_unsubtracted']
        print(f"    {name:56s}")
        print(f"      baseline window   {b['mean_hz']:+.4f} Hz "
              f"t({b['df']})={b['t']:+.2f} p={b['p_perm']:.4f}")
        print(f"      test window (raw) {t['mean_hz']:+.4f} Hz "
              f"t({t['df']})={t['t']:+.2f} p={t['p_perm']:.4f}")
    blocks['baseline window'] = bw

    payload = {'created': datetime.now().isoformat(timespec='seconds'),
               'script': os.path.basename(__file__),
               'question': ('does the ripple response to an uncovering differ '
                            'between task stages?'),
               'scheme': scheme,
               'settings': {'bundle': bundle, 'presses_csv': presses_csv,
                            'pad_s': pad_s, 'unit': unit,
                            'min_events': min_events, 'n_sign_flips': n_perm,
                            'seed': SEED, 'caliper_s': caliper_s,
                            'rel_caliper': rel_caliper,
                            'cells': CELLS,
                            'contrasts': {k: v for k, v in CONTRASTS.items()},
                            'primary_window_s': list(PRIMARY_WINDOW),
                            'named_windows_s': {k: list(v) for k, v
                                                in NAMED_WINDOWS.items()},
                            'baseline_window_s': list(rip.BASELINE_WIN),
                            'sliding_width_s': rip.SLIDE_WIDTHS_S[0],
                            'multiple_comparisons': {
                                'family': 'the three contrasts, per reading',
                                'size': len(CONTRASTS),
                                'method': 'Holm-Bonferroni on p_perm, '
                                          'primary window only'}},
               'readings': blocks}
    with open(os.path.join(out_dir, 'stage_interaction.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)

    if scheme == 'explore_known':
        figure(os.path.join(out_dir, 'stage_interaction.png'), payload)
        figure_correct_phase(os.path.join(out_dir, 'correct_phase.png'), payload)
        figure_phase_pairwise(os.path.join(out_dir, 'phase_pairwise.png'), payload)
        figure_rows_two_phase(
            os.path.join(out_dir, 'two_phase_main.png'), c_b, raw_b, prof_b,
            counts_b, c_m, prof_m, sides_m, meta_m, n_perm, unit, min_events,
            desc=descriptives(data))
    else:
        figure_three_stage(os.path.join(out_dir, 'stage_pairwise.png'), payload)
    print(f"\n  saved -> {out_dir}")
    return None


# ── Figure ────────────────────────────────────────────────────────────
# Panels are 3.2 x 3.2 cm on an A4 page. Type and line weights are set in
# points at that size -- no post-hoc scaling -- so what comes out is what is
# asked for. Colours come from `rip.condition_colour`, the pipeline's own rule:
# valence sets the hue, stage sets the lightness (dark = exploring,
# light = once known), so these panels match the main figure exactly.
PANEL_CM = 3.2
CM = 1 / 2.54
FS_TITLE, FS_LABEL, FS_TICK = 10, 9, 8.5
LW = 2.8
# DISPLAY ONLY. Every window and cluster test below reads the unsmoothed
# per-session profiles; this only stops a 0.1 s-bin difference trace from
# looking like noise.
SMOOTH_DISPLAY = 5


def _sm(y):
    return rip.triangle_smooth(np.asarray(y, float), SMOOTH_DISPLAY)

VALENCE_DARK = {'correct': rip.VALENCE_COLOUR['correct'],
                'error': rip.VALENCE_COLOUR['error']}
# 'exploring' takes the `first uncovers` lightness, 'once known' its own.
CELL_COLOUR = {
    'correct, exploring':  rip.condition_colour('correct, first uncovers'),
    'correct, once known': rip.condition_colour('correct, once known'),
    'error, exploring':    rip.condition_colour('error, first uncovers'),
    'error, once known':   rip.condition_colour('error, once known'),
}
STAGES_X = ('exploring', 'once known')
STAGE_LABEL = {'exploring': 'exploring', 'once known': 'once known'}
INT_KEY = 'INTERACTION (correct - error) x (exploring - once known)'


def _stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def _cell_panel(ax, common, interaction, ylabel, title):
    """The 2 x 2 itself, on the sessions the paired test actually uses.

    Hue is valence, fill lightness is stage -- the same coding as the main
    figure, so a dark marker is an exploring cell in both.
    """
    if not common:
        ax.set_visible(False)
        return
    for valence in ('correct', 'error'):
        y, lo, hi = [], [], []
        for stage in STAGES_X:
            w = common['cells'][f'{valence}, {stage}']
            y.append(w['mean_hz'])
            lo.append(w['mean_hz'] - w['ci_low_hz'])
            hi.append(w['ci_high_hz'] - w['mean_hz'])
        ax.plot([0, 1], y, color=VALENCE_DARK[valence], lw=LW, zorder=2)
        for i, stage in enumerate(STAGES_X):
            ax.errorbar([i], [y[i]], yerr=[[lo[i]], [hi[i]]],
                        color=VALENCE_DARK[valence], lw=1.4, capsize=3,
                        capthick=1.4, marker='o', ms=7,
                        mfc=CELL_COLOUR[f'{valence}, {stage}'],
                        mec=VALENCE_DARK[valence], mew=1.6, zorder=3)
    ax.axhline(0, color='0.6', lw=1.0, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['explore', 'known'], fontsize=FS_TICK)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    w = interaction['windows']['post (0..0.5)']
    ax.set_xlabel(f"n = {common['n_units']} sessions", fontsize=FS_LABEL)
    ax.set_title(f"interaction\np = {w['p_perm']:.3f} {_stars(w['p_perm'])}",
                 fontsize=FS_TITLE)


def _valence_panel(ax, rec, stage, ylim, cluster=(), ylabel=None):
    """correct − error at one stage, filled by who wins.

    Teal above zero means the correct uncover carries more ripples than the
    error one; pink below zero means the reverse. The INTERACTION is the
    difference between this panel and its neighbour -- nothing more.
    """
    p = rec['profile']
    x = np.asarray(p['centres_s']); m = _sm(p['mean_hz']); sd = _sm(p['sem_hz'])
    ax.axvspan(PRIMARY_WINDOW[0], PRIMARY_WINDOW[1], color='0.90', zorder=0, lw=0)
    for c in cluster:
        ax.axvspan(c['start_s'], c['stop_s'], color='0.72', zorder=0, lw=0)
    ax.fill_between(x, 0, m, where=m >= 0, interpolate=True,
                    color=VALENCE_DARK['correct'], alpha=0.55, lw=0, zorder=1)
    ax.fill_between(x, 0, m, where=m <= 0, interpolate=True,
                    color=VALENCE_DARK['error'], alpha=0.55, lw=0, zorder=1)
    ax.fill_between(x, m - sd, m + sd, color='0.35', alpha=0.16, lw=0, zorder=2)
    ax.plot(x, m, color='0.15', lw=2.0, zorder=3)
    ax.axhline(0, color='0.4', lw=1.0, zorder=4)
    ax.axvline(0, color='0.4', lw=1.0, ls=':', zorder=4)
    ax.set_ylim(ylim)
    ax.set_xlim(-2, 2)
    ax.set_xticks([-2, 0, 2])
    w = rec['window']
    ax.set_title(f"{stage}\np = {w['p_perm']:.3f} {_stars(w['p_perm'])}",
                 fontsize=FS_TITLE)
    ax.set_xlabel('Time from uncover (s)', fontsize=FS_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL - 0.5)


def figure(out_png, payload):
    """Redrawn from the saved payload, so it replots without refitting."""
    plt.rcParams.update({'font.family': 'Arial', 'font.size': FS_LABEL,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 1.0, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_CM * CM + 3.4 * CM,
                                            2 * PANEL_CM * CM + 3.0 * CM))

    rows = [('own baseline', 'Change vs own\nbaseline (Hz)'),
            ('stillness-matched', 'Uncover − matched\ncontrol (Hz)')]
    for r, (reading, ylab) in enumerate(rows):
        blk = payload['readings'][reading]
        vws = blk.get('common', {}).get('valence_within_stage', {})
        if not vws:
            continue
        # one y scale for both stages, or the eye reads the wrong difference
        lim = 0.0
        for rec in vws.values():
            p = rec['profile']
            m, sd = np.asarray(p['mean_hz']), np.asarray(p['sem_hz'])
            lim = max(lim, np.nanmax(np.abs(m) + sd))
        ylim = (-1.08 * lim, 1.08 * lim)

        _cell_panel(axes[r, 0], blk.get('common'), blk['contrasts'][INT_KEY],
                    ylab, reading)
        clusters = blk['contrasts'][INT_KEY]['clusters']
        for c, stage in enumerate(STAGES_X):
            if stage in vws:
                _valence_panel(axes[r, c + 1], vws[stage], stage, ylim,
                               cluster=clusters,
                               ylabel='correct − error (Hz)' if c == 0 else None)

    for ax in axes.ravel():
        ax.tick_params(labelsize=FS_TICK, width=1.0, length=2.5)
    # Valence identity lives in one shared legend: inside a 3.2 cm panel any
    # in-panel label lands on an error bar.
    handles = [plt.Line2D([], [], color=VALENCE_DARK[v], lw=LW, marker='o',
                          ms=6, mec=VALENCE_DARK[v], mew=1.6,
                          mfc=rip.condition_colour(f'{v}, first uncovers'),
                          label=f'{v} uncover')
               for v in ('correct', 'error')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.005),
               ncol=2, frameon=False, fontsize=FS_LABEL, handlelength=2.0,
               columnspacing=2.0)
    fig.suptitle('Positive feedback raises the ripple rate only while the '
                 'rewards are still being learned', fontsize=FS_TITLE, y=1.08)
    fig.tight_layout(w_pad=1.6, h_pad=2.0, rect=[0.06, 0.05, 1, 1])
    for y, name in ((0.73, 'own baseline'), (0.26, 'stillness-matched')):
        fig.text(0.005, y, name, rotation=90, va='center', ha='left',
                 fontsize=FS_LABEL, weight='bold')
    fig.text(0.5, -0.02,
             'Marker fill: dark = exploring, light = once known (same coding '
             'as the main figure).  Light grey = 0–0.5 s test window, '
             'dark grey = interaction cluster.\n'
             'Teal fill = correct uncover carries more ripples, '
             'pink fill = error uncover does.  Sign-flip p, session as unit; '
             'the interaction carries the Holm adjustment, the two per-stage '
             'p are an uncorrected decomposition of it.',
             ha='center', va='top', fontsize=FS_TICK - 1, color='0.35')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Figure 2: the phase effect within correct uncovers alone ──────────
# Not an interaction -- with two phases this is the simple effect of phase
# inside one valence. It is the top line of the 2 x 2 on its own.
CORR_KEY = 'correct: exploring - once known'
PHASE_COL = {'exploring': CELL_COLOUR['correct, exploring'],
             'once known': CELL_COLOUR['correct, once known']}


def _phase_points(ax, cells, contrast, ylabel):
    """Change vs baseline for the two phases, and the contrast's p."""
    y, lo, hi = [], [], []
    for stage in STAGES_X:
        w = cells[f'correct, {stage}']['windows']['post (0..0.5)']
        y.append(w['mean_hz'])
        lo.append(w['mean_hz'] - w['ci_low_hz'])
        hi.append(w['ci_high_hz'] - w['mean_hz'])
    ax.plot([0, 1], y, color=VALENCE_DARK['correct'], lw=LW, zorder=2)
    for i, stage in enumerate(STAGES_X):
        ax.errorbar([i], [y[i]], yerr=[[lo[i]], [hi[i]]],
                    color=VALENCE_DARK['correct'], lw=1.4, capsize=3,
                    capthick=1.4, marker='o', ms=7, mfc=PHASE_COL[stage],
                    mec=VALENCE_DARK['correct'], mew=1.6, zorder=3)
    ax.axhline(0, color='0.6', lw=1.0, zorder=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['explore', 'known'],
                                              fontsize=FS_TICK)
    ax.set_xlim(-0.4, 1.4)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.08 * (hi_ - lo_))
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    w = contrast['windows']['post (0..0.5)']
    ax.set_xlabel(f"n = {w['n_units']} sessions", fontsize=FS_LABEL)
    ax.set_title(f"explore − known\np = {w['p_perm']:.3f} {_stars(w['p_perm'])}",
                 fontsize=FS_TITLE)


def _phase_traces(ax, cells, ylabel):
    """Both phases' time courses, so the contrast is visible as a gap."""
    ax.axvspan(PRIMARY_WINDOW[0], PRIMARY_WINDOW[1], color='0.90', zorder=0, lw=0)
    for stage in STAGES_X:
        pr = cells[f'correct, {stage}']['profile']
        x = np.asarray(pr['centres_s']); m = _sm(pr['mean_hz'])
        sd = _sm(pr['sem_hz'])
        ax.fill_between(x, m - sd, m + sd, color=PHASE_COL[stage], alpha=0.30,
                        lw=0, zorder=1)
        ax.plot(x, m, color=PHASE_COL[stage], lw=LW, zorder=2,
                path_effects=None)
    ax.axhline(0, color='0.4', lw=1.0); ax.axvline(0, color='0.4', lw=1.0, ls=':')
    ax.set_xlim(-2, 2); ax.set_xticks([-2, 0, 2])
    ax.set_xlabel('Time from uncover (s)', fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL - 0.5)
    ax.set_title('correct uncovers\nboth phases', fontsize=FS_TITLE)


def _phase_diff(ax, contrast):
    """explore − known, filled by whichever phase carries more ripples."""
    pr = contrast['profile']
    x = np.asarray(pr['centres_s']); m = _sm(pr['mean_hz']); sd = _sm(pr['sem_hz'])
    ax.axvspan(PRIMARY_WINDOW[0], PRIMARY_WINDOW[1], color='0.90', zorder=0, lw=0)
    for c in contrast['clusters']:
        ax.axvspan(c['start_s'], c['stop_s'], color='0.72', zorder=0, lw=0)
    ax.fill_between(x, 0, m, where=m >= 0, interpolate=True,
                    color=PHASE_COL['exploring'], alpha=0.60, lw=0, zorder=1)
    ax.fill_between(x, 0, m, where=m <= 0, interpolate=True,
                    color=PHASE_COL['once known'], alpha=0.85, lw=0, zorder=1)
    ax.fill_between(x, m - sd, m + sd, color='0.35', alpha=0.16, lw=0, zorder=2)
    ax.plot(x, m, color='0.15', lw=2.0, zorder=3)
    ax.axhline(0, color='0.4', lw=1.0, zorder=4)
    ax.axvline(0, color='0.4', lw=1.0, ls=':', zorder=4)
    ax.set_xlim(-2, 2); ax.set_xticks([-2, 0, 2])
    cl = ('; '.join(f"{c['start_s']:+.2f}..{c['stop_s']:+.2f} s p={c['p']:.3f}"
                    for c in contrast['clusters']) or 'no cluster')
    ax.set_xlabel('Time from uncover (s)', fontsize=FS_LABEL)
    ax.set_ylabel('explore − known (Hz)', fontsize=FS_LABEL - 0.5)
    ax.set_title(f"difference\n{cl}", fontsize=FS_TITLE)


def figure_correct_phase(out_png, payload):
    """Does the correct-uncover response itself depend on phase?"""
    plt.rcParams.update({'font.family': 'Arial', 'font.size': FS_LABEL,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 1.0, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_CM * CM + 3.4 * CM,
                                            2 * PANEL_CM * CM + 3.0 * CM))
    rows = [('own baseline', 'Change vs own\nbaseline (Hz)'),
            ('stillness-matched', 'Uncover − matched\ncontrol (Hz)')]
    for r, (reading, ylab) in enumerate(rows):
        blk = payload['readings'][reading]
        if CORR_KEY not in blk['contrasts']:
            continue
        _phase_points(axes[r, 0], blk['cells'], blk['contrasts'][CORR_KEY], ylab)
        _phase_traces(axes[r, 1], blk['cells'], ylab)
        _phase_diff(axes[r, 2], blk['contrasts'][CORR_KEY])

    for ax in axes.ravel():
        ax.tick_params(labelsize=FS_TICK, width=1.0, length=2.5)
    handles = [plt.Line2D([], [], color=VALENCE_DARK['correct'], lw=LW,
                          marker='o', ms=6, mec=VALENCE_DARK['correct'],
                          mew=1.6, mfc=PHASE_COL[st], label=st)
               for st in STAGES_X]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.005),
               ncol=2, frameon=False, fontsize=FS_LABEL, handlelength=2.0,
               columnspacing=2.0)
    fig.suptitle('Correct uncovers alone: the ripple response is confined to '
                 'the explore phase', fontsize=FS_TITLE, y=1.08)
    fig.tight_layout(w_pad=1.6, h_pad=2.0, rect=[0.06, 0.05, 1, 1])
    for y, name in ((0.73, 'own baseline'), (0.26, 'stillness-matched')):
        fig.text(0.005, y, name, rotation=90, va='center', ha='left',
                 fontsize=FS_LABEL, weight='bold')
    fig.text(0.5, -0.02,
             'Simple effect of phase within correct uncovers -- with two '
             'phases this is a difference, not an interaction.  Light grey = '
             '0-0.5 s test window.\nFill in the right column marks whichever '
             'phase carries more ripples.  Sign-flip p, session as unit; Holm '
             'over the same three contrasts.',
             ha='center', va='top', fontsize=FS_TICK - 1, color='0.35')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Figure 3: every pairwise stage comparison, inside each valence ────
STAGE_SHORT = {'first uncovers': 'first', 'while learning': 'learning',
               'once known': 'known'}


def _stage_points(ax, cells, valence, ylabel, title, stages=None, short=None):
    """Change vs baseline at each stage, one valence."""
    stages = stages or rip.STAGES
    short = short or STAGE_SHORT
    labels = [f'{valence}, {st}' for st in stages if f'{valence}, {st}' in cells]
    for i, lab in enumerate(labels):
        w = cells[lab]['windows']['post (0..0.5)']
        ax.errorbar([i], [w['mean_hz']],
                    yerr=[[w['mean_hz'] - w['ci_low_hz']],
                          [w['ci_high_hz'] - w['mean_hz']]],
                    color=VALENCE_DARK[valence], lw=1.4, capsize=3, capthick=1.4,
                    marker='o', ms=7, mfc=rip.condition_colour(lab),
                    mec=VALENCE_DARK[valence], mew=1.6, zorder=3)
    ax.plot(range(len(labels)),
            [cells[l]['windows']['post (0..0.5)']['mean_hz'] for l in labels],
            color=VALENCE_DARK[valence], lw=LW, zorder=2)
    ax.axhline(0, color='0.6', lw=1.0, zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([short[st] for st in stages][:len(labels)],
                       fontsize=FS_TICK, rotation=30, ha='right',
                       rotation_mode='anchor')
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)


def _short_contrast(n):
    """Valence has to stay in the label: colour alone leaves two rows equal."""
    if n.startswith('INTERACTION'):
        return 'interaction'
    val, body = (n.split(': ', 1) if ': ' in n else ('', n))
    for a, b in (('first uncovers', 'first'), ('while learning', 'learning'),
                 ('once known', 'known'), ('exploring', 'explore')):
        body = body.replace(a, b)
    pre = {'correct': 'corr', 'error': 'err'}.get(val, val)
    return f'{pre}: {body}' if pre else body


def _forest(ax, recs, ylabel):
    """The six pairwise differences, with Holm-adjusted significance."""
    names = list(recs)
    for i, name in enumerate(names):
        w = recs[name]['windows']['post (0..0.5)']
        val = name.split(':')[0].strip()
        col = VALENCE_DARK.get(val, '#4a4a4a')
        sig = w['p_perm_holm'] < 0.05
        ax.errorbar([w['mean_hz']], [len(names) - 1 - i],
                    xerr=[[w['mean_hz'] - w['ci_low_hz']],
                          [w['ci_high_hz'] - w['mean_hz']]],
                    color=col, lw=1.4, capsize=2.5, capthick=1.4,
                    marker='o', ms=6, mfc=col if sig else 'white',
                    mec=col, mew=1.6)
        if recs[name]['clusters']:
            ax.plot([w['ci_high_hz']], [len(names) - 1 - i], marker='*',
                    ms=6, color='0.2', ls='none')
    ax.axvline(0, color='0.6', lw=1.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([_short_contrast(n) for n in names][::-1],
                       fontsize=FS_TICK - 0.5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel(ylabel, fontsize=FS_LABEL - 0.5)
    # three tick labels of width "-0.05" do not fit across a 3.2 cm panel
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='x', labelsize=FS_TICK - 1.5)
    ax.set_title('pairwise differences\nfilled = Holm p < 0.05', fontsize=FS_TITLE)


def figure_three_stage(out_png, payload):
    """Is the effect bigger at one stage than another? Tested, per valence."""
    plt.rcParams.update({'font.family': 'Arial', 'font.size': FS_LABEL,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 1.0, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_CM * CM + 4.6 * CM,
                                            2 * PANEL_CM * CM + 3.0 * CM))
    rows = [('own baseline', 'Change vs own\nbaseline (Hz)'),
            ('stillness-matched', 'Uncover − matched\ncontrol (Hz)')]
    for r, (reading, ylab) in enumerate(rows):
        blk = payload['readings'][reading]
        _stage_points(axes[r, 0], blk['cells'], 'correct', ylab,
                      'correct uncovers')
        _stage_points(axes[r, 1], blk['cells'], 'error', ylab,
                      'error uncovers')
        _forest(axes[r, 2], blk['contrasts'], 'difference (Hz)')
    for ax in axes.ravel():
        ax.tick_params(labelsize=FS_TICK, width=1.0, length=2.5)
    fig.suptitle('Is the ripple response bigger at one stage than another?',
                 fontsize=FS_TITLE, y=1.05)
    fig.tight_layout(w_pad=1.6, h_pad=2.0, rect=[0.06, 0.05, 1, 1])
    for y, name in ((0.73, 'own baseline'), (0.26, 'stillness-matched')):
        fig.text(0.005, y, name, rotation=90, va='center', ha='left',
                 fontsize=FS_LABEL, weight='bold')
    fig.text(0.5, -0.02,
             'Left and middle: each stage against its own baseline, 0-0.5 s. '
             'Right: the pairwise DIFFERENCE of those changes, mean +- 95% CI, '
             'Holm over the six contrasts.\nA star marks a contrast whose '
             'window-free sliding test also yields a cluster. Marker fill '
             'lightness codes stage, as in the main figure.',
             ha='center', va='top', fontsize=FS_TICK - 1, color='0.35')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


PHASE_SHORT = {'exploring': 'explore', 'once known': 'known'}


def figure_phase_pairwise(out_png, payload):
    """The explore-vs-known figure in the same layout as the three-stage one."""
    plt.rcParams.update({'font.family': 'Arial', 'font.size': FS_LABEL,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 1.0, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(3 * PANEL_CM * CM + 4.6 * CM,
                                            2 * PANEL_CM * CM + 3.0 * CM))
    rows = [('own baseline', 'Change vs own\nbaseline (Hz)'),
            ('stillness-matched', 'Uncover − matched\ncontrol (Hz)')]
    for r, (reading, ylab) in enumerate(rows):
        blk = payload['readings'][reading]
        _stage_points(axes[r, 0], blk['cells'], 'correct', ylab,
                      'correct uncovers', STAGES_X, PHASE_SHORT)
        _stage_points(axes[r, 1], blk['cells'], 'error', ylab,
                      'error uncovers', STAGES_X, PHASE_SHORT)
        _forest(axes[r, 2], blk['contrasts'], 'difference (Hz)')
    for ax in axes.ravel():
        ax.tick_params(labelsize=FS_TICK, width=1.0, length=2.5)
    fig.suptitle('Two phases: is the ripple response bigger while exploring '
                 'than once the grid is known?', fontsize=FS_TITLE, y=1.05)
    fig.tight_layout(w_pad=1.6, h_pad=2.0, rect=[0.06, 0.05, 1, 1])
    for y, name in ((0.73, 'own baseline'), (0.26, 'stillness-matched')):
        fig.text(0.005, y, name, rotation=90, va='center', ha='left',
                 fontsize=FS_LABEL, weight='bold')
    fig.text(0.5, -0.02,
             'Left and middle: each phase against its own baseline, 0-0.5 s. '
             'Right: the DIFFERENCE of those changes, mean +- 95% CI, Holm '
             'over the three contrasts.\nThe interaction (grey) is the two '
             'valence differences differenced. A star marks a contrast whose '
             'window-free sliding test also yields a cluster.',
             ha='center', va='top', fontsize=FS_TICK - 1, color='0.35')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Figure 4: the main-figure layout, in two phases ───────────────────
HEADLINE_2 = 'correct, exploring'
CONTROL_LABEL = 'matched arrow press'
CONTROL_COLOUR = '#5a5a5a'
SMOOTH_BINS = 5      # the main figure's display smooth; DISPLAY ONLY


def descriptives(data):
    """The footer numbers, counted as `swr_final_ripple_analysis` counts them."""
    qc = data['channel_qc']
    qc = qc[~qc.excluded.fillna(False)] if 'excluded' in qc else qc
    return {'n_sessions': int(data['ripples'].session.nunique()),
            'n_subjects': int(data['ripples'].subject_key.nunique()),
            'n_derivations': int(len(qc))}


def figure_rows_two_phase(out_png, c_b, raw_b, prof_b, counts_b, c_m, prof_m,
                          sides_m, meta_m, n_perm, unit, min_events, desc=None):
    """`ripple_main_figure` with two phases instead of three stages.

    Same three panels per row and the same `rip.plot_rows`, so the peri-event
    trace, the base-vs-window pair and the sliding test are drawn and tested
    exactly as in the published figure -- only the cells change.
    """
    # rows 1-2: own baseline, one sliding test over all four cells at once
    sliding = rip.sliding_window_test(prof_b, c_b,
                                      width_s=rip.SLIDE_WIDTHS_S[0],
                                      n_perm=n_perm, seed=SEED)
    # row 3: the headline cell against its stillness-matched control press
    row3, row3_sliding, row3_counts = {}, {}, {}
    if HEADLINE_2 in sides_m:
        row3 = {HEADLINE_2: sides_m[HEADLINE_2]['target'],
                CONTROL_LABEL: sides_m[HEADLINE_2]['control']}
        sl = rip.sliding_window_test({HEADLINE_2: prof_m[HEADLINE_2]}, c_m,
                                     width_s=rip.SLIDE_WIDTHS_S[0],
                                     n_perm=n_perm, seed=SEED)
        row3_sliding = {HEADLINE_2: sl.get(HEADLINE_2), CONTROL_LABEL: None}
        row3_counts = {CONTROL_LABEL: {
            'unit': unit,
            'n_units': len(sides_m[HEADLINE_2]['control']),
            'min_events': min_events, **(desc or {})}}

    rows = [('positive feedback',
             {l: raw_b[l] for l in CELLS if l.startswith('correct') and l in raw_b},
             {l: sliding.get(l) for l in CELLS if l.startswith('correct')
              and l in raw_b}),
            ('negative feedback',
             {l: raw_b[l] for l in CELLS if l.startswith('error') and l in raw_b},
             {l: sliding.get(l) for l in CELLS if l.startswith('error')
              and l in raw_b})]
    if row3:
        rows.append(('stillness-matched', row3, row3_sliding))

    rip.plot_rows(rows, out_png, width_s=rip.SLIDE_WIDTHS_S[0],
                  counts={**row3_counts, **counts_b},
                  smooth_bins=SMOOTH_BINS, legend_cm=1.6,
                  colours={CONTROL_LABEL: CONTROL_COLOUR},
                  suptitle='Ripples at uncovering, two phases: own baseline '
                           '(rows 1-2) and a stillness-matched control press '
                           '(row 3)')


def replot(out_dir):
    """Redraw the figure from an existing stage_interaction.json."""
    with open(os.path.join(out_dir, 'stage_interaction.json')) as f:
        payload = json.load(f)
    if payload.get('scheme', 'explore_known') == 'explore_known':
        figure(os.path.join(out_dir, 'stage_interaction.png'), payload)
        figure_correct_phase(os.path.join(out_dir, 'correct_phase.png'), payload)
        figure_phase_pairwise(os.path.join(out_dir, 'phase_pairwise.png'), payload)
    else:
        figure_three_stage(os.path.join(out_dir, 'stage_pairwise.png'), payload)
    print(f"  redrew -> {out_dir}")


if __name__ == "__main__":
    try:
        import fire
        fire.Fire({'run': run, 'replot': replot})
    except ImportError:
        run()
