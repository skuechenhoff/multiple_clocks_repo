#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How long does the subject sit still after an uncovering -- by phase and by
feedback?

WHY THIS IS NOT A SIDE QUESTION
    Ripple rate rises with how long the subject then sits still. So if correct
    and incorrect uncoverings are followed by different amounts of stillness,
    or if the same feedback is followed by different stillness in different
    phases, then a valence effect or a phase effect on ripple rate can be that
    behavioural difference rather than anything neural. This is the quantity
    `swr_final_ripple_analysis` matches on; this file asks what it looks like
    BEFORE the matching, across every phase definition in
    `exploratory/swr_robustness_grid`, so the two can be read together.

WHAT IS MEASURED
    `still_next_s` -- the gap from a press to the NEXT press of any kind. It is
    the same column the matching uses, taken from the same cached press table,
    so nothing here can drift from the matched analysis.

THE UNIT
    The session, as everywhere else. Each session contributes the MEDIAN
    stillness of its events in a cell (the quantity is a heavily skewed
    duration, so a session mean would be carried by its longest pause), and
    tests are paired across the sessions that have both sides. Wilcoxon is
    reported beside the paired t because the session medians are themselves
    skewed; both are on the same paired differences.

TWO QUESTIONS
    1  correct vs incorrect, WITHIN each phase
    2  the same feedback, ACROSS phases

    python scripts/swr_stillness_by_phase.py run

@author: Svenja Kuchenhoff

STATUS
    ◆ SUPPORTING — cited in the results and methods. Supplies the behavioural
    stillness numbers the stillness-matched control is justified by (+0.95 s
    correct vs incorrect during encoding). Not a neural result.
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
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import scripts.swr_final_ripple_analysis as fin
import mc.analyse.swr_phases as grid

MIN_EVENTS = fin.MIN_EVENTS
CM = 1 / 2.54
# Feedback colours as the HFB branch fixes them.
VAL_C = {'correct': '#0e3d3a', 'error': '#a30d6c'}


def cell_stillness(presses, phase_q, valence, min_events=MIN_EVENTS):
    """{session: median still_next_s} for one (phase, valence) cell."""
    q = f"valence == '{valence}' and ({phase_q})"
    g = presses.query(q)
    g = g[np.isfinite(g.still_next_s)]
    if not len(g):
        return {}, 0
    out, n = {}, 0
    for sess, gg in g.groupby('session'):
        if len(gg) < min_events:
            continue
        out[int(sess)] = float(gg.still_next_s.median())
        n += len(gg)
    return out, n


def paired(a, b, label):
    """Paired t and Wilcoxon on the sessions both cells have."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return None
    d = np.array([a[s] - b[s] for s in shared], float)
    t, p = stats.ttest_1samp(d, 0.0)
    try:
        w, pw = stats.wilcoxon(d)
    except ValueError:
        w, pw = np.nan, np.nan
    return {'label': label, 'n_sessions': len(shared),
            'mean_diff_s': float(d.mean()),
            'sem_s': float(d.std(ddof=1) / np.sqrt(len(d))),
            'median_diff_s': float(np.median(d)),
            't': float(t), 'df': len(d) - 1, 'p': float(p),
            'wilcoxon_p': float(pw)}


def run(bundle=None, out_dir=None, presses_csv=None, min_events=MIN_EVENTS):
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr')
    bundle = bundle or os.path.join(group, 'bundle_v2')
    presses_csv = presses_csv or os.path.join(group, 'press_categories.csv')
    out_dir = out_dir or os.path.join(
        group, f"stillness_by_phase_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_stillness_by_phase')

    print(f"\n  bundle : {bundle}\n  presses: {presses_csv}\n  out    : {out_dir}")
    # No re-pad: stillness is behavioural, and the artifact pad cannot move a
    # button press.
    data = rip.load_bundle(bundle)
    presses = fin.press_table(data, bundle, presses_csv, rebuild=False)
    presses = grid.add_exclusive_stage(data, presses)

    cells, counts = {}, {}
    print(f"\n  ── stillness per cell (session medians, >= {min_events} events) ──")
    for ph, phq in grid.PHASE_Q.items():
        for val in grid.VALENCES:
            k = f'{val}, {ph}'
            cells[k], counts[k] = cell_stillness(presses, phq, val, min_events)
            if cells[k]:
                v = np.array(list(cells[k].values()))
                print(f"    {k:22s} n={len(v):2d} sess, {counts[k]:6d} ev | "
                      f"median {np.median(v):.3f} s  IQR "
                      f"{np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}")

    out = {'q1_valence_within_phase': {}, 'q2_phase_within_valence': {}}

    print(f"\n  ── Q1: correct vs incorrect, WITHIN each phase ──")
    for ph in grid.PHASE_Q:
        got = paired(cells.get(f'correct, {ph}', {}), cells.get(f'error, {ph}', {}),
                     f'correct - error | {ph}')
        if got is None:
            continue
        out['q1_valence_within_phase'][ph] = got
        star = '*' if got['p'] < 0.05 else ' '
        print(f"   {star}{got['label']:34s} n={got['n_sessions']:2d} "
              f"{got['mean_diff_s']:+.3f} ± {got['sem_s']:.3f} s "
              f"t({got['df']})={got['t']:+.2f} p={got['p']:.4g} "
              f"(Wilcoxon p={got['wilcoxon_p']:.4g})")

    print(f"\n  ── Q2: the same feedback, ACROSS phases ──")
    for (bnd, grp), phases in grid.GROUPINGS.items():
        for i, a in enumerate(phases):
            for b in phases[i + 1:]:
                for val in grid.VALENCES:
                    key = f'{val}: {a} - {b}'
                    if key in out['q2_phase_within_valence']:
                        continue
                    got = paired(cells.get(f'{val}, {a}', {}),
                                 cells.get(f'{val}, {b}', {}), key)
                    if got is None:
                        continue
                    out['q2_phase_within_valence'][key] = got
                    star = '*' if got['p'] < 0.05 else ' '
                    print(f"   {star}{key:34s} n={got['n_sessions']:2d} "
                          f"{got['mean_diff_s']:+.3f} ± {got['sem_s']:.3f} s "
                          f"t({got['df']})={got['t']:+.2f} p={got['p']:.4g} "
                          f"(Wilcoxon p={got['wilcoxon_p']:.4g})")

    payload = {'created': datetime.now().isoformat(timespec='seconds'),
               'script': os.path.basename(__file__),
               'question': ('does post-uncover stillness differ by feedback '
                            'within a phase, or by phase within a feedback?'),
               'settings': {'bundle': bundle, 'presses_csv': presses_csv,
                            'min_events': min_events, 'unit': 'session',
                            'statistic': 'median still_next_s per session',
                            'phase_definitions': grid.PHASE_DEF,
                            'phase_queries': grid.PHASE_Q},
               'cells': {k: {'n_sessions': len(v), 'n_events': counts[k],
                             'median_s': (float(np.median(list(v.values())))
                                          if v else None),
                             'per_session': v}
                         for k, v in cells.items()},
               **out}
    with open(os.path.join(out_dir, 'stillness_by_phase.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    figure(os.path.join(out_dir, 'stillness_by_phase.png'), payload)
    print(f"\n  wrote stillness_by_phase.json + .png/.pdf to {out_dir}")
    return out_dir


def figure(out_png, payload):
    """Left: the session medians per cell. Right: the two contrast families.

    Each point on the left is one session, so the paired structure the tests
    use is visible rather than summarised away.
    """
    cells = payload['cells']
    order = [p for p in grid.PHASE_Q]
    # 24 contrast rows need ~0.5 cm each or the labels collide.
    n_rows = (len(payload['q1_valence_within_phase'])
              + len(payload['q2_phase_within_valence']))
    fig, axes = plt.subplots(1, 2, figsize=(21 * CM,
                                            max(9.0, 0.48 * n_rows + 3.5) * CM),
                             gridspec_kw={'width_ratios': [1.15, 1]})

    ax = axes[0]
    for pi, ph in enumerate(order):
        for vi, val in enumerate(grid.VALENCES):
            rec = cells.get(f'{val}, {ph}')
            if not rec or not rec['per_session']:
                continue
            v = np.array(list(rec['per_session'].values()), float)
            x = pi * 2.6 + vi
            rng = np.random.default_rng(42)
            ax.scatter(x + rng.uniform(-0.16, 0.16, v.size), v, s=5,
                       color=VAL_C[val], alpha=0.45, lw=0, zorder=2)
            ax.errorbar(x, np.median(v),
                        yerr=[[np.median(v) - np.percentile(v, 25)],
                              [np.percentile(v, 75) - np.median(v)]],
                        fmt='_', ms=16, capsize=3, lw=1.6,
                        color=VAL_C[val], zorder=3)
            ax.text(x, ax.get_ylim()[0], f"{rec['n_sessions']}", ha='center',
                    va='bottom', fontsize=6, color='0.4')
    ax.set_xticks([pi * 2.6 + 0.5 for pi in range(len(order))])
    ax.set_xticklabels(order, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('stillness after the press (s)\nsession median', fontsize=8.5)
    ax.set_title('post-uncover stillness, per session', fontsize=9, loc='left')
    ax.tick_params(labelsize=7.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    names, vals, sems, ps, cols = [], [], [], [], []
    for ph, rec in payload['q1_valence_within_phase'].items():
        names.append(f'corr − err | {ph}'); vals.append(rec['mean_diff_s'])
        sems.append(rec['sem_s']); ps.append(rec['p']); cols.append('#5C1027')
    for k, rec in payload['q2_phase_within_valence'].items():
        names.append(k); vals.append(rec['mean_diff_s'])
        sems.append(rec['sem_s']); ps.append(rec['p'])
        cols.append(VAL_C['correct' if k.startswith('correct') else 'error'])
    y = np.arange(len(names))[::-1]
    for yi, v, se, pv, c in zip(y, vals, sems, ps, cols):
        ax.errorbar(v, yi, xerr=se, fmt='o', ms=4, capsize=2, lw=1.0, color=c,
                    mfc=c if pv < 0.05 else 'white', mec=c, mew=1.0)
    ax.axvline(0, color='0.3', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel('Δ stillness (s)', fontsize=8.5)
    ax.set_title('contrasts (filled = p < 0.05)', fontsize=9, loc='left')
    ax.tick_params(labelsize=7.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    handles = [plt.Line2D([], [], marker='o', ls='none', ms=5,
                          color=VAL_C[v], label=f'{v} uncovers')
               for v in grid.VALENCES]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=8)
    fig.suptitle('How long the subject sits still after an uncovering, '
                 'by phase and feedback', fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


def figures(out_dir):
    """Redraw from a finished run."""
    with open(os.path.join(out_dir, 'stillness_by_phase.json')) as f:
        payload = json.load(f)
    figure(os.path.join(out_dir, 'stillness_by_phase.png'), payload)
    print(f"  redrew stillness_by_phase.png / .pdf")


if __name__ == '__main__':
    try:
        import fire
        fire.Fire({'run': run, 'figures': figures})
    except ImportError:
        run()
