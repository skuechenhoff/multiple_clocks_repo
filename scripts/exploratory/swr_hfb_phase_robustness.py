#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the medial frontal HFB-ripple locking depend on the phase split, and does
it survive a stillness control?

The companion question to `swr_robustness_grid.py`, on the other branch. Two
of the three factors that grid crosses do not exist here:

  * the BASELINE WINDOW is not a parameter of this measure. The locked value is
    peri-ripple (+-250 ms) minus the non-peri ring (250-750 ms) minus the
    shifted null. There is no -1.6..-1.1 s pre-event window anywhere in it, so
    a -1.5..-1.0 s window is not a setting that could be changed.
  * the PHASE SPLIT and the STILLNESS CONTROL do exist, and are what this asks.

Everything is built the way `swr_ripple_hfb_conditions.py` builds a condition:
per (session, derivation) the real minus the mean of its shifted nulls, then
the session mean, then a t-test across sessions. A contrast is the PAIRED
difference of two such session values on the sessions that have both sides --
so a contrast is tested by the same statistic as a single condition, on the
same unit, and cannot be read as "significant here, not there".

Needs a conditions run whose events carry stillness in EVERY phase and the
two-phase `*_learn` collapse, i.e. 2026-09-27 or later.

    python scripts/swr_hfb_phase_robustness.py run --cond_dir=<ripple_hfb_conditions_...>

@author: Svenja Kuchenhoff

STATUS
    ⚠ EXPLORATORY — NOT IN THE REPORTED RESULT. The HFB companion to
    `swr_robustness_grid.py`. Its one reportable finding is negative: a
    stillness control is not available in the HFB branch (>=2 s still periods
    with >=15 clean ripples exist in only 17/53 sessions during encoding).
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
import scripts.swr_ripple_hfb_conditions as cond

ROI = 'MedialFrontal'
KINDS = ('reward', 'error', 'move', 'still')
# The two splits, named as the ripple-rate grid names them.
SPLITS = {'three_phase': ('explore', 'plan', 'execute'),
          'two_phase': ('learn', 'execute'),
          'pooled': ('all',)}
CM = 1 / 2.54
FS_TITLE, FS_LABEL, FS_TICK = 11, 9, 8
# Feedback colours as `swr_ripple_hfb_conditions` fixes them; stillness is the
# palest grey because it is the weakest control, not a condition of interest.
KIND_C = {'reward': '#0e3d3a', 'error': '#a30d6c',
          'move': '#6E6E6E', 'still': '#B9B9B9'}
KIND_LABEL = {'reward': 'correct uncovers', 'error': 'incorrect uncovers',
              'move': 'navigation presses', 'still': 'stillness'}
PHASE_LABEL = {'explore': 'explore', 'plan': 'plan', 'execute': 'known',
               'learn': 'learning\n(expl+plan)', 'all': 'all'}


def session_values(cond_dir, roi=ROI):
    """{condition: {session: locked value}} -- real minus its shifted null."""
    d = pd.read_csv(os.path.join(cond_dir, 'per_cell.csv'))
    d = d[d.roi == roi]
    real = d[d.is_real].groupby(['session', 'cx_pair', 'cond'])['diff'].mean()
    null = d[~d.is_real].groupby(['session', 'cx_pair', 'cond'])['diff'].mean()
    locked = (real - null).dropna()
    return locked.groupby(['session', 'cond']).mean().unstack('cond')


def one_sample(v, label):
    v = pd.Series(v).dropna()
    if len(v) < 3:
        return None
    t, p = stats.ttest_1samp(v.to_numpy(float), 0.0)
    return {'label': label, 'n_sessions': int(len(v)), 'effect': float(v.mean()),
            'sem': float(v.std(ddof=1) / np.sqrt(len(v))),
            't': float(t), 'df': int(len(v) - 1), 'p': float(p)}


def paired(sess, a, b):
    if a not in sess or b not in sess:
        return None
    return one_sample(sess[a] - sess[b], f'{a} - {b}')


def run(cond_dir=None, out_dir=None, roi=ROI):
    group = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                         'group', 'swr')
    cond_dir = cond_dir or os.path.join(
        group, f"ripple_hfb_conditions_{datetime.now():%Y-%m-%d}")
    out_dir = out_dir or cond_dir
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_hfb_phase_robustness')

    sess = session_values(cond_dir, roi)
    print(f"\n  cond_dir: {cond_dir}\n  roi: {roi}\n"
          f"  {sess.shape[0]} sessions x {sess.shape[1]} conditions")

    out = {'per_condition': {}, 'vs_stillness': {}, 'across_phase': {},
           'within_phase': {}}

    # `execute` belongs to both splits, so iterate the phases, not the splits.
    all_phases = list(dict.fromkeys(p for v in SPLITS.values() for p in v))
    print(f"\n  ── each condition against its shifted null ──")
    if True:
        for ph in all_phases:
            for k in KINDS:
                c = f'{k}_{ph}'
                got = one_sample(sess[c], c) if c in sess else None
                if got is None:
                    continue
                out['per_condition'][c] = got
                star = '*' if got['p'] < 0.05 else ' '
                print(f"   {star}{c:16s} n={got['n_sessions']:2d} "
                      f"{got['effect']:+.4f} ± {got['sem']:.4f} "
                      f"t({got['df']})={got['t']:+.2f} p={got['p']:.4g}")

    print(f"\n  ── against STILLNESS in the SAME phase (the (c) question) ──")
    if True:
        for ph in all_phases:
            for k in ('reward', 'error', 'move'):
                got = paired(sess, f'{k}_{ph}', f'still_{ph}')
                if got is None:
                    continue
                out['vs_stillness'][got['label']] = got
                star = '*' if got['p'] < 0.05 else ' '
                print(f"   {star}{got['label']:34s} n={got['n_sessions']:2d} "
                      f"{got['effect']:+.4f} ± {got['sem']:.4f} "
                      f"t({got['df']})={got['t']:+.2f} p={got['p']:.4g}")

    print(f"\n  ── learning vs execution, within event type (the (a) question) ──")
    for k in KINDS:
        got = paired(sess, f'{k}_learn', f'{k}_execute')
        if got is None:
            continue
        out['across_phase'][got['label']] = got
        star = '*' if got['p'] < 0.05 else ' '
        print(f"   {star}{got['label']:34s} n={got['n_sessions']:2d} "
              f"{got['effect']:+.4f} t({got['df']})={got['t']:+.2f} p={got['p']:.4g}")

    print(f"\n  ── event type within phase ──")
    for ph in ('explore', 'plan', 'execute', 'learn', 'all'):
        for a, b in (('reward', 'move'), ('reward', 'error'), ('error', 'move')):
            got = paired(sess, f'{a}_{ph}', f'{b}_{ph}')
            if got is None:
                continue
            out['within_phase'][got['label']] = got
            star = '*' if got['p'] < 0.05 else ' '
            print(f"   {star}{got['label']:34s} n={got['n_sessions']:2d} "
                  f"{got['effect']:+.4f} t({got['df']})={got['t']:+.2f} "
                  f"p={got['p']:.4g}")

    payload = {'created': datetime.now().isoformat(timespec='seconds'),
               'script': os.path.basename(__file__),
               'question': ('does the medial frontal HFB-ripple locking depend '
                            'on the phase split, and does it survive a '
                            'stillness control?'),
               'settings': {'cond_dir': cond_dir, 'roi': roi, 'unit': 'session',
                            'peri_s': cond.PERI_S, 'nonperi_s': list(cond.NONPERI_S),
                            'min_ripples': cond.MIN_RIPPLES, 'post_s': cond.POST_S,
                            'baseline_window': ('not applicable -- this measure '
                                                'has no pre-event baseline'),
                            'splits': {k: list(v) for k, v in SPLITS.items()}},
               **out}
    with open(os.path.join(out_dir, 'hfb_phase_robustness.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    figure(os.path.join(out_dir, 'hfb_phase_robustness.png'), payload)
    print(f"\n  wrote hfb_phase_robustness.json + .png/.pdf to {out_dir}")
    return out_dir


def figure(out_png, payload):
    """Left: every condition against its null, grouped by phase. Right: the
    same conditions against STILLNESS in the same phase.

    Plotted straight from the json -- the point is the session-level mean the
    t-test used and the bar its SEM.
    """
    pc, vs = payload['per_condition'], payload['vs_stillness']
    phases = ['explore', 'plan', 'execute', 'learn', 'all']
    fig, axes = plt.subplots(1, 2, figsize=(18 * CM, 8.5 * CM),
                             gridspec_kw={'width_ratios': [1.25, 1]})

    for ax, (src, kinds, title, ylab) in zip(axes, [
            (pc, KINDS, 'each event type against its shifted null',
             'peri − non-peri HFB (z)'),
            (vs, ('reward', 'error', 'move'),
             'the same, against STILLNESS in that phase',
             'difference vs stillness (z)')]):
        xt, xl = [], []
        for pi, ph in enumerate(phases):
            base = pi * (len(kinds) + 1.2)
            for ki, k in enumerate(kinds):
                key = f'{k}_{ph}' if src is pc else f'{k}_{ph} - still_{ph}'
                rec = src.get(key)
                x = base + ki
                if rec is None:
                    continue
                sig = rec['p'] < 0.05
                ax.bar(x, rec['effect'], 0.8, color=KIND_C[k],
                       alpha=1.0 if sig else 0.45,
                       edgecolor=KIND_C[k], lw=0.8)
                ax.errorbar(x, rec['effect'], yerr=rec['sem'], fmt='none',
                            ecolor='0.25', capsize=2, lw=0.9)
                ax.text(x, rec['effect'] + np.sign(rec['effect'] or 1) * rec['sem'],
                        f"{rec['n_sessions']}", ha='center',
                        va='bottom' if rec['effect'] >= 0 else 'top',
                        fontsize=FS_TICK - 2.5, color='0.35')
            xt.append(base + (len(kinds) - 1) / 2)
            xl.append(PHASE_LABEL[ph])
        ax.axhline(0, color='0.3', lw=0.8)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=FS_TICK)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_LABEL, loc='left', pad=4)
        ax.tick_params(labelsize=FS_TICK)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=KIND_C[k], label=KIND_LABEL[k])
               for k in KINDS]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='white', ec='0.45', lw=0.8,
                                 alpha=0.45, label='pale = n.s. (p ≥ 0.05)'))
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               fontsize=FS_TICK)
    fig.suptitle(f"Medial frontal HFB-ripple locking: phase split and "
                 f"stillness control\n"
                 f"numbers above each bar = sessions contributing",
                 fontsize=FS_TITLE, y=0.99)
    fig.tight_layout(rect=(0, 0.11, 1, 0.88))
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    try:
        import fire
        fire.Fire({'run': run})
    except ImportError:
        run()
