#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the ripple-LOCKING bigger here than there? The contrasts, not two tests.

Both HFB branches currently support their "specific to X" claims with separate
tests: `swr_ripple_locked_hfb.py` clusters each ROI against its own shifted
null, `swr_ripple_hfb_conditions.py` clusters each condition against its own.
A cluster in one and none in the other is NOT evidence that the two differ --
that is the interaction fallacy, and it is exactly what a reviewer asks about.

This computes the DIFFERENCE, with the same machinery:

  * the per-unit locked curve is built exactly as the two figure functions
    build it -- real minus shifted, flank-normalised, 100 ms Gaussian smooth;
  * the contrast is the paired difference of two such curves on the units that
    have both sides;
  * the cluster test is `swr_sakon.cluster_perm_time`, the same function, the
    same sign-flip null, the same n_perm and seed.

Nothing is re-extracted: both `timecourses.npz` files already hold the traces,
so this reads what the published runs wrote.

    python scripts/swr_hfb_locking_contrasts.py \
        --region_dir=<ripple_locked_hfb_...> --cond_dir=<ripple_hfb_conditions_...>

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_sakon as sk

SEED, N_PERM, SMOOTH_MS = 42, 2000, 100.0
NONPERI_S = (0.25, 0.75)
PERI_MS = 250.0
MEDIAL = 'MedialFrontal'

# THE PHASE SPLIT, identical to the ripple-rate branch:
#   explore = the rewards are not yet all known  (first traversal + learning)
#   known   = the grid is known and being executed
# In this branch the stage labels are 'explore' / 'plan' / 'execute', which map
# onto 'first uncovers' / 'while learning' / 'once known'. So `plan` belongs on
# the EXPLORE side, exactly as `while learning` does there.
PHASES = {
    'explore': {'reward': ('reward_explore', 'reward_plan'),
                'error': ('error_explore', 'error_plan'),
                'move': ('move_explore', 'move_plan')},
    'known': {'reward': ('reward_execute',),
              'error': ('error_execute',),
              'move': ('move_execute',)},
}
COND_KEYS = ('reward', 'error', 'move')
COND_PAIRS = [('reward', 'error'), ('reward', 'move'), ('error', 'move')]
# Visual is not reported: its whole effect came from same-shaft contacts.
REGION_PAIRS = [(MEDIAL, 'TemporalLateral'), (MEDIAL, 'Auditory')]

COL = {MEDIAL: '#448363', 'TemporalLateral': '#8C8C8C', 'Auditory': '#4F4F4F',
       'reward': '#0e3d3a', 'error': '#a30d6c', 'move': '#6E6E6E'}
SHORT = {MEDIAL: 'med. frontal', 'TemporalLateral': 'lat. temp.',
         'Auditory': 'auditory', 'reward': 'reward', 'error': 'incorrect',
         'move': 'navigation'}
LABEL = {MEDIAL: 'medial frontal', 'TemporalLateral': 'lat. temporal',
         'Auditory': 'auditory', 'reward': 'correct uncovers',
         'error': 'incorrect uncovers', 'move': 'navigation presses'}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running = np.empty(p.size), 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def load(results_dir):
    z = np.load(os.path.join(results_dir, 'timecourses.npz'))
    t_ms, tr = z['t_ms'], z['traces']
    ix = pd.read_csv(os.path.join(results_dir, 'timecourse_index.csv'))
    ix['is_real'] = ix.is_real.astype(bool)
    if 'same_shaft' in ix.columns:
        ix['same_shaft'] = ix.same_shaft.astype(bool)
    if 'trace_row' not in ix.columns:
        ix['trace_row'] = np.arange(len(ix))
    return t_ms, tr, ix


def unit_curves(t_ms, tr, ix, sel, unit='session', smooth_ms=SMOOTH_MS):
    """{unit: locked curve}. Identical construction to the figure functions."""
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & (np.abs(t_ms) < NONPERI_S[1] * 1000)
    out = {}
    for u, gi in ix[sel].groupby(unit):
        rr = gi.trace_row.to_numpy()
        r = tr[rr[gi.is_real.to_numpy()]]
        n = tr[rr[~gi.is_real.to_numpy()]]
        if not len(r) or not len(n):
            continue
        c = r.mean(0) - n.mean(0)
        out[int(u)] = c - c[flank].mean()
    if smooth_ms and out:
        keys = list(out)
        A = gaussian_filter1d(np.stack([out[k] for k in keys]),
                              (smooth_ms / 1000.0 * fs) / 2.355, axis=-1)
        out = {k: A[i] for i, k in enumerate(keys)}
    return out


def pooled_cond_curves(t_ms, tr, ix, counts, conds, roi=MEDIAL,
                       unit='session', smooth_ms=SMOOTH_MS):
    """One curve per unit, pooling several conditions by their ripple counts.

    Pooling by count is what a re-run would compute: the saved trace of a
    condition is already the mean over its ripples, so a count-weighted mean of
    two conditions equals the mean over their union.
    """
    fs = 1000.0 / (t_ms[1] - t_ms[0])
    flank = (np.abs(t_ms) >= NONPERI_S[0] * 1000) & (np.abs(t_ms) < NONPERI_S[1] * 1000)
    w = {(int(r.session), r.cond): float(r.n_ripples)
         for r in counts.itertuples()}
    out = {}
    for u, gu in ix[(ix.roi == roi) & (ix.cond.isin(conds))].groupby(unit):
        num_r, num_n, den = 0.0, 0.0, 0.0
        for cond, gc in gu.groupby('cond'):
            rr = gc.trace_row.to_numpy()
            real = tr[rr[gc.is_real.to_numpy()]]
            null = tr[rr[~gc.is_real.to_numpy()]]
            if not len(real) or not len(null):
                continue
            wt = w.get((int(u), cond), float(len(real)))
            num_r += wt * real.mean(0)
            num_n += wt * null.mean(0)
            den += wt
        if den <= 0:
            continue
        c = num_r / den - num_n / den
        out[int(u)] = c - c[flank].mean()
    if smooth_ms and out:
        keys = list(out)
        A = gaussian_filter1d(np.stack([out[k] for k in keys]),
                              (smooth_ms / 1000.0 * fs) / 2.355, axis=-1)
        out = {k: A[i] for i, k in enumerate(keys)}
    return out


def ripple_duration(bundle_dir):
    """Mean ripple duration and its SE across sessions, from the bundle csv."""
    r = pd.read_csv(os.path.join(bundle_dir, 'ripples.csv'),
                    usecols=['session', 'duration_s'])
    per = r.groupby('session').duration_s.mean()
    return {'mean_ms': float(1000 * per.mean()),
            'sem_ms': float(1000 * per.std(ddof=1) / np.sqrt(len(per))),
            'n_sessions': int(len(per)),
            'n_ripples': int(len(r))}


def _cluster(A, t_ms):
    _, cl, pv, _ = sk.cluster_perm_time(A, n_perm=N_PERM, seed=SEED)
    return [{'start_ms': float(t_ms[a]), 'stop_ms': float(t_ms[b - 1]),
             'p': float(pp)} for (a, b), pp in zip(cl, pv) if pp < 0.05]


def describe(A, t_ms, label):
    """One side on its own: the test each branch already reports."""
    peri = np.abs(t_ms) < PERI_MS
    v = A[:, peri].mean(1)
    t, p = stats.ttest_1samp(v, 0.0)
    return {'label': label, 'n_units': int(len(A)),
            'peri_mean': float(v.mean()),
            'peri_sem': float(v.std(ddof=1) / np.sqrt(len(v))),
            't': float(t), 'p': float(p),
            'peak_z': float(A.mean(0).max()),
            'peak_at_ms': float(t_ms[int(np.argmax(A.mean(0)))]),
            'clusters': _cluster(A, t_ms)}


def contrast(ca, cb, t_ms, name):
    """The paired difference of two locked curves, same cluster test."""
    shared = sorted(set(ca) & set(cb))
    if len(shared) < 5:
        return None
    D = np.stack([ca[u] - cb[u] for u in shared])
    peri = np.abs(t_ms) < PERI_MS
    v = D[:, peri].mean(1)
    t, p = stats.ttest_1samp(v, 0.0)
    return {'name': name, 'n_units': int(len(shared)),
            'peri_mean': float(v.mean()),
            'peri_sem': float(v.std(ddof=1) / np.sqrt(len(v))),
            't': float(t), 'df': int(len(v) - 1), 'p': float(p),
            'cohens_d': float(v.mean() / v.std(ddof=1)),
            'clusters': _cluster(D, t_ms),
            'mean_curve': D.mean(0).tolist(),
            'sem_curve': (D.std(0, ddof=1) / np.sqrt(len(D))).tolist()}


def family(t_ms, tr, ix, keys, pairs, sel_fn, unit='season'):
    curves = {k: unit_curves(t_ms, tr, ix, sel_fn(k), unit=unit) for k in keys}
    curves = {k: v for k, v in curves.items() if len(v) >= 5}
    sides = {k: describe(np.stack([v[u] for u in sorted(v)]), t_ms, LABEL.get(k, k))
             for k, v in curves.items()}
    cons = {}
    for a, b in pairs:
        if a not in curves or b not in curves:
            continue
        got = contrast(curves[a], curves[b], t_ms, f'{a} - {b}')
        if got:
            cons[f'{a} - {b}'] = got
    if cons:
        names = list(cons)
        adj = holm([cons[n]['p'] for n in names])
        for n, a_ in zip(names, adj):
            cons[n]['p_holm'] = float(a_)
            cons[n]['family_size'] = len(names)
    return curves, sides, cons


def run(region_dir=None, cond_dir=None, bundle=None, out_dir=None,
        unit='session'):
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr')
    region_dir = region_dir or os.path.join(group, 'ripple_locked_hfb_2026-09-18')
    cond_dir = cond_dir or os.path.join(group, 'ripple_hfb_conditions_2026-09-21')
    bundle = bundle or os.path.join(group, 'bundle_v2')
    out_dir = out_dir or os.path.join(
        group, f"hfb_locking_contrasts_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_hfb_locking_contrasts')
    print(f"\n  regions from : {region_dir}\n  conditions   : {cond_dir}")
    print(f"  unit={unit}, peri +-{PERI_MS:.0f} ms, flank-normalised, "
          f"{SMOOTH_MS:.0f} ms smooth, {N_PERM} sign-flips, seed {SEED}")
    print(f"  phases: explore = {PHASES['explore']['reward']}, "
          f"known = {PHASES['known']['reward']}")

    dur = ripple_duration(bundle)
    print(f"  mean ripple duration {dur['mean_ms']:.1f} +- {dur['sem_ms']:.1f} ms "
          f"({dur['n_ripples']} ripples, {dur['n_sessions']} sessions)")

    payload = {'created': datetime.now().isoformat(timespec='seconds'),
               'script': os.path.basename(__file__),
               'settings': {'region_dir': region_dir, 'cond_dir': cond_dir,
                            'bundle': bundle, 'unit': unit,
                            'peri_ms': PERI_MS, 'nonperi_s': list(NONPERI_S),
                            'smooth_ms': SMOOTH_MS, 'n_perm': N_PERM,
                            'seed': SEED, 'phases': PHASES,
                            'multiple_comparisons': 'Holm within each family'},
               'ripple_duration': dur, 'families': {}}

    # -- regions -------------------------------------------------------
    t_ms, tr, ix = load(region_dir)
    keys = [MEDIAL, 'TemporalLateral', 'Auditory']
    cur = {k: unit_curves(t_ms, tr, ix, (ix.roi == k) & (~ix.same_shaft),
                          unit=unit) for k in keys}
    payload['families']['regions'] = _block(cur, t_ms, REGION_PAIRS)
    _report('REGIONS (different-shaft)', payload['families']['regions'])

    # -- conditions, one block per phase --------------------------------
    t2, tr2, ix2 = load(cond_dir)
    counts = pd.read_csv(os.path.join(cond_dir, 'ripple_counts.csv'))
    for phase, groups in PHASES.items():
        cur2 = {k: pooled_cond_curves(t2, tr2, ix2, counts, groups[k],
                                      unit=unit) for k in COND_KEYS}
        payload['families'][f'conditions_{phase}'] = _block(cur2, t2, COND_PAIRS)
        _report(f'CONDITIONS, {phase} (medial frontal)',
                payload['families'][f'conditions_{phase}'])

    with open(os.path.join(out_dir, 'locking_contrasts.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    figure(os.path.join(out_dir, 'locking_contrasts.png'), payload)
    print(f"\n  saved -> {out_dir}")


def _block(curves, t_ms, pairs):
    """Per-side stats, the contrasts, Holm, and the curves for drawing."""
    curves = {k: v for k, v in curves.items() if len(v) >= 5}
    sides = {k: describe(np.stack([v[u] for u in sorted(v)]), t_ms,
                         LABEL.get(k, k)) for k, v in curves.items()}
    cons = {}
    for a, b in pairs:
        if a not in curves or b not in curves:
            continue
        got = contrast(curves[a], curves[b], t_ms, f'{a} - {b}')
        if got:
            cons[f'{a} - {b}'] = got
    if cons:
        names = list(cons)
        adj = holm([cons[n]['p'] for n in names])
        for n, a_ in zip(names, adj):
            cons[n]['p_holm'] = float(a_)
            cons[n]['family_size'] = len(names)
    return {'sides': sides, 'contrasts': cons, 't_ms': np.asarray(t_ms).tolist(),
            'curves': {k: np.stack([v[u] for u in sorted(v)]).mean(0).tolist()
                       for k, v in curves.items()},
            'sem': {k: (np.stack([v[u] for u in sorted(v)]).std(0, ddof=1)
                        / np.sqrt(len(v))).tolist() for k, v in curves.items()}}


def _report(title, blk):
    sides, cons = blk['sides'], blk['contrasts']
    print(f"\n  ── {title} ──")
    print(f"    each side against its own shifted null:")
    for k, s in sides.items():
        cl = ('; '.join(f"{c['start_ms']:+.0f}..{c['stop_ms']:+.0f} ms p={c['p']:.4f}"
                        for c in s['clusters']) or 'no cluster')
        print(f"      {s['label']:24s} n={s['n_units']:3d} peri={s['peri_mean']:+.5f} "
              f"± {s['peri_sem']:.5f} t={s['t']:+.2f} p={s['p']:.4f} | {cl}")
    print(f"    the DIFFERENCE (paired, same cluster test):")
    for n, c in cons.items():
        cl = ('; '.join(f"{x['start_ms']:+.0f}..{x['stop_ms']:+.0f} ms p={x['p']:.4f}"
                        for x in c['clusters']) or 'no cluster')
        print(f"      {n:40s} n={c['n_units']:3d} {c['peri_mean']:+.5f} "
              f"t({c['df']})={c['t']:+.2f} p={c['p']:.4f} (Holm {c['p_holm']:.4f}) "
              f"d={c['cohens_d']:+.2f}")
        print(f"      {'':40s} {cl}")


# ── Figure ────────────────────────────────────────────────────────────
PANEL_CM, CM = 4.0, 1 / 2.54
FS_T, FS_L, FS_K, LW = 10, 9, 8.5, 2.6


def _duration_bar(ax, dur, y, colour='0.15'):
    """Mean ripple extent, drawn as peak +- duration/2, with its SE."""
    if not dur:
        return
    h = dur['mean_ms'] / 2.0
    se = dur['sem_ms'] / 2.0
    ax.plot([-h, h], [y, y], color=colour, lw=3.0, solid_capstyle='butt',
            zorder=5)
    ax.plot([-h - se, -h + se], [y, y], color=colour, lw=1.0, zorder=5)
    ax.plot([h - se, h + se], [y, y], color=colour, lw=1.0, zorder=5)
    ax.text(h + 30, y, 'ripple', ha='left', va='center',
            fontsize=FS_K - 2.5, color=colour)


def _sig_bars(ax, items, y0, dy, colour_of, clusters_of):
    """One horizontal bar per surviving cluster, stacked under the trace."""
    for i, k in enumerate(items):
        for c in clusters_of(k):
            ax.plot([c['start_ms'], c['stop_ms']], [y0 - i * dy] * 2,
                    color=colour_of(k), lw=3.0, solid_capstyle='butt', zorder=5)


def _curves_panel(ax, blk, keys, title, dur=None):
    t = np.asarray(blk['t_ms'])
    keys = [k for k in keys if k in blk['curves']]
    for k in keys:
        m = np.asarray(blk['curves'][k]); se = np.asarray(blk['sem'][k])
        ax.fill_between(t, m - se, m + se, color=COL[k], alpha=0.20, lw=0)
        ax.plot(t, m, color=COL[k], lw=LW, label=LABEL.get(k, k))
    ax.axhline(0, color='0.6', lw=0.9); ax.axvline(0, color='0.4', lw=0.9, ls=':')
    ax.set_xlim(-750, 750); ax.set_xticks([-500, 0, 500])
    ax.set_xlabel('ms from ripple', fontsize=FS_L)
    ax.set_ylabel('HFB (z)', fontsize=FS_L)
    ax.set_title(title, fontsize=FS_T)
    # everything annotative lives under the traces: a top-centre bar collides
    # with the legend as soon as a label is longer than a few characters.
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - 0.46 * span, hi + 0.42 * span)
    _duration_bar(ax, dur, lo - 0.07 * span)
    _sig_bars(ax, keys, lo - 0.17 * span, 0.09 * span,
              lambda k: COL[k], lambda k: blk['sides'][k]['clusters'])
    ax.legend(fontsize=FS_K - 2, frameon=False, loc='upper left',
              handlelength=1.3, borderpad=0.1, labelspacing=0.25)


def _contrast_panel(ax, blk, title):
    t = np.asarray(blk['t_ms'])
    names = list(blk['contrasts'])
    for i, n in enumerate(names):
        c = blk['contrasts'][n]
        m = np.asarray(c['mean_curve']); se = np.asarray(c['sem_curve'])
        a, b = n.split(' - ')
        col = COL[b]
        ax.fill_between(t, m - se, m + se, color=col, alpha=0.18, lw=0)
        ax.plot(t, m, color=col, lw=LW, ls=('-' if i < 2 else ':'),
                label=f"{SHORT.get(a, a)} − {SHORT.get(b, b)}")
    ax.axhline(0, color='0.6', lw=0.9); ax.axvline(0, color='0.4', lw=0.9, ls=':')
    ax.set_xlim(-750, 750); ax.set_xticks([-500, 0, 500])
    ax.set_xlabel('ms from ripple', fontsize=FS_L)
    ax.set_ylabel('difference (z)', fontsize=FS_L)
    ax.set_title(title, fontsize=FS_T)
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - 0.46 * span, hi + 0.30 * span)
    _sig_bars(ax, names, lo - 0.12 * span, 0.09 * span,
              lambda n: COL[n.split(' - ')[1]],
              lambda n: blk['contrasts'][n]['clusters'])
    ax.legend(fontsize=FS_K - 2, frameon=False, loc='upper left',
              handlelength=1.3, borderpad=0.1, labelspacing=0.25)


def figure(out_png, payload):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': FS_L,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 1.0, 'pdf.fonttype': 42})
    fams = [('regions', [MEDIAL, 'TemporalLateral', 'Auditory'],
             'ripple-locked HFB\nby region',
             'medial frontal minus\neach control region'),
            ('conditions_explore', list(COND_KEYS),
             'medial frontal HFB\nEXPLORE phase',
             'correct uncovers minus\neach control (explore)'),
            ('conditions_known', list(COND_KEYS),
             'medial frontal HFB\nKNOWN phase',
             'correct uncovers minus\neach control (known)')]
    fams = [f for f in fams if f[0] in payload['families']]
    fig, axes = plt.subplots(len(fams), 2,
                             figsize=(2 * PANEL_CM * CM + 3.2 * CM,
                                      len(fams) * PANEL_CM * CM + 2.8 * CM),
                             squeeze=False)
    dur = payload.get('ripple_duration')
    for r, (key, keys, t_left, t_right) in enumerate(fams):
        blk = payload['families'][key]
        _curves_panel(axes[r][0], blk, keys, t_left, dur=dur)
        _contrast_panel(axes[r][1], blk, t_right)
    for ax in axes.ravel():
        ax.tick_params(labelsize=FS_K, width=1.0, length=2.5)
    fig.suptitle('Ripple-locked HFB: each side, and the contrast',
                 fontsize=FS_T, y=1.02)
    fig.tight_layout(w_pad=1.8, h_pad=2.2)
    fig.text(0.5, -0.005,
             'Left: each curve against its own shifted null. Right: the PAIRED '
             'DIFFERENCE, same cluster test. Horizontal bars under the traces '
             'mark surviving clusters, in the trace colour.\n'
             'The short dark bar marks the mean ripple extent (peak +- '
             'duration/2) with its SE. Curves are real minus shifted, '
             'flank-normalised, 100 ms smooth; session as unit; Holm within '
             'each family.\nEXPLORE = first traversal + learning; '
             'KNOWN = route known and executed.',
             ha='center', va='top', fontsize=FS_K - 2, color='0.35')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    try:
        import fire
        fire.Fire({'run': run})
    except ImportError:
        run()
