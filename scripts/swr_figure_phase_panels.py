#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4f and 4g at the AGREED phase definition.

THE DEFINITION (fixed 2026-09-27, and the only one this file draws)
    fs = a grid's first fully correct (error-free) repeat.

      learning   rep <  fs      every repeat BEFORE the first correct one
      execution  rep >= fs      the first correct repeat and everything after

    This is `swr_windows.add_phase3` collapsed to two, and `learn_exc` /
    `known_exc` in `mc.analyse.swr_phases`. It is NOT `stage_of_repeat`,
    which puts the first correct repeat on the learning side.

TWO REFERENCES, BOTH DRAWN
    own baseline   each event against its own -1.5..-1.0 s pre-window
                   (Sakon Eq. 2). The reported reading.
    stillness      the event minus its stillness-matched arrow press. Drawn
                   beside it so a reader can see what the control costs;
                   where it does not survive, that is visible rather than
                   asserted.

Everything goes through the same `fin.window_stats` / `fin.cluster_test` /
`rip.contrast_profiles` as the rest of the pipeline.

    python scripts/swr_figure_phase_panels.py run

@author: Svenja Kuchenhoff

STATUS
    ★ KEY — REPORTED RESULT. Produces Figure 4f and 4g and every ripple-rate
    number in the results section. This is the ONLY script that should be
    quoted for ripple rate by phase; `swr_stage_interaction.py` and the
    `between` block of `swr_final_ripple_analysis.py` are superseded by it
    because they predate the agreed phase definition.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import scripts.swr_final_ripple_analysis as fin
import mc.analyse.swr_phases as grid

SEED = fin.SEED
WIN = fin.PRIMARY_WINDOW                 # (0.0, 0.5)
BASE = grid.BASE_B                       # (-1.5, -1.0), the agreed control
PHASES = {'learning': 'learn_exc', 'execution': 'known_exc'}
X_LABEL = {'learning': 'learn', 'execution': 'known'}   # as in Fig 4f/4g
CELLS = [(v, p) for p in PHASES for v in ('correct', 'error')]

# CLAUDE.md fixes these: correct/incorrect take the feedback greens and pinks,
# and the phase is carried by SHADE within a valence (dark = learning).
C = {('correct', 'learning'): '#0e3d3a', ('correct', 'execution'): '#9ec6b8',
     ('error', 'learning'): '#a30d6c', ('error', 'execution'): '#e3a7c8'}
# DISPLAY ONLY. Rates are binned at `rip.BIN_S` = 100 ms, which makes the raw
# trace look noisier than the effect is. The drawn line is Gaussian-smoothed
# with sigma = 1 bin (FWHM ~235 ms, narrower than the 300 ms sliding window the
# cluster test uses, so the peak is not widened away). EVERY statistic in this
# file -- window means, t tests, cluster tests -- is computed on the UNSMOOTHED
# profiles; smoothing touches nothing but the line.
SMOOTH_SIGMA_BINS = 1.0
CM = 1 / 2.54
# Panel f is ~8 x 4 cm on the page, panel g ~4 x 4 cm; at that size Arial 9-11
# is the house range and 8 is the floor for tick labels.
FS_T, FS_L, FS_K = 10, 9, 8


def build(data, presses, unit, min_events, caliper_s, rel_caliper):
    """Raw per-session rate profiles for the four cells, plus their controls."""
    unc = presses[presses.kind == 'uncover']
    out, centres = {}, None
    for val, ph in CELLS:
        q = f"valence == '{val}' and ({grid.PHASE_Q[PHASES[ph]]})"
        tgt, ctl, _, bal = fin.matched_sets(presses, q, caliper_s,
                                            rel_caliper, SEED)
        per_all = {int(s): g.t_s.to_numpy(float)
                   for s, g in unc.query(q).groupby('session')}
        c_t, raw_t, _ = fin.rates(data, tgt, unit, min_events)
        c_c, raw_c, _ = fin.rates(data, ctl, unit, min_events)
        c_a, raw_a, cnt_a = fin.rates(data, per_all, unit, min_events)
        centres = c_a
        out[(val, ph)] = {'all': raw_a, 'target': raw_t, 'control': raw_c,
                          'n_events': int(cnt_a['n_events_used']),
                          'n_matched': int(bal.n_matched.sum()),
                          'n_unmatched': int(bal.n_unmatched.sum()),
                          'still_s': float(bal.still_target_s.median())}
        print(f"    {val:8s} {ph:10s} {cnt_a['n_events_used']:6d} ev / "
              f"{len(raw_a):2d} sess | matched {bal.n_matched.sum():6d}"
              f"/{bal.n_matched.sum() + bal.n_unmatched.sum():6d} | still "
              f"{out[(val, ph)]['still_s']:.2f} s")
    return centres, out


def readings(cells, centres):
    """{reading: {cell: per-session profile}} -- the two we report."""
    own = {k: grid.base(v['all'], centres, BASE) for k, v in cells.items()}
    mat = {k: fin.difference(v['target'], v['control']) for k, v in cells.items()}
    return {'own baseline -1.5..-1.0': own, 'stillness-matched': mat}


def stats_for(prof, centres, n_perm):
    """Every cell and every contrast of the 2 x 2, one reading."""
    out = {}
    key = lambda v, p: f'{v}, {p}'
    for (v, p) in CELLS:
        d = prof.get((v, p))
        if d is None:
            continue
        s = fin.window_stats(d, centres, WIN)
        _, cl = fin.cluster_test(d, centres, key(v, p), n_perm)
        out[key(v, p)] = dict(s, clusters=cl)
    flat = {f'{v}, {p}': prof[(v, p)] for (v, p) in CELLS if (v, p) in prof}
    def con(name, w):
        if any(k not in flat for k in w):
            return
        d = rip.contrast_profiles(flat, w)
        if len(d) < 3:
            return
        s = fin.window_stats(d, centres, WIN)
        _, cl = fin.cluster_test(d, centres, name, n_perm)
        out[name] = dict(s, clusters=cl)
    for p in PHASES:
        con(f'valence | {p}', {f'correct, {p}': 1.0, f'error, {p}': -1.0})
    for v in ('correct', 'error'):
        con(f'phase | {v}', {f'{v}, learning': 1.0, f'{v}, execution': -1.0})
    con('INTERACTION', {'correct, learning': 1.0, 'error, learning': -1.0,
                        'correct, execution': -1.0, 'error, execution': 1.0})
    fam = [n for n in out if n.startswith(('valence |', 'phase |', 'INTER'))]
    if fam:
        adj = grid.holm([out[n]['p_perm'] for n in fam])
        for n, a in zip(fam, adj):
            out[n]['p_perm_holm'] = float(a)
    return out


def _trace(profiles, smooth=True):
    """Mean +- s.e.m. across sessions, smoothed for display only."""
    X = np.vstack([profiles[u] for u in sorted(profiles)])
    m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(X.shape[0])
    if smooth and SMOOTH_SIGMA_BINS:
        m = gaussian_filter1d(m, SMOOTH_SIGMA_BINS)
        se = gaussian_filter1d(se, SMOOTH_SIGMA_BINS)
    return m, se, X.shape[0]


def _mse(prof, cells_key, centres):
    X = np.vstack([prof[cells_key][u] for u in sorted(prof[cells_key])])
    return X.mean(0), X.std(0, ddof=1) / np.sqrt(X.shape[0]), X.shape[0]


YLAB = {'own baseline -1.5..-1.0': 'change vs own\nbaseline (Hz)',
        'stillness-matched': 'uncover −\nmatched control (Hz)'}


def panel_f(out_png, cells, centres, prof, st, reading):
    """Left: the raw rate around the event, learning vs execution, correct
    uncovers only. Right: the same two cells as the tested quantity."""
    wide = reading == 'stillness-matched'
    fig, axes = plt.subplots(1, 2,
                             figsize=((11.0 if wide else 9.5) * CM,
                                      (5.4 if wide else 4.2) * CM),
                             gridspec_kw={'width_ratios': [1.7, 1]})
    ax = axes[0]
    ax.axvspan(BASE[0], BASE[1], color='0.90', lw=0, zorder=0)
    ax.axvspan(WIN[0], WIN[1], color='0.90', lw=0, zorder=0)
    # WHICH TRACES. The two readings compare different things, so the left panel
    # must draw different things or it implies a comparison that was not made:
    #   own baseline  the uncover in each phase, against the shaded pre-window
    #   matched       the uncover AND the arrow press it was matched to, which
    #                 IS the reference in that reading
    if reading == 'stillness-matched':
        draw = [(('correct', ph), side, ls)
                for ph in PHASES
                for side, ls in (('target', '-'), ('control', (0, (1.4, 1.2))))]
    else:
        draw = [(('correct', ph), 'all', '-') for ph in PHASES]
    for key, side, ls in draw:
        m, se, _ = _trace(cells[key][side])
        ph = key[1]
        lab = (f'correct, {ph}' if side in ('all', 'target')
               else f'matched control, {ph}')
        ax.plot(centres, m, color=C[key], lw=1.3, ls=ls, label=lab, zorder=3)
        ax.fill_between(centres, m - se, m + se, color=C[key],
                        alpha=0.14 if side == 'control' else 0.22, lw=0,
                        zorder=2)
    for cl in st.get('correct, learning', {}).get('clusters', []):
        ax.plot([cl['start_s'], cl['stop_s']],
                [ax.get_ylim()[0]] * 2, lw=2.6,
                color=C[('correct', 'learning')], solid_capstyle='butt')
    ax.axvline(0, color='0.35', lw=0.8)
    ax.set_xlabel('time from uncover (s)', fontsize=FS_L)
    ax.set_ylabel('ripple rate (Hz)', fontsize=FS_L)
    ax.tick_params(labelsize=FS_K)
    if wide:
        ax.legend(fontsize=FS_K - 2, frameon=False, ncol=2, handlelength=1.4,
                  columnspacing=0.8, borderpad=0.1, loc='upper center',
                  bbox_to_anchor=(0.5, -0.46))
    else:
        ax.legend(fontsize=FS_K - 1, frameon=False, loc='upper left',
                  handlelength=1.2, borderpad=0.1)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    for xi, ph in enumerate(PHASES):
        r = st.get(f'correct, {ph}')
        if r is None:
            continue
        ax.errorbar(xi, r['mean_hz'],
                    yerr=[[r['mean_hz'] - r['ci_low_hz']],
                          [r['ci_high_hz'] - r['mean_hz']]],
                    fmt='o', ms=6, capsize=3, lw=1.3,
                    color=C[('correct', ph)], zorder=3)
    ax.plot([0, 1], [st[f'correct, {p}']['mean_hz'] for p in PHASES],
            color='0.5', lw=1.0, zorder=2)
    pc = st.get('phase | correct')
    if pc is not None:
        y = max(st[f'correct, {p}']['ci_high_hz'] for p in PHASES)
        ax.text(0.5, y, '*' if pc['p_perm'] < 0.05 else 'n.s.', ha='center',
                va='bottom', fontsize=FS_L)
    ax.axhline(0, color='0.35', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([X_LABEL[p] for p in PHASES], fontsize=FS_K)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel(YLAB[reading],
                  fontsize=FS_L - (1 if reading == 'stillness-matched' else 0))
    ax.tick_params(labelsize=FS_K)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    for e in ('.pdf', '.png'):
        fig.savefig(out_png.replace('.png', e), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_png)} (+ .pdf)")


def panel_g(out_png, st, reading):
    """The 2 x 2 that the interaction is made of -- the components, never the
    interaction on its own."""
    fig, ax = plt.subplots(figsize=(5.0 * CM, 4.4 * CM))
    for val in ('correct', 'error'):
        xs, ys = [], []
        for xi, ph in enumerate(PHASES):
            r = st.get(f'{val}, {ph}')
            if r is None:
                continue
            ax.errorbar(xi, r['mean_hz'],
                        yerr=[[r['mean_hz'] - r['ci_low_hz']],
                              [r['ci_high_hz'] - r['mean_hz']]],
                        fmt='o', ms=6, capsize=3, lw=1.3, color=C[(val, ph)],
                        zorder=3)
            xs.append(xi); ys.append(r['mean_hz'])
        ax.plot(xs, ys, color=C[(val, 'learning')], lw=1.1, zorder=2,
                label=f'{val} uncovers')
    it = st.get('INTERACTION')
    if it is not None:
        ax.set_title(f"interaction p={it['p_perm']:.3f}"
                     + ('*' if it['p_perm'] < 0.05 else ' (n.s.)'),
                     fontsize=FS_K, pad=3)
    ax.axhline(0, color='0.35', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([X_LABEL[p] for p in PHASES], fontsize=FS_K)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel(YLAB[reading],
                  fontsize=FS_L - (1 if reading == 'stillness-matched' else 0))
    ax.tick_params(labelsize=FS_K)
    # Below the axis: on a 4 cm panel an in-axes legend sits on the data.
    ax.legend(fontsize=FS_K - 1, frameon=False, ncol=2, handlelength=1.0,
              columnspacing=0.8, borderpad=0.1,
              loc='upper center', bbox_to_anchor=(0.5, -0.22))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(0.0, 1.16, reading, transform=ax.transAxes, ha='left',
            va='bottom', fontsize=FS_K - 1, color='0.35')
    fig.tight_layout(pad=0.4)
    for e in ('.pdf', '.png'):
        fig.savefig(out_png.replace('.png', e), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_png)} (+ .pdf)")


def run(bundle=None, out_dir=None, presses_csv=None, unit=fin.UNIT,
        min_events=fin.MIN_EVENTS, n_perm=rip.N_SIGN_FLIPS,
        caliper_s=fin.CALIPER_S, rel_caliper=fin.REL_CALIPER, pad_s=0.25):
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr')
    bundle = bundle or os.path.join(group, 'bundle_v2')
    presses_csv = presses_csv or os.path.join(group, 'press_categories.csv')
    out_dir = out_dir or os.path.join(
        group, f"figure4_panels_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_figure_phase_panels')
    np.random.seed(SEED)
    print(f"\n  phases: learning = rep < first_correct ; "
          f"execution = rep >= first_correct")
    print(f"  baseline {BASE}, test window {WIN}, pad {pad_s}, seed {SEED}")

    data = rip.load_bundle(bundle)
    if pad_s is not None:
        import mc.analyse.swr_bundle as sbu
        data = sbu.repad_bundle(data, float(pad_s))
    presses = fin.press_table(data, bundle, presses_csv, rebuild=False)
    presses = grid.add_exclusive_stage(data, presses)

    print("\n  ── cells ──")
    centres, cells = build(data, presses, unit, min_events, caliper_s,
                           rel_caliper)
    prof = readings(cells, centres)

    allst = {}
    for rname, p in prof.items():
        st = stats_for(p, centres, n_perm)
        allst[rname] = st
        print(f"\n  ══ {rname} ══")
        for n, r in st.items():
            star = '*' if r['p_perm'] < 0.05 else ' '
            cl = ('; '.join(f"{c['start_s']:+.2f}..{c['stop_s']:+.2f}s "
                            f"p={c['p']:.3f}" for c in r['clusters'])
                  or 'no cluster')
            hol = (f" holm={r['p_perm_holm']:.3f}" if 'p_perm_holm' in r else '')
            print(f"   {star}{n:24s} n={r['n_units']:3d} {r['mean_hz']:+.4f} "
                  f"[{r['ci_low_hz']:+.4f},{r['ci_high_hz']:+.4f}] "
                  f"t({r['df']})={r['t']:+.2f} p={r['p_perm']:.4f}{hol}  [{cl}]")

    tag = {'own baseline -1.5..-1.0': 'ownbase', 'stillness-matched': 'matched'}
    for rname, st in allst.items():
        panel_f(os.path.join(out_dir, f'panel_f_{tag[rname]}.png'),
                cells, centres, prof[rname], st, rname)
        panel_g(os.path.join(out_dir, f'panel_g_{tag[rname]}.png'), st, rname)

    payload = {'created': datetime.now().isoformat(timespec='seconds'),
               'script': os.path.basename(__file__),
               'phase_definition': {
                   'learning': 'rep < first_correct_repeat',
                   'execution': 'rep >= first_correct_repeat',
                   'note': 'agreed 2026-09-27; = swr_windows.add_phase3 '
                           'collapsed to two'},
               'settings': {'bundle': bundle, 'pad_s': pad_s, 'unit': unit,
                            'min_events': min_events, 'seed': SEED,
                            'n_sign_flips': n_perm, 'baseline_s': list(BASE),
                            'test_window_s': list(WIN),
                            'caliper_s': caliper_s, 'rel_caliper': rel_caliper,
                            'display_smoothing': {
                                'sigma_bins': SMOOTH_SIGMA_BINS,
                                'bin_s': rip.BIN_S,
                                'note': 'applied to the DRAWN LINE only; every '
                                        'statistic is computed on the '
                                        'unsmoothed profiles'}},
               'cell_counts': {f'{v}, {p}': {k: cells[(v, p)][k]
                                             for k in ('n_events', 'n_matched',
                                                       'n_unmatched', 'still_s')}
                               for (v, p) in CELLS},
               'stats': allst}
    with open(os.path.join(out_dir, 'figure4_panels.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    # Cache the per-session profiles so `figures` can redraw without rebuilding
    # the cells, which is the only slow step here.
    np.savez_compressed(
        os.path.join(out_dir, 'profiles.npz'), centres=centres,
        **{f'{sd}|{v}|{p}': np.vstack([cells[(v, p)][sd][u]
                                       for u in sorted(cells[(v, p)][sd])])
           for sd in ('all', 'target', 'control') for (v, p) in CELLS},
        **{f'{sd}u|{v}|{p}': np.array(sorted(cells[(v, p)][sd]))
           for sd in ('all', 'target', 'control') for (v, p) in CELLS},
        **{f'{r}|{v}|{p}': np.vstack([prof[r][(v, p)][u]
                                      for u in sorted(prof[r][(v, p)])])
           for r in prof for (v, p) in CELLS},
        **{f'{r}u|{v}|{p}': np.array(sorted(prof[r][(v, p)]))
           for r in prof for (v, p) in CELLS})
    print(f"\n  wrote figure4_panels.json + panels to {out_dir}")
    return out_dir


def figures(out_dir):
    """Redraw both panels from the cached profiles and stats."""
    z = np.load(os.path.join(out_dir, 'profiles.npz'))
    centres = z['centres']
    with open(os.path.join(out_dir, 'figure4_panels.json')) as f:
        allst = json.load(f)['stats']
    cells = {(v, p): {sd: {int(u): r for u, r in
                           zip(z[f'{sd}u|{v}|{p}'], z[f'{sd}|{v}|{p}'])}
                      for sd in ('all', 'target', 'control')}
             for (v, p) in CELLS}
    tag = {'own baseline -1.5..-1.0': 'ownbase', 'stillness-matched': 'matched'}
    for rname, st in allst.items():
        prof = {(v, p): {int(u): r for u, r in
                         zip(z[f'{rname}u|{v}|{p}'], z[f'{rname}|{v}|{p}'])}
                for (v, p) in CELLS}
        panel_f(os.path.join(out_dir, f'panel_f_{tag[rname]}.png'),
                cells, centres, prof, st, rname)
        panel_g(os.path.join(out_dir, f'panel_g_{tag[rname]}.png'), st, rname)


if __name__ == '__main__':
    try:
        import fire
        fire.Fire({'run': run, 'figures': figures})
    except ImportError:
        run()
