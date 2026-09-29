#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Every way of splitting the task into phases, crossed with every reference.

THE FOUR CHOICES
    a) WHERE THE BOUNDARY SITS. A grid's repeats run r0, r0+1, ... Let
       `first_solved` be the first fully correct (error-free) repeat. The two
       conventions in this codebase disagree about which side that repeat
       belongs to, and they disagree about 1432 repeats -- 39% of the learning
       side:

         inclusive   the first fully correct repeat is the LAST repeat of
                     learning.   `rip.stage_of_repeat`, and the definition the
                     project means: that repeat is where the route is first
                     demonstrated, not yet merely relied on.
         exclusive   the first fully correct repeat is the FIRST repeat of
                     execution.  `swr_windows.add_phase3`, which the HFB branch
                     used until 2026-09-27.

    b) HOW THE PHASES ARE GROUPED -- three phases, or one of two ways of
       collapsing to two.
    c) THE BASELINE WINDOW -- -1.6..-1.1 s or -1.5..-1.0 s.
    d) THE REFERENCE -- the event's own pre-window, or its stillness-matched
       arrow press.

THE PHASES, written out
    r0 = the grid's first repeat, fs = its first fully correct repeat.

      first         rep == r0                       (the first traversal)
      mid_inc       r0 <  rep <= fs                 while learning, INCLUSIVE
      known_inc     rep >  fs                       once known,     INCLUSIVE
      mid_exc       r0 <  rep <  fs                 while learning, EXCLUSIVE
      known_exc     rep >= fs                       once known,     EXCLUSIVE
      learn_inc     rep <= fs                       first + mid_inc
      learn_exc     rep <  fs                       first + mid_exc
      later         rep >  r0                       everything after the first

    `later` and `first` do not depend on the boundary, so `first vs later` is
    the SAME analysis under both conventions -- which is why its two rows
    below are identical rather than a coincidence.

THE GROUPINGS
      three_phase         first | mid | known
      first_vs_later      first | later
      learning_vs_known   learn | known

    Note `three_phase` is the only grouping in which the middle phase is tested
    at all; the two-phase groupings differ from each other only in which side
    the middle phase joins.

WHAT IS HELD FIXED
    Everything else: same bundle, same pad, same de-duplication, same
    per-session unit, same 10-event floor, same 0-0.5 s test window, and every
    number goes through the same `fin.window_stats` / `fin.cluster_test` /
    `rip.contrast_profiles` the reported values came from. Each cell and each
    contrast is computed ONCE per reading and reused wherever a grouping needs
    it, so two groupings that share a cell cannot disagree about it.

MULTIPLE COMPARISONS
    Uncorrected across the sweep on purpose: this asks whether a fixed claim is
    stable, not whether any cell is significant. Within one (boundary,
    grouping, reading) the contrasts are one family and carry Holm.

    python scripts/swr_robustness_grid.py run --pad_s=0.25

@author: Svenja Kuchenhoff

STATUS
    ⚠ EXPLORATORY — NOT IN THE REPORTED RESULT. A sensitivity analysis run to
    choose the phase definition, which is now fixed (learning = rep < first
    correct repeat). The reported numbers come from
    `swr_figure_phase_panels.py`. Kept because it is the evidence for that
    choice and for the statement that the interaction does not survive the
    stillness control.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import scripts.swr_final_ripple_analysis as fin
import mc.analyse.swr_phases as ph
from mc.analyse.swr_phases import (VALENCES, BASE_A, BASE_B, PHASE_Q,
                                   PHASE_DEF, PHASE_SHORT, BOUNDARIES,
                                   GROUPINGS, holm, base,
                                   add_exclusive_stage)

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

SEED = fin.SEED
PRIMARY_WINDOW = fin.PRIMARY_WINDOW

READINGS = {
    'own baseline -1.6..-1.1':    {'kind': 'own', 'win': BASE_A,
                                   'stillness': False, 'baseline': 'A'},
    'own baseline -1.5..-1.0':    {'kind': 'own', 'win': BASE_B,
                                   'stillness': False, 'baseline': 'B'},
    'stillness-matched':          {'kind': 'matched', 'win': None,
                                   'stillness': True, 'baseline': 'none'},
    # Supplementary. NOT "matching to a baseline": the stillness control is
    # still the matched arrow press, and each side is ADDITIONALLY referenced
    # to its own pre-window first. It is not a no-op because the two sides have
    # different pre-event rates -- control presses are chosen to sit in quiet
    # stretches, so their pre-window is ripple-richer than the uncover's.
    'stillness-matched + base A': {'kind': 'matched', 'win': BASE_A,
                                   'stillness': True, 'baseline': 'A'},
    'stillness-matched + base B': {'kind': 'matched', 'win': BASE_B,
                                   'stillness': True, 'baseline': 'B'},
}
PRIMARY_READINGS = ('own baseline -1.6..-1.1', 'own baseline -1.5..-1.0',
                    'stillness-matched')


# ══ 2) Build every cell once ══════════════════════════════════════════

def build_cells(data, presses, unit, min_events, caliper_s, rel_caliper):
    """For each (valence, phase set): matched target, its control, all events.

    Keyed on the phase-set id, and the QUERY is what is deduplicated -- so
    `first` is matched and rated once even though three groupings use it, and
    two groupings sharing a cell cannot disagree about its value.
    """
    unc = presses[presses.kind == 'uncover']
    out, centres = {}, None
    for ph, phq in PHASE_Q.items():
        for val in VALENCES:
            label = f'{val}, {ph}'
            query = f"valence == '{val}' and ({phq})"
            tgt, ctl, _, bal = fin.matched_sets(presses, query, caliper_s,
                                                rel_caliper, SEED)
            per_all = {int(s): g.t_s.to_numpy(float)
                       for s, g in unc.query(query).groupby('session')}
            if not tgt or not per_all:
                print(f"    {label:22s} -- nothing matched, skipped")
                continue
            c_t, raw_t, cnt_t = fin.rates(data, tgt, unit, min_events)
            c_c, raw_c, cnt_c = fin.rates(data, ctl, unit, min_events)
            c_a, raw_a, cnt_a = fin.rates(data, per_all, unit, min_events)
            if c_t is None or c_c is None or c_a is None:
                print(f"    {label:22s} -- not computable, skipped")
                continue
            centres = c_t
            n_m, n_u = int(bal.n_matched.sum()), int(bal.n_unmatched.sum())
            out[label] = {
                'target': raw_t, 'control': raw_c, 'all': raw_a,
                'meta': {'phase': ph, 'valence': val,
                         'definition': PHASE_DEF[ph], 'query': query,
                         'n_sessions_matched': len(raw_t),
                         'n_sessions_all': len(raw_a),
                         'n_matched_events': n_m, 'n_unmatched_events': n_u,
                         'match_rate': n_m / max(n_m + n_u, 1),
                         'n_events_all': int(cnt_a['n_events_used']),
                         'still_target_s': float(bal.still_target_s.median()),
                         'still_control_s': float(bal.still_control_s.median())}}
            m = out[label]['meta']
            print(f"    {label:22s} all {m['n_events_all']:6d} ev / "
                  f"{m['n_sessions_all']:2d} sess | matched {n_m:6d}/"
                  f"{n_m + n_u:6d} ({100 * m['match_rate']:3.0f}%) | still "
                  f"{m['still_target_s']:.2f} vs {m['still_control_s']:.2f} s")
    return centres, out


def profiles_for(cells, centres, spec):
    prof = {}
    for label, rec in cells.items():
        if spec['kind'] == 'matched':
            a, b = rec['target'], rec['control']
            if spec['win'] is not None:
                a, b = base(a, centres, spec['win']), base(b, centres, spec['win'])
            d = fin.difference(a, b)
        else:
            d = base(rec['all'], centres, spec['win'])
        if len(d) >= 3:
            prof[label] = d
    return prof


# ══ 3) Enumerate every test once ══════════════════════════════════════

def enumerate_tests():
    """{name: (block, weights or None)} -- the union over every grouping.

    A test that two groupings share appears once, so it is computed once and
    both groupings quote the same number.
    """
    tests = {}
    for ph in PHASE_Q:
        for val in VALENCES:
            tests[f'{val}, {ph}'] = ('effect', None)
    for (bnd, grp), phases in GROUPINGS.items():
        for p in phases:
            tests.setdefault(f'valence | {p}', ('contrast', {
                f'correct, {p}': 1.0, f'error, {p}': -1.0}))
        for i, a in enumerate(phases):
            for b in phases[i + 1:]:
                for val in VALENCES:
                    tests.setdefault(f'phase | {val} ({a} - {b})', ('contrast', {
                        f'{val}, {a}': 1.0, f'{val}, {b}': -1.0}))
                tests.setdefault(f'INTERACTION ({a} - {b})', ('contrast', {
                    f'correct, {a}': 1.0, f'error, {a}': -1.0,
                    f'correct, {b}': -1.0, f'error, {b}': 1.0}))
    return tests


def run_tests(prof, centres, n_perm, cluster_blocks):
    recs = {}
    for name, (block, weights) in enumerate_tests().items():
        if weights is None:
            d = prof.get(name)
        else:
            if any(k not in prof for k in weights):
                continue
            d = rip.contrast_profiles(prof, weights)
        if d is None or len(d) < 3:
            continue
        got = fin.window_stats(d, centres, PRIMARY_WINDOW)
        if got is None:
            continue
        rec = dict(got)
        rec['block'] = block
        if block in cluster_blocks:
            _, keep = fin.cluster_test(d, centres, name, n_perm)
            rec['clusters'] = keep
        recs[name] = rec
    return recs


def holm_within_groupings(recs):
    """Holm per (boundary, grouping): its contrasts are one family."""
    for (bnd, grp), phases in GROUPINGS.items():
        fam = []
        for i, a in enumerate(phases):
            for b in phases[i + 1:]:
                fam += [f'phase | {v} ({a} - {b})' for v in VALENCES]
                fam.append(f'INTERACTION ({a} - {b})')
        fam += [f'valence | {p}' for p in phases]
        fam = [n for n in fam if n in recs]
        if not fam:
            continue
        adj = holm([recs[n]['p_perm'] for n in fam])
        for n, a in zip(fam, adj):
            recs[n].setdefault('holm', {})[f'{bnd}/{grp}'] = float(a)
            recs[n].setdefault('family_size', {})[f'{bnd}/{grp}'] = len(fam)


# ══ 4) Driver ═════════════════════════════════════════════════════════

def run(bundle=None, out_dir=None, presses_csv=None, unit=fin.UNIT,
        min_events=fin.MIN_EVENTS, n_perm=rip.N_SIGN_FLIPS,
        caliper_s=fin.CALIPER_S, rel_caliper=fin.REL_CALIPER, pad_s=0.25,
        clusters='effect'):
    """`clusters`: which blocks get the sliding-window cluster test --
    'effect' (default), 'all', or 'none'. Clusters run only on the three
    primary readings."""
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr')
    bundle = bundle or os.path.join(group, 'bundle_v2')
    presses_csv = presses_csv or os.path.join(group, 'press_categories.csv')
    out_dir = out_dir or os.path.join(
        group, f"ripple_phase_sweep_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_robustness_grid')
    np.random.seed(SEED)
    cluster_blocks = {'effect': {'effect'}, 'all': {'effect', 'contrast'},
                      'none': set()}[clusters]

    print(f"\n  bundle : {bundle}\n  presses: {presses_csv}\n  out    : {out_dir}")
    print(f"  unit={unit}, min {min_events} events, pad {pad_s} s, "
          f"{n_perm} sign-flips, seed {SEED}, clusters={clusters}")
    print(f"  test window {PRIMARY_WINDOW}, baselines A={BASE_A} B={BASE_B}")

    data = rip.load_bundle(bundle)
    if pad_s is not None:
        import mc.analyse.swr_bundle as sbu
        n0 = len(data['ripples'])
        data = sbu.repad_bundle(data, float(pad_s))
        print(f"  re-padded to {pad_s} s: {n0} -> {len(data['ripples'])} ripples")
    presses = fin.press_table(data, bundle, presses_csv, rebuild=False)
    print("\n  ── adding the exclusive stage labelling ──")
    presses = add_exclusive_stage(data, presses)

    print("\n  ── building cells ──")
    centres, cells = build_cells(data, presses, unit, min_events,
                                 caliper_s, rel_caliper)

    readings = {}
    for rname, spec in READINGS.items():
        prof = profiles_for(cells, centres, spec)
        cb = cluster_blocks if rname in PRIMARY_READINGS else set()
        recs = run_tests(prof, centres, n_perm, cb)
        holm_within_groupings(recs)
        readings[rname] = {'spec': {k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in spec.items()},
                           'primary': rname in PRIMARY_READINGS,
                           'tests': recs}
        print(f"\n  ══ {rname}: {len(recs)} tests ══")

    rows = []
    for rname, blk in readings.items():
        for name, rec in blk['tests'].items():
            rows.append(dict(reading=rname, primary=blk['primary'],
                             stillness_control=blk['spec']['stillness'],
                             baseline=blk['spec']['baseline'], test=name,
                             block=rec['block'],
                             **{k: v for k, v in rec.items()
                                if k not in ('block', 'clusters', 'holm',
                                             'family_size', 'window_s')}))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'phase_sweep.csv'),
                              index=False)
    payload = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'script': os.path.basename(__file__),
        'question': ('how do the ripple-rate results change across the phase '
                     'boundary, the phase grouping, the baseline window and '
                     'the stillness control?'),
        'settings': {'bundle': bundle, 'presses_csv': presses_csv,
                     'pad_s': pad_s, 'unit': unit, 'min_events': min_events,
                     'seed': SEED, 'n_sign_flips': n_perm,
                     'caliper_s': caliper_s, 'rel_caliper': rel_caliper,
                     'primary_window_s': list(PRIMARY_WINDOW),
                     'baseline_A_s': list(BASE_A), 'baseline_B_s': list(BASE_B),
                     'bin_s': rip.BIN_S, 'half_s': rip.HALF_S,
                     'dedup_s': rip.DEDUP_S,
                     'sliding_width_s': rip.SLIDE_WIDTHS_S[0],
                     'cluster_alpha': rip.CLUSTER_ALPHA,
                     'clusters_on': clusters,
                     'control_keys': list(fin.CONTROL_KEYS),
                     'boundaries': BOUNDARIES,
                     'phase_definitions': PHASE_DEF,
                     'phase_queries': PHASE_Q,
                     'groupings': {f'{b}/{g}': p for (b, g), p in GROUPINGS.items()},
                     'multiple_comparisons': {
                         'across the sweep': 'none, on purpose',
                         'within a grouping': 'Holm on its contrast family'}},
        'cell_meta': {k: v['meta'] for k, v in cells.items()},
        'centres_s': centres.tolist(),
        'readings': readings}
    with open(os.path.join(out_dir, 'phase_sweep.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    report(out_dir)
    figure(os.path.join(out_dir, 'phase_sweep.png'), payload=payload)
    print(f"\n  wrote phase_sweep.csv / .json / .md / .png to {out_dir}")
    return out_dir


# ══ 5) Report ═════════════════════════════════════════════════════════

def _fmt(rec, key=None):
    if rec is None:
        return '—'
    b = '**' if rec['p_perm'] < 0.05 else ''
    hol = ''
    if key and isinstance(rec.get('holm'), dict) and key in rec['holm']:
        hol = f", Holm {rec['holm'][key]:.3f}"
    cl = ''
    if 'clusters' in rec:
        cl = (' · cl ' + '; '.join(f"{c['start_s']:+.2f}…{c['stop_s']:+.2f}s "
                                   f"p={c['p']:.3f}" for c in rec['clusters'])
              if rec['clusters'] else ' · no cl')
    return (f"{b}{rec['mean_hz']:+.4f}{b}, t({rec['df']})={rec['t']:+.2f}, "
            f"{b}p={rec['p_perm']:.4f}{b}{hol} (n={rec['n_units']}){cl}")


def report(out_dir):
    with open(os.path.join(out_dir, 'phase_sweep.json')) as f:
        payload = json.load(f)
    R = payload['readings']
    cols = [r for r in READINGS if R[r]['primary']]
    L = ['# Phase sweep — every grouping, every reference\n',
         '## The phases, written out\n',
         '| id | definition | events (correct) | events (error) |', '|---|---|---|---|']
    cm = payload['cell_meta']
    for ph, d in PHASE_DEF.items():
        c = cm.get(f'correct, {ph}', {}).get('n_events_all', '—')
        e = cm.get(f'error, {ph}', {}).get('n_events_all', '—')
        L.append(f'| `{ph}` | `{d}` | {c} | {e} |')
    L += ['\n`r0` = the grid\'s first repeat, `fs` = its first fully correct '
          'repeat.\n', '## The groupings\n',
          '| boundary | grouping | phases |', '|---|---|---|']
    for (b, g), p in GROUPINGS.items():
        L.append(f'| {b} | {g} | ' + ' · '.join(f'`{x}`' for x in p) + ' |')

    for (bnd, grp), phases in GROUPINGS.items():
        key = f'{bnd}/{grp}'
        L.append(f'\n---\n\n## {bnd} — {grp}\n')
        L.append('*' + BOUNDARIES[bnd] + '; phases: '
                 + ' · '.join(f'`{p}` ({PHASE_DEF[p].split("(")[0].strip()})'
                              for p in phases) + '*\n')
        for block, title, names in (
                ('effect', 'effects — each condition against its reference',
                 [f'{v}, {p}' for p in phases for v in VALENCES]),
                ('contrast', 'contrasts between effects',
                 [f'valence | {p}' for p in phases]
                 + [f'phase | {v} ({a} - {b})'
                    for i, a in enumerate(phases) for b in phases[i + 1:]
                    for v in VALENCES]
                 + [f'INTERACTION ({a} - {b})'
                    for i, a in enumerate(phases) for b in phases[i + 1:]])):
            L.append(f'\n### {title}\n')
            L.append('| test | ' + ' | '.join(cols) + ' |')
            L.append('|---|' + '---|' * len(cols))
            for n in names:
                if not any(n in R[c]['tests'] for c in cols):
                    continue
                L.append(f'| {n} | '
                         + ' | '.join(_fmt(R[c]['tests'].get(n), key)
                                      for c in cols) + ' |')
    txt = '\n'.join(L)
    with open(os.path.join(out_dir, 'phase_sweep.md'), 'w') as f:
        f.write(txt + '\n')
    print(f"  wrote phase_sweep.md")
    return txt


# ══ 6) Figure ═════════════════════════════════════════════════════════

READING_C = {'own baseline -1.6..-1.1': '#4F4F4F',
             'own baseline -1.5..-1.0': '#8C8C8C',
             'stillness-matched': '#0e3d3a',
             'stillness-matched + base A': '#175e62',
             'stillness-matched + base B': '#5b9b8d'}
CM = 1 / 2.54


def figure(out_png, payload=None, json_path=None):
    """One panel per (boundary, grouping); effects and contrasts stacked.

    Straight from the json -- the point is the 0-0.5 s window mean the test
    used and the bar its 95% CI.
    """
    if payload is None:
        with open(json_path) as f:
            payload = json.load(f)
    R = payload['readings']
    cols = [r for r in READINGS if R[r]['primary']]
    keys = list(GROUPINGS)
    rows_per = {}
    for (bnd, grp), phases in GROUPINGS.items():
        names = [f'{v}, {p}' for p in phases for v in VALENCES]
        names += [f'valence | {p}' for p in phases]
        names += [f'phase | {v} ({a} - {b})'
                  for i, a in enumerate(phases) for b in phases[i + 1:]
                  for v in VALENCES]
        names += [f'INTERACTION ({a} - {b})'
                  for i, a in enumerate(phases) for b in phases[i + 1:]]
        rows_per[(bnd, grp)] = [n for n in names
                                if any(n in R[c]['tests'] for c in cols)]
    nmax = max(len(v) for v in rows_per.values())
    fig, axes = plt.subplots(2, 3, figsize=(26 * CM, 0.40 * nmax * 2 * CM + 4 * CM),
                             sharex=True)
    for ax, k in zip(axes.T.ravel(order='F') if False else axes.ravel(), keys):
        names = rows_per[k]
        for yi, name in enumerate(names):
            y = len(names) - 1 - yi
            for ri, c in enumerate(cols):
                rec = R[c]['tests'].get(name)
                if rec is None:
                    continue
                off = (ri - (len(cols) - 1) / 2) * 0.26
                sig = rec['p_perm'] < 0.05
                ax.errorbar(rec['mean_hz'], y + off,
                            xerr=[[rec['mean_hz'] - rec['ci_low_hz']],
                                  [rec['ci_high_hz'] - rec['mean_hz']]],
                            fmt='o', ms=3.4, capsize=1.8, lw=0.9,
                            color=READING_C[c],
                            mfc=READING_C[c] if sig else 'white',
                            mec=READING_C[c], mew=1.0, zorder=3)
            if yi % 2:
                ax.axhspan(y - 0.5, y + 0.5, color='0.96', zorder=0)
        ax.axvline(0, color='0.3', lw=0.8, zorder=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace('INTERACTION', 'int')
                            .replace('valence | ', 'val|')
                            .replace('phase | ', 'ph|')
                            for n in names[::-1]], fontsize=6.2)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_title(f'{k[0]} — {k[1]}', fontsize=8.5, loc='left', pad=3)
        ax.tick_params(labelsize=6.5, length=2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Δ ripple rate (Hz, 0–0.5 s)', fontsize=8)
    handles = [plt.Line2D([], [], marker='o', ls='none', ms=4,
                          color=READING_C[c], label=c) for c in cols]
    handles.append(plt.Line2D([], [], marker='o', ls='none', ms=4, color='0.3',
                              mfc='white', label='open = n.s. (p ≥ 0.05)'))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Ripple rate: every phase boundary × grouping × reference\n'
                 'points are the 0–0.5 s window mean ± 95% CI', fontsize=10,
                 y=0.997)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {os.path.basename(out_png)} / .pdf")


def figures(out_dir):
    report(out_dir)
    figure(os.path.join(out_dir, 'phase_sweep.png'),
           json_path=os.path.join(out_dir, 'phase_sweep.json'))


if __name__ == '__main__':
    if fire is not None:
        fire.Fire({'run': run, 'report': report, 'figures': figures})
    else:
        run()
