#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One JSON with every ripple/HFB number needed to write the paper.

The results live in six directories written by five scripts, each with its own
schema. Writing a results section means hunting through all of them, and the
risk is quoting two numbers from two estimators in one sentence. This collapses
them into ONE flat list of records with a common shape:

    id, family, what, n, unit, estimate, estimate_unit, sem, ci, t, df,
    p, p_corrected, correction, cluster, source, apa

`apa` is a ready-to-paste string. `source` names the file the number came from,
so every record is traceable back to the run that produced it.

Nothing is recomputed: this is a reader, not an analysis.

    python scripts/swr_results_summary.py run

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io


def _f(x, n=4):
    return None if x is None else round(float(x), n)


def _apa(t=None, df=None, p=None, est=None, unit='Hz', d=None, extra=''):
    bits = []
    if est is not None:
        bits.append(f"{est:+.4f} {unit}")
    if t is not None and df is not None:
        bits.append(f"t({df}) = {t:+.2f}")
    if p is not None:
        bits.append(f"p = {p:.4f}" if p >= 1e-4 else "p < 0.0001")
    if d is not None:
        bits.append(f"d = {d:+.2f}")
    return ', '.join(bits) + (f"; {extra}" if extra else '')


def _cluster_str(clusters, unit='s'):
    if not clusters:
        return None
    out = []
    for c in clusters:
        a = c.get('start_s', c.get('start_ms'))
        b = c.get('stop_s', c.get('stop_ms'))
        out.append(f"{a:+.2f}..{b:+.2f} {unit}, p = {c['p']:.4f}")
    return '; '.join(out)


def rec(**kw):
    kw.setdefault('p_corrected', None)
    kw.setdefault('correction', None)
    kw.setdefault('cluster', None)
    return kw


def from_stage_interaction(path, tag):
    """The ripple-rate scheme files (two-phase and three-stage)."""
    out = []
    if not os.path.exists(path):
        return out
    j = json.load(open(path))
    src = os.path.basename(os.path.dirname(path)) + '/stage_interaction.json'
    for reading, blk in j['readings'].items():
        if reading == 'baseline window':
            for name, c in blk['contrasts'].items():
                for key, what in (('baseline_window', 'baseline window only'),
                                  ('test_window_unsubtracted',
                                   'test window, unsubtracted')):
                    w = c[key]
                    out.append(rec(
                        id=f'{tag}|{key}|{name}', family='ripple rate',
                        what=f'{name} — {what}', n=w['n_units'], unit='session',
                        estimate=_f(w['mean_hz']), estimate_unit='Hz',
                        sem=_f(w['sem_hz']),
                        ci=[_f(w['ci_low_hz']), _f(w['ci_high_hz'])],
                        t=_f(w['t'], 2), df=w['df'], p=_f(w['p_perm']),
                        p_corrected=_f(w.get('p_perm_holm')),
                        correction=('Holm over %d' % w['family_size']
                                    if 'family_size' in w else None),
                        source=src,
                        apa=_apa(w['t'], w['df'], w['p_perm'], w['mean_hz'],
                                 d=w.get('cohens_d'))))
            continue
        for label, c in blk.get('cells', {}).items():
            w = c['windows']['post (0..0.5)']
            out.append(rec(
                id=f'{tag}|{reading}|cell|{label}', family='ripple rate',
                what=f'{label} vs {reading}, 0–0.5 s', n=w['n_units'],
                unit='session', estimate=_f(w['mean_hz']), estimate_unit='Hz',
                sem=_f(w['sem_hz']),
                ci=[_f(w['ci_low_hz']), _f(w['ci_high_hz'])],
                t=_f(w['t'], 2), df=w['df'], p=_f(w['p_perm']),
                cluster=_cluster_str(c.get('clusters')), source=src,
                apa=_apa(w['t'], w['df'], w['p_perm'], w['mean_hz'],
                         d=w.get('cohens_d'),
                         extra=(_cluster_str(c.get('clusters')) or ''))))
        for name, c in blk.get('contrasts', {}).items():
            w = c['windows']['post (0..0.5)']
            out.append(rec(
                id=f'{tag}|{reading}|contrast|{name}', family='ripple rate',
                what=f'{name} ({reading})', n=w['n_units'], unit='session',
                estimate=_f(w['mean_hz']), estimate_unit='Hz',
                sem=_f(w['sem_hz']),
                ci=[_f(w['ci_low_hz']), _f(w['ci_high_hz'])],
                t=_f(w['t'], 2), df=w['df'], p=_f(w['p_perm']),
                p_corrected=_f(w.get('p_perm_holm')),
                correction=('Holm over %d' % w['family_size']
                            if 'family_size' in w else None),
                cluster=_cluster_str(c.get('clusters')), source=src,
                apa=_apa(w['t'], w['df'], w['p_perm'], w['mean_hz'],
                         d=w.get('cohens_d'),
                         extra=(_cluster_str(c.get('clusters')) or ''))))
    return out


def from_locking(path):
    """The HFB locking contrasts (regions and both phases)."""
    out = []
    if not os.path.exists(path):
        return out
    j = json.load(open(path))
    src = os.path.basename(os.path.dirname(path)) + '/locking_contrasts.json'
    out.append(rec(id='hfb|ripple_duration', family='descriptive',
                   what='mean ripple duration',
                   n=j['ripple_duration']['n_sessions'], unit='session',
                   estimate=_f(j['ripple_duration']['mean_ms'], 1),
                   estimate_unit='ms', sem=_f(j['ripple_duration']['sem_ms'], 1),
                   ci=None, t=None, df=None, p=None, source=src,
                   apa=f"{j['ripple_duration']['mean_ms']:.1f} ± "
                       f"{j['ripple_duration']['sem_ms']:.1f} ms "
                       f"({j['ripple_duration']['n_ripples']} ripples)"))
    for fam, blk in j['families'].items():
        for k, s in blk['sides'].items():
            out.append(rec(
                id=f'hfb|{fam}|side|{k}', family='HFB locking',
                what=f"{s['label']} ({fam}) vs shifted null, peri ±250 ms",
                n=s['n_units'], unit='session', estimate=_f(s['peri_mean'], 5),
                estimate_unit='z', sem=_f(s['peri_sem'], 5), ci=None,
                t=_f(s['t'], 2), df=s['n_units'] - 1, p=_f(s['p']),
                cluster=_cluster_str(s['clusters'], 'ms'), source=src,
                apa=_apa(s['t'], s['n_units'] - 1, s['p'], s['peri_mean'],
                         unit='z', extra=(_cluster_str(s['clusters'], 'ms') or ''))))
        for n_, c in blk['contrasts'].items():
            out.append(rec(
                id=f'hfb|{fam}|contrast|{n_}', family='HFB locking',
                what=f'{n_} ({fam}), paired difference',
                n=c['n_units'], unit='session', estimate=_f(c['peri_mean'], 5),
                estimate_unit='z', sem=_f(c['peri_sem'], 5), ci=None,
                t=_f(c['t'], 2), df=c['df'], p=_f(c['p']),
                p_corrected=_f(c.get('p_holm')),
                correction=('Holm over %d' % c['family_size']
                            if 'family_size' in c else None),
                cluster=_cluster_str(c['clusters'], 'ms'), source=src,
                apa=_apa(c['t'], c['df'], c['p'], c['peri_mean'], unit='z',
                         d=c.get('cohens_d'),
                         extra=(_cluster_str(c['clusters'], 'ms') or ''))))
    return out


def from_published(path):
    """Descriptives and the published headline cells."""
    out = []
    if not os.path.exists(path):
        return out
    j = json.load(open(path))
    src = os.path.basename(os.path.dirname(path)) + '/ripple_statistics.json'
    d = j['descriptives']
    out.append(rec(id='desc|dataset', family='descriptive', what='dataset',
                   n=d['n_sessions'], unit='session', estimate=None,
                   estimate_unit=None, sem=None, ci=None, t=None, df=None,
                   p=None, source=src,
                   apa=f"{d['n_sessions']} sessions, {d['n_subjects']} "
                       f"subjects, {d['n_derivations']} derivations, "
                       f"{d['n_ripples']} ripples, "
                       f"{d['clean_hours']:.1f} clean hours"))
    for label, r in j.get('feedback_stage', {}).items():
        w = r['windows']['post (0..0.5)']
        out.append(rec(
            id=f'published|feedback_stage|{label}', family='ripple rate',
            what=f'{label} vs own baseline (published 3-stage)',
            n=w['n_units'], unit='session', estimate=_f(w['mean_hz']),
            estimate_unit='Hz', sem=_f(w['sem_hz']),
            ci=[_f(w['ci_low_hz']), _f(w['ci_high_hz'])],
            t=_f(w['t'], 2), df=w['df'], p=_f(w['p_perm']),
            cluster=_cluster_str(r.get('clusters')), source=src,
            apa=_apa(w['t'], w['df'], w['p_perm'], w['mean_hz'],
                     d=w.get('cohens_d'),
                     extra=(_cluster_str(r.get('clusters')) or ''))))
    return out


def run(out_dir=None):
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr')

    def newest(pattern, fname):
        hits = sorted(glob.glob(os.path.join(group, pattern)), reverse=True)
        for h in hits:
            f = os.path.join(h, fname)
            if os.path.exists(f):
                return f
        return os.path.join(group, pattern, fname)

    records = []
    records += from_published(os.path.join(group, 'ripple_final_pad0.25',
                                           'ripple_statistics.json'))
    records += from_stage_interaction(
        newest('ripple_stage_interaction_2*', 'stage_interaction.json'),
        'two_phase')
    records += from_stage_interaction(
        newest('ripple_stage_interaction_three_stage_2*',
               'stage_interaction.json'), 'three_stage')
    records += from_locking(
        newest('hfb_locking_contrasts_2*', 'locking_contrasts.json'))

    out_dir = out_dir or os.path.join(
        group, f"results_summary_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'script': os.path.basename(__file__),
        'phase_definition': {
            'explore': 'first uncovers + while learning (rewards not yet all known)',
            'known': 'once known (route known, being executed)',
            'note': 'in the HFB branch the same split is explore+plan vs execute'},
        'how_to_read': {
            'estimate': 'mean of the per-session value, in estimate_unit',
            'p': 'sign-flip permutation p where available, else parametric',
            'p_corrected': 'Holm within the family named in `correction`',
            'cluster': 'window-free sliding test; absent means none survived',
            'apa': 'ready-to-paste string'},
        'n_records': len(records),
        'records': records,
        'by_id': {r['id']: r for r in records}}
    f = os.path.join(out_dir, 'results_summary.json')
    with open(f, 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n  {len(records)} records")
    for r in records:
        star = ' *' if (r.get('p_corrected') or r.get('p') or 1) < 0.05 else ''
        print(f"    {r['id']:62s} {r['apa']}{star}")
    print(f"\n  saved -> {f}")


if __name__ == "__main__":
    try:
        import fire
        fire.Fire({'run': run})
    except ImportError:
        run()
