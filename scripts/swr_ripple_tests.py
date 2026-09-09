#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripple tests — the conditions we settled on, tested and plotted.

The question: hippocampal ripples are how the hippocampus tells mPFC what the
plan is. So ripples should appear when a piece of route information is acquired,
and not when nothing is learned.

Each test below defines a set of conditions, aligns ripples to those events, and
asks one question: does ripple rate depart from THAT SAME TRIAL'S baseline?
Comparing a condition with its own baseline rather than with another condition
is what makes conditions of different overall rate comparable — an earlier
version of this analysis compared rates between conditions and missed the effect
entirely, because the first traversal has a lower floor.

No window is chosen. Every position of a sliding window is tested and the
multiple comparisons across positions are corrected by a cluster permutation.

Tests:
    1) stage          uncovering D, by learning stage
    2) reward         each reward A-D, first traversal only
    3) feedback       correct vs error, pooled
    4) feedback_stage feedback valence crossed with learning stage
    5) reward_feedback  correct vs error for each reward
    6) full           reward x valence x stage (cells with enough events)

Outputs, per test, in <out_dir>/:
    <test>.png            rate over time, the sliding test, the null
    <test>_result.json    hypothesis, every number, the conclusion
    <test>_counts.csv     subjects, sessions, derivations, events per condition

    python scripts/swr_ripple_tests.py --bundle=<bundle dir>
    python scripts/swr_ripple_tests.py --bundle=<dir> --tests="['feedback_stage']"

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import itertools
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mc.analyse.ripples as rip

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


# ── Settings ──────────────────────────────────────────────────────────
ALL_TESTS = ('stage', 'reward', 'feedback', 'feedback_stage',
             'reward_feedback', 'full')
NAMED_WINDOWS = {'pre  (-0.6..-0.1)': (-0.6, -0.1),
                 'post (0..0.5)': (0.0, 0.5),
                 'post (0.5..1.0)': (0.5, 1.0)}

QUESTIONS = {
    'stage':  'Do ripples rise after uncovering D, and only the first time?',
    'reward': 'Is the rise specific to D, or does any first reward do it?',
    'feedback': 'Does learning something (correct) raise ripples and learning '
                'nothing (error) lower them?',
    'feedback_stage': 'Is that valence effect specific to when the route is '
                      'still unknown?',
    'reward_feedback': 'Does it matter which reward was being sought?',
    'full': 'Reward x valence x stage, for cells with enough events.',
}


# ── Conditions ────────────────────────────────────────────────────────
# Each returns {condition label: {session: event times}}. Nothing else in this
# script knows how a condition is built, so adding one is a single function.

def conditions_stage(bundle):
    """Uncovering D, split by learning stage."""
    out = {f'D, {s}': {} for s in rip.STAGES}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        hit = table.query("valence == 'correct' and reward == 'D'")
        for stage, g in hit.groupby('stage'):
            out[f'D, {stage}'][session] = g.t_s.to_numpy(float)
    return out


def conditions_reward(bundle):
    """Each reward A-D, on the first traversal only."""
    out = {f'first {r}': {} for r in rip.REWARDS}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        hit = table.query("valence == 'correct' and stage == 'first uncovers'")
        for reward, g in hit.groupby('reward'):
            out[f'first {reward}'][session] = g.t_s.to_numpy(float)
    return out


def conditions_feedback(bundle):
    """Correct vs error uncoverings, pooled over stage and reward."""
    out = {v: {} for v in rip.VALENCE}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        for valence, g in table.groupby('valence'):
            out[valence][session] = g.t_s.to_numpy(float)
    return out


def conditions_feedback_stage(bundle):
    """Valence crossed with learning stage."""
    out = {f'{v}, {s}': {} for v in rip.VALENCE for s in rip.STAGES}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        for (valence, stage), g in table.groupby(['valence', 'stage']):
            out[f'{valence}, {stage}'][session] = g.t_s.to_numpy(float)
    return out


def conditions_reward_feedback(bundle):
    """Correct vs error for each reward being sought."""
    out = {f'{v} {r}': {} for r in rip.REWARDS for v in rip.VALENCE}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        for (valence, reward), g in table.groupby(['valence', 'reward']):
            out[f'{valence} {reward}'][session] = g.t_s.to_numpy(float)
    return out


def conditions_full(bundle):
    """Reward x valence x stage."""
    out = {}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        for (valence, reward, stage), g in table.groupby(
                ['valence', 'reward', 'stage']):
            out.setdefault(f'{valence} {reward}, {stage}', {})[session] = \
                g.t_s.to_numpy(float)
    return out


BUILDERS = {'stage': conditions_stage, 'reward': conditions_reward,
            'feedback': conditions_feedback,
            'feedback_stage': conditions_feedback_stage,
            'reward_feedback': conditions_reward_feedback,
            'full': conditions_full}


# ── One test ──────────────────────────────────────────────────────────

# How each test is split into figure ROWS. A row is one panel-triple, 16 cm x
# 4 cm; grouping conditions into rows keeps a 24-condition test readable and
# puts the contrast the row is about on one pair of axes.
#   feedback_stage  -> positive feedback on one row, negative on the other
#   full            -> one row per (valence, stage), all four rewards in it
ROW_SPLITS = {
    'feedback_stage': [('positive feedback', lambda c: c.startswith('correct')),
                       ('negative feedback', lambda c: c.startswith('error'))],
    'full': [(f'{v}, {s}',
              (lambda v_, s_: (lambda c: c.startswith(v_) and c.endswith(s_)))(v, s))
             for v in ('correct', 'error') for s in rip.STAGES],
}


def _rows_for(name, profiles, sliding_for_plot):
    """[(row_title, profiles, sliding)] -- one entry if the test is not split."""
    split = ROW_SPLITS.get(name)
    if not split:
        return [(QUESTIONS[name][:60], profiles, sliding_for_plot)]
    rows = []
    for row_title, belongs in split:
        pr = {k: v for k, v in profiles.items() if belongs(k)}
        if not pr:
            continue
        sl = {k: sliding_for_plot.get(k) for k in pr}
        rows.append((row_title, pr, sl))
    return rows


def run_one_test(bundle, name, out_dir, n_perm=rip.N_SIGN_FLIPS,
                 correct_over='time', unit='subject',
                 min_events=rip.MIN_EVENTS, make_figures=True):
    print(f"\n{'=' * 74}\n {name}: {QUESTIONS[name]}\n{'=' * 74}")
    by_condition = BUILDERS[name](bundle)

    profiles, counts, centres = {}, {}, None
    for label, per_session in by_condition.items():
        centres_i, per_subject, count = rip.rate_by_unit(
            bundle, per_session, unit=unit, min_events=min_events)
        if len(per_subject) < 3:
            print(f"  {label:34s} skipped ({count['n_events_used']} events, "
                  f"{len(per_subject)} subjects)")
            continue
        centres = centres_i
        profiles[label] = per_subject
        counts[label] = count
    if not profiles:
        print("  nothing computable")
        return None

    print(f"  {'condition':34s} {'subj':>5s} {'sess':>5s} {'deriv':>6s} "
          f"{'events':>7s}")
    for label, count in counts.items():
        print(f"  {label:34s} {count['n_subjects']:5d} {count['n_sessions']:5d} "
              f"{count['n_derivations']:6d} {count['n_events_used']:7d}")

    results = {'test': name, 'question': QUESTIONS[name],
               'conditions': {}, 'baseline_window_s': list(rip.BASELINE_WIN)}
    sliding_for_plot = {}

    what = ('window positions AND conditions'
            if correct_over == 'time_and_conditions' else 'window positions')
    print(f"\n  sliding window, cluster-corrected over {what} "
          f"({n_perm} sign-flips, no window chosen)")
    baselined = {}
    for label, per_subject in profiles.items():
        subjects, X = rip.baseline_subtract(per_subject, centres)
        baselined[label] = {s: X[i] for i, s in enumerate(subjects)}
    for width in rip.SLIDE_WIDTHS_S:
        sliding = rip.sliding_window_test(baselined, centres, width_s=width,
                                          n_perm=n_perm,
                                          correct_over=correct_over)
        for label, res in sliding.items():
            keep = [c for c in res['clusters'] if c['p'] < 0.05]
            results['conditions'].setdefault(label, {}) \
                .setdefault('sliding', {})[f'{width:g}s'] = keep
            if width == rip.SLIDE_WIDTHS_S[0]:
                sliding_for_plot[label] = res
            if keep:
                for c in keep:
                    print(f"    {width:g}s  {label:32s} {c['direction']:8s} "
                          f"{c['start_s']:+.2f}..{c['stop_s']:+.2f} s "
                          f"(peak {c['peak_s']:+.2f}, p = {c['p']:.4f})")
            else:
                print(f"    {width:g}s  {label:32s} none")

    print(f"\n  named windows vs the same trial's baseline "
          f"(reported, not primary)")
    for label, per_subject in profiles.items():
        for wname, window in NAMED_WINDOWS.items():
            got = rip.window_test(per_subject, centres, window)
            if got is None:
                continue
            results['conditions'][label].setdefault('windows', {})[wname] = got
            star = ' *' if got['p_perm'] < 0.05 else ''
            print(f"    {label:32s} {wname:20s} t = {got['t']:+6.2f}  "
                  f"p_perm = {got['p_perm']:.4g}{star}")

    survived = {lab: [c for w in d.get('sliding', {}).values() for c in w]
                for lab, d in results['conditions'].items()}
    survived = {k: v for k, v in survived.items() if v}
    conclusion = ('; '.join(
        f"{k}: {v[0]['direction']} {v[0]['start_s']:+.2f}..{v[0]['stop_s']:+.2f} s "
        f"(p = {v[0]['p']:.4f})" for k, v in survived.items())
        or 'no condition shows a cluster surviving correction')

    os.makedirs(out_dir, exist_ok=True)
    swr_io.write_result(out_dir, name, hypothesis=QUESTIONS[name],
                        tests=results['conditions'], conclusion=conclusion,
                        extra={'baseline_window_s': list(rip.BASELINE_WIN),
                               'bin_s': rip.BIN_S, 'dedup_s': rip.DEDUP_S,
                               'sliding_widths_s': list(rip.SLIDE_WIDTHS_S),
                               'n_sign_flips': n_perm,
                               'corrected_over': what,
                               'min_events_per_condition': rip.MIN_EVENTS})
    contrasts = condition_contrasts(profiles, centres, counts)
    if len(contrasts):
        contrasts.insert(0, 'test', name)
        contrasts.to_csv(os.path.join(out_dir, f'{name}_contrasts.csv'),
                         index=False)
        sig = contrasts[(contrasts.baseline_p < 0.05) | (contrasts.effect_p < 0.05)]
        if len(sig):
            print("\n  between-condition contrasts with p < 0.05:")
            for _, rr in sig.iterrows():
                bits = []
                if rr.baseline_p < 0.05:
                    bits.append(f"BASELINE differs {rr.baseline_diff_hz:+.4f} Hz "
                                f"(p={rr.baseline_p:.3f})")
                if rr.effect_p < 0.05:
                    bits.append(f"effect differs {rr.effect_diff_hz:+.4f} Hz "
                                f"(p={rr.effect_p:.3f})")
                print(f"    {rr.cond_a} vs {rr.cond_b}: " + "; ".join(bits))
    counts_df = pd.DataFrame(counts).T.reset_index().rename(
        columns={'index': 'condition'})
    counts_df.to_csv(os.path.join(out_dir, f'{name}_counts.csv'), index=False)
    rows = _rows_for(name, profiles, sliding_for_plot)
    # `full` contrasts the four rewards within a row -> A-D ramp.
    # `stage`/`reward` contrast the three stages of D -> stage scale.
    scheme = {'full': 'reward', 'reward': 'reward', 'stage': 'stage'}.get(name)
    # Two versions of every figure: each row on its own y-scale, which shows
    # the shape of a small effect, and one scale per column across rows, which
    # is the only way to compare magnitudes between rows honestly.
    if not make_figures:
        print(f"\n  conclusion: {conclusion}")
        return results
    for suffix, share in (('', False), ('_sharedy', True)):
        rip.plot_rows(rows, os.path.join(out_dir, f'{name}{suffix}.png'),
                      width_s=rip.SLIDE_WIDTHS_S[0], scheme=scheme,
                      counts=counts, share_y=share,
                      suptitle=f'{name}: {QUESTIONS[name]}')
    print(f"\n  conclusion: {conclusion}")
    print(f"  wrote {name}.png + {name}_sharedy.png, "
          f"{name}_result.json, {name}_counts.csv")
    return results

def condition_contrasts(profiles, centres, counts, win=(0.0, 0.5),
                        baseline=rip.BASELINE_WIN):
    """Between-condition tests, which the vs-own-baseline test does not give.

    Two per pair of conditions, both paired across the units both conditions
    share:

      baseline_diff  do the two conditions differ in their BASELINE rate? This
                     is not cosmetic. The baseline window sits 1.1-1.6 s before
                     the event, and the median gap between uncoverings is
                     1.25 s, so 16-28% of baselines contain the previous
                     uncovering -- and that percentage differs by learning
                     stage. A condition with a cleaner baseline shows a larger
                     rise for the same neural signal.
      effect_diff    do the two conditions differ in (window - baseline)? This
                     is the contrast the hypothesis actually implies -- "D and
                     only the first time" is a claim about a DIFFERENCE between
                     stages, not about each one separately exceeding its own
                     baseline.
    """
    mb = (centres >= baseline[0]) & (centres <= baseline[1])
    mw = (centres >= win[0]) & (centres <= win[1])
    rows = []
    labels = list(profiles)
    for a, b in itertools.combinations(labels, 2):
        shared = sorted(set(profiles[a]) & set(profiles[b]))
        if len(shared) < 3:
            continue
        ba = np.array([np.nanmean(profiles[a][u][mb]) for u in shared])
        bb = np.array([np.nanmean(profiles[b][u][mb]) for u in shared])
        ea = np.array([np.nanmean(profiles[a][u][mw]) for u in shared]) - ba
        eb = np.array([np.nanmean(profiles[b][u][mw]) for u in shared]) - bb
        ok = np.isfinite(ba) & np.isfinite(bb) & np.isfinite(ea) & np.isfinite(eb)
        if ok.sum() < 3:
            continue
        tb, pb = stats.ttest_rel(ba[ok], bb[ok])
        te, pe = stats.ttest_rel(ea[ok], eb[ok])
        rows.append(dict(
            cond_a=a, cond_b=b, n_paired=int(ok.sum()),
            baseline_a_hz=round(float(ba[ok].mean()), 4),
            baseline_b_hz=round(float(bb[ok].mean()), 4),
            baseline_diff_hz=round(float((ba[ok] - bb[ok]).mean()), 4),
            baseline_t=round(float(tb), 2), baseline_p=round(float(pb), 4),
            effect_a_hz=round(float(ea[ok].mean()), 4),
            effect_b_hz=round(float(eb[ok].mean()), 4),
            effect_diff_hz=round(float((ea[ok] - eb[ok]).mean()), 4),
            effect_t=round(float(te), 2), effect_p=round(float(pe), 4)))
    return pd.DataFrame(rows)


def descriptive_json(data, out_dir):
    """Every descriptive number the methods section needs, in one file.

    Modelled on what Chen et al. report for human hippocampal ripples: the
    sample at each level, the detector's output (rate, duration, peak
    frequency, amplitude), how much recording survived artifact rejection, and
    the behavioural counts. Written as JSON so the manuscript quotes a file
    rather than a number retyped from a console.

    Every value is `median [IQR]` across the stated unit, not a mean across
    pooled events -- an event-pooled mean is dominated by whichever session
    contributed most events.
    """
    rip_ev, qc, pairs = data['ripples'], data['channel_qc'], data['pairs']
    unc, beh = data['uncover'], data['behaviour']
    usable = qc[~qc.excluded.fillna(False)] if 'excluded' in qc else qc

    def iqr(x, nd=3):
        x = np.asarray(pd.to_numeric(x, errors='coerce'), float)
        x = x[np.isfinite(x)]
        if not x.size:
            return None
        return {'median': round(float(np.median(x)), nd),
                'q25': round(float(np.percentile(x, 25)), nd),
                'q75': round(float(np.percentile(x, 75)), nd),
                'n': int(x.size)}

    per_deriv = rip_ev.groupby(['session', 'pair_id'])
    out = {
        'sample': {
            'sessions': int(rip_ev.session.nunique()),
            'subjects': int(rip_ev.subject_key.nunique()),
            'derivations_total': int(len(qc)),
            'derivations_used': int(len(usable)),
            'derivations_excluded': int(len(qc) - len(usable)),
            'sessions_by_site': rip_ev.groupby('recording_site'
                                               ).session.nunique().to_dict(),
            'derivations_per_session': iqr(usable.groupby('session').size(), 1),
        },
        'ripples': {
            'n_total': int(len(rip_ev)),
            'n_per_derivation': iqr(per_deriv.size(), 1),
            'rate_hz_per_derivation': iqr(usable.rate_hz)
            if 'rate_hz' in usable else None,
            'duration_ms': iqr(rip_ev.duration_s * 1000, 1)
            if 'duration_s' in rip_ev else None,
            'peak_freq_hz': iqr(rip_ev.peak_freq_hz, 1)
            if 'peak_freq_hz' in rip_ev else None,
            'amplitude_uv': iqr(rip_ev.amp_peak_uv, 1)
            if 'amp_peak_uv' in rip_ev else None,
            'peak_rms_z': iqr(rip_ev.rms_peak_z, 2)
            if 'rms_peak_z' in rip_ev else None,
        },
        'artifact_rejection': {
            'clean_fraction_per_derivation': iqr(1 - usable.contaminated_frac)
            if 'contaminated_frac' in usable else None,
            'clean_hours_total': round(float(usable.clean_s.sum() / 3600), 1)
            if 'clean_s' in usable else None,
            'spectral_passed_strict_pct': round(
                100 * float(rip_ev.spectral_passed_strict.mean()), 1)
            if 'spectral_passed_strict' in rip_ev else None,
        },
        'anatomy': {
            'roi_counts': pairs.pair_roi.value_counts().to_dict()
            if 'pair_roi' in pairs else {},
            'hemisphere': pairs.hemisphere.value_counts().to_dict()
            if 'hemisphere' in pairs else {},
        },
        'behaviour': {
            'grids_per_session': iqr(beh.groupby('session').grid_no.nunique(), 1),
            'repeats_per_session': iqr(beh.groupby('session').size(), 1),
            'uncoverings_total': int(len(unc)),
            'uncoverings_correct': int((unc.correct == 1).sum()),
            'uncoverings_error': int((unc.correct == 0).sum()),
            'inter_uncovering_interval_s': iqr(
                unc.sort_values(['session', 't_s']).groupby('session').t_s.diff(), 2),
            'repeats_by_phase': beh.phase3.value_counts().to_dict()
            if 'phase3' in beh else {},
        },
        'analysis_settings': {
            'bin_s': rip.BIN_S, 'half_s': rip.HALF_S,
            'baseline_window_s': list(rip.BASELINE_WIN),
            'dedup_s': rip.DEDUP_S, 'min_events': rip.MIN_EVENTS,
            'min_clean_frac': rip.MIN_CLEAN_FRAC,
            'sliding_widths_s': list(rip.SLIDE_WIDTHS_S),
            'n_sign_flips': rip.N_SIGN_FLIPS,
            'cluster_alpha': rip.CLUSTER_ALPHA,
        },
        'created': datetime.now().isoformat(timespec='seconds'),
    }
    path = os.path.join(out_dir, 'descriptive_statistics.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  wrote descriptive_statistics.json")
    return out


def descriptive_rates(data, out_dir, unit='session', win_s=0.5):
    """Overall ripple rate by reward, by phase and by feedback valence.

    A sanity panel, not a test. If the rate differs between A, B, C and D, or
    between explore/plan/execute, or between correct and error trials, then any
    peri-event effect could be riding on a difference in the underlying rate
    rather than on the event. The expectation is that these are flat.

    Rewards and valence use a `win_s` window from the event. Phases use the
    WHOLE phase -- every repeat labelled with that phase, start to finish --
    because a phase is a period, not an event.

    Rate is always events per ARTIFACT-FREE second: counts over
    `clean_seconds`, per derivation, then averaged within unit.
    """
    unc, beh = data['uncover'], data['behaviour']
    groups = {}

    for state in rip.REWARDS:
        e = unc[(unc.state == state) & (unc.correct == 1)]
        groups[('reward', state)] = {s: (g.t_s.values, g.t_s.values + win_s)
                                     for s, g in e.groupby('session')}
    for val, sel in (('correct', unc.correct == 1), ('error', unc.correct == 0)):
        e = unc[sel]
        groups[('valence', val)] = {s: (g.t_s.values, g.t_s.values + win_s)
                                    for s, g in e.groupby('session')}
    if 'phase3' in beh:
        for ph in ('explore', 'plan', 'execute'):
            d = beh[beh.phase3 == ph]
            groups[('phase', ph)] = {
                s: (g.new_grid_onset.values.astype(float),
                    g.t_D.values.astype(float))
                for s, g in d.groupby('session')
                if np.isfinite(g.new_grid_onset).all() and np.isfinite(g.t_D).all()}

    rows = []
    for (kind, label), per_session in groups.items():
        per_unit = {}
        for session, (starts, stops) in per_session.items():
            ok = np.isfinite(starts) & np.isfinite(stops) & (stops > starts)
            starts, stops = starts[ok], stops[ok]
            if not starts.size:
                continue
            key = (rip.subject_of(data, session) if unit == 'subject'
                   else int(session))
            for pair_id, ripples, intervals in rip.derivations(data, session):
                tt = np.sort(np.asarray(ripples, float))
                n = (np.searchsorted(tt, stops, 'right')
                     - np.searchsorted(tt, starts, 'left')).sum()
                exposure = rip.clean_seconds(intervals, starts, stops).sum()
                if exposure > 1.0:
                    per_unit.setdefault(key, []).append(n / exposure)
        vals = np.array([np.mean(v) for v in per_unit.values()], float)
        rows.append(dict(kind=kind, label=label, n_units=len(vals),
                         mean_hz=float(np.mean(vals)) if vals.size else np.nan,
                         sem_hz=float(np.std(vals) / max(np.sqrt(vals.size), 1))
                         if vals.size else np.nan,
                         values=vals))
    df = pd.DataFrame(rows)
    df.drop(columns='values').to_csv(
        os.path.join(out_dir, f'descriptive_rates_{unit}.csv'), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(rip.ROW_W_CM * rip.CM,
                                           (rip.ROW_H_CM + 0.6) * rip.CM),
                             gridspec_kw=dict(width_ratios=[1.2, 1.0, 0.8]))
    panels = [('reward', rip.REWARDS, f'Reward (event +{win_s:g} s)'),
              ('phase', ('explore', 'plan', 'execute'), 'Phase (whole phase)'),
              ('valence', ('correct', 'error'), f'Feedback (event +{win_s:g} s)')]
    for ax, (kind, order, title) in zip(axes, panels):
        sub = df[df.kind == kind].set_index('label')
        order = [o for o in order if o in sub.index]
        for x, lab in enumerate(order):
            r = sub.loc[lab]
            c = (rip.REWARD_COLOUR.get(lab) if kind == 'reward'
                 else dict(zip(('explore', 'plan', 'execute'),
                               rip.STAGE_COLOUR.values())).get(lab)
                 if kind == 'phase' else rip.VALENCE_COLOUR.get(lab))
            ax.errorbar([x], [r.mean_hz], yerr=[r.sem_hz], color=c, marker='o',
                        ms=5, capsize=3, elinewidth=1.2, lw=0)
            ax.annotate(f"n={int(r.n_units)}", xy=(x, r.mean_hz), xytext=(0, -14),
                        textcoords='offset points', ha='center',
                        fontsize=rip.FS - 3, color='0.4')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=rip.FS)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(title, fontsize=rip.FS, pad=3)
        ax.tick_params(labelsize=rip.FS - 1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
    axes[0].set_ylabel('Ripple rate (Hz)', fontsize=rip.FS)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)                       # one scale: the point is flatness
    fig.suptitle(f'Overall ripple rate ({unit}s as the unit) '
                 f'-- expected to be flat', fontsize=rip.FS)
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=[0, 0, 1, 0.90])
    out = os.path.join(out_dir, f'descriptive_rates_{unit}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote descriptive_rates_{unit}.png")
    return df


# ── Main ──────────────────────────────────────────────────────────────

def run(bundle=None, tests=None, out_dir=None, n_perm=rip.N_SIGN_FLIPS,
        unit='subject', min_events=rip.MIN_EVENTS,
        correct_over='time'):
    """correct_over: 'time' (per condition) or 'time_and_conditions' (family)."""
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                              'group', 'swr', 'bundle')
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                               'group', 'swr', 'ripple_tests')
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_ripple_tests')
    wanted = [tests] if isinstance(tests, str) else (list(tests) if tests
                                                     else list(ALL_TESTS))

    data = rip.load_bundle(bundle)
    ripples = data['ripples']
    qc = data['channel_qc']
    qc = qc[~qc.excluded.fillna(False)] if 'excluded' in qc else qc
    print(f"\n  bundle: {bundle}")
    print(f"  {ripples.session.nunique()} sessions | "
          f"{ripples.subject_key.nunique()} subjects | {len(qc)} derivations | "
          f"{len(ripples)} ripples | {qc.clean_s.sum() / 3600:.1f} h clean")

    summary = {}
    for name in wanted:
        summary[name] = run_one_test(data, name, out_dir, n_perm=n_perm,
                                     correct_over=correct_over, unit=unit,
                                     min_events=min_events)

    with open(os.path.join(out_dir, 'all_tests.json'), 'w') as f:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   'bundle': bundle, 'tests': summary}, f, indent=2,
                  default=str)
    descriptive_json(data, out_dir)
    for u in ('session', 'subject'):
        descriptive_rates(data, out_dir, unit=u)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
