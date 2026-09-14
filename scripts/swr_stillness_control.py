#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is a stage difference in ripple rate a stillness difference in disguise?

The contrast that prompted this: positive feedback raises ripple rate more on
the FIRST traversal than later (+0.035 Hz, 0-0.5 s, p_perm = 0.049, n = 61 --
no surviving cluster). Stillness is not balanced across those two cells. After
a correct uncovering on the first traversal the subject waits a median 1.375 s
before pressing anything again; later in the grid, 0.425 s. F1 says ripple rate
rises with stillness, so some of that 0.035 Hz is bought rather than earned.

Three checks, in increasing order of how much they are worth:

  1) STRATIFY.  Split every event by how long the subject then sat still and
     run the same contrast INSIDE each stillness bin. Like is then compared
     with like. If the stage difference is stillness, it vanishes within bins;
     if it survives in the bins where both stages are well sampled, it is not
     stillness. This is the primary check -- it throws nothing away and it also
     shows the SHAPE of the effect at matched stillness, which a single number
     cannot.

  2) PREDICT.  Measure the rate-vs-stillness curve on these very events, then
     ask what rate difference the observed difference in stillness
     DISTRIBUTIONS alone would produce. Comparing that prediction with the
     observed difference says how much of the effect stillness can account for.

  3) STANDARDISE.  Average the per-bin contrasts within each session, which is
     the stage contrast at a common stillness distribution, and test it with
     the pipeline's own sliding cluster test.

Every rate, contrast and test is the pipeline's: `rip.rate_by_unit`,
`rip.contrast_profiles`, `rip.window_test`, `rip.sliding_window_test`. Nothing
here re-implements a statistic.

    python scripts/swr_stillness_control.py
    python scripts/swr_stillness_control.py --valence=error --min_events=10

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
from swr_valence_stage_interaction import (event_table, profiles_of, _rc,
                                           _axes, PRIMARY_WINDOW, NAMED_WINDOWS)

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


# ── Settings ──────────────────────────────────────────────────────────
QUESTION = ('Does the first-traversal ripple increase survive when events are '
            'matched on how long the subject then sits still?')

# Bin edges in seconds. The first two straddle the 0-0.5 s test window: an
# event followed by a press inside 0.5 s has that press IN the window, so it is
# not comparable with one followed by silence, and binning is what keeps those
# two apart rather than averaging them.
BIN_EDGES = (0.0, 0.5, 1.0, 1.5, 2.5, np.inf)
BIN_LABELS = ('0-0.5 s', '0.5-1 s', '1-1.5 s', '1.5-2.5 s', '≥2.5 s')

STAGE2 = {'first uncovers': 'first uncovers',
          'while learning': 'later', 'once known': 'later'}
STAGES2 = ('first uncovers', 'later')

SEED = 42

# Stillness is a NUISANCE variable, not one of the project's categorical
# variables, so it takes a neutral light-to-dark grey ramp rather than
# borrowing the phase, state, location or ROI scales.
BIN_COLOURS = ('#c9c9c9', '#a0a0a0', '#777777', '#4e4e4e', '#252525')


def bin_events(events, edges=BIN_EDGES, labels=BIN_LABELS):
    """Label every event with the stillness bin it falls in."""
    return events.assign(
        still_bin=pd.cut(events.still_next_s, list(edges), labels=list(labels),
                         right=False),
        stage2=events.stage.map(STAGE2))


def cells_for(events, stages=STAGES2):
    """{f'{stage}': {session: event times}} for one slice of the data."""
    out = {}
    for stage, g in events.groupby('stage2', observed=True):
        if stage in stages:
            out[stage] = {int(s): gg.t_s.to_numpy(float)
                          for s, gg in g.groupby('session')}
    return out


def window_value(profiles, centres, win=PRIMARY_WINDOW,
                 baseline=rip.BASELINE_WIN):
    """Per-unit (window - own baseline) scalar, as `window_test` computes it."""
    units, X = rip.baseline_subtract(profiles, centres)
    m = (centres >= win[0]) & (centres < win[1])
    return units, np.nanmean(X[:, m], axis=1)


# ── 1) The contrast, raw and per stillness bin ────────────────────────

def contrast_per_bin(data, events, unit, min_events, stages=STAGES2):
    """The stage contrast computed separately inside each stillness bin.

    Returns (centres, {bin: contrast profiles}, {bin: counts}) plus the same
    for the unstratified data under the key `all`.
    """
    weights = {stages[0]: 1.0, stages[1]: -1.0}
    profiles, counts, centres = {}, {}, None
    for label in ('all',) + tuple(BIN_LABELS):
        sub = events if label == 'all' else events[events.still_bin == label]
        if not len(sub):
            continue
        c, raw, cnt = profiles_of(data, cells_for(sub, stages), unit,
                                  min_events)
        if c is None or any(s not in raw or not raw[s] for s in stages):
            print(f"    {label:12s} not computable")
            continue
        centres = c
        p = rip.contrast_profiles(raw, weights)
        if len(p) < 3:
            print(f"    {label:12s} only {len(p)} paired {unit}s")
            continue
        profiles[label] = p
        counts[label] = {s: cnt[s] for s in stages}
        counts[label]['n_paired'] = len(p)
    return centres, profiles, counts


def standardised_contrast(profiles):
    """Average each session's per-bin contrasts -> the contrast at a common
    stillness distribution.

    Equal weight per bin, i.e. direct standardisation to a FLAT stillness
    distribution. Equal weights are the neutral choice: weighting by either
    stage's own distribution would standardise onto that stage and make the
    comparison asymmetric. Per-bin values are reported alongside, so a reader
    who prefers other weights can apply them.
    """
    per_unit = {}
    for label in BIN_LABELS:
        for unit, profile in profiles.get(label, {}).items():
            per_unit.setdefault(unit, []).append(profile)
    return {u: np.nanmean(np.vstack(v), axis=0)
            for u, v in per_unit.items() if len(v) >= 2}


# ── 2) What stillness alone predicts ──────────────────────────────────

def predicted_from_stillness(data, events, unit, min_events, centres,
                             stages=STAGES2):
    """Direct standardisation: what the stillness imbalance alone would give.

    rate(bin) is measured on BOTH stages pooled, so it carries no stage
    information at all. Multiplying it by each stage's own stillness
    distribution and differencing gives the rate difference that the stillness
    imbalance produces on its own. Compared with the observed difference, it
    says how much of the effect is bought rather than earned.
    """
    rate_by_bin = {}
    for label in BIN_LABELS:
        sub = events[events.still_bin == label]
        if not len(sub):
            continue
        pooled = {int(s): g.t_s.to_numpy(float) for s, g in sub.groupby('session')}
        c, per_unit, _ = rip.rate_by_unit(data, pooled, unit=unit,
                                          min_events=min_events)
        if c is None or len(per_unit) < 3:
            continue
        units, vals = window_value(per_unit, c)
        rate_by_bin[label] = dict(zip(units, vals))

    share = (events.groupby(['session', 'stage2'], observed=True)
             .still_bin.value_counts(normalize=True).rename('p')
             .reset_index())
    rows = []
    for session, g in share.groupby('session'):
        session = int(session)
        got = {}
        for stage in stages:
            s = g[g.stage2 == stage].set_index('still_bin').p
            got[stage] = {b: float(s.get(b, 0.0)) for b in BIN_LABELS}
        total = 0.0
        used = 0
        for b in BIN_LABELS:
            if b not in rate_by_bin or session not in rate_by_bin[b]:
                continue
            r = rate_by_bin[b][session]
            if not np.isfinite(r):
                continue
            total += (got[stages[0]][b] - got[stages[1]][b]) * r
            used += 1
        if used >= 3:
            rows.append({'session': session, 'predicted_hz': total,
                         'n_bins_used': used})
    return pd.DataFrame(rows), rate_by_bin


# ── 3) Figure ─────────────────────────────────────────────────────────

def figure(out_png, events, centres, profiles, counts, standardised, sliding,
           predicted, observed, rate_by_bin, stages, suptitle, footnote):
    legend_cm, title_cm = 1.5, 0.7
    with plt.rc_context(_rc()):
        fig, axes = _axes(2, extra_cm=legend_cm + title_cm + 0.6)

        # ---- (0,0) the imbalance that motivates all of this --------------
        ax = axes[0][0]
        frac = (events.groupby('stage2', observed=True).still_bin
                .value_counts(normalize=True).rename('p').reset_index())
        width = 0.38
        for k, stage in enumerate(stages):
            s = frac[frac.stage2 == stage].set_index('still_bin').p
            ax.bar(np.arange(len(BIN_LABELS)) + (k - 0.5) * width,
                   [100 * float(s.get(b, 0.0)) for b in BIN_LABELS],
                   width=width, color=rip.STAGE_COLOUR.get(
                       stage, rip.STAGE_COLOUR['once known']),
                   label=stage, lw=0)
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, fontsize=rip.FS - 3, rotation=30,
                           ha='right', rotation_mode='anchor')
        ax.set_ylabel('Events (%)', fontsize=rip.FS)
        ax.set_title('Stillness is not balanced', fontsize=rip.FS, pad=3)

        # ---- (0,1) the rate-vs-stillness curve on these events -----------
        ax = axes[0][1]
        ax.axhline(0, color='0.45', lw=0.8)
        xs, ys, es = [], [], []
        for i, b in enumerate(BIN_LABELS):
            if b not in rate_by_bin:
                continue
            v = np.array([x for x in rate_by_bin[b].values() if np.isfinite(x)])
            if not v.size:
                continue
            xs.append(i)
            ys.append(v.mean())
            es.append(v.std() / max(np.sqrt(v.size), 1))
        ax.errorbar(xs, ys, yerr=es, color='#252525', lw=rip.LW, marker='o',
                    ms=4, capsize=2, elinewidth=1.0)
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, fontsize=rip.FS - 3, rotation=30,
                           ha='right', rotation_mode='anchor')
        ax.set_ylabel('Δ rate %.1f–%.1f s\n(Hz, both stages)' % PRIMARY_WINDOW,
                      fontsize=rip.FS)
        ax.set_title('Rate rises with stillness', fontsize=rip.FS, pad=3)

        # ---- (0,2) observed vs what stillness alone predicts -------------
        ax = axes[0][2]
        ax.axhline(0, color='0.45', lw=0.8)
        for i, (name, v, colour) in enumerate(
                (('observed', observed, rip.STAGE_COLOUR['first uncovers']),
                 ('predicted by\nstillness alone', predicted, '#252525'))):
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            ax.errorbar([i], [v.mean()],
                        yerr=[v.std() / max(np.sqrt(v.size), 1)], color=colour,
                        marker='o', ms=5, capsize=3, elinewidth=1.4, lw=0)
            ax.annotate(f"n={v.size}", xy=(i, 0.02),
                        xycoords=('data', 'axes fraction'), ha='center',
                        va='bottom', fontsize=rip.FS - 3, color='0.45')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['observed', 'predicted by\nstillness alone'],
                           fontsize=rip.FS - 3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel('%s − %s\n(Hz, %.1f–%.1f s)'
                      % (stages[0], stages[1], *PRIMARY_WINDOW),
                      fontsize=rip.FS - 1)
        ax.set_title('How much is stillness?', fontsize=rip.FS, pad=3)

        # ---- (1,0) the contrast inside each stillness bin ----------------
        ax = axes[1][0]
        ax.axvspan(*rip.BASELINE_WIN, color='0.88', lw=0, zorder=0)
        ax.axhline(0, color='0.45', lw=0.8)
        for colour, b in zip(BIN_COLOURS, BIN_LABELS):
            if b not in profiles:
                continue
            units, X = rip.baseline_subtract(profiles[b], centres)
            ax.plot(centres, np.nanmean(X, axis=0), color=colour,
                    lw=rip.LW_RATE, zorder=3, label=f'{b} (n={len(units)})')
        ax.axvline(0, color='0.35', lw=1.0)
        ax.set_xlabel('Time from uncovering (s)', fontsize=rip.FS, labelpad=1)
        ax.set_ylabel('%s − %s\n(Hz, vs own baseline)' % stages,
                      fontsize=rip.FS - 1)
        ax.set_title('Within each stillness bin', fontsize=rip.FS, pad=3)

        # ---- (1,1) the effect per bin, against the raw value -------------
        ax = axes[1][1]
        ax.axhline(0, color='0.45', lw=0.8)
        if 'all' in profiles:
            _, v = window_value(profiles['all'], centres)
            v = v[np.isfinite(v)]
            ax.axhline(v.mean(), color=rip.STAGE_COLOUR['first uncovers'],
                       lw=1.2, ls='--')
            ax.annotate('unstratified', xy=(0.98, v.mean()),
                        xycoords=('axes fraction', 'data'), ha='right',
                        va='bottom', fontsize=rip.FS - 3,
                        color=rip.STAGE_COLOUR['first uncovers'])
        for i, (colour, b) in enumerate(zip(BIN_COLOURS, BIN_LABELS)):
            if b not in profiles:
                continue
            _, v = window_value(profiles[b], centres)
            v = v[np.isfinite(v)]
            ax.errorbar([i], [v.mean()],
                        yerr=[v.std() / max(np.sqrt(v.size), 1)], color=colour,
                        marker='o', ms=5, capsize=3, elinewidth=1.4, lw=0)
            ax.annotate(f"{v.size}", xy=(i, 0.02),
                        xycoords=('data', 'axes fraction'), ha='center',
                        va='bottom', fontsize=rip.FS - 3, color='0.45')
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, fontsize=rip.FS - 3, rotation=30,
                           ha='right', rotation_mode='anchor')
        ax.set_xlim(-0.5, len(BIN_LABELS) - 0.5)
        ax.set_ylabel('Δ rate %.1f–%.1f s (Hz)' % PRIMARY_WINDOW,
                      fontsize=rip.FS)
        ax.set_title('Effect per bin (n below)', fontsize=rip.FS, pad=3)

        # ---- (1,2) the standardised contrast, sliding test ---------------
        ax = axes[1][2]
        ax.axhline(0, color='0.45', lw=0.8)
        res = sliding
        if res is not None:
            ax.plot(res['times'], res['t'], color='#252525', lw=1.6)
            for cl in res['clusters']:
                if cl['p'] < 0.05:
                    ax.axvspan(cl['start_s'], cl['stop_s'], color='#252525',
                               alpha=0.16, lw=0)
                    ax.annotate(f"p={cl['p']:.3f}", xy=(cl['peak_s'], 0.97),
                                xycoords=('data', 'axes fraction'),
                                ha='center', va='top', fontsize=rip.FS - 3)
            for sign in (1, -1):
                ax.axhline(sign * res['threshold'], color='0.6', lw=0.7, ls=':')
        ax.axvline(0, color='0.35', lw=1.0)
        ax.set_xlabel('Sliding window centre (s)', fontsize=rip.FS, labelpad=1)
        ax.set_ylabel('t, stillness-matched\ncontrast vs 0', fontsize=rip.FS - 1)
        ax.set_title('Matched, cluster test', fontsize=rip.FS, pad=3)

        handles, labels = [], []
        for ax in (axes[0][0], axes[1][0]):
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        total = rip.ROW_H_CM * 2 + legend_cm + title_cm + 0.6
        fig.tight_layout(pad=0.4, h_pad=2.0, w_pad=1.8,
                         rect=[0, legend_cm / total, 1, 1 - title_cm / total])
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.035), ncol=4, fontsize=rip.FS - 2,
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


# ── 4) Main ───────────────────────────────────────────────────────────

def run(bundle=None, out_dir=None, valence='correct', unit='session',
        min_events=10, n_perm=rip.N_SIGN_FLIPS, stillness_cache=None):
    """`valence` selects which feedback the stage contrast is run on."""
    root = swr_io.get_data_root()
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                              'bundle')
    if out_dir is None:
        out_dir = os.path.join(
            swr_io.derivatives_dir(root), 'group', 'swr',
            f"stillness_control_{valence}_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_stillness_control')
    np.random.seed(SEED)

    data = rip.load_bundle(bundle)
    print(f"\n  bundle: {bundle}")
    print(f"  {QUESTION}")
    print(f"  contrast: {valence} feedback, first uncovers - later "
          f"| unit = {unit}, min {min_events} events\n")

    if stillness_cache is None:
        stillness_cache = os.path.join(out_dir, 'events_with_stillness.csv')
    events = bin_events(event_table(data, cache_path=stillness_cache))
    events = events[events.valence == valence]
    events = events[events.still_bin.notna()]
    print(f"  {len(events)} {valence} uncoverings with a stillness value")
    share = (events.groupby('stage2', observed=True).still_bin
             .value_counts(normalize=True).unstack().round(3))
    print("\n  fraction of events per stillness bin")
    print(share.to_string())

    print("\n  contrast per stillness bin")
    centres, profiles, counts = contrast_per_bin(data, events, unit,
                                                 min_events)
    if centres is None or 'all' not in profiles:
        print("  nothing computable")
        return None

    results = {'question': QUESTION, 'bundle': bundle, 'valence': valence,
               'unit': unit, 'min_events': min_events,
               'bin_edges_s': [float(e) for e in BIN_EDGES],
               'primary_window_s': list(PRIMARY_WINDOW),
               'baseline_window_s': list(rip.BASELINE_WIN),
               'n_sign_flips': n_perm, 'seed': SEED,
               'stillness_share': share.to_dict(), 'bins': {}}

    print(f"\n  {'stratum':12s} {'n':>4s} {'mean Hz':>9s} {'t':>7s} "
          f"{'p_perm':>8s}   window")
    for label in ('all',) + tuple(BIN_LABELS):
        if label not in profiles:
            continue
        entry = {'counts': counts[label], 'windows': {}}
        for wname, win in NAMED_WINDOWS.items():
            got = rip.window_test(profiles[label], centres, win)
            if got is None:
                continue
            entry['windows'][wname] = got
            if win == tuple(PRIMARY_WINDOW) or wname.startswith('post (0..'):
                star = ' *' if got['p_perm'] < 0.05 else ''
                print(f"  {label:12s} {got['n_subjects']:4d} "
                      f"{got['mean_hz']:+9.4f} {got['t']:+7.2f} "
                      f"{got['p_perm']:8.4f}   {wname}{star}")
        results['bins'][label] = entry

    # ---- the stillness-standardised contrast -----------------------------
    std = standardised_contrast(profiles)
    print(f"\n  stillness-standardised (per-session mean over bins, "
          f"n = {len(std)})")
    results['standardised'] = {'n_units': len(std), 'windows': {}}
    for wname, win in NAMED_WINDOWS.items():
        got = rip.window_test(std, centres, win)
        if got is None:
            continue
        results['standardised']['windows'][wname] = got
        star = ' *' if got['p_perm'] < 0.05 else ''
        print(f"    {wname:20s} n={got['n_subjects']:3d} "
              f"mean={got['mean_hz']:+.4f} t={got['t']:+6.2f} "
              f"p_perm={got['p_perm']:.4g}{star}")
    units, X = rip.baseline_subtract(std, centres)
    sliding = rip.sliding_window_test(
        {'standardised': {u: X[i] for i, u in enumerate(units)}}, centres,
        width_s=rip.SLIDE_WIDTHS_S[0], n_perm=n_perm, seed=SEED)
    sliding = sliding.get('standardised')
    keep = [c for c in sliding['clusters'] if c['p'] < 0.05] if sliding else []
    print(f"    sliding {rip.SLIDE_WIDTHS_S[0]:g}s        " +
          ('; '.join(f"{c['direction']} {c['start_s']:+.2f}..{c['stop_s']:+.2f} "
                     f"p={c['p']:.4f}" for c in keep) if keep else 'none'))
    results['standardised']['sliding'] = {
        'clusters': keep, 'width_s': rip.SLIDE_WIDTHS_S[0]} if sliding else None

    # ---- how much of it stillness alone predicts -------------------------
    print("\n  what the stillness imbalance alone predicts")
    pred, rate_by_bin = predicted_from_stillness(data, events, unit,
                                                 min_events, centres)
    obs_units, obs_vals = window_value(profiles['all'], centres)
    observed = pd.DataFrame({'session': obs_units, 'observed_hz': obs_vals})
    joined = observed.merge(pred, on='session', how='inner').dropna()
    joined.to_csv(os.path.join(out_dir, 'observed_vs_predicted.csv'),
                  index=False)
    if len(joined) >= 3:
        o, p_ = joined.observed_hz.to_numpy(), joined.predicted_hz.to_numpy()
        t_o, _ = stats.ttest_1samp(o, 0.0)
        t_p, _ = stats.ttest_1samp(p_, 0.0)
        t_d, p_d = stats.ttest_rel(o, p_)
        share_explained = float(p_.mean() / o.mean()) if o.mean() else np.nan
        print(f"    observed            {o.mean():+.4f} Hz (t = {t_o:+.2f}, "
              f"n = {len(o)})")
        print(f"    predicted by stillness {p_.mean():+.4f} Hz "
              f"(t = {t_p:+.2f})")
        print(f"    observed - predicted   {(o - p_).mean():+.4f} Hz "
              f"(t = {t_d:+.2f}, p = {p_d:.4g})")
        print(f"    stillness accounts for {100 * share_explained:.0f}% "
              f"of the observed difference")
        results['decomposition'] = {
            'n_units': len(o), 'observed_hz': float(o.mean()),
            'predicted_hz': float(p_.mean()),
            'residual_hz': float((o - p_).mean()),
            'residual_t': float(t_d), 'residual_p': float(p_d),
            'fraction_explained': share_explained}
    else:
        results['decomposition'] = None

    for label, per_unit in rate_by_bin.items():
        results.setdefault('rate_by_stillness_bin', {})[label] = {
            'n_units': len(per_unit),
            'mean_hz': float(np.nanmean(list(per_unit.values())))}

    figure(os.path.join(out_dir, f'stillness_control_{valence}.png'), events,
           centres, profiles, counts, std, sliding,
           joined.predicted_hz.to_numpy() if len(joined) else np.array([]),
           joined.observed_hz.to_numpy() if len(joined) else np.array([]),
           rate_by_bin, STAGES2,
           f'{valence.capitalize()} feedback, first traversal vs later: '
           f'is it stillness?',
           f'Stillness = time to the next key press of any kind. Bins applied '
           f'to both stages alike; unit = {unit}, min {min_events} events per '
           f'{unit} per cell per bin; {n_perm} sign-flips.')

    with open(os.path.join(out_dir, 'stillness_control_result.json'), 'w') as f:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   **results}, f, indent=2, default=str)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
