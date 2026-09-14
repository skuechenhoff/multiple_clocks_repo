#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do ripples at a correct uncovering predict how few mistakes follow?

`swr_valence_stage_interaction.py` asks whether ripple rate DIFFERS between
conditions. This asks something stronger and behavioural: within a subject,
does a grid on which the hippocampus rippled more during discovery turn out to
be a grid the subject then solves with fewer wrong uncoverings?

Four predictors, exactly the comparison SK asked for:

    all            ripple rate at every correct uncovering in the grid
    first uncovers ripple rate at the correct uncoverings of the FIRST
                   traversal -- the only strictly predictive one, since the
                   whole outcome still lies in the future
    while learning correct uncoverings from repeat 2 up to and including the
                   first error-free repeat
    once known     correct uncoverings after the route is demonstrated

and two outcomes per predictor:

    errors_total   every wrong uncovering in the grid. The quantity asked
                   for, and the only one defined for all four predictors --
                   but it is strictly predictive only for `first uncovers`,
                   and concurrent or retrospective for the later stages.
    errors_after   wrong uncoverings made AFTER the last event that went into
                   the predictor. Nothing in the outcome then precedes the
                   ripples. This is the clean test for `first uncovers`; for
                   `once known` and `all` it is near-degenerate by
                   construction (almost no errors follow the grid's last
                   correct uncovering), and the printed n says so.

Unit of analysis: the GRID, correlated WITHIN a session, then the per-session
correlations tested across sessions. Correlating across sessions instead would
compare patients, electrodes and anatomy, none of which this question is about.
Spearman, because error counts are skewed and bounded below by zero.

The null is built by shuffling the outcome across grids WITHIN each session and
re-running the identical function, so the permuted statistic is produced by the
same code path as the empirical one.

Control. F1 says ripple rate rises with stillness, and a subject who pauses
longer on a reward may well learn better for reasons that have nothing to do
with ripples. The same analysis is therefore run with STILLNESS as the
predictor; if stillness predicts errors as well as ripples do, the ripple
result carries no independent information. Ripple rate is additionally tested
after stillness is partialled out within session.

    python scripts/swr_ripples_predict_errors.py --bundle=<bundle dir>

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import warnings
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
from swr_valence_stage_interaction import (event_table, stage_colour,
                                           _rc, _axes, _stage_ticks)

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


# ── Settings ──────────────────────────────────────────────────────────
QUESTION = ('Does a higher ripple rate at correct uncoverings predict fewer '
            'wrong uncoverings afterwards, and is that specific to the first '
            'traversal?')

EVENT_WINDOW = (0.0, 0.5)        # the window the rate is read from
PREDICTORS = ('all', 'first uncovers', 'while learning', 'once known')
OUTCOMES = ('errors_total', 'errors_after')
MIN_GRIDS = 6                    # a session needs this many grids to correlate
MIN_EVENTS_PER_GRID = 1          # a grid needs at least one usable event
N_PERM = 1000
SEED = 42

OUTCOME_LABEL = {'errors_after': 'wrong uncoverings AFTER the predictor window',
                 'errors_total': 'wrong uncoverings in the whole grid'}
OUTCOME_AXIS = {'errors_after': 'Later wrong uncoverings',
                'errors_total': 'Wrong uncoverings in the grid'}
ALL_COLOUR = '#0e3d3a'           # the pooled predictor; stages keep STAGE_COLOUR


# ── 1) One row per grid ───────────────────────────────────────────────

def grid_table(data, events, window=EVENT_WINDOW, baseline=rip.BASELINE_WIN):
    """Per grid and predictor: ripple rate, stillness, and the error counts.

    The ripple rate of one event is its rate in `window` minus its rate in the
    same trial's baseline, averaged over the session's derivations -- the same
    quantity `window_test` tests, read one event at a time so it can be
    attributed to a grid. `rate_raw_hz` keeps the uncorrected window rate.

    `errors_after` counts wrong uncoverings later in the SAME grid than the
    last event that went into the predictor, so the outcome never contains
    anything that preceded the ripples.
    """
    rows = []
    for session in rip.sessions_in(data):
        ev = events[events.session == session]
        if not len(ev):
            continue
        derivs = rip.derivations(data, session)
        if not derivs:
            continue
        t_all = ev.t_s.to_numpy(float)
        # one pass per derivation over ALL of the session's events, then
        # averaged -- the per-event value is the session's, not an electrode's
        win_rate, base_rate = [], []
        for _, ripples, intervals in derivs:
            r_w, _, _ = rip.rate_in_window(t_all, ripples, intervals, window)
            r_b, _, _ = rip.rate_in_window(t_all, ripples, intervals, baseline)
            win_rate.append(r_w)
            base_rate.append(r_b)
        # An event too artifact-contaminated on EVERY derivation averages an
        # all-NaN column; that is the intended NaN, not a problem to report.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            win_rate = np.nanmean(np.vstack(win_rate), axis=0)
            base_rate = np.nanmean(np.vstack(base_rate), axis=0)
        ev = ev.assign(rate_raw_hz=win_rate, rate_hz=win_rate - base_rate)

        for grid, g in ev.groupby('grid_no'):
            correct = g[g.valence == 'correct']
            error = g[g.valence == 'error']
            for predictor in PREDICTORS:
                sel = correct if predictor == 'all' else \
                    correct[correct.stage == predictor]
                sel = sel[np.isfinite(sel.rate_hz)]
                if len(sel) < MIN_EVENTS_PER_GRID:
                    continue
                t_last = float(sel.t_s.max())
                rows.append(dict(
                    session=int(session), grid_no=int(grid),
                    subject=rip.subject_of(data, session),
                    predictor=predictor, n_events=int(len(sel)),
                    rate_hz=float(sel.rate_hz.mean()),
                    rate_raw_hz=float(sel.rate_raw_hz.mean()),
                    still_next_s=float(sel.still_next_s.mean()),
                    errors_total=int(len(error)),
                    errors_after=int((error.t_s > t_last).sum()),
                    n_repeats=int(g.rep_overall.nunique()),
                    last_event_t_s=t_last))
    return pd.DataFrame(rows)


# ── 2) The statistic, used for the data and for every permutation ─────
# `prepare` pulls the per-session arrays out of the table ONCE. Everything
# after it is numpy, and `group_statistic` is the single code path that turns
# those arrays into a number -- the empirical value and all 1000 permuted
# values come out of the same call, the permutation only reordering `y`.

def _resid(x, z):
    """x with z regressed out; both 1-D and finite."""
    zc = z - z.mean()
    var = float(zc @ zc)
    if var == 0:
        return x - x.mean()
    return (x - x.mean()) - (float(zc @ (x - x.mean())) / var) * zc


def prepare(table, predictor, outcome, x_col='rate_hz', partial_col=None,
            min_grids=MIN_GRIDS):
    """[(session, x, y, z)] -- one entry per session with enough usable grids.

    Rows with a non-finite value in any column the test needs are dropped here
    rather than inside the statistic, so the permutation cannot change which
    grids take part.
    """
    sub = table[table.predictor == predictor]
    cols = [x_col, outcome] + ([partial_col] if partial_col else [])
    out = []
    for session, g in sub.groupby('session'):
        g = g[np.isfinite(g[cols]).all(axis=1)]
        if len(g) < min_grids:
            continue
        out.append((int(session), g[x_col].to_numpy(float),
                    g[outcome].to_numpy(float),
                    g[partial_col].to_numpy(float) if partial_col else None))
    return out


def permute(prepared, rng):
    """Shuffle the outcome across grids WITHIN each session.

    The null the question needs: every session keeps its grids, its ripple
    rates and its error counts, and only the pairing between them is destroyed.
    Shuffling across sessions would instead ask whether sessions differ, which
    is not the claim.
    """
    return [(s, x, rng.permutation(y), z) for s, x, y, z in prepared]


def session_correlations(prepared):
    """Spearman rho per session, with `z` partialled out of both ranks if given."""
    out = []
    for session, x, y, z in prepared:
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        if rx.std() == 0 or ry.std() == 0:
            continue
        if z is not None:
            rz = stats.rankdata(z)
            if rz.std() == 0:
                continue
            rx, ry = _resid(rx, rz), _resid(ry, rz)
            if rx.std() == 0 or ry.std() == 0:
                continue
        out.append((session, float(np.corrcoef(rx, ry)[0, 1]), int(x.size)))
    return out


def group_statistic(prepared):
    """Fisher-z the per-session rhos, then one t across sessions."""
    per = session_correlations(prepared)
    if len(per) < 3:
        return None
    rhos = np.array([r for _, r, _ in per], float)
    z = np.arctanh(np.clip(rhos, -0.999999, 0.999999))
    t, p = stats.ttest_1samp(z, 0.0)
    return {'n_sessions': len(per), 'n_grids': int(sum(n for _, _, n in per)),
            'mean_rho': float(rhos.mean()), 'median_rho': float(np.median(rhos)),
            'mean_z': float(z.mean()), 't': float(t), 'p': float(p),
            'per_session': [{'session': s, 'rho': r, 'n_grids': n}
                            for s, r, n in per]}


def test_predictor(table, predictor, outcome, x_col='rate_hz',
                   partial_col=None, n_perm=N_PERM, seed=SEED,
                   min_grids=MIN_GRIDS):
    prepared = prepare(table, predictor, outcome, x_col=x_col,
                       partial_col=partial_col, min_grids=min_grids)
    got = group_statistic(prepared)
    if got is None:
        return None
    rng = np.random.default_rng(seed)
    null = [g['t'] for g in (group_statistic(permute(prepared, rng))
                             for _ in range(n_perm)) if g is not None]
    null = np.asarray(null, float)
    got['n_perm'] = int(null.size)
    got['p_perm'] = float((1 + np.sum(np.abs(null) >= abs(got['t'])))
                          / (1 + null.size)) if null.size else np.nan
    got['null_t_sd'] = float(null.std()) if null.size else np.nan
    return got


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running = np.empty(p.size), 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def contrast_first_vs_later(table, outcome, x_col='rate_hz',
                            min_grids=MIN_GRIDS, n_perm=N_PERM, seed=SEED):
    """Is the first traversal's correlation LARGER than the later stages'?

    "Predictive, and only at the start" is a claim about a DIFFERENCE between
    stages, so it needs its own paired test rather than one significant stage
    standing beside two non-significant ones.

    The null shuffles the outcome within session for each stage independently,
    so it is the null of no relation anywhere rather than of equal relations.
    """
    stages = ('first uncovers', 'while learning', 'once known')
    prepared = {s_: prepare(table, s_, outcome, x_col=x_col,
                            min_grids=min_grids) for s_ in stages}

    def statistic(prep):
        rho = {s_: dict((ss, r) for ss, r, _ in session_correlations(prep[s_]))
               for s_ in stages}
        shared = sorted(set(rho[stages[0]]) & set(rho[stages[1]])
                        & set(rho[stages[2]]))
        if len(shared) < 3:
            return None, None
        clip = lambda v: np.arctanh(np.clip(v, -0.999999, 0.999999))
        za = clip([rho[stages[0]][s_] for s_ in shared])
        zb = clip([(rho[stages[1]][s_] + rho[stages[2]][s_]) / 2
                   for s_ in shared])
        return stats.ttest_rel(za, zb), za - zb

    res, diff = statistic(prepared)
    if res is None:
        return None
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        r, _ = statistic({s_: permute(p_, rng) for s_, p_ in prepared.items()})
        if r is not None:
            null.append(r.statistic)
    null = np.asarray(null, float)
    return {'n_sessions': int(diff.size), 'mean_z_difference': float(diff.mean()),
            't': float(res.statistic), 'p': float(res.pvalue),
            'p_perm': float((1 + np.sum(np.abs(null) >= abs(res.statistic)))
                            / (1 + null.size)) if null.size else np.nan,
            'contrast': 'first uncovers  vs  mean(while learning, once known)'}


# ── 3) Figure ─────────────────────────────────────────────────────────

def figure_prediction(results, table, out_png, outcome, suptitle, footnote):
    """Per-session rhos by predictor | the effect sizes | the grids themselves."""
    legend_cm, title_cm = 1.4, 0.7
    with plt.rc_context(_rc()):
        fig, axes = _axes(1, extra_cm=legend_cm + title_cm + 0.5)

        # ---- left: every session's rho, per predictor -------------------
        ax = axes[0][0]
        ax.axhline(0, color='0.45', lw=0.8)
        rng = np.random.default_rng(SEED)
        for i, predictor in enumerate(PREDICTORS):
            got = results['ripples'][outcome].get(predictor)
            if not got:
                continue
            rhos = np.array([d['rho'] for d in got['per_session']], float)
            colour = (ALL_COLOUR if predictor == 'all'
                      else stage_colour(predictor))
            ax.scatter(i + rng.uniform(-0.16, 0.16, rhos.size), rhos, s=7,
                       color=colour, alpha=0.45, lw=0, zorder=2)
            ax.errorbar([i], [rhos.mean()],
                        yerr=[rhos.std() / max(np.sqrt(rhos.size), 1)],
                        color=colour, marker='o', ms=5, capsize=3,
                        elinewidth=1.4, lw=0, zorder=3)
            star = ('**' if got['p_perm'] < 0.01 else
                    '*' if got['p_perm'] < 0.05 else '')
            if star:
                ax.annotate(star, xy=(i, 0.97), xycoords=('data', 'axes fraction'),
                            ha='center', va='top', fontsize=rip.FS, color=colour)
        _stage_ticks(ax, PREDICTORS)
        ax.set_ylabel('Spearman ρ per session\nripple rate vs errors',
                      fontsize=rip.FS)
        ax.set_title('Every session', fontsize=rip.FS, pad=3)

        # ---- middle: the group effect, ripples vs stillness -------------
        ax = axes[0][1]
        ax.axhline(0, color='0.45', lw=0.8)
        for which, colour, marker, label in (
                ('ripples', ALL_COLOUR, 'o', 'ripple rate'),
                ('stillness', rip.VALENCE_COLOUR['error'], 's', 'stillness')):
            xs, ys, es = [], [], []
            for i, predictor in enumerate(PREDICTORS):
                got = results[which][outcome].get(predictor)
                if not got:
                    continue
                z = np.arctanh(np.clip([d['rho'] for d in got['per_session']],
                                       -0.999999, 0.999999))
                xs.append(i)
                ys.append(z.mean())
                es.append(z.std() / max(np.sqrt(z.size), 1))
            ax.errorbar(xs, ys, yerr=es, color=colour, marker=marker, ms=4,
                        lw=rip.LW, capsize=2, elinewidth=1.0, label=label)
        _stage_ticks(ax, PREDICTORS)
        ax.set_ylabel('Fisher z (mean ± SEM)', fontsize=rip.FS)
        ax.set_title('vs stillness control', fontsize=rip.FS, pad=3)

        # ---- right: the grids themselves, first traversal ---------------
        ax = axes[0][2]
        sub = table[table.predictor == 'first uncovers'].copy()
        sub = sub[np.isfinite(sub[['rate_hz', outcome]]).all(axis=1)]
        # z-scored WITHIN session, because the correlation is within session:
        # raw values would show between-patient spread the test never sees.
        for col in ('rate_hz', outcome):
            sub[col + '_z'] = sub.groupby('session')[col].transform(
                lambda v: (v - v.mean()) / (v.std() if v.std() else np.nan))
        sub = sub[np.isfinite(sub[['rate_hz_z', outcome + '_z']]).all(axis=1)]
        ax.axhline(0, color='0.85', lw=0.6)
        ax.axvline(0, color='0.85', lw=0.6)
        ax.scatter(sub.rate_hz_z, sub[outcome + '_z'], s=4,
                   color=stage_colour('first uncovers'), alpha=0.25, lw=0)
        if len(sub) > 10:
            b = np.polyfit(sub.rate_hz_z, sub[outcome + '_z'], 1)
            xx = np.linspace(sub.rate_hz_z.min(), sub.rate_hz_z.max(), 20)
            ax.plot(xx, np.polyval(b, xx), color='0.15', lw=1.4)
        got = results['ripples'][outcome].get('first uncovers')
        if got:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.22 * (hi - lo))
            ax.annotate(f"ρ = {got['mean_rho']:+.3f}, p = {got['p_perm']:.3f}\n"
                        f"{got['n_grids']} grids, {got['n_sessions']} sessions",
                        xy=(0.03, 0.98), xycoords='axes fraction', va='top',
                        ha='left', fontsize=rip.FS - 3, color='0.25')
        ax.set_xlabel('Ripple rate, first traversal\n(z within session)',
                      fontsize=rip.FS, labelpad=1)
        ax.set_ylabel(f'{OUTCOME_AXIS[outcome]}\n(z within session)',
                      fontsize=rip.FS)
        ax.set_title('Grid by grid', fontsize=rip.FS, pad=3)

        total = rip.ROW_H_CM + legend_cm + title_cm + 0.5
        fig.tight_layout(pad=0.4, h_pad=1.6, w_pad=2.0,
                         rect=[0, legend_cm / total, 1, 1 - title_cm / total])
        h, l = axes[0][1].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.04),
                   ncol=2, fontsize=rip.FS - 2, frameon=False,
                   handlelength=1.4, handletextpad=0.4, borderaxespad=0.0)
        fig.suptitle(suptitle, fontsize=rip.FS, y=0.998, va='top')
        fig.text(0.5, 0.002, footnote, ha='center', va='bottom',
                 fontsize=rip.FS - 3, color='0.4')
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.splitext(out_png)[0] + '.pdf', bbox_inches='tight')
        plt.close(fig)
    return out_png


# ── 4) Main ───────────────────────────────────────────────────────────

def run(bundle=None, out_dir=None, n_perm=N_PERM, min_grids=MIN_GRIDS,
        stillness_cache=None):
    root = swr_io.get_data_root()
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                              'bundle')
    if out_dir is None:
        out_dir = os.path.join(
            swr_io.derivatives_dir(root), 'group', 'swr',
            f"ripples_predict_errors_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_ripples_predict_errors')
    np.random.seed(SEED)

    data = rip.load_bundle(bundle)
    print(f"\n  bundle: {bundle}")
    print(f"  {QUESTION}\n")

    if stillness_cache is None:
        stillness_cache = os.path.join(out_dir, 'events_with_stillness.csv')
    events = event_table(data, cache_path=stillness_cache)
    table = grid_table(data, events)
    table.to_csv(os.path.join(out_dir, 'grid_table.csv'), index=False)
    n_grids = table[['session', 'grid_no']].drop_duplicates().shape[0]
    print(f"  {n_grids} grids x {len(PREDICTORS)} predictors "
          f"-> {len(table)} rows, {table.session.nunique()} sessions")
    print(f"  grids per session: median "
          f"{table[table.predictor == 'all'].groupby('session').size().median():.0f}")

    results = {'question': QUESTION, 'bundle': bundle,
               'event_window_s': list(EVENT_WINDOW),
               'baseline_window_s': list(rip.BASELINE_WIN),
               'min_grids_per_session': min_grids, 'n_perm': n_perm,
               'seed': SEED, 'unit': 'grid, correlated within session',
               'ripples': {}, 'stillness': {}, 'ripples_partial_stillness': {},
               'first_vs_later': {}}
    rows = []

    for outcome in OUTCOMES:
        print(f"\n{'=' * 78}\n outcome: {OUTCOME_LABEL[outcome]}\n{'=' * 78}")
        for which, x_col, partial in (('ripples', 'rate_hz', None),
                                      ('stillness', 'still_next_s', None),
                                      ('ripples_partial_stillness', 'rate_hz',
                                       'still_next_s')):
            results[which].setdefault(outcome, {})
            got_all = {}
            for predictor in PREDICTORS:
                got = test_predictor(table, predictor, outcome, x_col=x_col,
                                     partial_col=partial, n_perm=n_perm,
                                     seed=SEED, min_grids=min_grids)
                if got:
                    got_all[predictor] = got
            if not got_all:
                continue
            adj = holm([g['p_perm'] for g in got_all.values()])
            for g, a in zip(got_all.values(), adj):
                g['p_perm_holm'] = float(a)
                g['predictor_family'] = len(got_all)
            results[which][outcome] = got_all
            print(f"\n  {which}")
            print(f"    {'predictor':16s} {'sess':>5s} {'grids':>6s} "
                  f"{'mean rho':>9s} {'t':>7s} {'p_perm':>8s} {'p_holm':>8s}")
            for predictor, g in got_all.items():
                star = ' *' if g['p_perm_holm'] < 0.05 else ''
                print(f"    {predictor:16s} {g['n_sessions']:5d} "
                      f"{g['n_grids']:6d} {g['mean_rho']:+9.4f} {g['t']:+7.2f} "
                      f"{g['p_perm']:8.4f} {g['p_perm_holm']:8.4f}{star}")
                rows.append(dict(outcome=outcome, measure=which,
                                 predictor=predictor, **{
                                     k: v for k, v in g.items()
                                     if k != 'per_session'}))

        contrast = contrast_first_vs_later(table, outcome, n_perm=n_perm,
                                           seed=SEED, min_grids=min_grids)
        results['first_vs_later'][outcome] = contrast
        if contrast:
            print(f"\n    first traversal vs the two later stages: "
                  f"Δz = {contrast['mean_z_difference']:+.4f}, "
                  f"t({contrast['n_sessions'] - 1}) = {contrast['t']:+.2f}, "
                  f"p_perm = {contrast['p_perm']:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'prediction_summary.csv'),
                              index=False)

    for outcome in OUTCOMES:
        if not results['ripples'].get(outcome):
            continue
        figure_prediction(
            results, table,
            os.path.join(out_dir, f'ripples_predict_{outcome}.png'), outcome,
            f'Ripples at correct uncoverings vs {OUTCOME_LABEL[outcome]}',
            f'Spearman within session over grids, Fisher-z, t across sessions; '
            f'{n_perm} within-session shuffles of the outcome. '
            f'Rate = {EVENT_WINDOW[0]:.1f}–{EVENT_WINDOW[1]:.1f} s minus the '
            f'same trial\'s baseline. Sessions need ≥ {min_grids} grids.')

    with open(os.path.join(out_dir, 'predict_errors_result.json'), 'w') as f:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   **results}, f, indent=2, default=str)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
