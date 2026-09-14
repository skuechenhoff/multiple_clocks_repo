#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLORATORY sweep: does the ripple response to an uncovering predict mistakes?

Nothing here is confirmatory. Figures are .png only, no publication styling; the
point is to see the shape of the data before deciding what is worth testing
properly.

Four questions, in order:

  1  D-specific.  The first time D is uncovered in a grid, does the ripple
     response predict whether the subject then gets D wrong -- ever again in
     that grid, or only in the next few repeats?

  2  General.  Averaged over the first uncovering of all four rewards, does the
     ripple response predict how many wrong uncoverings follow -- over all
     remaining repeats, or only the next few?

  3  Do people repeat the SAME mistake?  Confusion matrices: which location is
     uncovered in error while each reward is sought, and whether consecutive
     errors for the same reward land on the same location.

  4  Mistake specificity.  At the moment of a wrong uncovering, does the ripple
     response predict whether that EXACT mistake (same reward, same location)
     is made again?

Two measures throughout, because SK wanted both:
    delta  rate in the test window minus the same trial's baseline
    raw    rate in the test window alone, no baseline

The test window is +0.35 to +0.75 s, where the stillness-matched cluster peaked.

Tests are paired (matched) t-tests at the session level: within each session,
the mean ripple measure over grids/events WITH the outcome is compared against
the mean over those WITHOUT, and the per-session differences are tested across
sessions. That keeps every comparison within subject.

    python scripts/swr_explore_ripples_and_mistakes.py

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

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


TEST_WIN = (0.35, 0.75)      # where the matched-control cluster peaked
BASE_WIN = rip.BASELINE_WIN  # (-1.6, -1.1)
NEXT_FEW = 3                 # "the next few repeats"
MIN_PER_CELL = 3             # a session needs this many grids/events per side
SEED = 42
REWARDS = ('A', 'B', 'C', 'D')

C_YES, C_NO = '#B03A5B', '#0E3D3A'   # outcome: mistake follows / does not


# ── 1) Events with their ripple response ──────────────────────────────

def event_table(data):
    """Every uncovering with its ripple response, location and task labels.

    `delta` is the test window minus the same trial's baseline; `raw` is the
    test window alone. Both are averaged over the session's derivations, so the
    value belongs to the session rather than to an electrode.
    """
    rows = []
    for session in rip.sessions_in(data):
        tab = rip.uncover_table(data, session)
        if not len(tab):
            continue
        unc = data['uncover']
        unc = unc[unc.session == session][['t_s', 'loc']].sort_values('t_s')
        tab = pd.merge_asof(tab.sort_values('t_s'), unc, on='t_s',
                            direction='nearest', tolerance=0.03)
        derivs = rip.derivations(data, session)
        if not derivs:
            continue
        t = tab.t_s.to_numpy(float)
        win, base = [], []
        for _, ripples, intervals in derivs:
            w, _, _ = rip.rate_in_window(t, ripples, intervals, TEST_WIN)
            b, _, _ = rip.rate_in_window(t, ripples, intervals, BASE_WIN)
            win.append(w)
            base.append(b)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            win = np.nanmean(np.vstack(win), axis=0)
            base = np.nanmean(np.vstack(base), axis=0)
        rows.append(tab.assign(session=int(session), raw=win, delta=win - base))
    return pd.concat(rows, ignore_index=True)


# ── 2) Outcomes ───────────────────────────────────────────────────────

def label_outcomes(ev, next_few=NEXT_FEW):
    """Attach, to each first-uncovering, whether mistakes follow.

    Two horizons, because "later" is ambiguous: `_all` looks at every remaining
    repeat of that grid, `_few` only at the next `next_few` repeats. A grid the
    subject never repeats contributes nothing, since there is no "later".
    """
    out = []
    for (session, grid), g in ev.groupby(['session', 'grid_no']):
        g = g.sort_values('t_s')
        errs = g[g.valence == 'error']
        last_rep = int(g.rep_overall.max())

        # -- Q1: the first correct D of the grid ------------------------
        d_first = g[(g.valence == 'correct') & (g.reward == 'D')
                    & (g.stage == 'first uncovers')]
        if len(d_first):
            r0 = int(d_first.rep_overall.iloc[0])
            if last_rep > r0:
                later_d = errs[(errs.rep_overall > r0) & (errs.reward == 'D')]
                few_d = later_d[later_d.rep_overall <= r0 + next_few]
                out.append(dict(
                    session=session, grid_no=grid, question='D-specific',
                    raw=float(d_first.raw.iloc[0]),
                    delta=float(d_first.delta.iloc[0]),
                    n_later_all=int(len(later_d)), n_later_few=int(len(few_d)),
                    err_all=int(len(later_d) > 0), err_few=int(len(few_d) > 0),
                    n_repeats_after=last_rep - r0))

        # -- Q2: all four first uncoverings, averaged -------------------
        first = g[(g.valence == 'correct') & (g.stage == 'first uncovers')]
        first = first[np.isfinite(first.delta)]
        if len(first) >= 2:
            r0 = int(first.rep_overall.max())
            if last_rep > r0:
                later = errs[errs.rep_overall > r0]
                few = later[later.rep_overall <= r0 + next_few]
                out.append(dict(
                    session=session, grid_no=grid, question='all first',
                    raw=float(first.raw.mean()), delta=float(first.delta.mean()),
                    n_later_all=int(len(later)), n_later_few=int(len(few)),
                    err_all=int(len(later) > 0), err_few=int(len(few) > 0),
                    n_repeats_after=last_rep - r0))
    return pd.DataFrame(out)


def repeated_mistakes(ev):
    """Every wrong uncovering, and whether that exact mistake recurs.

    "The same mistake" means the same grid, the same reward being sought and
    the same location uncovered. The outcome looks only forwards in time, so an
    error can predict its own repetition but never its own history.
    """
    rows = []
    for (session, grid), g in ev.groupby(['session', 'grid_no']):
        errs = g[(g.valence == 'error') & np.isfinite(g.loc_)].sort_values('t_s')
        for i, e in enumerate(errs.itertuples()):
            later = errs.iloc[i + 1:]
            same = later[(later.reward == e.reward) & (later.loc_ == e.loc_)]
            rows.append(dict(session=session, grid_no=grid, t_s=e.t_s,
                             reward=e.reward, location=int(e.loc_),
                             raw=e.raw, delta=e.delta,
                             repeated=int(len(same) > 0),
                             n_later_errors=int(len(later))))
    return pd.DataFrame(rows)


# ── 3) The paired test ────────────────────────────────────────────────

def paired_by_session(df, measure, outcome, min_per_cell=MIN_PER_CELL):
    """Within session: mean measure WITH the outcome vs WITHOUT, then paired t.

    A session contributes only if it has at least `min_per_cell` observations on
    each side, so a single grid cannot define a session's mean.
    """
    rows = []
    for session, g in df.groupby('session'):
        g = g[np.isfinite(g[measure])]
        yes, no = g[g[outcome] == 1], g[g[outcome] == 0]
        if len(yes) < min_per_cell or len(no) < min_per_cell:
            continue
        rows.append(dict(session=int(session), mean_yes=yes[measure].mean(),
                         mean_no=no[measure].mean(), n_yes=len(yes),
                         n_no=len(no)))
    per = pd.DataFrame(rows)
    if len(per) < 3:
        return per, None
    d = (per.mean_yes - per.mean_no).to_numpy(float)
    t, p = stats.ttest_1samp(d, 0.0)
    rng = np.random.default_rng(SEED)
    null = (rng.choice([-1.0, 1.0], size=(10000, d.size)) * d).mean(axis=1)
    sem = d.std(ddof=1) / np.sqrt(d.size)
    crit = stats.t.ppf(0.975, d.size - 1)
    return per, {'n_sessions': int(d.size), 'mean_yes': float(per.mean_yes.mean()),
                 'mean_no': float(per.mean_no.mean()),
                 'diff_hz': float(d.mean()), 'sem_hz': float(sem),
                 'ci_low': float(d.mean() - crit * sem),
                 'ci_high': float(d.mean() + crit * sem),
                 't': float(t), 'df': int(d.size - 1), 'p': float(p),
                 'p_perm': float((1 + np.sum(np.abs(null) >= abs(d.mean())))
                                 / 10001),
                 'n_obs_yes': int(per.n_yes.sum()), 'n_obs_no': int(per.n_no.sum())}


# ── 4) Plots (exploratory) ────────────────────────────────────────────

def plot_prediction(out_png, results, pers, title):
    """One column per test: every session's two means, joined, plus the summary."""
    keys = list(results)
    fig, axes = plt.subplots(2, len(keys), figsize=(2.5 * len(keys), 5.6),
                             squeeze=False)
    for j, k in enumerate(keys):
        per, res = pers[k], results[k]
        ax = axes[0][j]
        if len(per):
            for _, r in per.iterrows():
                ax.plot([0, 1], [r.mean_no, r.mean_yes], color='0.75', lw=0.6,
                        zorder=1)
            ax.errorbar([0], [per.mean_no.mean()],
                        yerr=[per.mean_no.sem()], color=C_NO, marker='o',
                        ms=7, capsize=3, lw=0, elinewidth=1.6, zorder=3)
            ax.errorbar([1], [per.mean_yes.mean()],
                        yerr=[per.mean_yes.sem()], color=C_YES, marker='o',
                        ms=7, capsize=3, lw=0, elinewidth=1.6, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['no\nmistake', 'mistake\nfollows'], fontsize=8)
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(k, fontsize=9)
        if j == 0:
            ax.set_ylabel(f'Ripple rate {TEST_WIN[0]}–{TEST_WIN[1]} s (Hz)',
                          fontsize=9)
        if res:
            ax.annotate(f"Δ={res['diff_hz']:+.4f}\nt({res['df']})={res['t']:+.2f}\n"
                        f"p={res['p_perm']:.3f}  n={res['n_sessions']}",
                        xy=(0.5, 0.02), xycoords='axes fraction', ha='center',
                        va='bottom', fontsize=7, color='0.25')

        ax = axes[1][j]
        if res and len(per):
            d = (per.mean_yes - per.mean_no)
            ax.hist(d, bins=14, color='0.7', edgecolor='w')
            ax.axvline(0, color='0.3', lw=1.0)
            ax.axvline(d.mean(), color=C_YES, lw=2.0)
        ax.set_xlabel('Δ (mistake − no mistake), Hz', fontsize=8)
        if j == 0:
            ax.set_ylabel('Sessions', fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion(out_png, ev, rep):
    """Which locations are chosen in error, and whether errors repeat."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    errs = ev[(ev.valence == 'error') & np.isfinite(ev.loc_)]
    m = np.zeros((4, 9))
    for i, r in enumerate(REWARDS):
        sub = errs[errs.reward == r]
        for l in range(1, 10):
            m[i, l - 1] = (sub.loc_ == l).sum()
    mn = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    ax = axes[0]
    im = ax.imshow(100 * mn, cmap='RdPu', aspect='auto')
    ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10), fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(REWARDS, fontsize=8)
    ax.set_xlabel('location uncovered in error', fontsize=9)
    ax.set_ylabel('reward being sought', fontsize=9)
    ax.set_title(f'Where errors land (n={int(m.sum())})', fontsize=9)
    fig.colorbar(im, ax=ax, label='% of that reward\'s errors')

    # consecutive errors for the same (grid, reward): same location twice?
    trans = np.zeros((9, 9))
    for (_, _, _), g in errs.groupby(['session', 'grid_no', 'reward']):
        locs = g.sort_values('t_s').loc_.to_numpy(int)
        for a, b in zip(locs[:-1], locs[1:]):
            trans[a - 1, b - 1] += 1
    tn = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)
    ax = axes[1]
    im = ax.imshow(100 * tn, cmap='RdPu', aspect='auto')
    ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10), fontsize=8)
    ax.set_yticks(range(9)); ax.set_yticklabels(range(1, 10), fontsize=8)
    ax.set_xlabel('next error location', fontsize=9)
    ax.set_ylabel('this error location', fontsize=9)
    diag = 100 * np.trace(trans) / max(trans.sum(), 1)
    ax.set_title(f'Consecutive errors, same reward\ndiagonal = {diag:.1f}% '
                 f'(chance ≈ {100 / 8:.1f}%)', fontsize=9)
    fig.colorbar(im, ax=ax, label='% of transitions')

    ax = axes[2]
    frac = rep.repeated.mean() if len(rep) else np.nan
    by_r = rep.groupby('reward').repeated.mean() if len(rep) else pd.Series()
    ax.bar(range(len(by_r)), 100 * by_r.values, color=C_YES)
    ax.axhline(100 * frac, color='0.3', ls='--', lw=1.0,
               label=f'overall {100 * frac:.0f}%')
    ax.set_xticks(range(len(by_r))); ax.set_xticklabels(by_r.index, fontsize=8)
    ax.set_ylabel('% of errors repeated later\n(same reward, same location)',
                  fontsize=9)
    ax.set_xlabel('reward being sought', fontsize=9)
    ax.set_title(f'Do mistakes repeat? (n={len(rep)})', fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── 5) Main ───────────────────────────────────────────────────────────

def run(bundle=None, out_dir=None):
    root = swr_io.get_data_root()
    group = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr')
    bundle = bundle or os.path.join(group, 'bundle')
    out_dir = out_dir or os.path.join(
        group, f"explore_ripples_mistakes_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_explore_ripples_and_mistakes')
    np.random.seed(SEED)

    data = rip.load_bundle(bundle)
    print(f"\n  bundle: {bundle}")
    print(f"  window {TEST_WIN[0]}-{TEST_WIN[1]} s, baseline "
          f"{BASE_WIN[0]}..{BASE_WIN[1]} s, EXPLORATORY\n")

    ev = event_table(data).rename(columns={'loc': 'loc_'})
    ev.to_csv(os.path.join(out_dir, 'events.csv'), index=False)
    print(f"  {len(ev)} uncoverings, {ev.session.nunique()} sessions, "
          f"{ev.loc_.notna().mean():.1%} with a location")

    grids = label_outcomes(ev)
    grids.to_csv(os.path.join(out_dir, 'grid_outcomes.csv'), index=False)
    rep = repeated_mistakes(ev)
    rep.to_csv(os.path.join(out_dir, 'repeated_mistakes.csv'), index=False)
    print(f"  {len(grids)} grid-level observations | {len(rep)} errors")

    payload, all_rows = {}, []

    # -- Q1 and Q2 --------------------------------------------------------
    for question in ('D-specific', 'all first'):
        sub = grids[grids.question == question]
        for measure in ('delta', 'raw'):
            results, pers = {}, {}
            for horizon, col in (('all later repeats', 'err_all'),
                                 (f'next {NEXT_FEW} repeats', 'err_few')):
                key = f'{horizon}'
                per, res = paired_by_session(sub, measure, col)
                pers[key], results[key] = per, res
                if res:
                    print(f"  {question:12s} {measure:5s} {horizon:18s} "
                          f"no {res['mean_no']:+.4f} vs mistake "
                          f"{res['mean_yes']:+.4f}  Δ={res['diff_hz']:+.4f} "
                          f"t({res['df']})={res['t']:+.2f} "
                          f"p={res['p_perm']:.4f}  n={res['n_sessions']}")
                    all_rows.append(dict(question=question, measure=measure,
                                         horizon=horizon, **res))
                else:
                    print(f"  {question:12s} {measure:5s} {horizon:18s} "
                          f"too few sessions")
            payload[f'{question} | {measure}'] = results
            tag = f"{question.replace(' ', '_')}_{measure}"
            plot_prediction(os.path.join(out_dir, f'predict_{tag}.png'),
                            results, pers,
                            f'{question}, {measure} — does a bigger ripple '
                            f'response mean fewer later mistakes?')

    # -- Q3 and Q4 --------------------------------------------------------
    plot_confusion(os.path.join(out_dir, 'mistake_confusion.png'), ev, rep)
    print(f"\n  mistakes repeated (same reward, same location): "
          f"{100 * rep.repeated.mean():.1f}% of {len(rep)}")

    results, pers = {}, {}
    for measure in ('delta', 'raw'):
        per, res = paired_by_session(rep, measure, 'repeated')
        pers[measure], results[measure] = per, res
        if res:
            print(f"  error ripple {measure:5s}: not-repeated "
                  f"{res['mean_no']:+.4f} vs repeated {res['mean_yes']:+.4f}  "
                  f"Δ={res['diff_hz']:+.4f} t({res['df']})={res['t']:+.2f} "
                  f"p={res['p_perm']:.4f}  n={res['n_sessions']}")
            all_rows.append(dict(question='error repeats', measure=measure,
                                 horizon='same reward+location', **res))
    payload['error repeats'] = results
    plot_prediction(os.path.join(out_dir, 'predict_error_repeats.png'),
                    results, pers,
                    'Ripple response at a wrong uncovering — does it predict '
                    'making that exact mistake again?')

    pd.DataFrame(all_rows).to_csv(
        os.path.join(out_dir, 'explore_statistics.csv'), index=False)
    with open(os.path.join(out_dir, 'explore_statistics.json'), 'w') as f:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   'status': 'EXPLORATORY — nothing here is confirmatory',
                   'test_window_s': list(TEST_WIN),
                   'baseline_window_s': list(BASE_WIN),
                   'next_few_repeats': NEXT_FEW, 'seed': SEED,
                   'results': payload}, f, indent=2, default=str)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
