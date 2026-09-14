#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What "stillness" actually is here, and a task-free reference for it.

SK's objections, all fair, and each one gets a section:

  Q1  Every event is a BUTTON PRESS, which is the opposite of stillness. What
      does "0-0.5 s after a press" have to do with being still?

      It is not a claim that the window is still relative to baseline. The
      variable is `still_next_s` = time from that press to the NEXT key press
      of any kind, taken from the 25 ms button series. It measures the length
      of the quiet GAP the window falls into. For an event with a 2 s gap the
      whole 0-0.5 s window sits inside a period with no further presses; for
      one with a 0.3 s gap the window contains the next press. That is the
      difference being controlled -- not "stillness vs movement" but "how long
      until the subject does anything again". It is BUTTON stillness: there is
      no eye or body tracking here, so a motionless subject thinking hard and
      a motionless subject resting are the same thing to this measure.

  Q2  Inside a long still period, where are the ripples -- beginning, end,
      evenly? Section 2 aligns to the start of every gap >= 2.5 s and to its
      end, so the profile is read directly instead of assumed.

  Q3  Is there a stillness period with no task on it? Yes: after the last D of
      a grid the subject has finished and the next grid has not started.
      1,431 grid endings, median gap 4.3 s to the next press, 68% over 2 s,
      ~16 per session. Section 3 uses the LATE part of that gap (1.5-2.5 s in),
      by which point any feedback transient has passed, as a task-free
      stillness reference, and compares it with in-task pauses of the same
      length sampled at the same depth.

  Q4  Test the event against a stillness-matched CONTROL rather than against
      its own pre-event baseline. Section 4 does the sliding cluster test on
      the raw rate difference (feedback minus matched pause), with no baseline
      window anywhere in it -- which also removes the pre-event imbalance that
      the 2026-09-13 entry flagged as a limitation.

    python scripts/swr_stillness_anatomy.py

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
from swr_valence_stage_interaction import event_table, _rc, _axes
from swr_feedback_vs_matched_stillness import pause_table, per_session

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


QUESTION = ('What is stillness here, how are ripples distributed inside it, '
            'and is a task-free still period any different?')

GAP_MIN_S = 2.5          # a "long" gap: every included period is this quiet
GAP_MAX_S = 60.0         # beyond this it is a break in the session, not a pause
LATE_WINDOW = (1.5, 2.5)  # read the gap here: past any event transient
HALF_S = 2.5             # peri-event half width for the within-gap profiles
SEED = 42

# The three ways a long gap can start. Colours: the two task events keep the
# project's valence/stage hues, the task-free one takes the neutral dark grey
# used for stillness throughout this set of scripts.
SOURCES = ('correct uncovering', 'movement press', 'grid end (task-free)')
SOURCE_COLOUR = {'correct uncovering': rip.STAGE_COLOUR['first uncovers'],
                 'movement press': '#252525',
                 'grid end (task-free)': '#448363'}


# ── 1) The three kinds of long gap ────────────────────────────────────

def grid_end_table(data, presses_by_session):
    """The last D of every grid, with the gap that follows it.

    After the final repeat of a grid the subject has nothing left to do and the
    next grid has not appeared, so the gap is a still period with no task on
    it -- the only one this paradigm contains.
    """
    beh = data['behaviour']
    rows = []
    for session in rip.sessions_in(data):
        b = beh[beh.session == session].sort_values(['grid_no', 'rep_overall'])
        if not len(b) or session not in presses_by_session:
            continue
        allp = presses_by_session[session]
        last = b.groupby('grid_no').tail(1)
        for r in last.itertuples():
            t_d = float(r.t_D)
            if not np.isfinite(t_d):
                continue
            i = np.searchsorted(allp, t_d, 'right')
            nxt = float(allp[i]) if i < allp.size else np.nan
            rows.append({'session': int(session), 'grid_no': int(r.grid_no),
                         't_s': t_d, 'still_next_s': nxt - t_d})
    return pd.DataFrame(rows)


def long_gaps(events, pauses, grid_ends, gap_min=GAP_MIN_S, gap_max=GAP_MAX_S):
    """{source: frame} -- every gap at least `gap_min` long, by what opened it.

    The same length criterion is applied to all three, so they differ in what
    the subject had just done and in nothing else that this analysis controls.
    """
    def keep(frame):
        ok = (frame.still_next_s >= gap_min) & (frame.still_next_s <= gap_max)
        return frame[ok]

    correct = events[(events.valence == 'correct')]
    return {'correct uncovering': keep(correct),
            'movement press': keep(pauses),
            'grid end (task-free)': keep(grid_ends)}


# ── 2) Where the ripples sit inside a long gap ────────────────────────

def within_gap_profiles(data, gaps, unit, min_events, align='start'):
    """Peri-event rate around the press that opens (or closes) each long gap.

    `align='start'` locks to the opening press, so 0..+GAP_MIN_S is entirely
    inside the quiet period. `align='end'` locks to the press that ends it, so
    -GAP_MIN_S..0 is inside. Reading both says whether ripples cluster at the
    start of a pause, at its end, or sit evenly across it.
    """
    out, counts, centres = {}, {}, None
    for source, frame in gaps.items():
        if not len(frame):
            continue
        f = frame.copy()
        if align == 'end':
            f['t_s'] = f.t_s + f.still_next_s
        c, per_unit, cnt = rip.rate_by_unit(data, per_session(f), unit=unit,
                                            min_events=min_events,
                                            half_s=HALF_S)
        if c is None or len(per_unit) < 3:
            print(f"    {source:24s} not computable")
            continue
        centres = c
        out[source], counts[source] = per_unit, cnt
        print(f"    {source:24s} n={cnt['n_units']:3d} "
              f"events={cnt['n_events_used']:5d}")
    return centres, out, counts


def window_rate(per_unit, centres, win):
    """Per-unit RAW rate in a window -- no baseline subtraction anywhere."""
    units = sorted(per_unit)
    m = (centres >= win[0]) & (centres < win[1])
    return units, np.array([np.nanmean(per_unit[u][m]) for u in units], float)


def paired_difference(per_unit_a, per_unit_b, centres, win):
    """Paired t between two sources on the units they share."""
    shared = sorted(set(per_unit_a) & set(per_unit_b))
    if len(shared) < 3:
        return None
    m = (centres >= win[0]) & (centres < win[1])
    a = np.array([np.nanmean(per_unit_a[u][m]) for u in shared], float)
    b = np.array([np.nanmean(per_unit_b[u][m]) for u in shared], float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    t, p = stats.ttest_rel(a[ok], b[ok])
    rng = np.random.default_rng(SEED)
    d = a[ok] - b[ok]
    null = (rng.choice([-1.0, 1.0], size=(10000, d.size)) * d).mean(axis=1)
    return {'n_units': int(ok.sum()), 'mean_a_hz': float(a[ok].mean()),
            'mean_b_hz': float(b[ok].mean()), 'diff_hz': float(d.mean()),
            't': float(t), 'p': float(p),
            'p_perm': float((1 + np.sum(np.abs(null) >= abs(d.mean())))
                            / 10001)}


# ── 3) Raw contrast against a matched control, no baseline window ─────

def raw_contrast_sliding(data, events, pauses, unit, min_events, n_perm,
                         gap_min=GAP_MIN_S, gap_max=GAP_MAX_S):
    """Feedback minus matched pause, in absolute Hz, tested at every position.

    Both event classes are restricted to gaps of the same length, so the
    comparison is stillness-matched, and NEITHER is baseline-subtracted. That
    is what SK asked for: the control replaces the pre-event baseline instead
    of both being measured against their own. It also sidesteps the pre-event
    stillness imbalance, since no pre-event window enters the statistic.
    """
    sel = lambda f: f[(f.still_next_s >= gap_min) & (f.still_next_s <= gap_max)]
    ctrl = sel(pauses)
    c, pause_profiles, pause_counts = None, None, None
    c, pause_profiles, pause_counts = rip.rate_by_unit(
        data, per_session(ctrl), unit=unit, min_events=min_events,
        half_s=rip.HALF_S)
    if c is None or not pause_profiles:
        return None, None, None
    centres = c

    out, counts = {}, {'matched pause': pause_counts}
    for label, valence, stage in (('correct, first uncovers', 'correct',
                                   'first uncovers'),
                                  ('correct, later', 'correct', 'later'),
                                  ('error, first uncovers', 'error',
                                   'first uncovers'),
                                  ('error, later', 'error', 'later')):
        sub = sel(events[(events.valence == valence)
                         & (events.stage2 == stage)])
        if not len(sub):
            continue
        _, per_unit, cnt = rip.rate_by_unit(data, per_session(sub), unit=unit,
                                            min_events=min_events,
                                            half_s=rip.HALF_S)
        if not per_unit:
            continue
        counts[label] = cnt
        raw = {label: per_unit, 'matched pause': pause_profiles}
        diff = rip.contrast_profiles(raw, {label: 1.0, 'matched pause': -1.0})
        if len(diff) >= 3:
            out[label] = diff
    if not out:
        return centres, None, counts
    # NOT baseline-subtracted: the matched pause IS the reference.
    sliding = rip.sliding_window_test(out, centres,
                                      width_s=rip.SLIDE_WIDTHS_S[0],
                                      n_perm=n_perm, seed=SEED)
    return centres, (out, sliding), counts


# ── 4) Figure ─────────────────────────────────────────────────────────

def figure(out_png, centres_gap, start_profiles, end_profiles, late_stats,
           centres_raw, raw_pack, suptitle, footnote):
    legend_cm, title_cm = 2.3, 0.7
    with plt.rc_context(_rc()):
        fig, axes = _axes(2, extra_cm=legend_cm + title_cm + 0.6)

        # ---- (0,0) inside the gap, aligned to its start ------------------
        ax = axes[0][0]
        ax.axvspan(0, GAP_MIN_S, color='#f2f2f2', lw=0, zorder=0)
        for source in SOURCES:
            if source not in start_profiles:
                continue
            units, X = sorted(start_profiles[source]), None
            X = np.vstack([start_profiles[source][u] for u in units])
            mean = np.nanmean(X, axis=0)
            sem = np.nanstd(X, axis=0) / max(np.sqrt(X.shape[0]), 1)
            ax.plot(centres_gap, mean, color=SOURCE_COLOUR[source],
                    lw=rip.LW_RATE, zorder=3, label=f'{source} (n={len(units)})')
            ax.fill_between(centres_gap, mean - sem, mean + sem,
                            color=SOURCE_COLOUR[source], alpha=0.20, lw=0,
                            zorder=2)
        ax.axvline(0, color='0.35', lw=1.0)
        ax.set_xlabel('Time from the press that starts the gap (s)',
                      fontsize=rip.FS - 1, labelpad=1)
        ax.set_ylabel('Ripple rate (Hz)', fontsize=rip.FS)
        ax.set_title(f'Inside a gap ≥ {GAP_MIN_S:g} s', fontsize=rip.FS, pad=3)

        # ---- (0,1) the same, aligned to the end of the gap ---------------
        ax = axes[0][1]
        ax.axvspan(-GAP_MIN_S, 0, color='#f2f2f2', lw=0, zorder=0)
        for source in SOURCES:
            if source not in end_profiles:
                continue
            units = sorted(end_profiles[source])
            X = np.vstack([end_profiles[source][u] for u in units])
            mean = np.nanmean(X, axis=0)
            sem = np.nanstd(X, axis=0) / max(np.sqrt(X.shape[0]), 1)
            ax.plot(centres_gap, mean, color=SOURCE_COLOUR[source],
                    lw=rip.LW_RATE, zorder=3)
            ax.fill_between(centres_gap, mean - sem, mean + sem,
                            color=SOURCE_COLOUR[source], alpha=0.20, lw=0,
                            zorder=2)
        ax.axvline(0, color='0.35', lw=1.0)
        ax.set_xlabel('Time from the press that ends the gap (s)',
                      fontsize=rip.FS - 1, labelpad=1)
        ax.set_ylabel('Ripple rate (Hz)', fontsize=rip.FS)
        ax.set_title('Aligned to the gap end', fontsize=rip.FS, pad=3)

        # ---- (0,2) the late window: task-free vs in-task ------------------
        ax = axes[0][2]
        for i, source in enumerate(SOURCES):
            if source not in start_profiles:
                continue
            units, v = window_rate(start_profiles[source], centres_gap,
                                   LATE_WINDOW)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            ax.errorbar([i], [v.mean()],
                        yerr=[v.std() / max(np.sqrt(v.size), 1)],
                        color=SOURCE_COLOUR[source], marker='o', ms=5,
                        capsize=3, elinewidth=1.4, lw=0)
            ax.annotate(f"{v.size}", xy=(i, 0.02),
                        xycoords=('data', 'axes fraction'), ha='center',
                        va='bottom', fontsize=rip.FS - 3, color='0.45')
        ax.set_xticks(range(len(SOURCES)))
        ax.set_xticklabels([s.replace(' (', '\n(').replace(' press', '\npress')
                            .replace(' uncovering', '\nuncovering')
                            for s in SOURCES], fontsize=rip.FS - 3)
        ax.set_xlim(-0.5, len(SOURCES) - 0.5)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.18 * (hi - lo), hi)
        ax.set_ylabel('Ripple rate (Hz)\n%.1f–%.1f s into the gap'
                      % LATE_WINDOW, fontsize=rip.FS - 1)
        ax.set_title('Late in the gap', fontsize=rip.FS, pad=3)

        # ---- (1,0) and (1,1) raw difference vs matched pause -------------
        if raw_pack is not None:
            diffs, sliding = raw_pack
            ax = axes[1][0]
            ax.axhline(0, color='0.45', lw=0.8)
            for label, per_unit in diffs.items():
                units = sorted(per_unit)
                X = np.vstack([per_unit[u] for u in units])
                mean = np.nanmean(X, axis=0)
                sem = np.nanstd(X, axis=0) / max(np.sqrt(X.shape[0]), 1)
                colour = (rip.VALENCE_COLOUR['correct']
                          if label.startswith('correct')
                          else rip.VALENCE_COLOUR['error'])
                if 'later' in label:
                    base = np.array(plt.matplotlib.colors.to_rgb(colour))
                    colour = tuple(base + (1.0 - base) * 0.45)
                ax.plot(centres_raw, mean, color=colour, lw=rip.LW_RATE,
                        zorder=3, label=f'{label} (n={len(units)})')
                ax.fill_between(centres_raw, mean - sem, mean + sem,
                                color=colour, alpha=0.18, lw=0, zorder=2)
            ax.axvline(0, color='0.35', lw=1.0)
            ax.set_xlabel('Time from press (s)', fontsize=rip.FS, labelpad=1)
            ax.set_ylabel('Feedback − matched pause\n(Hz, no baseline)',
                          fontsize=rip.FS - 1)
            ax.set_title('Raw difference, no baseline', fontsize=rip.FS, pad=3)

            ax = axes[1][1]
            ax.axhline(0, color='0.45', lw=0.8)
            for label, res in sliding.items():
                colour = (rip.VALENCE_COLOUR['correct']
                          if label.startswith('correct')
                          else rip.VALENCE_COLOUR['error'])
                if 'later' in label:
                    base = np.array(plt.matplotlib.colors.to_rgb(colour))
                    colour = tuple(base + (1.0 - base) * 0.45)
                ax.plot(res['times'], res['t'], color=colour, lw=1.5)
                for cl in res['clusters']:
                    if cl['p'] < 0.05:
                        ax.axvspan(cl['start_s'], cl['stop_s'], color=colour,
                                   alpha=0.16, lw=0)
                        ax.annotate(f"p={cl['p']:.3f}",
                                    xy=(cl['peak_s'], 0.97),
                                    xycoords=('data', 'axes fraction'),
                                    ha='center', va='top',
                                    fontsize=rip.FS - 3, color=colour)
                for sign in (1, -1):
                    ax.axhline(sign * res['threshold'], color='0.6', lw=0.7,
                               ls=':')
            ax.axvline(0, color='0.35', lw=1.0)
            ax.set_xlabel('Sliding window centre (s)', fontsize=rip.FS,
                          labelpad=1)
            ax.set_ylabel('t, feedback − matched pause', fontsize=rip.FS - 1)
            ax.set_title('Cluster test, no baseline', fontsize=rip.FS, pad=3)

        # ---- (1,2) the late-window pairwise tests, written out ------------
        ax = axes[1][2]
        ax.axis('off')
        short = {'correct uncovering': 'correct unc.',
                 'movement press': 'movement',
                 'grid end (task-free)': 'grid end'}
        lines = ['Late in the gap (%.1f–%.1f s), paired:' % LATE_WINDOW, '']
        for name, got in late_stats.items():
            if got is None:
                continue
            a, b = [short.get(x.strip().lstrip('vs ').strip(), x.strip())
                    for x in name.split('\n')]
            star = ' *' if got['p_perm'] < 0.05 else ''
            lines.append(f"{a} vs {b}{star}")
            lines.append(f"   {got['mean_a_hz']:.3f} vs {got['mean_b_hz']:.3f} Hz"
                         f"  (Δ={got['diff_hz']:+.3f})")
            lines.append(f"   t({got['n_units'] - 1})={got['t']:+.2f}, "
                         f"p={got['p_perm']:.3f}")
            lines.append('')
        ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, va='top',
                ha='left', fontsize=rip.FS - 3, family='sans-serif',
                color='0.15')

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
                   bbox_to_anchor=(0.5, 0.03), ncol=3, fontsize=rip.FS - 2,
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


# ── 5) Main ───────────────────────────────────────────────────────────

def run(bundle=None, out_dir=None, unit='session', min_events=8,
        n_perm=rip.N_SIGN_FLIPS, stillness_cache=None, pause_cache=None):
    root = swr_io.get_data_root()
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                              'bundle')
    if out_dir is None:
        out_dir = os.path.join(
            swr_io.derivatives_dir(root), 'group', 'swr',
            f"stillness_anatomy_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_stillness_anatomy')
    np.random.seed(SEED)

    data = rip.load_bundle(bundle)
    print(f"\n  bundle: {bundle}\n  {QUESTION}")
    print(f"  unit = {unit}, min {min_events} events per {unit}\n")

    if stillness_cache is None:
        stillness_cache = os.path.join(out_dir, 'events_with_stillness.csv')
    if pause_cache is None:
        pause_cache = os.path.join(out_dir, 'movement_presses.csv')
    events = event_table(data, cache_path=stillness_cache)
    events = events.assign(stage2=events.stage.map(
        {'first uncovers': 'first uncovers', 'while learning': 'later',
         'once known': 'later'}))
    pauses = pause_table(data, cache_path=pause_cache)

    # The full press train is the union of the two tables already in hand --
    # movement presses and uncovering attempts. Rebuilding it from the 25 ms
    # button series would re-read every raw file for a table we already have.
    presses_by_session = {}
    for session in rip.sessions_in(data):
        m = pauses.loc[pauses.session == session, 't_s'].to_numpy(float)
        u = events.loc[events.session == session, 't_s'].to_numpy(float)
        if m.size or u.size:
            presses_by_session[session] = np.sort(np.concatenate([m, u]))
    grid_ends = grid_end_table(data, presses_by_session)
    grid_ends.to_csv(os.path.join(out_dir, 'grid_end_gaps.csv'), index=False)

    results = {'question': QUESTION, 'bundle': bundle, 'unit': unit,
               'min_events': min_events, 'gap_min_s': GAP_MIN_S,
               'gap_max_s': GAP_MAX_S, 'late_window_s': list(LATE_WINDOW),
               'n_sign_flips': n_perm, 'seed': SEED,
               'stillness_definition': (
                   'time from the event to the next key press of any kind '
                   '(movement or uncover), from the 25 ms button series; '
                   'BUTTON stillness only, no eye or body tracking')}

    # ---- how long is the gap, by what opened it --------------------------
    gaps = long_gaps(events, pauses, grid_ends)
    print("  long gaps (%.1f-%.0f s), by what opened them" % (GAP_MIN_S,
                                                              GAP_MAX_S))
    desc = {}
    for source, frame in gaps.items():
        if not len(frame):
            continue
        d = frame.still_next_s
        desc[source] = {'n': int(len(frame)), 'median_s': float(d.median()),
                        'q25_s': float(d.quantile(.25)),
                        'q75_s': float(d.quantile(.75)),
                        'n_sessions': int(frame.session.nunique())}
        print(f"    {source:24s} n={len(frame):6d} "
              f"median={d.median():.2f}s IQR {d.quantile(.25):.2f}-"
              f"{d.quantile(.75):.2f}  sessions={frame.session.nunique()}")
    results['gap_lengths'] = desc

    # ---- Q2: where are the ripples inside the gap ------------------------
    print("\n  Q2) inside a long gap, aligned to its START")
    centres_gap, start_profiles, start_counts = within_gap_profiles(
        data, gaps, unit, min_events, align='start')
    print("  Q2) the same, aligned to its END")
    _, end_profiles, _ = within_gap_profiles(data, gaps, unit, min_events,
                                             align='end')
    if centres_gap is not None:
        prof_rows = []
        for source, per_unit in start_profiles.items():
            units = sorted(per_unit)
            X = np.vstack([per_unit[u] for u in units])
            for t, m, s in zip(centres_gap, np.nanmean(X, axis=0),
                               np.nanstd(X, axis=0) / np.sqrt(X.shape[0])):
                prof_rows.append(dict(source=source, align='start',
                                      t_s=float(t), mean_hz=float(m),
                                      sem_hz=float(s), n_units=len(units)))
        for source, per_unit in end_profiles.items():
            units = sorted(per_unit)
            X = np.vstack([per_unit[u] for u in units])
            for t, m, s in zip(centres_gap, np.nanmean(X, axis=0),
                               np.nanstd(X, axis=0) / np.sqrt(X.shape[0])):
                prof_rows.append(dict(source=source, align='end',
                                      t_s=float(t), mean_hz=float(m),
                                      sem_hz=float(s), n_units=len(units)))
        pd.DataFrame(prof_rows).to_csv(
            os.path.join(out_dir, 'within_gap_profiles.csv'), index=False)

        print(f"\n  rate across the gap (aligned to start), "
              f"0.5 s slices, mean Hz")
        slices = [(a, a + 0.5) for a in np.arange(0.0, GAP_MIN_S, 0.5)]
        head = '  '.join(f"{a:.1f}-{b:.1f}" for a, b in slices)
        print(f"    {'source':24s} {head}")
        for source, per_unit in start_profiles.items():
            vals = []
            for win in slices:
                _, v = window_rate(per_unit, centres_gap, win)
                v = v[np.isfinite(v)]
                vals.append(v.mean() if v.size else np.nan)
            line = '  '.join(f"{v:7.3f}" for v in vals)
            print(f"    {source:24s} {line}")
            results.setdefault('within_gap', {})[source] = {
                f'{a:.1f}-{b:.1f}s': float(v)
                for (a, b), v in zip(slices, vals)}

    # ---- Q3: task-free vs in-task stillness ------------------------------
    print(f"\n  Q3) late in the gap ({LATE_WINDOW[0]:.1f}-{LATE_WINDOW[1]:.1f} s), "
          f"raw rate, paired")
    late_stats = {}
    if centres_gap is not None:
        for source in SOURCES:
            if source not in start_profiles:
                continue
            _, v = window_rate(start_profiles[source], centres_gap, LATE_WINDOW)
            v = v[np.isfinite(v)]
            print(f"    {source:24s} n={v.size:3d} mean={v.mean():.4f} Hz")
            results.setdefault('late_window', {})[source] = {
                'n_units': int(v.size), 'mean_hz': float(v.mean()),
                'sem_hz': float(v.std() / max(np.sqrt(v.size), 1))}
        pairs = [('correct uncovering', 'grid end (task-free)'),
                 ('movement press', 'grid end (task-free)'),
                 ('correct uncovering', 'movement press')]
        for a, b in pairs:
            if a not in start_profiles or b not in start_profiles:
                continue
            got = paired_difference(start_profiles[a], start_profiles[b],
                                    centres_gap, LATE_WINDOW)
            late_stats[f'{a}\n  vs {b}'] = got
            if got:
                star = ' *' if got['p_perm'] < 0.05 else ''
                print(f"    {a} vs {b}: {got['mean_a_hz']:.4f} vs "
                      f"{got['mean_b_hz']:.4f} Hz, Δ={got['diff_hz']:+.4f}, "
                      f"t({got['n_units'] - 1})={got['t']:+.2f}, "
                      f"p_perm={got['p_perm']:.4g}{star}")
        results['late_window_contrasts'] = {k: v for k, v in late_stats.items()}

    # ---- Q4: raw contrast against a matched control ----------------------
    print(f"\n  Q4) feedback minus matched pause, NO baseline window, "
          f"gaps ≥ {GAP_MIN_S:g} s")
    centres_raw, raw_pack, raw_counts = raw_contrast_sliding(
        data, events, pauses, unit, min_events, n_perm)
    if raw_pack is not None:
        diffs, sliding = raw_pack
        for label, res in sliding.items():
            keep = [c for c in res['clusters'] if c['p'] < 0.05]
            txt = ('; '.join(f"{c['direction']} {c['start_s']:+.2f}.."
                             f"{c['stop_s']:+.2f} s (p={c['p']:.4f})"
                             for c in keep) if keep else 'none')
            print(f"    {label:26s} n={res['n_subjects']:3d}  {txt}")
            results.setdefault('raw_vs_matched_pause', {})[label] = {
                'n_units': res['n_subjects'], 'clusters': keep}
        for label, per_unit in diffs.items():
            units = sorted(per_unit)
            X = np.vstack([per_unit[u] for u in units])
            m = (centres_raw >= 0.0) & (centres_raw < 0.5)
            v = np.nanmean(X[:, m], axis=1)
            v = v[np.isfinite(v)]
            t, p = stats.ttest_1samp(v, 0.0)
            print(f"      {label:26s} 0-0.5 s: {v.mean():+.4f} Hz, "
                  f"t={t:+.2f}, p={p:.4g}")
            results['raw_vs_matched_pause'][label].update(
                {'window_0_0.5_hz': float(v.mean()), 't': float(t),
                 'p': float(p)})
    else:
        print("    not computable")

    figure(os.path.join(out_dir, 'stillness_anatomy.png'), centres_gap,
           start_profiles, end_profiles, late_stats, centres_raw, raw_pack,
           'What stillness is, where ripples sit in it, and a task-free '
           'reference',
           f'Stillness = time to the next key press of any kind. Gaps '
           f'{GAP_MIN_S:g}–{GAP_MAX_S:g} s only, same criterion for all three '
           f'sources. Rates are RAW (no baseline subtraction). unit = {unit}, '
           f'min {min_events} events per {unit}.')

    with open(os.path.join(out_dir, 'stillness_anatomy_result.json'), 'w') as f:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   **results}, f, indent=2, default=str)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
