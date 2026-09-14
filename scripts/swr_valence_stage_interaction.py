#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valence x stage: is the feedback effect on ripple rate specific to the first
traversal?

`swr_ripple_tests.py --tests="['feedback_stage']"` tests each of the six
valence x stage cells against its OWN baseline, which answers "does this cell
move?" but never "do the cells differ from each other?". By eye the six cells
look crossed -- positive feedback raises the rate early in a grid and negative
feedback, if anything, late -- and that shape is an INTERACTION, which needs
its own test.

What this script adds:

  1) the simple valence effect (correct - error) inside each stage, as a time
     course, not just a window;
  2) the interaction itself as a single per-session contrast -- linear and
     quadratic over the three stages, plus a two-level (first vs later)
     version that keeps far more sessions;
  3) the omnibus 2 x 3 interaction (both contrasts jointly, Hotelling T^2);
  4) the stillness control that F1 forces on any contrast in this project.

Everything is a LINEAR COMBINATION of the same per-session rate profiles the
rest of the pipeline uses, so it goes through `rip.baseline_subtract`,
`rip.window_test` and `rip.sliding_window_test` unchanged -- the interaction is
tested by the same functions, against the same sign-flip null, as any ordinary
condition. Nothing here re-implements a test.

Why the stillness control is not optional. F1 established that ripple rate
rises with stillness, and stillness is not balanced across these cells: after a
correct uncovering on the first traversal the subject sits still for a median
1.4 s, in every other cell for under 0.6 s. That difference has the same
crossed shape as the ripple effect, so the interaction is recomputed under a
common stillness criterion applied identically to all cells, swept over
thresholds.

    python scripts/swr_valence_stage_interaction.py --bundle=<bundle dir>
    python scripts/swr_valence_stage_interaction.py --min_events=15 --unit=session

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
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import mc.analyse.swr_probes as prb

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


# ── Settings ──────────────────────────────────────────────────────────
QUESTION = ('Does the feedback-valence effect on ripple rate depend on the '
            'learning stage -- a valence x stage interaction?')

NAMED_WINDOWS = {'pre  (-0.6..-0.1)': (-0.6, -0.1),
                 'post (0..0.5)': (0.0, 0.5),
                 'post (0.5..1.0)': (0.5, 1.0)}
PRIMARY_WINDOW = (0.0, 0.5)

# The three stages pooled to two. `while learning` and `once known` are both
# "the route has been seen at least once", and pooling them at the EVENT level
# -- before the min-events filter -- is what rescues the sessions that have too
# few errors in either one alone to be estimated separately.
STAGE2 = {'first uncovers': 'first uncovers',
          'while learning': 'later', 'once known': 'later'}
STAGES2 = ('first uncovers', 'later')

# Common stillness criteria, applied identically to every cell.
STILL_THRESHOLDS_S = (0.0, 0.5, 1.0, 1.5, 2.0)
CONTROL_THRESHOLD_S = 1.0

SEED = 42

# The simple effects are per-stage and take the project's stage scale. An
# interaction is not one of the project's categorical variables, so it is drawn
# in neutral greys rather than borrowing a stage or valence hue -- otherwise the
# same colour would mean "while learning" in one panel and "quadratic" in the
# next.
INTERACTION_COLOUR = {'linear': '#2b2b2b', 'quadratic': '#9a9a9a',
                      'two-level': '#2b2b2b'}

# `later` is the two stages pooled, so it takes the colour of the later of them
# rather than a new hue -- the same stage keeps the same colour in every panel.
STAGE_COLOUR2 = {'first uncovers': rip.STAGE_COLOUR['first uncovers'],
                 'later': rip.STAGE_COLOUR['once known']}


def stage_colour(stage):
    return rip.STAGE_COLOUR.get(stage, STAGE_COLOUR2.get(stage, '0.4'))


LEFT_MARGIN = 0.045              # figure fraction kept free for row titles


def _axes(n_rows, extra_cm):
    """Like `rip._row_axes`, but the middle panel is wide enough for three
    stage names and the left edge is free for a rotated row title."""
    fig, axes = plt.subplots(n_rows, 3, squeeze=False,
                             figsize=(rip.ROW_W_CM * rip.CM,
                                      (rip.ROW_H_CM * n_rows + extra_cm)
                                      * rip.CM),
                             gridspec_kw=dict(width_ratios=[1.20, 1.00, 1.15]))
    for ax in axes.ravel():
        ax.tick_params(labelsize=rip.FS - 1, length=2.5, width=0.8)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
    return fig, axes


def _stage_ticks(ax, stages):
    """Stage names on a 3 cm axis: angled, so three of them still fit."""
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=rip.FS - 2, rotation=22, ha='right',
                       rotation_mode='anchor')
    ax.set_xlim(-0.4, len(stages) - 0.6)


def _rc():
    return {'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.spines.top': False, 'axes.spines.right': False}


# ── 1) Events, with how long the subject then sits still ──────────────

def event_table(data, cache_path=None):
    """Every uncovering attempt, labelled, with the stillness around it.

    `still_next_s` is the time from the attempt to the NEXT key press of any
    kind -- movement presses included, which is what makes it the stillness
    measure F1 uses and not merely the gap to the next uncovering. The two are
    very different here: after a correct uncovering the subject must still walk
    to the next location, so the gap between uncoverings counts that walk as
    rest.

    Press times come from the 25 ms button series, so they are on the ripple
    clock. That read is the slow part (~2 s per session), hence `cache_path`.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  stillness: reusing {os.path.basename(cache_path)}")
        return pd.read_csv(cache_path)

    beh = data['behaviour']
    rows = []
    for session in rip.sessions_in(data):
        table = rip.uncover_table(data, session)
        if not len(table):
            continue
        move_t, uncover_t = prb.press_times(session, beh[beh.session == session])
        presses = np.sort(np.concatenate([move_t, uncover_t]))
        t = table.t_s.to_numpy(float)
        if presses.size:
            i_next = np.searchsorted(presses, t, 'right')
            i_prev = np.searchsorted(presses, t, 'left') - 1
            nxt = np.where(i_next < presses.size,
                           presses[np.clip(i_next, 0, presses.size - 1)],
                           np.nan) - t
            prv = t - np.where(i_prev >= 0, presses[np.clip(i_prev, 0, None)],
                               np.nan)
        else:
            nxt = prv = np.full(t.size, np.nan)
        rows.append(table.assign(session=session, still_next_s=nxt,
                                 still_prev_s=prv,
                                 stage2=table.stage.map(STAGE2)))
    events = pd.concat(rows, ignore_index=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        events.to_csv(cache_path, index=False)
    return events


def cells(events, stage_col, min_still_next_s=None):
    """{cell label: {session: event times}} for one valence x stage crossing.

    The stillness criterion is applied to EVERY cell or to none. Filtering one
    cell to make it comparable to another would be the data snooping the whole
    control exists to avoid.
    """
    e = events if not min_still_next_s else \
        events[events.still_next_s >= min_still_next_s]
    out = {}
    for (valence, stage), g in e.groupby(['valence', stage_col]):
        out[f'{valence}, {stage}'] = {int(s): gg.t_s.to_numpy(float)
                                      for s, gg in g.groupby('session')}
    return out


def profiles_of(data, cells_dict, unit, min_events):
    """Peri-event rate per unit for every cell -- the pipeline's own routine."""
    raw, counts, centres = {}, {}, None
    for label, per_session in cells_dict.items():
        c, per_unit, count = rip.rate_by_unit(data, per_session, unit=unit,
                                              min_events=min_events)
        if c is not None:
            centres = c
        raw[label], counts[label] = per_unit, count
    return centres, raw, counts


# ── 2) The contrasts ──────────────────────────────────────────────────
# A contrast is {cell label: weight}. `rip.contrast_profiles` combines the raw
# per-unit profiles and keeps only units present in every weighted cell, which
# is what makes a difference of differences paired.

VAL_W = {'correct': +1.0, 'error': -1.0}


def _cross(stage_weights, stages):
    """Valence contrast crossed with a weighting over stages."""
    return {f'{v}, {s}': wv * ws
            for v, wv in VAL_W.items()
            for s, ws in zip(stages, stage_weights) if ws}


def contrast_set(stages):
    """Simple valence effects, the interaction contrasts, the main effect."""
    out = {f'correct - error, {s}': _cross(
        [1.0 if x == s else 0.0 for x in stages], stages) for s in stages}
    if len(stages) == 3:
        out['valence x stage, linear (first - once known)'] = \
            _cross((1.0, 0.0, -1.0), stages)
        out['valence x stage, quadratic (first - 2*learning + known)'] = \
            _cross((1.0, -2.0, 1.0), stages)
    else:
        out['valence x stage (first - later)'] = _cross((1.0, -1.0), stages)
    out['valence main effect (mean over stages)'] = \
        _cross([1.0 / len(stages)] * len(stages), stages)
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the order given."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running = np.empty(p.size), 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def test_contrasts(raw, centres, con_defs, n_perm, correct_over):
    """Every contrast through the pipeline's own window and sliding tests.

    The named windows are a FAMILY of three per contrast, so they carry a Holm
    adjustment across that family. The sliding test is the primary one: it
    chooses no window at all and corrects across positions by cluster
    permutation.
    """
    profiles = {}
    for label, weights in con_defs.items():
        if any(w not in raw for w in weights):
            continue
        p = rip.contrast_profiles(raw, weights)
        if len(p) >= 3:
            profiles[label] = p

    windows = {}
    for label, p in profiles.items():
        got = {n: rip.window_test(p, centres, w)
               for n, w in NAMED_WINDOWS.items()}
        got = {n: v for n, v in got.items() if v is not None}
        if got:
            adj = holm([v['p_perm'] for v in got.values()])
            for v, a in zip(got.values(), adj):
                v['p_perm_holm'] = float(a)
                v['window_family'] = len(got)
        windows[label] = got

    baselined = {}
    for label, p in profiles.items():
        units, X = rip.baseline_subtract(p, centres)
        baselined[label] = {u: X[i] for i, u in enumerate(units)}
    sliding = {f'{w:g}s': rip.sliding_window_test(
        baselined, centres, width_s=w, n_perm=n_perm, seed=SEED,
        correct_over=correct_over) for w in rip.SLIDE_WIDTHS_S}
    return profiles, windows, sliding


def omnibus_interaction(raw, centres, stages, n_perm, win=PRIMARY_WINDOW):
    """The 2 x 3 interaction as one 2-df test, on the fully crossed sessions.

    Linear and quadratic over stage span the interaction, so testing them
    jointly IS the omnibus. Sessions must supply all six cells, which is the
    price of an omnibus and the reason the two-level version is also reported.
    """
    defs = {'linear': _cross((1.0, 0.0, -1.0), stages),
            'quadratic': _cross((1.0, -2.0, 1.0), stages)}
    if any(w not in raw for d in defs.values() for w in d):
        return None
    per = {k: rip.contrast_profiles(raw, d) for k, d in defs.items()}
    units = sorted(set(per['linear']) & set(per['quadratic']))
    if len(units) < 5:
        return None
    mw = (centres >= win[0]) & (centres < win[1])
    cols = []
    for k in ('linear', 'quadratic'):
        _, X = rip.baseline_subtract({u: per[k][u] for u in units}, centres)
        cols.append(np.nanmean(X[:, mw], axis=1))
    got = rip.hotelling_signflip(np.column_stack(cols), n_perm=n_perm, seed=SEED)
    if got:
        got['window_s'] = list(win)
        got['contrasts'] = ['linear', 'quadratic']
    return got


def cell_window_means(raw, centres, labels, win=PRIMARY_WINDOW,
                      baseline=rip.BASELINE_WIN):
    """Baselined window mean per cell, on the units present in ALL labels.

    The paired subset, because that is what the interaction test sees. A cell
    plotted at its own larger n would not add up to the interaction drawn
    beside it.
    """
    labels = [l for l in labels if l in raw and raw[l]]
    if not labels:
        return [], {}
    units = sorted(set.intersection(*[set(raw[l]) for l in labels]))
    mw = (centres >= win[0]) & (centres < win[1])
    mb = (centres >= baseline[0]) & (centres < baseline[1])
    out = {l: np.array([np.nanmean(raw[l][u][mw]) - np.nanmean(raw[l][u][mb])
                        for u in units], float) for l in labels}
    return units, out


# ── 3) The stillness the cells actually differ in ─────────────────────

def stillness_table(events, stage_col):
    """Median stillness per cell, and the same interaction contrast on it.

    If the behaviour shows the crossed shape the ripples are supposed to show,
    the ripple interaction is not evidence about memory. Tested with the same
    sign-flip null used for the rates, at the same unit.
    """
    g = events.groupby(['valence', stage_col]).still_next_s
    per_cell = g.agg(n='size', median='median', mean='mean').round(3).reset_index()

    per_session = events.groupby(['session', 'valence', stage_col]) \
        .still_next_s.median().unstack([1, 2])
    stages = list(dict.fromkeys(events[stage_col]))
    stages = [s for s in (rip.STAGES if len(stages) == 3 else STAGES2)
              if s in stages]
    inter = None
    need = [('correct', stages[0]), ('error', stages[0]),
            ('correct', stages[-1]), ('error', stages[-1])]
    if all(c in per_session.columns for c in need):
        d = per_session[need].dropna()
        v = (d[need[0]] - d[need[1]]) - (d[need[2]] - d[need[3]])
        if len(v) >= 3:
            t, p = stats.ttest_1samp(v, 0.0)
            rng = np.random.default_rng(SEED)
            null = (rng.choice([-1.0, 1.0], size=(10000, len(v))) * v.to_numpy()
                    ).mean(axis=1)
            inter = {'n_sessions': int(len(v)), 'mean_s': float(v.mean()),
                     't': float(t), 'p': float(p),
                     'p_perm': float((1 + np.sum(np.abs(null) >= abs(v.mean())))
                                     / 10001),
                     'contrast': f'(correct - error | {stages[0]}) - '
                                 f'(correct - error | {stages[-1]})'}
    return per_cell, inter


# ── 4) Figures ────────────────────────────────────────────────────────

def _diff_panel(ax, profiles, centres, stages, title=None):
    """correct - error over time, one line per stage."""
    ax.axvspan(*rip.BASELINE_WIN, color='0.88', lw=0, zorder=0)
    ax.axhline(0, color='0.45', lw=0.8)
    for stage in stages:
        label = f'correct - error, {stage}'
        if label not in profiles:
            continue
        units, X = rip.baseline_subtract(profiles[label], centres)
        mean = np.nanmean(X, axis=0)
        sem = np.nanstd(X, axis=0) / max(np.sqrt(X.shape[0]), 1)
        c = stage_colour(stage)
        ax.plot(centres, mean, color=c, lw=rip.LW_RATE, zorder=3,
                label=f'{stage} (n={len(units)})')
        ax.fill_between(centres, mean - sem, mean + sem, color=c, alpha=0.22,
                        lw=0, zorder=2)
    ax.axvline(0, color='0.35', lw=1.0)
    ax.set_xlabel('Time from uncovering (s)', fontsize=rip.FS, labelpad=1)
    ax.set_ylabel('Δ rate, correct − error\n(Hz, vs own baseline)',
                  fontsize=rip.FS, labelpad=2)
    if title:
        ax.set_title(title, fontsize=rip.FS, pad=3)


def _interaction_panel(ax, raw, centres, stages, title=None, note=None):
    """The 2 x k interaction plot: window rate per cell, a line per valence."""
    labels = [f'{v}, {s}' for v in rip.VALENCE for s in stages]
    units, means = cell_window_means(raw, centres, labels)
    ax.axhline(0, color='0.45', lw=0.8)
    for valence in rip.VALENCE:
        xs, ys, es = [], [], []
        for i, stage in enumerate(stages):
            key = f'{valence}, {stage}'
            if key not in means:
                continue
            v = means[key][np.isfinite(means[key])]
            xs.append(i)
            ys.append(v.mean())
            es.append(v.std() / max(np.sqrt(v.size), 1))
        ax.errorbar(xs, ys, yerr=es, color=rip.VALENCE_COLOUR[valence],
                    lw=rip.LW, marker='o', ms=4, capsize=2, elinewidth=1.0,
                    label=f'{valence} feedback')
    _stage_ticks(ax, stages)
    ax.set_ylabel('Δ rate %.1f–%.1f s\n(Hz, vs own baseline)'
                  % PRIMARY_WINDOW, fontsize=rip.FS)
    # Sample and omnibus live inside the frame: at 3 cm wide there is no room
    # for a second x label under angled ticks.
    lines = [t for t in (note, f'n = {len(units)} sessions, all '
                               f'{len(labels)} cells') if t]
    if lines:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.30 * (hi - lo))
        ax.annotate('\n'.join(lines), xy=(0.02, 0.98),
                    xycoords='axes fraction', ha='left', va='top',
                    fontsize=rip.FS - 3, color='0.3')
    if title:
        ax.set_title(title, fontsize=rip.FS, pad=3)


def _sliding_panel(ax, sliding, keys, width_s, title=None):
    """t at every window position for the interaction contrasts."""
    ax.axhline(0, color='0.45', lw=0.8)
    drawn = []
    for key, colour, label in keys:
        res = sliding.get(key)
        if res is None:
            continue
        ax.plot(res['times'], res['t'], color=colour, lw=1.6, label=label)
        for cl in res['clusters']:
            if cl['p'] < 0.05:
                ax.axvspan(cl['start_s'], cl['stop_s'], color=colour,
                           alpha=0.16, lw=0)
                drawn.append((cl, colour))
        for sign in (1, -1):
            ax.axhline(sign * res['threshold'], color='0.6', lw=0.7, ls=':')
    if drawn:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.15 * len(drawn) * (hi - lo))
    for j, (cl, colour) in enumerate(drawn):
        ax.annotate(f"p={cl['p']:.3f}", xy=(cl['peak_s'], 0.97 - 0.10 * j),
                    xycoords=('data', 'axes fraction'), ha='center', va='top',
                    fontsize=rip.FS - 3, color=colour)
    ax.axvline(0, color='0.35', lw=1.0)
    ax.set_xlabel('Sliding window centre (s)', fontsize=rip.FS, labelpad=1)
    ax.set_ylabel('t, interaction vs 0', fontsize=rip.FS)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.16 * (hi - lo), hi)
    lo, hi = ax.get_ylim()
    y = lo + 0.06 * (hi - lo)
    ax.plot([-1.9, -1.9 + width_s], [y] * 2, color='0.25', lw=3.0,
            solid_capstyle='butt')
    ax.annotate(f'{width_s:g} s', xy=(-1.9 + width_s / 2, y), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=rip.FS - 3,
                color='0.25')
    if title:
        ax.set_title(title, fontsize=rip.FS, pad=3)


def figure_interaction(blocks, out_png, stages, width_s, suptitle, footnote):
    """One row per event set: simple effects | interaction | sliding test."""
    legend_cm, title_cm = 1.7, 0.7
    with plt.rc_context(_rc()):
        fig, axes = _axes(len(blocks), extra_cm=legend_cm + title_cm + 0.5)
        for r, block in enumerate(blocks):
            top = r == 0
            _diff_panel(axes[r][0], block['profiles'], block['centres'], stages,
                        'Valence effect per stage' if top else None)
            _interaction_panel(axes[r][1], block['raw'], block['centres'],
                               stages, 'Interaction' if top else None,
                               note=block.get('note'))
            _sliding_panel(axes[r][2], block['sliding'], block['keys'], width_s,
                           'Sliding test' if top else None)
        handles, labels = [], []
        for ax in axes[0]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        total = rip.ROW_H_CM * len(blocks) + legend_cm + title_cm + 0.5
        fig.tight_layout(pad=0.4, h_pad=1.8, w_pad=1.6,
                         rect=[LEFT_MARGIN, legend_cm / total, 1,
                               1 - title_cm / total])
        # After tight_layout, so the title sits on the row it names whatever
        # the panels ended up doing.
        for r, block in enumerate(blocks):
            pos = axes[r][0].get_position()
            fig.text(0.004, (pos.y0 + pos.y1) / 2, block['title'], rotation=90,
                     va='center', ha='left', fontsize=rip.FS,
                     fontweight='bold')
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.035), ncol=3, fontsize=rip.FS - 2,
                   frameon=False, handlelength=1.4, handletextpad=0.4,
                   columnspacing=1.6, borderaxespad=0.0)
        fig.suptitle(suptitle, fontsize=rip.FS, y=0.998, va='top')
        fig.text(0.5, 0.002, footnote, ha='center', va='bottom',
                 fontsize=rip.FS - 3, color='0.4')
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.splitext(out_png)[0] + '.pdf', bbox_inches='tight')
        plt.close(fig)
    return out_png


def figure_confound(still_cells, sweep, out_png, stages, suptitle, footnote):
    """What the behaviour does, and what the effect does as stillness is fixed."""
    legend_cm, title_cm = 1.4, 0.7
    with plt.rc_context(_rc()):
        fig, axes = _axes(1, extra_cm=legend_cm + title_cm + 0.5)
        ax = axes[0][0]
        for valence in rip.VALENCE:
            sub = still_cells[still_cells.valence == valence].set_index('stage')
            xs = [i for i, s_ in enumerate(stages) if s_ in sub.index]
            ys = [sub.loc[s_, 'median'] for s_ in stages if s_ in sub.index]
            ax.plot(xs, ys, color=rip.VALENCE_COLOUR[valence], lw=rip.LW,
                    marker='o', ms=4, label=f'{valence} feedback')
        _stage_ticks(ax, stages)
        ax.set_ylabel('Time to next key press\n(s, median)', fontsize=rip.FS)
        ax.set_title('The behaviour', fontsize=rip.FS, pad=3)

        for ax, key, name in ((axes[0][1], 'simple_first',
                               'Valence effect, first uncovers'),
                              (axes[0][2], 'interaction',
                               'Interaction (first − later)')):
            s_ = sweep[sweep.quantity == key]
            colour = (rip.VALENCE_COLOUR['correct'] if key == 'simple_first'
                      else rip.STAGE_COLOUR['once known'])
            ax.axhline(0, color='0.45', lw=0.8)
            ax.errorbar(s_.threshold_s, s_.mean_hz, yerr=s_.sem_hz,
                        color=colour, lw=rip.LW, marker='o', ms=4, capsize=2,
                        elinewidth=1.0)
            # n on one line along the bottom of the frame -- placed at the data
            # point they sat on top of the trace.
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.18 * (hi - lo), hi)
            for _, row in s_.iterrows():
                ax.annotate(f"{int(row.n_sessions)}",
                            xy=(row.threshold_s, 0.02),
                            xycoords=('data', 'axes fraction'), ha='center',
                            va='bottom', fontsize=rip.FS - 3, color='0.45')
            ax.set_xlabel('Min. stillness after event (s)', fontsize=rip.FS,
                          labelpad=1)
            ax.set_ylabel('Δ rate %.1f–%.1f s (Hz)' % PRIMARY_WINDOW,
                          fontsize=rip.FS)
            ax.set_title(name, fontsize=rip.FS, pad=3)

        total = rip.ROW_H_CM + legend_cm + title_cm + 0.5
        fig.tight_layout(pad=0.4, h_pad=1.6, w_pad=1.6,
                         rect=[0, legend_cm / total, 1, 1 - title_cm / total])
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.07),
                   ncol=2, fontsize=rip.FS - 2, frameon=False,
                   handlelength=1.4, handletextpad=0.4, borderaxespad=0.0)
        fig.suptitle(suptitle, fontsize=rip.FS, y=0.998, va='top')
        fig.text(0.5, 0.002, footnote, ha='center', va='bottom',
                 fontsize=rip.FS - 3, color='0.4')
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.splitext(out_png)[0] + '.pdf', bbox_inches='tight')
        plt.close(fig)
    return out_png


# ── 5) Reporting ──────────────────────────────────────────────────────

def _print_block(name, counts, windows, sliding):
    print(f"\n  {name}")
    print(f"    {'cell':34s} {'units':>6s} {'events':>8s}")
    for label, c in counts.items():
        print(f"    {label:34s} {c['n_units']:6d} {c['n_events_used']:8d}")
    print(f"\n    {'contrast':52s} {'n':>4s} {'mean Hz':>9s} {'t':>7s} "
          f"{'p_perm':>8s} {'p_holm':>8s}   window")
    for label, got in windows.items():
        for wname, v in got.items():
            star = ' *' if v['p_perm_holm'] < 0.05 else ''
            print(f"    {label:52s} {v['n_subjects']:4d} {v['mean_hz']:+9.4f} "
                  f"{v['t']:+7.2f} {v['p_perm']:8.4f} {v['p_perm_holm']:8.4f}   "
                  f"{wname}{star}")
    print()
    for width, per_label in sliding.items():
        for label, res in per_label.items():
            keep = [c for c in res['clusters'] if c['p'] < 0.05]
            txt = ('; '.join(f"{c['direction']} {c['start_s']:+.2f}"
                             f"..{c['stop_s']:+.2f} s (p = {c['p']:.4f})"
                             for c in keep) if keep else 'none')
            print(f"    slide {width:>4s}  {label:52s} {txt}")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()
                if k not in ('null_mass',)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ── 6) Main ───────────────────────────────────────────────────────────

def run(bundle=None, out_dir=None, unit='session', min_events=15,
        n_perm=rip.N_SIGN_FLIPS, correct_over='time',
        control_threshold_s=CONTROL_THRESHOLD_S, stillness_cache=None):
    """Valence x stage interaction, with the stillness control F1 demands."""
    root = swr_io.get_data_root()
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                              'bundle')
    if out_dir is None:
        out_dir = os.path.join(
            swr_io.derivatives_dir(root), 'group', 'swr',
            f"valence_x_stage_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_valence_stage_interaction')
    np.random.seed(SEED)

    data = rip.load_bundle(bundle)
    qc = data['channel_qc']
    qc = qc[~qc.excluded.fillna(False)] if 'excluded' in qc else qc
    print(f"\n  bundle: {bundle}")
    print(f"  {data['ripples'].session.nunique()} sessions | "
          f"{data['ripples'].subject_key.nunique()} subjects | {len(qc)} "
          f"derivations | {len(data['ripples'])} ripples")
    print(f"  unit = {unit}, min {min_events} events per {unit} per cell\n")
    print(f"  {QUESTION}")

    if stillness_cache is None:
        stillness_cache = os.path.join(out_dir, 'events_with_stillness.csv')
    events = event_table(data, cache_path=stillness_cache)
    print(f"  {len(events)} uncovering attempts, "
          f"{events.still_next_s.notna().sum()} with a stillness value")

    results = {'question': QUESTION, 'bundle': bundle, 'unit': unit,
               'min_events': min_events, 'n_sign_flips': n_perm,
               'corrected_over': correct_over,
               'baseline_window_s': list(rip.BASELINE_WIN),
               'primary_window_s': list(PRIMARY_WINDOW),
               'named_windows_s': {k: list(v) for k, v in NAMED_WINDOWS.items()},
               'seed': SEED, 'blocks': {}}
    blocks_for_fig, counts_rows = [], []

    # ---- the two crossings, with and without the stillness criterion -----
    schemes = [('3 stages', 'stage', rip.STAGES),
               ('2 stages (first vs later)', 'stage2', STAGES2)]
    settings = [('all events', None), (f'stillness ≥ '
                                       f'{control_threshold_s:g} s',
                                       control_threshold_s)]
    store = {}
    for scheme_name, stage_col, stages in schemes:
        for set_name, threshold in settings:
            key = f'{scheme_name} | {set_name}'
            print(f"\n{'=' * 78}\n {key}\n{'=' * 78}")
            centres, raw, counts = profiles_of(
                data, cells(events, stage_col, threshold), unit, min_events)
            if centres is None:
                print("  nothing computable")
                continue
            profiles, windows, sliding = test_contrasts(
                raw, centres, contrast_set(stages), n_perm, correct_over)
            _print_block(key, counts, windows, sliding)
            omni = (omnibus_interaction(raw, centres, stages, n_perm)
                    if len(stages) == 3 else None)
            if omni:
                print(f"\n    omnibus 2 x 3 interaction, {PRIMARY_WINDOW} s: "
                      f"F({omni['df1']},{omni['df2']}) = {omni['F']:.2f}, "
                      f"p_perm = {omni['p_perm']:.4f} (n = {omni['n_units']})")
            store[key] = dict(centres=centres, raw=raw, counts=counts,
                              profiles=profiles, sliding=sliding,
                              stages=stages, threshold=threshold)
            results['blocks'][key] = _jsonable(
                {'counts': counts, 'windows': windows,
                 'sliding': {w: {l: {'clusters': r['clusters'],
                                     'n_units': r['n_subjects'],
                                     'corrected_over': r['corrected_over']}
                                 for l, r in per.items()}
                             for w, per in sliding.items()},
                 'omnibus_interaction': omni,
                 'stillness_threshold_s': threshold})
            for label, c in counts.items():
                counts_rows.append(dict(scheme=scheme_name, event_set=set_name,
                                        cell=label, **c))

    pd.DataFrame(counts_rows).to_csv(
        os.path.join(out_dir, 'cell_counts.csv'), index=False)

    # ---- what the behaviour does ----------------------------------------
    print(f"\n{'=' * 78}\n stillness -- the confound F1 forces us to check\n{'=' * 78}")
    still_rows = []
    for scheme_name, stage_col, stages in schemes:
        per_cell, inter = stillness_table(events, stage_col)
        per_cell = per_cell.rename(columns={stage_col: 'stage'})
        per_cell.insert(0, 'scheme', scheme_name)
        still_rows.append(per_cell)
        print(f"\n  {scheme_name}: time to the next key press of ANY kind (s)")
        print(per_cell.to_string(index=False))
        if inter:
            print(f"    same interaction on the BEHAVIOUR: "
                  f"{inter['mean_s']:+.3f} s, t = {inter['t']:+.2f}, "
                  f"p_perm = {inter['p_perm']:.4g} (n = {inter['n_sessions']})")
        results.setdefault('stillness', {})[scheme_name] = _jsonable(
            {'per_cell': per_cell.to_dict('records'), 'interaction': inter})
    still_cells = pd.concat(still_rows, ignore_index=True)
    still_cells.to_csv(os.path.join(out_dir, 'stillness_by_cell.csv'),
                       index=False)

    # ---- the sweep: fix stillness, watch the effect ----------------------
    print(f"\n  sweeping the stillness criterion (2-stage crossing)")
    sweep_rows = []
    for threshold in STILL_THRESHOLDS_S:
        centres, raw, _ = profiles_of(
            data, cells(events, 'stage2', threshold or None), unit, min_events)
        if centres is None:
            continue
        for name, weights in (
                ('simple_first', _cross((1.0, 0.0), STAGES2)),
                ('interaction', _cross((1.0, -1.0), STAGES2))):
            if any(w not in raw for w in weights):
                continue
            p = rip.contrast_profiles(raw, weights)
            got = rip.window_test(p, centres, PRIMARY_WINDOW)
            if got is None:
                continue
            _, X = rip.baseline_subtract(p, centres)
            mw = (centres >= PRIMARY_WINDOW[0]) & (centres < PRIMARY_WINDOW[1])
            v = np.nanmean(X[:, mw], axis=1)
            v = v[np.isfinite(v)]
            sweep_rows.append(dict(
                quantity=name, threshold_s=threshold,
                n_sessions=got['n_subjects'], mean_hz=got['mean_hz'],
                sem_hz=float(v.std() / max(np.sqrt(v.size), 1)),
                t=got['t'], p_perm=got['p_perm']))
            print(f"    still >= {threshold:.1f} s  {name:13s} "
                  f"n={got['n_subjects']:3d} mean={got['mean_hz']:+.4f} "
                  f"t={got['t']:+6.2f} p_perm={got['p_perm']:.4g}")
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(out_dir, 'stillness_sweep.csv'), index=False)
    results['stillness_sweep'] = _jsonable(sweep.to_dict('records'))

    # ---- figures ---------------------------------------------------------
    width = rip.SLIDE_WIDTHS_S[0]
    keys3 = [('valence x stage, linear (first - once known)',
              INTERACTION_COLOUR['linear'], 'linear (first − once known)'),
             ('valence x stage, quadratic (first - 2*learning + known)',
              INTERACTION_COLOUR['quadratic'], 'quadratic')]
    keys2 = [('valence x stage (first - later)',
              INTERACTION_COLOUR['two-level'], 'first − later')]
    for scheme_name, stage_col, stages in schemes:
        blocks = []
        for set_name, _ in settings:
            key = f'{scheme_name} | {set_name}'
            if key not in store:
                continue
            b = store[key]
            keys = keys3 if len(stages) == 3 else keys2
            omni = results['blocks'][key].get('omnibus_interaction')
            note = (f"omnibus F({omni['df1']},{omni['df2']}) = {omni['F']:.2f}, "
                    f"p = {omni['p_perm']:.3f}" if omni else None)
            blocks.append(dict(title=set_name, centres=b['centres'],
                               raw=b['raw'], profiles=b['profiles'],
                               sliding=b['sliding'][f'{width:g}s'],
                               keys=keys, note=note))
        if not blocks:
            continue
        tag = 'three_stages' if len(stages) == 3 else 'two_stages'
        c0 = next(iter(store[f'{scheme_name} | {settings[0][0]}']['counts']
                       .values()))
        figure_interaction(
            blocks, os.path.join(out_dir, f'valence_x_stage_{tag}.png'), stages,
            width, f'Valence × stage: {QUESTION}',
            f"unit: {unit}   |   min {min_events} events per {unit} per cell   "
            f"|   {c0.get('n_sessions', '?')} sessions, "
            f"{c0.get('n_subjects', '?')} subjects, "
            f"{c0.get('n_derivations', '?')} derivations   |   "
            f"cluster-corrected over {correct_over}, {n_perm} sign-flips")
        blocks_for_fig.append(tag)

    figure_confound(
        still_cells[still_cells.scheme == '2 stages (first vs later)'],
        sweep, os.path.join(out_dir, 'valence_x_stage_stillness.png'), STAGES2,
        'Stillness, not valence: the behaviour shows the same interaction',
        'Stillness = time from the uncovering to the next key press of any kind '
        '(movement included); the criterion is applied to every cell. Numbers '
        'along the bottom of the right two panels are contributing sessions.')

    with open(os.path.join(out_dir, 'valence_x_stage_result.json'), 'w') as f:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   **results}, f, indent=2, default=str)
    print(f"\n saved -> {out_dir}")
    print(f"  figures: {', '.join('valence_x_stage_' + t for t in blocks_for_fig)}"
          f", valence_x_stage_stillness")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
