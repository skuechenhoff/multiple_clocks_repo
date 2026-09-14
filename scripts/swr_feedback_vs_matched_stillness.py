#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is a feedback moment anything more than a pause of the same length?

SK's question, and it is the right one. Ripple rate rises with stillness (F1),
and feedback moments differ in how much stillness follows them, so before any
claim about feedback we need two numbers:

  1) HOW MUCH DOES STILLNESS ALONE DO?  Measured on MOVEMENT presses -- presses
     that uncover nothing, so no information arrives -- binned by how long the
     subject then sits still. That is a dose-response curve for stillness with
     no feedback anywhere in it.

  2) IS FEEDBACK MORE THAN THAT?  Inside each stillness bin, contrast feedback
     events against movement presses from the SAME session and the SAME bin.
     Both are button presses, both are followed by the same amount of
     stillness; the only thing left is whether information arrived. If feedback
     exceeds the matched pause, it is not stillness. If it does not, the
     "feedback effect" is a pause.

Why movement presses rather than random time points: a random time point has no
motor onset, so it would differ from an uncovering in two ways at once. A
movement press is the same motor act with nothing revealed. N1 in this project
found no ripple modulation at movement presses (+0.013 Hz, p = 0.14), so the
control is close to neutral on its own.

Why stratify rather than 1:1 match: stratification uses every event and needs
no sampling, so there is no seed and no discarded data. The long-stillness bins
are where the control pool is thinnest (1,608 movement presses beyond 2.5 s),
which is exactly why 1:1 matching would have to sample with replacement there.

All four feedback cells are run, so "positive feedback early" and "negative
feedback late" -- the two SK flagged -- are tested the same way as the others.

Rates, contrasts and tests are the pipeline's own: `rip.rate_by_unit`,
`rip.contrast_profiles`, `rip.window_test`, `rip.sliding_window_test`.

    python scripts/swr_feedback_vs_matched_stillness.py

@author: Svenja Kuchenhoff
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip
import mc.analyse.swr_probes as prb
from swr_valence_stage_interaction import event_table, _rc, _axes, PRIMARY_WINDOW
from swr_stillness_control import (BIN_EDGES, BIN_LABELS, BIN_COLOURS, STAGE2,
                                   window_value)

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


QUESTION = ('Does a feedback moment carry more ripples than a movement press '
            'followed by the same amount of stillness?')

# The four feedback cells, plus the control. `pause` is the reference every
# feedback cell is contrasted against inside its own stillness bin.
CELLS = (('correct, first uncovers', 'correct', 'first uncovers'),
         ('correct, later', 'correct', 'later'),
         ('error, first uncovers', 'error', 'first uncovers'),
         ('error, later', 'error', 'later'))
PAUSE = 'pause (movement press)'
PAUSE_COLOUR = '#252525'
SHAPE_BIN = '1.5-2.5 s'          # the bin the time-course panels are drawn from
SEED = 42


def cell_colour(label):
    """Valence sets the hue, stage the lightness -- the project's own rule."""
    valence = 'correct' if label.startswith('correct') else 'error'
    base = np.array(plt.matplotlib.colors.to_rgb(rip.VALENCE_COLOUR[valence]))
    light = 0.0 if 'first' in label else 0.45
    return tuple(base + (1.0 - base) * light)


# ── 1) The control events ─────────────────────────────────────────────

def pause_table(data, cache_path=None):
    """Every MOVEMENT press, with how long the subject then sits still.

    Stillness is measured against the full press train (movement and uncover
    together), so it means the same thing here as it does for a feedback event.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  pauses: reusing {os.path.basename(cache_path)}")
        return pd.read_csv(cache_path)

    beh = data['behaviour']
    rows = []
    for session in rip.sessions_in(data):
        move_t, uncover_t = prb.press_times(session, beh[beh.session == session])
        presses = np.sort(np.concatenate([move_t, uncover_t]))
        t = np.sort(np.asarray(move_t, float))
        if not t.size or not presses.size:
            continue
        i_next = np.searchsorted(presses, t, 'right')
        nxt = np.where(i_next < presses.size,
                       presses[np.clip(i_next, 0, presses.size - 1)],
                       np.nan) - t
        rows.append(pd.DataFrame({'session': session, 't_s': t,
                                  'still_next_s': nxt}))
    out = pd.concat(rows, ignore_index=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out.to_csv(cache_path, index=False)
    return out


def binned(frame):
    return frame.assign(still_bin=pd.cut(frame.still_next_s, list(BIN_EDGES),
                                         labels=list(BIN_LABELS), right=False))


def per_session(frame):
    return {int(s): g.t_s.to_numpy(float) for s, g in frame.groupby('session')}


# ── 2) Rates inside each stillness bin ────────────────────────────────

def rates_by_bin(data, events, pauses, unit, min_events):
    """{bin: {cell label: per-unit profile}} for the four cells and the control."""
    out, counts, centres = {}, {}, None
    for b in BIN_LABELS:
        got, cnt = {}, {}
        pool = pauses[pauses.still_bin == b]
        if len(pool):
            c, per_unit, n = rip.rate_by_unit(data, per_session(pool),
                                              unit=unit, min_events=min_events)
            if c is not None and per_unit:
                centres = c
                got[PAUSE], cnt[PAUSE] = per_unit, n
        for label, valence, stage in CELLS:
            sub = events[(events.valence == valence) & (events.stage2 == stage)
                         & (events.still_bin == b)]
            if not len(sub):
                continue
            c, per_unit, n = rip.rate_by_unit(data, per_session(sub),
                                              unit=unit, min_events=min_events)
            if c is not None and per_unit:
                centres = c
                got[label], cnt[label] = per_unit, n
        out[b], counts[b] = got, cnt
        line = '  '.join(f"{k.split(',')[0][:7]}:{v['n_units']}"
                         for k, v in cnt.items())
        print(f"    {b:10s} {line}")
    return centres, out, counts


def contrast_vs_pause(raw_by_bin, centres, label):
    """{bin: contrast profiles} for one feedback cell minus the matched pause."""
    out = {}
    for b, raw in raw_by_bin.items():
        if label not in raw or PAUSE not in raw:
            continue
        p = rip.contrast_profiles(raw, {label: 1.0, PAUSE: -1.0})
        if len(p) >= 3:
            out[b] = p
    return out


def standardise(per_bin):
    """Each session's mean over the bins it has -- a flat stillness weighting."""
    per_unit = {}
    for b, profiles in per_bin.items():
        for unit, profile in profiles.items():
            per_unit.setdefault(unit, []).append(profile)
    return {u: np.nanmean(np.vstack(v), axis=0)
            for u, v in per_unit.items() if len(v) >= 2}


# ── 3) Figure ─────────────────────────────────────────────────────────

def _timecourse(ax, raw, centres, labels, colours, title, ylabel):
    ax.axvspan(*rip.BASELINE_WIN, color='0.88', lw=0, zorder=0)
    ax.axhline(0, color='0.45', lw=0.8)
    for label, colour in zip(labels, colours):
        if label not in raw:
            continue
        units, X = rip.baseline_subtract(raw[label], centres)
        mean = np.nanmean(X, axis=0)
        sem = np.nanstd(X, axis=0) / max(np.sqrt(X.shape[0]), 1)
        short = PAUSE if label == PAUSE else label
        ax.plot(centres, mean, color=colour, lw=rip.LW_RATE, zorder=3,
                label=f'{short} (n={len(units)})')
        ax.fill_between(centres, mean - sem, mean + sem, color=colour,
                        alpha=0.20, lw=0, zorder=2)
    ax.axvline(0, color='0.35', lw=1.0)
    ax.set_xlabel('Time from press (s)', fontsize=rip.FS, labelpad=1)
    ax.set_ylabel(ylabel, fontsize=rip.FS - 1)
    ax.set_title(title, fontsize=rip.FS, pad=3)


def _bin_ticks(ax):
    ax.set_xticks(range(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS, fontsize=rip.FS - 3, rotation=30,
                       ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.5, len(BIN_LABELS) - 0.5)


def figure(out_png, centres, raw_by_bin, per_cell, standardised, sliding,
           suptitle, footnote):
    legend_cm, title_cm = 1.8, 0.7
    with plt.rc_context(_rc()):
        fig, axes = _axes(2, extra_cm=legend_cm + title_cm + 0.6)

        # ---- (0,0) stillness alone: the dose-response at movement presses --
        ax = axes[0][0]
        ax.axhline(0, color='0.45', lw=0.8)
        xs, ys, es, ns = [], [], [], []
        for i, b in enumerate(BIN_LABELS):
            raw = raw_by_bin.get(b, {})
            if PAUSE not in raw:
                continue
            _, v = window_value(raw[PAUSE], centres)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            xs.append(i)
            ys.append(v.mean())
            es.append(v.std() / max(np.sqrt(v.size), 1))
            ns.append(v.size)
        ax.errorbar(xs, ys, yerr=es, color=PAUSE_COLOUR, lw=rip.LW, marker='o',
                    ms=4, capsize=2, elinewidth=1.0)
        for x, n in zip(xs, ns):
            ax.annotate(f"{n}", xy=(x, 0.02), xycoords=('data', 'axes fraction'),
                        ha='center', va='bottom', fontsize=rip.FS - 3,
                        color='0.45')
        _bin_ticks(ax)
        ax.set_ylabel('Δ rate %.1f–%.1f s\n(Hz, vs own baseline)'
                      % PRIMARY_WINDOW, fontsize=rip.FS - 1)
        ax.set_title('Stillness alone (movement presses)', fontsize=rip.FS,
                     pad=3)

        # ---- (0,1) feedback minus matched pause, per bin -------------------
        ax = axes[0][1]
        ax.axhline(0, color='0.45', lw=0.8)
        for label, _, _ in CELLS:
            per_bin = per_cell.get(label, {})
            xs, ys, es = [], [], []
            for i, b in enumerate(BIN_LABELS):
                if b not in per_bin:
                    continue
                _, v = window_value(per_bin[b], centres)
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                xs.append(i)
                ys.append(v.mean())
                es.append(v.std() / max(np.sqrt(v.size), 1))
            ax.errorbar(xs, ys, yerr=es, color=cell_colour(label), lw=rip.LW,
                        marker='o', ms=3.5, capsize=2, elinewidth=0.9,
                        label=label)
        _bin_ticks(ax)
        ax.set_ylabel('Feedback − matched pause\n(Hz, %.1f–%.1f s)'
                      % PRIMARY_WINDOW, fontsize=rip.FS - 1)
        ax.set_title('Feedback − matched pause', fontsize=rip.FS, pad=3)

        # ---- (0,2) the same, standardised over bins ------------------------
        ax = axes[0][2]
        ax.axhline(0, color='0.45', lw=0.8)
        for i, (label, _, _) in enumerate(CELLS):
            std = standardised.get(label)
            if not std:
                continue
            _, v = window_value(std, centres)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            ax.errorbar([i], [v.mean()],
                        yerr=[v.std() / max(np.sqrt(v.size), 1)],
                        color=cell_colour(label), marker='o', ms=5, capsize=3,
                        elinewidth=1.4, lw=0)
            ax.annotate(f"{v.size}", xy=(i, 0.02),
                        xycoords=('data', 'axes fraction'), ha='center',
                        va='bottom', fontsize=rip.FS - 3, color='0.45')
        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels([c[0].replace(', ', '\n') for c in CELLS],
                           fontsize=rip.FS - 3, rotation=30, ha='right',
                           rotation_mode='anchor')
        ax.set_xlim(-0.5, len(CELLS) - 0.5)
        ax.set_ylabel('Feedback − matched pause\n(Hz, stillness-standardised)',
                      fontsize=rip.FS - 1)
        ax.set_title('Standardised over bins', fontsize=rip.FS, pad=3)

        # ---- (1,0) and (1,1) the shape at matched stillness ----------------
        raw = raw_by_bin.get(SHAPE_BIN, {})
        _timecourse(axes[1][0], raw, centres,
                    ['correct, first uncovers', PAUSE],
                    [cell_colour('correct, first uncovers'), PAUSE_COLOUR],
                    f'Matched stillness ({SHAPE_BIN})',
                    'Δ rate (Hz, vs own baseline)')
        _timecourse(axes[1][1], raw, centres, ['error, later', PAUSE],
                    [cell_colour('error, later'), PAUSE_COLOUR],
                    f'Matched stillness ({SHAPE_BIN})',
                    'Δ rate (Hz, vs own baseline)')

        # ---- (1,2) sliding test on the standardised contrast ---------------
        ax = axes[1][2]
        ax.axhline(0, color='0.45', lw=0.8)
        for label, res in (sliding or {}).items():
            if res is None:
                continue
            ax.plot(res['times'], res['t'], color=cell_colour(label), lw=1.5)
            for cl in res['clusters']:
                if cl['p'] < 0.05:
                    ax.axvspan(cl['start_s'], cl['stop_s'],
                               color=cell_colour(label), alpha=0.16, lw=0)
                    ax.annotate(f"p={cl['p']:.3f}", xy=(cl['peak_s'], 0.97),
                                xycoords=('data', 'axes fraction'),
                                ha='center', va='top', fontsize=rip.FS - 3,
                                color=cell_colour(label))
            for sign in (1, -1):
                ax.axhline(sign * res['threshold'], color='0.6', lw=0.7, ls=':')
        ax.axvline(0, color='0.35', lw=1.0)
        ax.set_xlabel('Sliding window centre (s)', fontsize=rip.FS, labelpad=1)
        ax.set_ylabel('t, feedback − pause\n(standardised)',
                      fontsize=rip.FS - 1)
        ax.set_title('Cluster test', fontsize=rip.FS, pad=3)

        handles, labels = [], []
        for ax in (axes[0][1], axes[1][0], axes[1][1]):
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        total = rip.ROW_H_CM * 2 + legend_cm + title_cm + 0.6
        fig.tight_layout(pad=0.4, h_pad=2.0, w_pad=1.8,
                         rect=[0, legend_cm / total, 1, 1 - title_cm / total])
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.035), ncol=3, fontsize=rip.FS - 2,
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

def run(bundle=None, out_dir=None, unit='session', min_events=10,
        n_perm=rip.N_SIGN_FLIPS, stillness_cache=None, pause_cache=None):
    root = swr_io.get_data_root()
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(root), 'group', 'swr',
                              'bundle')
    if out_dir is None:
        out_dir = os.path.join(
            swr_io.derivatives_dir(root), 'group', 'swr',
            f"feedback_vs_matched_stillness_{datetime.now():%Y-%m-%d}")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_feedback_vs_matched_stillness')
    np.random.seed(SEED)

    data = rip.load_bundle(bundle)
    print(f"\n  bundle: {bundle}\n  {QUESTION}")
    print(f"  unit = {unit}, min {min_events} events per {unit} per cell "
          f"per bin\n")

    if stillness_cache is None:
        stillness_cache = os.path.join(out_dir, 'events_with_stillness.csv')
    if pause_cache is None:
        pause_cache = os.path.join(out_dir, 'movement_presses.csv')
    events = binned(event_table(data, cache_path=stillness_cache))
    events = events.assign(stage2=events.stage.map(STAGE2))
    pauses = binned(pause_table(data, cache_path=pause_cache))
    print(f"  {len(events)} uncoverings | {len(pauses)} movement presses")
    print("\n  events per stillness bin")
    print(pd.crosstab(pauses.still_bin, 'pause').to_string())

    print("\n  usable sessions per bin")
    centres, raw_by_bin, counts = rates_by_bin(data, events, pauses, unit,
                                               min_events)
    if centres is None:
        print("  nothing computable")
        return None

    results = {'question': QUESTION, 'bundle': bundle, 'unit': unit,
               'min_events': min_events, 'n_sign_flips': n_perm, 'seed': SEED,
               'bin_edges_s': [float(e) for e in BIN_EDGES],
               'primary_window_s': list(PRIMARY_WINDOW),
               'baseline_window_s': list(rip.BASELINE_WIN),
               'stillness_alone': {}, 'cells': {}}

    # ---- 1) how much does stillness alone do? ----------------------------
    print("\n  1) stillness alone -- movement presses vs their own baseline")
    for b in BIN_LABELS:
        raw = raw_by_bin.get(b, {})
        if PAUSE not in raw:
            continue
        got = rip.window_test(raw[PAUSE], centres, PRIMARY_WINDOW)
        if got is None:
            continue
        results['stillness_alone'][b] = got
        star = ' *' if got['p_perm'] < 0.05 else ''
        print(f"    {b:10s} n={got['n_subjects']:3d} "
              f"mean={got['mean_hz']:+.4f} t={got['t']:+6.2f} "
              f"p_perm={got['p_perm']:.4g}{star}")

    # ---- 2) is feedback more than a matched pause? ------------------------
    print("\n  2) feedback minus matched pause, inside each stillness bin")
    per_cell, standardised, sliding = {}, {}, {}
    rows = []
    for label, _, _ in CELLS:
        per_bin = contrast_vs_pause(raw_by_bin, centres, label)
        if not per_bin:
            continue
        per_cell[label] = per_bin
        results['cells'].setdefault(label, {'bins': {}})
        print(f"\n    {label}")
        for b in BIN_LABELS:
            if b not in per_bin:
                continue
            got = rip.window_test(per_bin[b], centres, PRIMARY_WINDOW)
            if got is None:
                continue
            star = ' *' if got['p_perm'] < 0.05 else ''
            print(f"      {b:10s} n={got['n_subjects']:3d} "
                  f"mean={got['mean_hz']:+.4f} t={got['t']:+6.2f} "
                  f"p_perm={got['p_perm']:.4g}{star}")
            rows.append(dict(cell=label, stratum=b, **{
                k: v for k, v in got.items() if k != 'window_s'}))
            results['cells'][label]['bins'][b] = got

        std = standardise(per_bin)
        if len(std) < 3:
            continue
        standardised[label] = std
        got = rip.window_test(std, centres, PRIMARY_WINDOW)
        if got is not None:
            star = ' *' if got['p_perm'] < 0.05 else ''
            print(f"      {'STANDARDISED':10s} n={got['n_subjects']:3d} "
                  f"mean={got['mean_hz']:+.4f} t={got['t']:+6.2f} "
                  f"p_perm={got['p_perm']:.4g}{star}")
            rows.append(dict(cell=label, stratum='standardised', **{
                k: v for k, v in got.items() if k != 'window_s'}))
            results['cells'][label]['standardised'] = got
        units, X = rip.baseline_subtract(std, centres)
        res = rip.sliding_window_test(
            {label: {u: X[i] for i, u in enumerate(units)}}, centres,
            width_s=rip.SLIDE_WIDTHS_S[0], n_perm=n_perm, seed=SEED)
        sliding[label] = res.get(label)
        keep = ([c for c in sliding[label]['clusters'] if c['p'] < 0.05]
                if sliding[label] else [])
        print(f"      {'sliding':10s}     " +
              ('; '.join(f"{c['direction']} {c['start_s']:+.2f}.."
                         f"{c['stop_s']:+.2f} p={c['p']:.4f}" for c in keep)
               if keep else 'none'))
        results['cells'][label]['sliding_clusters'] = keep

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'feedback_vs_pause.csv'),
                              index=False)
    figure(os.path.join(out_dir, 'feedback_vs_matched_stillness.png'), centres,
           raw_by_bin, per_cell, standardised, sliding,
           'Is a feedback moment more than a pause of the same length?',
           f'Control = movement presses (nothing uncovered) from the same '
           f'session and the same stillness bin. Stillness = time to the next '
           f'key press of any kind. unit = {unit}, min {min_events} events per '
           f'{unit} per cell per bin.')

    with open(os.path.join(out_dir, 'matched_stillness_result.json'), 'w') as f:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   **results}, f, indent=2, default=str)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
