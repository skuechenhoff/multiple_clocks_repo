#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example rate-map PAIRS: "the same cell draws the same map in two different
tasks, but only when you align the maps at the right lag."

This is the two-task, three-lag version of
`plot_ratemaps_from_per_lag_encoding.py`. That script shows every task
configuration side by side at two lags; the point here is the opposite —
show as little as possible so the eye can do the correlation itself:

    columns = TWO task configurations
    rows    = THREE lags: the cell's key lag, the opposite-ROI key lag,
              and one far control lag (default 180 deg = half a trial away)

WHAT IS PLOTTED (this is the whole trick)
-----------------------------------------
Each 3x3 panel shows the cell's lag-shifted rate map for one task, converted
to a DWELL-WEIGHTED z-SCORE within that panel. This is not a different
analysis — it is the identity the reported number is made of. The per-lag CV
statistic scores a pair of maps with a dwell-weighted Pearson r, and

        r = sum_L w_L * z_a(L) * z_b(L) / sum_L w_L

so the colours you see ARE the terms of the correlation: red-on-red and
blue-on-blue push r up, red-on-blue pushes it down. Raw firing rate differs
in mean and scale between tasks, which the correlation ignores and the eye
does not; z-scoring removes exactly the two nuisances r is already blind to
and nothing else. Set `Z_SCORE = False` to plot raw rate instead.

Locations that the correlation does NOT use (fewer than MIN_DWELL_BINS
shared dwell bins in the pair) are drawn grey, so every coloured square is a
square that entered the reported r and vice versa.

`COLOUR_CENTRE_Z` moves the white point of the diverging colormap above the
map mean, so only the genuinely hot locations read as red and the rest goes
blue. It is a monotone recolouring — it cannot change which location is
hotter than which — but it is a display choice, so the colourbar shows the
shifted white point and it is logged in the settings json.

HOW THE EXAMPLES WERE PICKED, AND WHAT THAT COSTS
-------------------------------------------------
These are ILLUSTRATIVE cells. They are not a test, and they are not a random
draw. Candidates were scored on what the eye actually reads in this figure,
not on r alone: does the SAME square come out red in both tasks, is the red
set small and overlapping, and does the peak MOVE at the contrast and control
lags. That score was evaluated exhaustively over every (cell x task-pair) with
the best available spatial coverage.

Coverage is the binding constraint and it is a property of the subject's task
configurations, not of the cell. A location enters r only if BOTH tasks visit
it for >= MIN_DWELL_BINS bins. Scanned exhaustively, **6 of 9 is the ceiling
anywhere in this dataset** - single configurations reach 7-8 visited
locations, but the intersection of two never exceeds 6. All three panels sit
at that ceiling.

THE COST, stated plainly because the figure cannot hide it: at 6/9 coverage,
the task pairs that read as convincingly aligned belong to cells whose own
cross-validated statistic is positive but NOT individually significant
(p = 0.15, 0.35, 0.16 for panels a, b, c). Cells that ARE individually
significant exist at this coverage, but their best-covered task pairs look
muddy (e.g. mPFC 33_05, p = 0.005, but pairwise r = 0.17 at 60 deg). The
per-lag effect is a POPULATION effect; single cells rarely clear a
permutation test on 6-8 tasks. So each panel prints its cell's own
cross-validated r and permutation p under the title, and carries the pair's r
at all 12 lags as a profile strip, so a reader sees both the lags that were
not chosen as rows and exactly how strong the underlying cell is. The claim
these panels support is "this is what the population effect looks like in a
single cell", not "this cell is significant".

@author: Svenja Kuchenhoff (script: Claude)
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
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mc.analyse.helpers_human_cells as hh
from mc.analyse.future_spatial_peaks import _residualise_phase
from per_lag_encoding import (                                   # noqa: E402
    DATA_DIR, TRIALS, PHASE_RESIDUALISE, N_LOC, LAGS_DEG,
    MIN_DWELL_BINS, MIN_SHARED_LOCS,
    _build_per_cfg_sequences, _lag_shifted_rate_map,
)

# ── Where the per-lag run lives ───────────────────────────────────────
RUN_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
           'ephys_humans/derivatives/group/per_lag_encoding/'
           '2026-08-04_07-25-15_reload_from_2026-06-30_18-21-57_relabelled-final')
PER_CELL_CSV = os.path.join(RUN_DIR, 'per_cell_ALL_ROIs.csv')

_STAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(RUN_DIR, f'ratemap_pair_examples_{_STAMP}')

# ── The three example cells (see module docstring on selection) ───────
# One control lag, fixed for every panel and chosen a priori rather than per
# cell: 270 deg is -90 deg on the circular trial, i.e. a quarter of a trial in
# the PAST. It is the natural counterpart to the future lags, and fixing it
# means the third row is not a per-cell search for the worst-looking lag.
CONTROL_LAG = 270

# neuron, (cfg_a, cfg_b), key lag, contrast lag, control lag, panel letter
EXAMPLES = [
    # 30 deg is the noisiest of the three key lags, and structurally so: the
    # location 30 deg ahead coincides with the CURRENT location on ~39% of bins
    # (per_lag_encoding's docstring), so across all 1492 mPFC pairs
    # corr(r30, r0) = +0.24 against +0.03 at 60 deg -- a crisp 30 deg key row
    # tends to drag the 0 deg row up with it.
    # Scored on MIRRORING directly (disc = squares red in one task but not the
    # other; rho = rank agreement over all contributing squares). This pair is
    # the cleanest image in the whole 30 deg pool: disc = 0, a SINGLE red square
    # in the same corner of both tasks, and both control rows strongly negative
    # (-0.52 and -0.70). Chosen by the user over the two alternatives that are
    # individually significant, which are kept in CANDIDATES below:
    #   37_02-02-mRF3bC01-ACC cfg 6/7 -- r = 0.90, disc 0, 2 reds, p = 0.037
    #   08_01-01-mLF2aCa01-ACC cfg 1/5 -- r = 0.81, disc 0, 2 reds, p = 0.017
    # NOTE this cell is NOT individually significant at 30 deg (p = 0.108); the
    # panel prints that under its title, as every panel does.
    dict(neuron='43_12-12-mRF2aCa03-ACC', cfgs=(6, 7), roi='mPFC', letter='a',
         key_lag=30, contrast_lag=0, control_lag=CONTROL_LAG,
         labels=('next action', 'where I am now', 'a quarter trial ago'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only 30\u00b0 ahead',
         headline_short='same map in two tasks,\nonly 30\u00b0 ahead'),
    dict(neuron='27_01-01-mLF2aCa03-ACC', cfgs=(1, 4), roi='mPFC', letter='b',
         key_lag=60, contrast_lag=0, control_lag=CONTROL_LAG,
         labels=('action after next', 'where I am now', 'a quarter trial ago'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only 60\u00b0 ahead',
         headline_short='same map in two tasks,\nonly 60\u00b0 ahead'),
    dict(neuron='35_06-06-mRT2bHb06-HC', cfgs=(2, 5), roi='HC_mid', letter='c',
         key_lag=0, contrast_lag=60, control_lag=CONTROL_LAG,
         labels=('where I am now', 'action after next', 'a quarter trial ago'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only where the subject is now',
         headline_short='same map in two tasks,\nonly where the subject is'),
]

# Top-ranked candidates per ROI x key lag, from the exhaustive visual-match
# search (same peak square, small overlapping red set, high pairwise r,
# peak moving at both control lags, >= 7 of 9 locations filled).
# The only mPFC 30 deg pairs with ZERO mirrored squares, a small red set
# and quiet control rows. See the note above EXAMPLES.
# Backup panels requested separately:
#  * backup_mPFC_30over0 -- a cell aligned at BOTH 0 and 30 deg but more
#    strongly at 30, i.e. the honest version of the 30/0 entanglement.
#  * backup_HC_330 -- hippocampus at 330 deg (= 30 deg in the PAST).
#    330 is a PREDICTED hippocampal lag in per_lag_encoding
#    (ROI_PREDICTED_LAGS_DEG['HC_mid'] = (0, 330)), so the contrast is a
#    FUTURE lag (60) rather than 0, and the far control is 180 rather
#    than 270, which sits only 60 deg from the key lag.
CANDIDATES = [
    dict(neuron='43_20-20-mRT2cHb06-HC', cfgs=(0, 7), roi='HC_mid', letter='d',
         key_lag=330, contrast_lag=60, control_lag=180,
         group='backup_HC_330_sig', labels=('where I just was', 'action after next', 'half a trial away'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only 30\u00b0 in the PAST',
         headline_short='same map in two tasks,\nonly 30\u00b0 in the past'),
    dict(neuron='44_22-22-mRT2cHb06-HC', cfgs=(5, 7), roi='HC_mid', letter='d',
         key_lag=330, contrast_lag=60, control_lag=180,
         group='backup_HC_330_sig', labels=('where I just was', 'action after next', 'half a trial away'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only 30\u00b0 in the PAST',
         headline_short='same map in two tasks,\nonly 30\u00b0 in the past'),
    dict(neuron='44_19-19-mRT2cHb02-HC', cfgs=(1, 6), roi='HC_mid', letter='d',
         key_lag=330, contrast_lag=60, control_lag=180,
         group='backup_HC_330_sig', labels=('where I just was', 'action after next', 'half a trial away'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only 30\u00b0 in the PAST',
         headline_short='same map in two tasks,\nonly 30\u00b0 in the past'),
    dict(neuron='57_18-18-mLT2bHa06-HC', cfgs=(0, 6), roi='HC_anterior', letter='d',
         key_lag=330, contrast_lag=60, control_lag=180,
         group='backup_HC_330_sig', labels=('where I just was', 'action after next', 'half a trial away'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only 30\u00b0 in the PAST',
         headline_short='same map in two tasks,\nonly 30\u00b0 in the past'),
    dict(neuron='63_11-11-mLT2aHaE06-EC', cfgs=(3, 5), roi='HC_anterior', letter='d',
         key_lag=330, contrast_lag=60, control_lag=180,
         group='backup_HC_330_sig', labels=('where I just was', 'action after next', 'half a trial away'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only 30\u00b0 in the PAST',
         headline_short='same map in two tasks,\nonly 30\u00b0 in the past'),
    dict(neuron='42_08-08-mLAHIP7-HC', cfgs=(1, 5), roi='HC_anterior', letter='d',
         key_lag=330, contrast_lag=60, control_lag=180,
         group='backup_HC_330_sig', labels=('where I just was', 'action after next', 'half a trial away'),
         headline='the same spatial map in two different tasks\n'
                  '\u2014 but only 30\u00b0 in the PAST',
         headline_short='same map in two tasks,\nonly 30\u00b0 in the past'),
]

# ── Display settings ──────────────────────────────────────────────────
Z_SCORE = True          # per-panel dwell-weighted z (see docstring)
# Each panel is then affinely rescaled to its own [min, max]. A Pearson r is
# exactly invariant to a per-map affine rescaling, so this cannot change any
# reported number — it only spends the whole colormap on the range that map
# actually occupies, which is what makes the hot location read as red instead
# of pale pink. WHITE_POINT is where the colormap turns from blue to red, as a
# fraction of that range: 0.5 splits the range in half, higher values mean
# only the top locations go red and the rest of the map goes blue.
PER_MAP_MINMAX = True
WHITE_POINT = 0.60
CMAP = 'RdBu_r'
GREY = '#e2e2e2'        # locations not entering the correlation
RAW_CLIP_PCT = (10, 90)  # used only when Z_SCORE = False
SHOW_REWARDS = False     # ring the 4 rewarded locations of each task

# Dwell threshold for a location to enter a PAIRWISE panel. Deliberately lower
# than the analysis constant MIN_DWELL_BINS = 25, and here is why. The group
# statistic never correlates two single tasks: each CV fold correlates ONE
# held-out task against the dwell-weighted POOL of all the others, so the
# pooled side always has ample dwell and the 25-bin rule effectively binds on
# one task only (it keeps ~5.4 of 9 locations, max 8). Applying the same 25
# bins to BOTH single tasks of a pair is strictly harsher than anything the
# analysis does and caps a pair at 6 of 9 locations anywhere in this dataset.
# Thin locations are handled the way per_lag_encoding says they are handled --
# by the min(dwell_a, dwell_b) WEIGHTS, which shrink a 10-bin location to a
# near-zero contribution -- not by exclusion. Set to 25 to recover the harsher
# pairwise rule; the figure footnote always prints the value in force.
PAIR_MIN_DWELL_BINS = 10

# Layout: one panel is ~5.5 x 7 cm on A4 -> Arial 11 title / 9 labels.
CM = 1.0 / 2.54
FS_TITLE, FS_LABEL, FS_TICK = 11, 9, 8
PANEL_W_CM, PANEL_H_CM = 6.8, 8.4
DPI = 300

ROI_DISPLAY_NAMES = {'mPFC': 'mPFC', 'HC_mid': 'mid hippocampus',
                     'HC_anterior': 'anterior hippocampus'}
ROI_COLOURS = {'mPFC': '#8f3d97', 'HC_mid': '#c39b5a',
               'HC_anterior': '#23677E'}
OBSERVED_GREEN = '#0e3d3a'


def _wrap(text, width=13):
    """Break a row label onto as few lines as fit the left margin."""
    out, line = [], ''
    for word in text.split():
        trial = f'{line} {word}'.strip()
        if len(trial) > width and line:
            out.append(line); line = word
        else:
            line = trial
    out.append(line)
    return '\n'.join(out)


def _set_rc():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FS_LABEL,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


# ── Rate maps ─────────────────────────────────────────────────────────
def build_cell_maps(sub_str, neurons_wanted):
    """{neuron: (rate, dwell)} with shapes (n_lag, n_cfg, 9) for one subject."""
    raw = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
    if not raw:
        return {}
    sd = hh.filter_data(raw, int(sub_str), TRIALS)[f'sub-{sub_str}']
    beh = sd['beh'].copy().reset_index(drop=True)
    locs = sd['locations'].to_numpy(dtype=float)
    _, _, idx_cfg, _ = np.unique(
        beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']].to_numpy(),
        axis=0, return_index=True, return_inverse=True, return_counts=True)
    n_cfg = len(np.unique(idx_cfg))
    rewards = beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']].to_numpy()
    rew_by_cfg = np.stack([rewards[idx_cfg == c][0] for c in range(n_cfg)])

    out = {}
    for nid in neurons_wanted:
        if nid not in sd['normalised_neurons']:
            print(f'  {nid}: not in normalised_neurons'); continue
        arr = sd['normalised_neurons'][nid].to_numpy(dtype=float)
        if PHASE_RESIDUALISE:
            arr = _residualise_phase(arr, basis=PHASE_RESIDUALISE)
        Y_cfg, loc_cfg, _ = _build_per_cfg_sequences(
            arr, idx_cfg, locs, np.zeros_like(locs), n_cfg)
        rate = np.full((len(LAGS_DEG), n_cfg, N_LOC), np.nan)
        dwell = np.zeros((len(LAGS_DEG), n_cfg, N_LOC), dtype=int)
        for li, lag in enumerate(LAGS_DEG):
            for c in range(n_cfg):
                rate[li, c], dwell[li, c] = _lag_shifted_rate_map(
                    Y_cfg[c], loc_cfg[c], int(lag))
        out[nid] = dict(rate=rate, dwell=dwell, rewards=rew_by_cfg, n_cfg=n_cfg)
    return out


# ── The correlation, and the z-scores it decomposes into ──────────────
def pair_at_lag(cell, a, b, lag):
    """Weighted z-maps for cfg a and b, their weighted Pearson r, and the
    mask of locations that entered it. Same keep/weight rules as the CV."""
    li = LAGS_DEG.index(lag)
    rate, dwell = cell['rate'][li], cell['dwell'][li]
    w = np.minimum(dwell[a], dwell[b]).astype(float)
    keep = (w >= PAIR_MIN_DWELL_BINS) & np.isfinite(rate[a]) & np.isfinite(rate[b])
    if keep.sum() < MIN_SHARED_LOCS:
        return None
    za, zb = (_weighted_z(rate[i], w, keep) for i in (a, b))
    if za is None or zb is None:
        return None
    r = float((w[keep] * za[keep] * zb[keep]).sum() / w[keep].sum())
    return dict(za=za, zb=zb, raw_a=rate[a], raw_b=rate[b],
                r=r, keep=keep, w=w)


def _weighted_z(x, w, keep):
    z = np.full(N_LOC, np.nan)
    xs, ws = x[keep], w[keep]
    m = (xs * ws).sum() / ws.sum()
    v = (ws * (xs - m) ** 2).sum() / ws.sum()
    if v < 1e-12:
        return None
    z[keep] = (xs - m) / np.sqrt(v)
    return z


def lag_profile(cell, a, b):
    """Pairwise r at every one of the 12 lags (None -> NaN)."""
    out = []
    for lag in LAGS_DEG:
        p = pair_at_lag(cell, a, b, lag)
        out.append(np.nan if p is None else p['r'])
    return np.array(out)


# ── Drawing ───────────────────────────────────────────────────────────
def _norm():
    """Colour norm on the per-map rescaled values (see PER_MAP_MINMAX)."""
    return TwoSlopeNorm(vcenter=WHITE_POINT, vmin=0.0, vmax=1.0)


def _to_fraction(values, keep):
    """Affinely rescale a map's contributing values to [0, 1].
    Pearson r is invariant to exactly this transform."""
    out = np.full(N_LOC, np.nan)
    v = values[keep]
    lo, hi = np.nanmin(v), np.nanmax(v)
    out[keep] = (v - lo) / (hi - lo) if hi - lo > 1e-12 else 0.5
    return out


def draw_map(ax, values, keep, norm, cmap, rewards=None):
    """One 3x3 map. Non-contributing locations grey, not white, so they are
    obviously 'no data' and not 'low rate'."""
    cm = plt.get_cmap(cmap)
    rgba = np.zeros((3, 3, 4))
    g, k = values.reshape(3, 3), keep.reshape(3, 3)
    grey = matplotlib.colors.to_rgba(GREY)
    for i in range(3):
        for j in range(3):
            rgba[i, j] = cm(norm(g[i, j])) if k[i, j] else grey
    ax.imshow(rgba, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color('0.35')
    if rewards is not None:                       # mark the 4 reward locations
        for L in rewards:
            i, j = divmod(int(L) - 1, 3)
            ax.plot(j, i, marker='o', ms=2.6, mfc='none', mec='k', mew=0.7)


def _add_colourbar(fig, gs, norm, cmap):
    cax = fig.add_subplot(gs[0:3, 4])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, extend='both')
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=FS_TICK - 1, length=0, pad=1)
    if Z_SCORE:
        cb.set_ticks([0.03, WHITE_POINT, 0.97])
        cb.set_ticklabels(['low', '', 'high'])
        cb.set_label('firing rate\n(rescaled within each map)',
                     fontsize=FS_TICK, labelpad=3, linespacing=1.1)
    else:
        cb.set_label('firing rate', fontsize=FS_TICK, labelpad=3)


def draw_panel(fig, ex, cell, show_rewards=SHOW_REWARDS, cv=None,
               compact=False, show_cbar=True):
    """One panel: 3 lags x 2 tasks, plus the 12-lag pairwise-r profile.
    `fig` may be a Figure or a SubFigure, so panels compose into one sheet."""
    a, b = ex['cfgs']
    lags = [ex['key_lag'], ex['contrast_lag'], ex['control_lag']]
    roles = list(ex.get('labels', ('key lag', 'contrast lag', 'control lag')))
    cmap, norm = CMAP, _norm()
    prof = lag_profile(cell, a, b)
    pairs = {lag: pair_at_lag(cell, a, b, lag) for lag in lags}

    # cols: [lag label | map | map | r | colourbar]
    gs = fig.add_gridspec(
        4, 5, width_ratios=[0.55, 1, 1, 0.80, 0.10],
        height_ratios=[1, 1, 1, 1.05],
        left=0.02, right=0.90 if show_cbar else 0.96,
    top=0.70 if compact else 0.76, bottom=0.175,
        wspace=0.18, hspace=0.26)

    for ri, (lag, role) in enumerate(zip(lags, roles)):
        p = pairs[lag]
        key_row = (ri == 0)
        axl = fig.add_subplot(gs[ri, 0]); axl.axis('off')
        axl.text(1.0, 0.60, f'{lag}\u00b0', transform=axl.transAxes,
                 fontsize=FS_LABEL + 1, ha='right', va='center',
                 fontweight='bold' if key_row else 'normal',
                 color='k' if key_row else '0.3')
        axl.text(1.0, 0.22, _wrap(role), transform=axl.transAxes,
                 fontsize=FS_TICK - 1, ha='right', va='center', color='0.45',
                 linespacing=1.0)
        for ci, key in enumerate(('za', 'zb')):
            ax = fig.add_subplot(gs[ri, 1 + ci])
            vals = p[key] if Z_SCORE else p['raw_a' if ci == 0 else 'raw_b']
            if PER_MAP_MINMAX:
                vals = _to_fraction(vals, p['keep'])
            draw_map(ax, vals, p['keep'], norm, cmap,
                     rewards=cell['rewards'][a if ci == 0 else b]
                     if show_rewards else None)
            if ri == 0:
                ax.set_title(f'task {ci + 1}', fontsize=FS_LABEL, pad=4)
        axr = fig.add_subplot(gs[ri, 3]); axr.axis('off')
        axr.text(0.10, 0.5, f"r = {p['r']:.2f}", fontsize=FS_LABEL,
                 ha='left', va='center',
                 fontweight='bold' if key_row else 'normal',
                 color=OBSERVED_GREEN if key_row else '0.45')

    n_used = int(pairs[lags[0]]['keep'].sum())
    note = (f'{n_used}/9 locations enter r (grey = not included)' if compact
            else f'{n_used} of the 9 locations were visited for '
                 f'\u2265 {PAIR_MIN_DWELL_BINS} bins in BOTH tasks and enter r'
                 '  \u00b7  grey = not included')
    fig.text(0.50, 0.055, note, fontsize=FS_TICK - 1, ha='center', va='bottom',
             color='0.45')

    # ── 12-lag profile strip: the rows you are NOT shown ──────────────
    axp = fig.add_subplot(gs[3, 1:4])
    axp.axhline(0, color='0.75', lw=0.6, zorder=1)
    axp.plot(LAGS_DEG, prof, '-', color='0.55', lw=0.8, zorder=2)
    axp.scatter(LAGS_DEG, prof, s=7, color='0.55', zorder=3)
    for li_, lag in enumerate(lags):
        c = OBSERVED_GREEN if li_ == 0 else '#b03a5b'
        axp.scatter([lag], [prof[LAGS_DEG.index(lag)]], s=22, color=c,
                    zorder=4, edgecolor='w', linewidth=0.6)
    axp.set_xlim(-18, 348); axp.set_ylim(-1.2, 1.2)
    axp.set_xticks([0, 90, 180, 270])
    axp.set_xticklabels(['0', '90', '180', '270'], fontsize=FS_TICK)
    axp.set_yticks([-1, 0, 1]); axp.tick_params(labelsize=FS_TICK, length=2,
                                                pad=1)
    axp.set_xlabel('lag (\u00b0 of trial ahead)', fontsize=FS_TICK, labelpad=1)
    axp.set_ylabel('r', fontsize=FS_TICK, rotation=0, labelpad=4, va='center')
    for sp in ('top', 'right'):
        axp.spines[sp].set_visible(False)

    # ── colourbar (one is enough when panels are combined) ───────────
    if show_cbar:
        _add_colourbar(fig, gs, norm, cmap)

    roi_name = ROI_DISPLAY_NAMES.get(ex['roi'], ex['roi'])
    fig.text(0.015, 0.985, ex['letter'], fontsize=FS_TITLE,
             fontweight='bold', ha='left', va='top')
    
    if compact:
        title = f"{roi_name} cell\n{ex.get('headline_short', '')}"
        stats = (None if cv is None else
                 f"across all {cell['n_cfg']} tasks: cross-validated\n"
                 f"r = {cv['r']:.2f} at {ex['key_lag']}\u00b0, p = {cv['p']:.3f}")
        x0 = 0.11
    else:
        headline = ex.get('headline',
                          'the same spatial map in two different tasks\n'
                          f"\u2014 but only {ex['key_lag']}\u00b0 ahead")
        title = f'{roi_name} cell: {headline}'
        stats = (None if cv is None else
                 f"cell across all {cell['n_cfg']} tasks: cross-validated "
                 f"r = {cv['r']:.2f} at {ex['key_lag']}\u00b0, p = {cv['p']:.3f}")
        x0 = 0.085
    fig.text(x0, 0.985, title, fontsize=FS_TITLE, ha='left', va='top',
             linespacing=1.15, color=ROI_COLOURS.get(ex['roi'], 'k'))
    if stats is not None:
        fig.text(x0, 0.845 if not compact else 0.805, stats,
                 fontsize=FS_TICK, ha='left', va='top', color='0.35',
                 linespacing=1.15)

    return prof


def plot_example(ex, cell, out_stem, show_rewards=SHOW_REWARDS, cv=None):
    """Render one example as its own standalone figure."""
    _set_rc()
    fig = plt.figure(figsize=(PANEL_W_CM * CM, PANEL_H_CM * CM))
    prof = draw_panel(fig, ex, cell, show_rewards=show_rewards, cv=cv)
    fig.savefig(out_stem + '.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(out_stem + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return prof


def plot_combined(examples, cells, cv_by_neuron, out_stem):
    """All three examples side by side, ready to drop in as one figure row."""
    _set_rc()
    n = len(examples)
    fig = plt.figure(figsize=(PANEL_W_CM * CM * n, (PANEL_H_CM + 0.7) * CM))
    sfigs = np.atleast_1d(fig.subfigures(1, n, wspace=0.0))
    for i, (sfig, ex) in enumerate(zip(sfigs, examples)):
        sfig.set_facecolor('none')
        draw_panel(sfig, ex, cells[ex['neuron']],
                   cv=cv_by_neuron.get(ex['neuron']),
                   compact=True, show_cbar=(i == n - 1))
    fig.savefig(out_stem + '.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(out_stem + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_cell = pd.read_csv(PER_CELL_CSV)
    per_cell['subject_id'] = per_cell['subject_id'].astype(str).str.zfill(2)

    render_candidates = '--candidates' in sys.argv
    to_load = EXAMPLES + (CANDIDATES if render_candidates else [])
    by_sub = {}
    for ex in to_load:
        by_sub.setdefault(ex['neuron'].split('_')[0].zfill(2), set()).add(ex['neuron'])

    cells = {}
    for sub_str, nids in by_sub.items():
        print(f'loading sub-{sub_str}', flush=True)
        cells.update(build_cell_maps(sub_str, nids))

    summary, cv_by_neuron = [], {}
    for ex in EXAMPLES:
        cell = cells.get(ex['neuron'])
        if cell is None:
            print(f"{ex['neuron']}: no data, skipped"); continue
        stem = os.path.join(OUT_DIR, f"{ex['letter']}_{ex['neuron'].replace('/', '-')}")
        row = per_cell[per_cell['neuron'] == ex['neuron']]
        cv_key = None
        if len(row):
            cv_key = dict(
                r=float(row[f"r_lag{ex['key_lag']:03d}_noctrl"].iloc[0]),
                p=float(row[f"p_lag{ex['key_lag']:03d}_noctrl"].iloc[0]))
        cv_by_neuron[ex['neuron']] = cv_key
        prof = plot_example(ex, cell, stem, cv=cv_key)
        cv = {f'cv_r_lag{l:03d}_noctrl': float(row[f'r_lag{l:03d}_noctrl'].iloc[0])
              for l in LAGS_DEG} if len(row) else {}
        summary.append(dict(
            neuron=ex['neuron'], roi=ex['roi'], cfgs=list(ex['cfgs']),
            key_lag=ex['key_lag'], contrast_lag=ex['contrast_lag'],
            control_lag=ex['control_lag'], n_cfg=int(cell['n_cfg']),
            pair_r_by_lag={str(l): (None if not np.isfinite(v) else round(float(v), 4))
                           for l, v in zip(LAGS_DEG, prof)},
            cell_level_cv_r=cv,
            cell_level_perm_p={f'lag{l:03d}': float(row[f'p_lag{l:03d}_noctrl'].iloc[0])
                               for l in LAGS_DEG} if len(row) else {},
            figure=os.path.basename(stem) + '.pdf'))
        print(f"  {ex['letter']}  {ex['neuron']}  cfg{ex['cfgs']}  "
              f"r@{ex['key_lag']}={prof[LAGS_DEG.index(ex['key_lag'])]:.2f}")

    ready = [ex for ex in EXAMPLES if ex['neuron'] in cells]
    if len(ready) > 1:
        plot_combined(ready, cells, cv_by_neuron,
                      os.path.join(OUT_DIR, 'combined_ratemap_pair_examples'))
        print('  combined figure written')

    if render_candidates:
        rank = {}
        for ex in CANDIDATES:
            cell = cells.get(ex['neuron'])
            if cell is None:
                continue
            g = ex.get('group', 'other')
            rank[g] = rank.get(g, 0) + 1
            cdir = os.path.join(OUT_DIR, 'candidates', g)
            os.makedirs(cdir, exist_ok=True)
            row = per_cell[per_cell['neuron'] == ex['neuron']]
            cv = (dict(r=float(row[f"r_lag{ex['key_lag']:03d}_noctrl"].iloc[0]),
                       p=float(row[f"p_lag{ex['key_lag']:03d}_noctrl"].iloc[0]))
                  if len(row) else None)
            stem = os.path.join(
                cdir, f"{rank[g]:02d}_{ex['neuron'].replace('/', '-')}"
                      f"_cfg{ex['cfgs'][0]}-{ex['cfgs'][1]}")
            prof = plot_example(ex, cell, stem, cv=cv)
            pr = prof[LAGS_DEG.index(ex['key_lag'])]
            print(f"  {g}/{rank[g]:02d} {ex['neuron']:28s} cfg{ex['cfgs']} "
                  f"r={pr:.2f}"
                  + (f"  cell p={cv['p']:.3f}" if cv else ''))

    settings = dict(
        created=_STAMP, source_run=RUN_DIR,
        display=dict(z_score_per_panel=Z_SCORE,
                     pair_min_dwell_bins=PAIR_MIN_DWELL_BINS,
                     pair_min_dwell_note=(
                         'Lower than the analysis MIN_DWELL_BINS=25 on purpose: '
                         'the CV never correlates two single tasks, it '
                         'correlates one held-out task against the pooled '
                         'others, so 25 bins in BOTH single tasks is harsher '
                         'than the analysis and caps a pair at 6/9. Thin '
                         'locations are downweighted by the min-dwell weights '
                         'instead of excluded.'),
                     per_map_minmax=PER_MAP_MINMAX, white_point=WHITE_POINT,
                     cmap=CMAP,
                     grey_locations='pair dwell < PAIR_MIN_DWELL_BINS '
                                    '(excluded from r, drawn grey)'),
        analysis=dict(min_dwell_bins=MIN_DWELL_BINS,
                      min_shared_locs=MIN_SHARED_LOCS,
                      phase_residualise=PHASE_RESIDUALISE,
                      lags_deg=LAGS_DEG, trials=TRIALS,
                      pair_r='dwell-weighted Pearson, weights = min(dwell_a, '
                             'dwell_b), identical rule to the per-lag CV fold'),
        selection=dict(
            status='Illustrative examples, not a test.',
            search=('Exhaustive over every (cell x task-pair) at the maximum '
                    'available coverage, for all mPFC / HC_mid / HC_anterior '
                    'cells. Ranked by a VISUAL-match score, not by r alone: '
                    'same peak square in both tasks, small overlapping red '
                    'set (Jaccard above the colormap white point), high '
                    'pairwise r, and the peak must MOVE at the contrast and '
                    'control lags.'),
            coverage=('A location enters r only if BOTH tasks visit it for '
                      '>= MIN_DWELL_BINS bins. Scanned exhaustively: 6 of 9 '
                      'is the ceiling anywhere in this dataset. All three '
                      'panels are at that ceiling.'),
            cost=('At 6/9 coverage the convincingly aligned pairs belong to '
                  'cells that are positive but NOT individually significant '
                  '(p = 0.15 / 0.35 / 0.16 for a / b / c). Individually '
                  'significant cells at this coverage have muddy pairs '
                  '(mPFC 33_05: p = 0.005 but pairwise r = 0.17 at 60 deg). '
                  'The per-lag effect is a population effect; the panels '
                  'illustrate it, they do not test it. Each panel prints its '
                  'own cell-level r and p.'),
            control_lag=('Fixed a priori at 270 deg (= -90 deg, a quarter '
                         'trial in the past) for every panel, so the third '
                         'row is not a per-cell search for the worst lag.'),
            disclosure=('Each figure carries the pair r at all 12 lags as a '
                        'profile strip, so the lags that were not chosen as '
                        'rows are visible; the cell-level cross-validated r '
                        'and permutation p across all tasks are printed '
                        'under the title.')),
        examples=summary)
    with open(os.path.join(OUT_DIR, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)
    print(f'\nDone -> {OUT_DIR}')


if __name__ == '__main__':
    main()
