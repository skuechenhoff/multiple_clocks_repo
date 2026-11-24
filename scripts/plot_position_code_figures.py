#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chapter 5 figures: the position-in-sequence index.

Both levels of analysis are run on traces from which the two signals of no
interest have been removed at the data level, in this order:
  1. goal progress   (within-state cosine basis)
  2. the per-loop monotone ramp (least-squares linear trend per trial)
What survives is an ABSOLUTE index of position, not a relative code.

  Fig A  population RSA (5 ROIs; EC not testable)
  Fig B  single cells (6 ROIs), all cells and the ramp-robust subset
  Fig C  preferred position among position-coding cells, with and without
         the residualisation, which makes the cost of the control explicit

Inference is Benjamini-Hochberg FDR across ROIs everywhere. For the cells the
estimator is the Hodges-Lehmann location with its distribution-free interval,
because that is the quantity the Wilcoxon signed-rank test actually tests --
a mean with its CI can cross zero while the rank test is significant, which
made an earlier version of this figure contradict itself.
"""
import os, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
import era_brewer

G = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
RSA_RUN  = G + "/DSR_RSA_simple_ROI/2026-10-02_13-57-52"           # phase + ramp
ENC_RUN  = G + "/encoding_state_sustained_cv/2026-10-02_13-54-14"  # phase + ramp
ENC_RAW  = G + "/encoding_state_sustained_cv/2026-09-24_18-35-23"  # no residualisation
OUT      = G + "/position_index_2026-10-02"
os.makedirs(OUT, exist_ok=True)

CM = 1/2.54
_p = era_brewer.era_brew("Showgirl2", n=7)
ROI_COLOURS = {'EC': _p[0], 'mPFC': _p[1], 'HC_anterior': _p[2], 'PCC': _p[3],
               'mOFC': _p[4], 'HC_mid': '#a30d6c'}
STATE_COLOURS = {'A': '#F15A29', 'B': '#F7931E', 'C': '#C7C6E2', 'D': '#6B60AA'}
LABEL = {'HC_anterior': 'HC ant', 'HC_mid': 'HC mid'}
lab = lambda r: LABEL.get(r, r)
FIXED = ['EC', 'mOFC', 'mPFC', 'HC_anterior', 'HC_mid', 'PCC']
ordered = lambda avail: [r for r in FIXED if r in set(avail)]
SIG = 0.05

def _rc():
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.fontsize': 8, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})

def _save(fig, stem):
    fig.savefig(stem + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig); print("wrote", stem + '.pdf')

def hodges_lehmann(v, alpha=0.05):
    """HL location estimate + its distribution-free CI (Walsh averages).

    This is the location the Wilcoxon signed-rank test tests, so the interval
    and the p-value can never disagree.
    """
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    n = v.size
    if n < 6:
        return np.nan, np.nan, np.nan
    w = np.sort(((v[:, None] + v[None, :]) / 2)[np.triu_indices(n)])
    M = w.size
    z = stats.norm.ppf(1 - alpha / 2)
    k = int(np.floor(n*(n+1)/4 - z*np.sqrt(n*(n+1)*(2*n+1)/24)))
    k = max(k, 0)
    return float(np.median(w)), float(w[k]), float(w[M-1-k])

# ------------------------------------------------------------------ helpers
def rsa_position():
    d = pd.read_csv(RSA_RUN + "/results_summary_combos.csv")
    d = d[(d.test == 'split_halves_z') & (d.combo == 'S_P_ctrl')
          & (d.sub_model == 'state')]
    return d.set_index('roi')

def cell_position(keep_pref=None):
    """Per-ROI Hodges-Lehmann on the held-out position r, FDR across ROIs."""
    r = pd.read_csv(ENC_RUN + "/state_sustained_cv_results.csv")
    order = ordered(r['roi'].unique())
    rows = []
    for roi in order:
        d = r[r['roi'] == roi]
        if keep_pref is not None:
            d = d[d['pref_state_mode'].isin(keep_pref)]
        v = d['r_state'].dropna().to_numpy()
        e, lo, hi = hodges_lehmann(v)
        pv = (stats.wilcoxon(v, alternative='greater').pvalue
              if v.size > 5 else np.nan)
        rows.append(dict(roi=roi, n=int(v.size), hl=e, lo=lo, hi=hi, p=pv))
    out = pd.DataFrame(rows)
    out['q'] = multipletests(out['p'].fillna(1.0), method='fdr_bh')[1]
    return out

def _style_axis(ax, order, ylabel, title):
    ax.set_xticks(range(len(order))); ax.set_xlim(-0.65, len(order) - 0.35)
    ax.set_xticklabels([lab(x) for x in order], rotation=30, ha='right')
    ax.set_ylabel(ylabel, labelpad=2, fontsize=8.5)
    ax.set_title(title, loc='left', fontweight='bold', pad=6)
    ax.margins(y=0.20)

# ------------------------------------------------------------------ Fig 1
def fig_main():
    """Population RSA and single cells, side by side, matched ROI order."""
    _rc()
    rsa = rsa_position()
    cells = cell_position()
    # keep one shared x-axis across the two panels; ROIs the RSA could not
    # test (too few subjects contribute electrodes) are left empty and marked,
    # rather than dropped, so the panels stay readable against each other
    rsa_order = list(cells['roi'])
    fig, axes = plt.subplots(1, 2, figsize=(13.5 * CM, 3.9 * CM),
                             gridspec_kw={'wspace': 0.42})

    boot = {}
    bpath = OUT + '/rsa_roi_bootstrap_summary.json'
    if os.path.exists(bpath):
        with open(bpath) as f:
            boot = json.load(f)['per_roi']

    ax = axes[0]
    for i, roi in enumerate(rsa_order):
        if roi not in rsa.index:
            ax.text(i, .004, 'not\ntested', ha='center', va='bottom',
                    fontsize=6.5, color='0.55', linespacing=1.1)
            continue
        b = float(rsa.loc[roi, 'beta']); sig = float(rsa.loc[roi, 'q_fdr_per_combo']) < SIG
        c = ROI_COLOURS.get(roi, '#888')
        if roi in boot:
            ax.plot([i, i], [boot[roi]['ci_lo'], boot[roi]['ci_hi']],
                    color='0.05' if sig else '0.55', lw=2.3 if sig else 1.1,
                    alpha=1.0 if sig else 0.45, zorder=3, solid_capstyle='round')
            ax.scatter([i], [b], s=72, facecolor=c, zorder=4,
                       edgecolor='0.05' if sig else '0.55',
                       linewidth=1.3 if sig else 0.9, alpha=1.0 if sig else 0.45)
            top = boot[roi]['ci_hi']
        else:
            ax.bar(i, b, width=0.72, facecolor=c, edgecolor=c, lw=1.0,
                   alpha=1.0 if sig else 0.3, zorder=2)
            top = b
        if not sig:
            ax.text(i, top + .004, 'n.s.', ha='center', va='bottom',
                    fontsize=7, color='0.45')
    ax.axhline(0, color='0.45', ls=':', lw=0.9, zorder=0)
    _style_axis(ax, rsa_order, r'RSA $\beta$ (95% CI)', 'a   Population (RSA)')

    ax = axes[1]
    for i, row in cells.iterrows():
        sig = row['q'] < SIG
        c = ROI_COLOURS.get(row['roi'], '#888')
        ax.plot([i, i], [row['lo'], row['hi']], color='0.05' if sig else '0.55',
                lw=2.3 if sig else 1.1, alpha=1.0 if sig else 0.45,
                zorder=3, solid_capstyle='round')
        ax.scatter([i], [row['hl']], s=72, facecolor=c, zorder=4,
                   edgecolor='0.05' if sig else '0.55',
                   linewidth=1.3 if sig else 0.9, alpha=1.0 if sig else 0.45)
    ax.axhline(0, color='0.45', ls=':', lw=0.9, zorder=0)
    _style_axis(ax, list(cells['roi']), 'held-out r  (HL, 95% CI)',
                'b   Single cells')
    for i, row in cells.iterrows():
        if not row['q'] < SIG:
            ax.text(i, row['hi'] + .002, 'n.s.', ha='center', va='bottom',
                    fontsize=7, color='0.45')
    _save(fig, OUT + "/fig1_position_rsa_and_cells")
    return rsa, cells

# ------------------------------------------------------------------ Fig 2
def fig_ramp_robust():
    """Cells whose preferred position is 2 or 3 -- the two positions that
    share only 7% of their variance with a full-loop ramp, so this subset
    cannot be produced by any residual drift."""
    _rc()
    c = cell_position(keep_pref=['B', 'C'])
    fig, ax = plt.subplots(figsize=(7.2 * CM, 3.9 * CM))
    for i, row in c.iterrows():
        sig = row['q'] < SIG
        col = ROI_COLOURS.get(row['roi'], '#888')
        ax.plot([i, i], [row['lo'], row['hi']], color='0.05' if sig else '0.55',
                lw=2.3 if sig else 1.1, alpha=1.0 if sig else 0.45,
                zorder=3, solid_capstyle='round')
        ax.scatter([i], [row['hl']], s=72, facecolor=col, zorder=4,
                   edgecolor='0.05' if sig else '0.55',
                   linewidth=1.3 if sig else 0.9, alpha=1.0 if sig else 0.45)
        ax.text(i, row['hi'] + .0015, f"{int(row['n'])}", ha='center',
                va='bottom', fontsize=6.5, color='0.35')
    ax.axhline(0, color='0.45', ls=':', lw=0.9, zorder=0)
    _style_axis(ax, list(c['roi']), 'held-out r  (HL, 95% CI)',
                'Cells preferring position 2 or 3')
    _save(fig, OUT + "/fig2_ramp_robust_cells")
    return c

# ------------------------------------------------------------------ Fig C
def _pale(hex_colour, f=0.48):
    """Mix a colour towards white, so the uncorrected bars read as a faded
    version of the corrected ones rather than as four new categories."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_colour))
    return tuple(rgb + (1.0 - rgb) * f)

def fig_preferred():
    """Preferred position among position-coding cells, paired within ROI.

    Pale bar  = no residualisation, solid bar = goal progress and the per-loop
    ramp removed. Pairing them on one axis makes the cost of the control
    directly readable: position 4 is what the ramp correction takes away.
    """
    _rc()
    runs = [('no residualisation', ENC_RAW, True),
            ('phase + ramp removed', ENC_RUN, False)]
    shares, counts = {}, {}
    for tag, run, _ in runs:
        r = pd.read_csv(run + "/state_sustained_cv_results.csv")
        sig = r[r['sig_r_state']]
        shares[tag], counts[tag] = {}, {}
        for roi, d in sig.groupby('roi'):
            c = d['pref_state_mode'].value_counts().reindex(list('ABCD')).fillna(0)
            if c.sum() == 0:
                continue
            shares[tag][roi] = c / c.sum()
            counts[tag][roi] = int(c.sum())
    order = ordered(set().union(*[set(v) for v in shares.values()]))

    W, OFF = 0.36, 0.20
    fig, ax = plt.subplots(figsize=(15.5 * CM, 4.2 * CM))
    for i, roi in enumerate(order):
        for (tag, _, is_pale), sgn in zip(runs, (-1, +1)):
            if roi not in shares[tag]:
                continue
            x, bottom = i + sgn * OFF, 0.0
            for st in 'ABCD':
                col = _pale(STATE_COLOURS[st]) if is_pale else STATE_COLOURS[st]
                ax.bar(x, shares[tag][roi][st], bottom=bottom, width=W,
                       facecolor=col, edgecolor='white', lw=0.5, zorder=2)
                bottom += shares[tag][roi][st]
            if is_pale:
                ax.bar(x, 1.0, width=W, facecolor='none', edgecolor='0.5',
                       lw=0.7, zorder=4)
            ax.text(x, 1.015, f'{counts[tag][roi]}', ha='center', va='bottom',
                    fontsize=6.5, color='0.55' if is_pale else '0.3')
    ax.axhline(0.25, color='0.25', ls='--', lw=0.8, zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_xticklabels([lab(x) for x in order])
    ax.set_ylim(0, 1.0); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(['0', '25', '50', '75', '100'])
    ax.set_ylabel('share of position-coding\ncells (%)', labelpad=2, fontsize=8.5)
    ax.set_title('Preferred position, with and without the drift control',
                 loc='left', fontweight='bold', pad=11)

    pos_h = [plt.Rectangle((0, 0), 1, 1, facecolor=STATE_COLOURS[st],
                           edgecolor='white', lw=0.5, label=f'position {k + 1}')
             for k, st in enumerate('ABCD')]
    cond_h = [plt.Rectangle((0, 0), 1, 1, facecolor=_pale('#6E6E6E'),
                            edgecolor='0.5', lw=0.7, label='no residualisation'),
              plt.Rectangle((0, 0), 1, 1, facecolor='#6E6E6E',
                            edgecolor='white', lw=0.5, label='phase + ramp removed')]
    leg = ax.legend(handles=pos_h, frameon=False, fontsize=7, loc='upper left',
                    bbox_to_anchor=(1.01, 1.04), handlelength=1.0,
                    labelspacing=0.28, borderpad=0.2)
    ax.add_artist(leg)
    ax.legend(handles=cond_h, frameon=False, fontsize=7, loc='lower left',
              bbox_to_anchor=(1.01, -0.04), handlelength=1.0,
              labelspacing=0.28, borderpad=0.2)
    _save(fig, OUT + "/fig3_preferred_position")

def write_summary(rsa, cells, bc):
    """Machine-readable record of exactly what the figures show."""
    out = {
        'runs': {'rsa': os.path.basename(RSA_RUN), 'glm': os.path.basename(ENC_RUN),
                 'glm_no_residualisation': os.path.basename(ENC_RAW)},
        'residualisation': 'goal progress (within-state cosine) then per-loop '
                           'linear ramp, both removed from the traces',
        'inference': 'Benjamini-Hochberg FDR across ROIs',
        'rsa_position': {r: {'beta': float(rsa.loc[r, 'beta']),
                             'p_perm': float(rsa.loc[r, 'p_perm']),
                             'q_fdr': float(rsa.loc[r, 'q_fdr_per_combo'])}
                         for r in rsa.index},
        'cells_position': cells.set_index('roi').to_dict('index'),
        'cells_position_pref_2or3': bc.set_index('roi').to_dict('index'),
    }
    r_raw = pd.read_csv(ENC_RAW + "/state_sustained_cv_results.csv")
    r_res = pd.read_csv(ENC_RUN + "/state_sustained_cv_results.csv")
    pref = {}
    for tag, df in [('no_residualisation', r_raw), ('phase_plus_ramp', r_res)]:
        sig = df[df['sig_r_state']]
        counts = sig['pref_state_mode'].value_counts().reindex(list('ABCD')).fillna(0)
        pref[tag] = {'n': int(counts.sum()),
                     'share': {k: float(v / counts.sum()) for k, v in counts.items()},
                     'chi2_p_vs_uniform': float(
                         stats.chisquare(counts.to_numpy()).pvalue)}
    out['preferred_position'] = pref
    with open(OUT + '/results_summary.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print('wrote', OUT + '/results_summary.json')

if __name__ == '__main__':
    rsa, cells = fig_main()
    bc = fig_ramp_robust()
    fig_preferred()
    write_summary(rsa, cells, bc)
