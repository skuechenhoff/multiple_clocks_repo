#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig 2c: where the analysed single units were recorded, per ROI.

Glass brain, three views (right sagittal, left sagittal, axial), one marker
per recording SITE at its exact MNI coordinate -- never jittered. Cells on
one microwire bundle share a coordinate exactly, so they are one marker.
Top right, in place of a legend, a bar panel gives per ROI the number of
CELLS (solid bars, bottom axis) and of SESSIONS with >= 1 cell there
(transparent bars, grey top axis running 0 .. all sessions of the sample,
so it reads as the share of sessions). The ROI names on that panel are the
colour key.

One figure per cell set, all from one run:
  all_in_roi_table  every cell with an ROI label
  rsa               cells from the 28 sessions in the population-RSA
                    grouping JSON (cell_selection.load_rsa_subjects)
  not_in_rsa        cells from every other session

Coordinates and ROI labels come from the canonical anatomy table
(`neurons_with_ROI_labels.csv`, `MNI_*_final`, `alt_final_roi`) through
`mc.analyse.cell_selection.load_cells`, i.e. the same source as every
ROI-grouped analysis and the Fig 3d gradient panel. Colours are the
project ROI colours (`mc.plotting.cell_results._roi_color_map`).

Sizes are for the panel AT PRINT SIZE (~6 x 5.75 cm); place the PDF
without rescaling.

Draw order: ROIs with the most sites first, so the small ROIs (EC, PCC)
sit on top where sites overlap in projection.

OUTPUTS (data/ephys_humans/derivatives/group/cell_coverage_glassbrain/<stamp>/)
  cell_coverage_glassbrain_<cell_set>.{pdf,png}
  coverage_summary.json   cells / sites / sessions per ROI, per cell set
  config.json
"""
import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.ticker import MaxNLocator
from nilearn import plotting

from mc.analyse import cell_selection as cs
from mc.plotting.cell_results import _roi_color_map, roi_display

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['pdf.fonttype'] = 42

# ── Settings ─────────────────────────────────────────────────────────
CELL_SETS = ['all_in_roi_table', 'rsa', 'not_in_rsa']
SET_TITLES = {'all_in_roi_table': 'All cells',
              'rsa': 'RSA sample',
              'not_in_rsa': 'Not in RSA sample'}
ROIS = ['EC', 'HC_anterior', 'HC_mid', 'mPFC', 'mOFC', 'PCC']   # legend order
LABELS = {'HC_anterior': 'HC ant', 'HC_mid': 'HC mid'}

FIG_W_CM, FIG_H_CM = 6.0, 5.75
BRAIN_H_CM = 4.5           # glass-brain block, as in the original panel
BAR_RECT_CM = (4.2, 3.05, 1.5, 1.85)   # x, y, w, h of the bar axes (top right)
SESSION_ALPHA = 0.35       # fill of the transparent session bars
SESSION_AXIS_COLOUR = '#777777'
MARKER_DIAM_PT = 3.2       # marker diameter at print size
MARKER_EDGE_PT = 0.3       # rim in a darker shade of the ROI colour
EDGE_DARKEN = 0.6
OUTLINE_SCALE = 0.5        # thin nilearn's glass-brain outlines for a small panel
FS = 9

OUT_ROOT = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                '/ephys_humans/derivatives/group/cell_coverage_glassbrain')
CM = 1 / 2.54


def cm_rect(x, y, w, h):
    """Axes rect in figure fractions from cm (origin bottom-left)."""
    return [x / FIG_W_CM, y / FIG_H_CM, w / FIG_W_CM, h / FIG_H_CM]


def roi_sites(cells, roi):
    """Unique (x, y, z) recording sites of one ROI."""
    return np.unique(cells.loc[cells['roi'] == roi,
                               ['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float),
                     axis=0)


def draw_view(fig, rect, mode, sites, colours, draw_order):
    ax = fig.add_axes(rect)
    display = plotting.plot_glass_brain(None, display_mode=mode, axes=ax,
                                        figure=fig, black_bg=False,
                                        plot_abs=False, annotate=False)
    for roi in draw_order:
        if len(sites[roi]) == 0:
            continue
        display.add_markers(sites[roi], marker_color=colours[roi],
                            marker_size=MARKER_DIAM_PT ** 2,
                            edgecolors=[np.array(to_rgb(colours[roi]))
                                        * EDGE_DARKEN],
                            linewidths=MARKER_EDGE_PT)
    for a in display.axes.values():
        for p in a.ax.patches:
            p.set_linewidth(p.get_linewidth() * OUTLINE_SCALE)
    if mode == 'z':
        display.annotate(left_right=True, positions=False, size=FS - 2)
    return display


def draw_count_bars(fig, summary, colours):
    """Per ROI: cells (solid, bottom axis) and sessions (transparent, top
    axis), as two thin bars per row -- the session bar above, next to its
    axis. ROI names are the y tick labels, so the panel is also the legend."""
    ax_cells = fig.add_axes(cm_rect(*BAR_RECT_CM))
    ax_sess = ax_cells.twiny()
    y = np.arange(len(ROIS))
    n_cells = [summary[r]['cells'] for r in ROIS]
    n_sess = [summary[r]['sessions'] for r in ROIS]
    edges = [np.array(to_rgb(colours[r])) * EDGE_DARKEN for r in ROIS]
    ax_cells.barh(y + 0.19, n_cells, height=0.36,
                  color=[colours[r] for r in ROIS],
                  edgecolor=edges, linewidth=MARKER_EDGE_PT)
    ax_sess.barh(y - 0.19, n_sess, height=0.36,
                 color=[to_rgba(colours[r], SESSION_ALPHA) for r in ROIS],
                 edgecolor=edges, linewidth=MARKER_EDGE_PT)

    ax_cells.set_yticks(y, [LABELS.get(r, roi_display(r)) for r in ROIS],
                        fontsize=FS)
    ax_cells.set_ylim(len(ROIS) - 0.5, -0.5)          # first ROI on top
    ax_cells.tick_params(axis='y', length=0, pad=2)
    # cells: own range; sessions: 0 .. all sessions of this sample, so a
    # session bar reads as the share of sessions with cells in that ROI.
    n_sess_total = summary['total']['sessions']
    ax_cells.set_xlim(0, max(n_cells) * 1.05)
    ax_cells.xaxis.set_major_locator(MaxNLocator(nbins=2,
                                                      steps=[1, 2, 5, 10]))
    ax_sess.set_xlim(0, n_sess_total)
    ax_sess.set_xticks([0, n_sess_total])
    for ax, label, col, pad in ((ax_cells, 'cells (n)', 'black', 1),
                                (ax_sess, 'sessions (n)', SESSION_AXIS_COLOUR,
                                 3)):
        ax.set_xlabel(label, fontsize=FS, labelpad=pad, color=col)
        ax.tick_params(axis='x', labelsize=FS, length=2, pad=1,
                       colors=col)
    ax_cells.spines[['top', 'right']].set_visible(False)
    ax_sess.spines[['bottom', 'left', 'right']].set_visible(False)
    ax_sess.spines['top'].set_color(SESSION_AXIS_COLOUR)
    return ax_cells, ax_sess


def coverage_summary(cells, sites):
    summary = {roi: dict(cells=int((cells['roi'] == roi).sum()),
                         sites=int(len(sites[roi])),
                         sessions=int(cells.loc[cells['roi'] == roi,
                                                'subject_int'].nunique()))
               for roi in ROIS}
    summary['total'] = dict(cells=int(len(cells)),
                            sites=int(sum(len(s) for s in sites.values())),
                            sessions=int(cells['subject_int'].nunique()))
    return summary


def make_figure(cell_set, colours, out_dir):
    cells = cs.load_cells(cell_set=cell_set, rois_keep=ROIS)
    sites = {roi: roi_sites(cells, roi) for roi in ROIS}
    draw_order = sorted(ROIS, key=lambda r: -len(sites[r]))  # stable sort
    summary = coverage_summary(cells, sites)
    print(f'\n{cell_set}')
    for roi, v in summary.items():
        print(f'  {roi:12s} {v}')

    fig = plt.figure(figsize=(FIG_W_CM * CM, FIG_H_CM * CM))
    half = BRAIN_H_CM / 2
    draw_view(fig, cm_rect(0.0, half, 3.0, half), 'r',
              sites, colours, draw_order)
    draw_view(fig, cm_rect(0.0, 0.0, 3.0, half), 'l',
              sites, colours, draw_order)
    draw_view(fig, cm_rect(3.1, 0.0, 1.9, half), 'z',
              sites, colours, draw_order)
    draw_count_bars(fig, summary, colours)
    tot = summary['total']
    fig.text(0.1 / FIG_W_CM, 1 - 0.05 / FIG_H_CM,
             f"{SET_TITLES[cell_set]}\n{tot['cells']} cells, "
             f"{tot['sessions']} sessions",
             ha='left', va='top', fontsize=FS, linespacing=1.1)

    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'cell_coverage_glassbrain_{cell_set}.{ext}',
                    dpi=600)
    plt.close(fig)
    return summary, draw_order


def main():
    colours = _roi_color_map(ROIS)
    out_dir = OUT_ROOT / time.strftime('%Y-%m-%d_%H-%M-%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, draw_orders = {}, {}
    for cell_set in CELL_SETS:
        summaries[cell_set], draw_orders[cell_set] = make_figure(
            cell_set, colours, out_dir)

    (out_dir / 'coverage_summary.json').write_text(
        json.dumps(summaries, indent=2))
    (out_dir / 'config.json').write_text(json.dumps(dict(
        cell_sets=CELL_SETS, rois=ROIS,
        rsa_sessions=sorted(int(s) for s in cs.load_rsa_subjects()),
        rsa_definition='sessions in all_sessions_dsrRSA_grouping_summary.json',
        roi_table='neurons_with_ROI_labels.csv '
        '(MNI_*_final, alt_final_roi) via cell_selection.load_cells',
        jitter_mm=0, one_marker_per_site=True, draw_order=draw_orders,
        bar_panel='per ROI: cells (solid, bottom axis) and sessions with '
        '>= 1 cell in the ROI (transparent, top axis, 0..total sessions of the '
        'sample); no bar labels',
        fig_cm=[FIG_W_CM, FIG_H_CM], bar_rect_cm=BAR_RECT_CM,
        session_alpha=SESSION_ALPHA, marker_diam_pt=MARKER_DIAM_PT,
        marker_edge_pt=MARKER_EDGE_PT, outline_scale=OUTLINE_SCALE,
        colours=colours), indent=2))
    print(f'\nAll outputs in {out_dir}')


if __name__ == '__main__':
    main()
