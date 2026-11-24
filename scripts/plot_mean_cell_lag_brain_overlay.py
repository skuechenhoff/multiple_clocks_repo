#!/usr/bin/env python3
"""Render one quarters overlay with all cells coloured by mean argmax lag.

The background settings match the requested reference:

    quarters / angle_deg / gradient_thr1.5 / circular_gated /
    bilateral / right hemisphere / medial view

The cell colour is the unweighted circular mean of the cell-wise argmax lag
across all mPFC cells in ``per_cell_ALL_ROIs.csv``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex

import harmonic_maps_brain_overlay as overlay


CELL_CORRELATION_TABLE = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/ephys_humans/derivatives/group/per_lag_encoding'
    '/2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled'
    '/per_cell_ALL_ROIs.csv'
)
LAG_ANGLES_DEG = np.arange(0.0, 360.0, 30.0)
CELL_ROI = 'mPFC'
CORRELATION_KIND = 'noctrl'


def mean_cell_argmax_angle():
    table = read_per_cell_table(CELL_CORRELATION_TABLE)
    cells = table.loc[table['roi'].eq(CELL_ROI)].copy()
    correlation_columns = [
        f'r_lag{int(angle):03d}_{CORRELATION_KIND}'
        for angle in LAG_ANGLES_DEG
    ]
    correlations = cells[correlation_columns].to_numpy(dtype=float)
    usable = np.isfinite(correlations).any(axis=1)
    correlations = correlations[usable]

    best_index = np.nanargmax(correlations, axis=1)
    argmax_angles = LAG_ANGLES_DEG[best_index]
    theta = np.radians(argmax_angles)
    mean_cosine = float(np.mean(np.cos(theta)))
    mean_sine = float(np.mean(np.sin(theta)))
    mean_angle = float(
        np.degrees(np.arctan2(mean_sine, mean_cosine)) % 360.0
    )
    resultant_length = float(np.hypot(mean_cosine, mean_sine))
    return mean_angle, resultant_length, len(argmax_angles)


def angle_to_overlay_colour(angle_deg):
    """Use the exact circular wheel employed by the brain overlay."""
    signed_angle = (angle_deg + 180.0) % 360.0 - 180.0
    cmap = LinearSegmentedColormap.from_list(
        'circular_wheel', overlay.CIRCULAR_ANCHORS_HEX
    )
    rgba = cmap(Normalize(vmin=-180.0, vmax=180.0)(signed_angle))
    return to_hex(rgba, keep_alpha=False)


def add_cell_mean_label(image_path, mean_angle, cell_colour):
    """Add a compact label without changing the underlying brain render."""
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 300, height / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis('off')
    ax.text(
        0.02, 0.97,
        f'all cells: circular mean argmax = {mean_angle:.1f}°',
        transform=ax.transAxes, ha='left', va='top', fontsize=8,
        color='black',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=cell_colour, linewidth=1.5, alpha=0.9),
    )
    fig.savefig(image_path, dpi=300, pad_inches=0)
    plt.close(fig)


def main():
    mean_angle, resultant_length, n_cells = mean_cell_argmax_angle()
    cell_colour = angle_to_overlay_colour(mean_angle)

    # Explicitly match the magnitude-weighted reference branch.
    overlay.USE_UNIT_VECTOR_MAPS = False
    overlay.BASE_HARMONIC = overlay.HARMONIC_RESULTS_ROOT
    overlay.CELL_COLOR = cell_colour
    overlay.SAVE_PDF = False
    overlay.OUT_ROOT = (
        overlay.HARMONIC_RESULTS_ROOT
        / 'brain_overlays_with_mPFC_cells_mean_argmax'
    )
    overlay.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    overlay.RENDER_FILTER = {
        'datasets': ('quarters',),
        'combinations': (
            ('angle_deg', 'gradient_thr1.5', 'circular_gated'),
        ),
        'hemis': ('rh',),
        'views': ('medial',),
    }

    summary_path = overlay.OUT_ROOT / 'mean_cell_argmax_summary.csv'
    pd.DataFrame([{
        'n_cells': n_cells,
        'circular_mean_argmax_deg': mean_angle,
        'mean_resultant_length': resultant_length,
        'cell_colour_hex': cell_colour,
    }]).to_csv(summary_path, index=False)

    print(
        f'Mean cell-wise argmax: {mean_angle:.3f}°; '
        f'R={resultant_length:.4f}; n={n_cells}; colour={cell_colour}'
    )
    overlay.main()

    image_path = (
        overlay.OUT_ROOT / 'quarters'
        / 'angle_deg__gradient_thr1.5__circg0_bil__rh_medial.png'
    )
    if image_path.exists():
        add_cell_mean_label(image_path, mean_angle, cell_colour)
        print(f'Final labelled overlay: {image_path}')


if __name__ == '__main__':
    main()
