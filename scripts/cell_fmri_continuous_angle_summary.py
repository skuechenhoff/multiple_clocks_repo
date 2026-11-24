#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare four definitions of cell preference with the fMRI angle.

Cell methods
------------
1. ``subject_mean_argmax``: average the 12-lag correlation profiles across
   cells in each subject/session, then take that mean profile's argmax.
2. ``cell_harmonic``: continuous first-harmonic angle of each cell's profile.
3. ``cell_argmax``: lag with the largest correlation for each cell.
4. ``cell_top3``: whichever of a cell's three largest-correlation lag angles
   lies closest to the fMRI angle.

Significance is established with a circular-shift null. In every permutation,
each cell's complete 12-lag profile is independently rolled by a random number
of lag bins. All four cell methods are then recomputed from the shifted
profiles while the fMRI angles remain fixed.

The fMRI value is read directly from the unit-vector-derived
``angle_deg.nii.gz`` at the nearest voxel to each cell's MNI coordinate. No
smoothing, sphere averaging, thresholds, or p-value weighting are applied.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table


# ---------------------------------------------------------------------
# Inputs and settings
# ---------------------------------------------------------------------
CELL_TABLE = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/ephys_humans/derivatives/group/per_lag_encoding'
    '/2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled'
    '/per_cell_ALL_ROIs.csv'
)
FMRI_RESULTS_ROOT = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps'
    '/unit_vector_derived'
)
FMRI_DATASETS = ('quarters', 'eighths')
OUTPUT_DIR = FMRI_RESULTS_ROOT / 'cell_continuous_angle_summary'

CELL_ROI = 'mPFC'
CORRELATION_KIND = 'noctrl'
LAG_ANGLES_DEG = np.arange(0.0, 360.0, 30.0)
N_PERMUTATIONS = 5000
RANDOM_SEED = 42
SAVE_PDF = True

METHOD_ORDER = (
    'subject_mean_argmax',
    'cell_harmonic',
    'cell_argmax',
    'cell_top3',
)
METHOD_LABELS = {
    'subject_mean_argmax': 'Subject mean\nprofile argmax',
    'cell_harmonic': 'Cell-wise\nharmonic angle',
    'cell_argmax': 'Cell-wise\nargmax',
    'cell_top3': 'Closest of cell\ntop 3',
}
METHOD_COLOURS = {
    'subject_mean_argmax': '#7E57C2',
    'cell_harmonic': '#00897B',
    'cell_argmax': '#F9A825',
    'cell_top3': '#E64A19',
}


# ---------------------------------------------------------------------
# Angle definitions
# ---------------------------------------------------------------------
def harmonic_angles(correlation_profiles):
    """Continuous first-harmonic angle and vector length for every row."""
    profiles = np.asarray(correlation_profiles, dtype=float)
    theta = np.radians(LAG_ANGLES_DEG)
    finite = np.isfinite(profiles)
    count = finite.sum(axis=1)
    safe = np.where(finite, profiles, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine = np.sum(safe * np.cos(theta), axis=1) / count
        sine = np.sum(safe * np.sin(theta), axis=1) / count
    length = np.hypot(cosine, sine)
    angle = np.degrees(np.arctan2(sine, cosine)) % 360.0
    valid = (count > 0) & np.isfinite(length) & (length > 0)
    return np.where(valid, angle, np.nan), length


def ranked_lag_indices(correlation_profiles):
    """Lag indices ordered from largest to smallest raw correlation."""
    profiles = np.asarray(correlation_profiles, dtype=float)
    return np.argsort(
        np.where(np.isfinite(profiles), profiles, -np.inf), axis=1
    )[:, ::-1]


def subject_mean_profile_argmax(correlation_profiles, subject_codes):
    """Average profiles within subject/session, then take their argmax."""
    profiles = np.asarray(correlation_profiles, dtype=float)
    subject_codes = np.asarray(subject_codes, dtype=int)
    output = np.full(len(profiles), np.nan)
    for code in np.unique(subject_codes):
        members = subject_codes == code
        mean_profile = np.nanmean(profiles[members], axis=0)
        if np.isfinite(mean_profile).any():
            best_index = int(np.nanargmax(mean_profile))
            output[members] = LAG_ANGLES_DEG[best_index]
    return output


def cell_methods(correlation_profiles, subject_codes):
    """Return one or more candidate angles per cell for all four methods."""
    harmonic, _ = harmonic_angles(correlation_profiles)
    ranked = ranked_lag_indices(correlation_profiles)
    return {
        'subject_mean_argmax': subject_mean_profile_argmax(
            correlation_profiles, subject_codes
        )[:, None],
        'cell_harmonic': harmonic[:, None],
        'cell_argmax': LAG_ANGLES_DEG[ranked[:, :1]],
        'cell_top3': LAG_ANGLES_DEG[ranked[:, :3]],
    }


def absolute_circular_difference(angle_a, angle_b):
    return np.abs(
        (np.asarray(angle_a) - np.asarray(angle_b) + 180.0) % 360.0 - 180.0
    )


def snap_to_lag_index(angles):
    """Nearest 30-degree lag bin; the +15 avoids banker's rounding."""
    angles = np.asarray(angles, dtype=float) % 360.0
    return np.floor((angles + 15.0) / 30.0).astype(int) % 12


def comparison_metrics(correlation_profiles, subject_codes, fmri_angles):
    """Mean closest-angle error and lag-bin match rate for each method."""
    candidates = cell_methods(correlation_profiles, subject_codes)
    fmri_angles = np.asarray(fmri_angles, dtype=float)
    fmri_bins = snap_to_lag_index(fmri_angles)
    metrics = {}

    for method, candidate_angles in candidates.items():
        errors = absolute_circular_difference(
            candidate_angles, fmri_angles[:, None]
        )
        closest_error = np.nanmin(errors, axis=1)
        candidate_bins = snap_to_lag_index(candidate_angles)
        bin_match = np.any(candidate_bins == fmri_bins[:, None], axis=1)
        metrics[method] = {
            'mean_abs_error_deg': float(np.nanmean(closest_error)),
            'median_abs_error_deg': float(np.nanmedian(closest_error)),
            'match_rate': float(np.nanmean(bin_match)),
        }
    return metrics


# ---------------------------------------------------------------------
# Input table and direct fMRI sampling
# ---------------------------------------------------------------------
def sample_angle_map(angle_img, mni_coordinates):
    inverse_affine = np.linalg.inv(angle_img.affine)
    data = angle_img.get_fdata()
    shape = np.asarray(data.shape)
    sampled = np.full(len(mni_coordinates), np.nan)
    for row, coordinate in enumerate(np.asarray(mni_coordinates, dtype=float)):
        voxel = np.round(
            nib.affines.apply_affine(inverse_affine, coordinate)
        ).astype(int)
        if (voxel >= 0).all() and (voxel < shape).all():
            value = data[tuple(voxel)]
            if np.isfinite(value):
                sampled[row] = value % 360.0
    return sampled


def add_recording_site_ids(table):
    site_columns = ['subject_id', 'MNI_x', 'MNI_y', 'MNI_z']
    sites = (
        table[site_columns]
        .drop_duplicates()
        .sort_values(site_columns, kind='stable')
        .copy()
    )
    sites['recording_site_number'] = sites.groupby('subject_id').cumcount() + 1
    table = table.merge(
        sites, on=site_columns, how='left', validate='many_to_one'
    )
    table['recording_site_id'] = [
        f'sub-{int(subject):02d}_site-{int(site):02d}'
        for subject, site in zip(
            table['subject_id'], table['recording_site_number']
        )
    ]
    return table


def load_cell_data():
    source = read_per_cell_table(CELL_TABLE)
    cells = source.loc[source['roi'].eq(CELL_ROI)].copy().reset_index(drop=True)
    correlation_columns = [
        f'r_lag{int(angle):03d}_{CORRELATION_KIND}'
        for angle in LAG_ANGLES_DEG
    ]
    profiles = cells[correlation_columns].to_numpy(dtype=float)
    usable = np.isfinite(profiles).any(axis=1)
    cells = cells.loc[usable].reset_index(drop=True)
    profiles = profiles[usable]

    table = add_recording_site_ids(
        cells[['neuron', 'subject_id', 'MNI_x', 'MNI_y', 'MNI_z']].copy()
    )
    subject_codes, _ = pd.factorize(table['subject_id'], sort=True)

    harmonic, harmonic_length = harmonic_angles(profiles)
    ranked = ranked_lag_indices(profiles)
    subject_argmax = subject_mean_profile_argmax(profiles, subject_codes)
    rows = np.arange(len(table))

    table['cell_harmonic_angle_deg'] = harmonic
    table['cell_harmonic_vector_length'] = harmonic_length
    table['subject_mean_profile_argmax_deg'] = subject_argmax
    for rank, name in enumerate(('argmax', 'second_best', 'third_best')):
        index = ranked[:, rank]
        table[f'cell_{name}_angle_deg'] = LAG_ANGLES_DEG[index]
        table[f'cell_{name}_correlation'] = profiles[rows, index]

    coordinates = table[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(dtype=float)
    for dataset in FMRI_DATASETS:
        angle_path = FMRI_RESULTS_ROOT / dataset / 'angle_deg.nii.gz'
        if not angle_path.exists():
            raise FileNotFoundError(f'Missing fMRI angle map: {angle_path}')
        table[f'fmri_{dataset}_angle_deg'] = sample_angle_map(
            nib.load(str(angle_path)), coordinates
        )
    return table, profiles, subject_codes


# ---------------------------------------------------------------------
# Circular-shift permutation test
# ---------------------------------------------------------------------
def circularly_shift_profiles(profiles, shifts):
    """Equivalent to independently calling np.roll(row, shift) per cell."""
    profiles = np.asarray(profiles)
    shifts = np.asarray(shifts, dtype=int)
    source_columns = (
        np.arange(profiles.shape[1])[None, :] - shifts[:, None]
    ) % profiles.shape[1]
    return np.take_along_axis(profiles, source_columns, axis=1)


def permutation_comparison(profiles, subject_codes, fmri_angles, dataset):
    valid = np.isfinite(fmri_angles)
    profiles = profiles[valid]
    subject_codes = subject_codes[valid]
    fmri_angles = np.asarray(fmri_angles)[valid]
    observed = comparison_metrics(profiles, subject_codes, fmri_angles)

    rng = np.random.default_rng(RANDOM_SEED)
    null_error = {
        method: np.empty(N_PERMUTATIONS) for method in METHOD_ORDER
    }
    null_match = {
        method: np.empty(N_PERMUTATIONS) for method in METHOD_ORDER
    }

    for permutation in range(N_PERMUTATIONS):
        shifts = rng.integers(0, len(LAG_ANGLES_DEG), size=len(profiles))
        shifted_profiles = circularly_shift_profiles(profiles, shifts)
        shifted = comparison_metrics(
            shifted_profiles, subject_codes, fmri_angles
        )
        for method in METHOD_ORDER:
            null_error[method][permutation] = shifted[method][
                'mean_abs_error_deg'
            ]
            null_match[method][permutation] = shifted[method]['match_rate']

    rows = []
    for method in METHOD_ORDER:
        error_null = null_error[method]
        match_null = null_match[method]
        error_observed = observed[method]['mean_abs_error_deg']
        match_observed = observed[method]['match_rate']
        error_sd = float(np.std(error_null, ddof=1))
        match_sd = float(np.std(match_null, ddof=1))
        rows.append({
            'dataset': dataset,
            'method': method,
            'n_cells': len(profiles),
            'observed_mean_abs_error_deg': error_observed,
            'observed_median_abs_error_deg': observed[method][
                'median_abs_error_deg'
            ],
            'null_mean_abs_error_deg': float(np.mean(error_null)),
            'null_error_ci_low_deg': float(np.percentile(error_null, 2.5)),
            'null_error_ci_high_deg': float(np.percentile(error_null, 97.5)),
            'error_improvement_vs_null_deg': (
                float(np.mean(error_null)) - error_observed
            ),
            'error_improvement_z': (
                (float(np.mean(error_null)) - error_observed) / error_sd
                if error_sd > 0 else np.nan
            ),
            'p_circular_shift_error': (
                1 + int(np.sum(error_null <= error_observed))
            ) / (N_PERMUTATIONS + 1),
            'observed_match_rate': match_observed,
            'null_mean_match_rate': float(np.mean(match_null)),
            'null_match_ci_low': float(np.percentile(match_null, 2.5)),
            'null_match_ci_high': float(np.percentile(match_null, 97.5)),
            'match_improvement_vs_null': (
                match_observed - float(np.mean(match_null))
            ),
            'match_improvement_z': (
                (match_observed - float(np.mean(match_null))) / match_sd
                if match_sd > 0 else np.nan
            ),
            'p_circular_shift_match': (
                1 + int(np.sum(match_null >= match_observed))
            ) / (N_PERMUTATIONS + 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Direct comparison figure
# ---------------------------------------------------------------------
def plot_method_comparison(results, dataset):
    results = results.set_index('method').loc[list(METHOD_ORDER)].reset_index()
    x = np.arange(len(results))
    colours = [METHOD_COLOURS[m] for m in results['method']]
    labels = [METHOD_LABELS[m] for m in results['method']]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    observed_error = results['observed_mean_abs_error_deg'].to_numpy()
    null_error = results['null_mean_abs_error_deg'].to_numpy()
    null_error_yerr = np.vstack([
        null_error - results['null_error_ci_low_deg'].to_numpy(),
        results['null_error_ci_high_deg'].to_numpy() - null_error,
    ])
    axes[0].bar(x, observed_error, color=colours, alpha=0.88)
    axes[0].errorbar(
        x, null_error, yerr=null_error_yerr, fmt='ko', ms=4,
        capsize=4, lw=1.0, label='shift-null mean and 95% interval',
    )
    for index, row in results.iterrows():
        text_y = max(
            observed_error[index], row['null_error_ci_high_deg']
        ) + 2.0
        axes[0].text(
            index, text_y,
            f"p_shift={row['p_circular_shift_error']:.3g}\n"
            f"Δ={row['error_improvement_vs_null_deg']:+.1f}°",
            ha='center', va='bottom', fontsize=8,
        )
    axes[0].set_ylim(
        0,
        max(observed_error.max(), results['null_error_ci_high_deg'].max())
        + 14,
    )
    axes[0].set_ylabel('mean minimum |fMRI − cell angle| (°)')
    axes[0].set_title('Angular error: lower is better')
    axes[0].legend(fontsize=8, frameon=False, loc='upper right')

    observed_match = 100 * results['observed_match_rate'].to_numpy()
    null_match = 100 * results['null_mean_match_rate'].to_numpy()
    null_match_yerr = 100 * np.vstack([
        results['null_mean_match_rate'].to_numpy()
        - results['null_match_ci_low'].to_numpy(),
        results['null_match_ci_high'].to_numpy()
        - results['null_mean_match_rate'].to_numpy(),
    ])
    axes[1].bar(x, observed_match, color=colours, alpha=0.88)
    axes[1].errorbar(
        x, null_match, yerr=null_match_yerr, fmt='ko', ms=4,
        capsize=4, lw=1.0, label='shift-null mean and 95% interval',
    )
    for index, row in results.iterrows():
        text_y = max(
            observed_match[index], 100 * row['null_match_ci_high']
        ) + 1.5
        axes[1].text(
            index, text_y,
            f"p_shift={row['p_circular_shift_match']:.3g}",
            ha='center', va='bottom', fontsize=8,
        )
    axes[1].set_ylim(
        0,
        max(observed_match.max(), 100 * results['null_match_ci_high'].max())
        + 8,
    )
    axes[1].set_ylabel('nearest-lag-bin match (%)')
    axes[1].set_title('Lag-bin matching: higher is better')
    axes[1].legend(fontsize=8, frameon=False, loc='upper left')

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#dddddd', lw=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        f'Which cell-angle definition matches fMRI best? — {dataset}\n'
        f'{N_PERMUTATIONS} null permutations: independent circular shift of '
        'each 12-lag cell profile',
        fontsize=12,
    )
    stem = OUTPUT_DIR / f'angle_method_comparison_{dataset}'
    fig.savefig(f'{stem}.png', dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table, profiles, subject_codes = load_cell_data()

    cell_table_path = OUTPUT_DIR / 'cell_continuous_angles.csv'
    table.to_csv(cell_table_path, index=False)
    print(f'Wrote {cell_table_path}')

    all_results = []
    for dataset in FMRI_DATASETS:
        results = permutation_comparison(
            profiles,
            subject_codes,
            table[f'fmri_{dataset}_angle_deg'].to_numpy(dtype=float),
            dataset,
        )
        all_results.append(results)
        plot_method_comparison(results, dataset)
        best = results.loc[results['error_improvement_z'].idxmax()]
        print(
            f"{dataset}: strongest null-standardised error improvement = "
            f"{best['method']} (z={best['error_improvement_z']:.2f}, "
            f"p={best['p_circular_shift_error']:.4f})"
        )

    comparison = pd.concat(all_results, ignore_index=True)
    comparison_path = OUTPUT_DIR / 'angle_method_comparison.csv'
    comparison.to_csv(comparison_path, index=False)
    print(f'Wrote {comparison_path}')
    print(f'All outputs in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
