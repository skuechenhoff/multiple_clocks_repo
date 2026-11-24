#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Can the fMRI harmonic-angle map predict each mPFC cell's preferred lag?

For every mPFC single-unit in
    per_cell_ALL_ROIs.csv
we have:
  * 12 spatial-tuning correlations, one per lag (0°, 30°, ..., 330°) →
    read out the cell's preferred lag using one of three methods:
      - discrete argmax (the original analysis),
      - continuous first-harmonic angle of the 12 correlation values,
      - the same harmonic angle with exploratory (1 - permutation-p)
        confidence weighting.
  * an MNI coordinate → read out the fMRI-predicted preferred phase from
    ``angle_deg.nii.gz`` (both ``quarters`` and ``eighths`` datasets).

We then compare cell-preferred vs. fMRI-predicted angle at three levels:
  1. Circular-circular correlation (Fisher & Lee).
  2. Angular error distribution (|Δangle| in circular sense), against a
     uniform-random baseline of 90°.
  3. Exact-match / top-K match rates against a uniform baseline of
     1/12 = 8.3% (exact) and 3/12 = 25% (top-3).
  4. Confusion matrix: 12 × 12 counts of (cell argmax bin) × (fMRI
     nearest-lag bin).
Add a per-cell shuffle null so we can report a p-value for each metric.

Set ``USE_UNIT_VECTOR_MAPS`` to select either the original magnitude-weighted
harmonic maps or the separately stored unit-vector-derived maps. Visualisations
are saved beneath the selected results branch, so modes never overwrite one
another.

Visualisations (saved into ``<selected_harmonic_maps>/cell_lag_prediction/``):
  * ``confusion_matrix_{dataset}.pdf/png``           — 12 × 12 heatmap
  * ``angular_error_hist_{dataset}.pdf/png``         — |Δangle| distribution
  * ``scatter_cell_vs_fmri_{dataset}.pdf/png``       — angle-vs-angle scatter
  * ``glass_brain_cells_by_pref_lag.pdf/png``        — each mPFC cell
    plotted on a glass brain, coloured by its own preferred lag using
    the circular cyclic colormap (blue=±180°, green=−90°, yellow=0°,
    red=+90°).
  * ``per_cell_summary_{dataset}.csv``               — per-cell prediction
    numbers for follow-up.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats


# ── Settings ─────────────────────────────────────────────────────────
CSV_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                '/ephys_humans/derivatives/group/per_lag_encoding'
                '/2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled'
                '/per_cell_ALL_ROIs.csv')
# Keep this flag aligned with ``harmonic_angle_maps.py``. True reads maps in
# ``unit_vector_derived/``; False reads the original magnitude-weighted maps.
USE_UNIT_VECTOR_MAPS = True
UNIT_VECTOR_RESULTS_DIRNAME = 'unit_vector_derived'

HARMONIC_RESULTS_ROOT = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps')
BASE_HARMONIC = (HARMONIC_RESULTS_ROOT / UNIT_VECTOR_RESULTS_DIRNAME
                 if USE_UNIT_VECTOR_MAPS else HARMONIC_RESULTS_ROOT)
OUT_BASE_DIR = BASE_HARMONIC / 'cell_lag_prediction'
OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
# Per-config output subdir; reset by the config loop.
OUT_DIR = OUT_BASE_DIR

# ── Masks for the MNE surface backdrops ──────────────────────────────
MPFC_MASK_PATH   = Path('/Users/xpsy1114/Documents/projects/multiple_clocks'
                        '/data/masks/mask_PFC_LR_smoothed_resampled.nii.gz')
GRAD15_MASK_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks'
                        '/data/masks/gradient_thr_1.5.nii.gz')

CELL_ROI     = 'mPFC'
LAG_COL_KIND = 'noctrl'                       # 'ctrl' or 'noctrl'
LAG_ANGLES   = np.arange(0, 360, 30)         # 12 lags, 30° apart
DATASETS     = ['quarters', 'eighths']
N_PERM       = int(os.environ.get('CELL_LAG_N_PERM', '5000'))
SEED         = 42
SIG_P_THRESHOLD = 0.05         # p-threshold for "significantly tuned" cells
SAVE_PDF     = os.environ.get('CELL_LAG_SAVE_PDF', '1') != '0'  # PNG + PDF
# MNE/fsaverage surface rendering is deliberately opt-in: each call is slow
# and expands across datasets, masks, and hemispheres. The lightweight glass
# brain + comparison plots are sufficient for the routine method comparison.
RUN_MNE_SURFACE_PLOTS = os.environ.get('CELL_LAG_RUN_MNE', '0') != '0'
RUN_TOPK_GLASS_BRAINS = os.environ.get('CELL_LAG_RUN_TOPK_BRAINS', '0') != '0'

# How each neuron's 12-point correlation profile becomes one preferred angle.
# Each method gets a separate output folder. ``harmonic_r`` is the preferred
# continuous analysis. ``harmonic_r_pweighted`` is exploratory because a
# p-value is an evidence measure rather than a naturally linear weight; the
# bounded (1-p) transform avoids extreme -log10(p) leverage.
CELL_ANGLE_METHODS = [
    'argmax',
    'harmonic_r',
    'harmonic_r_pweighted',
]
if os.environ.get('CELL_LAG_METHODS'):
    CELL_ANGLE_METHODS = [
        x.strip() for x in os.environ['CELL_LAG_METHODS'].split(',')
        if x.strip()
    ]
CELL_ANGLE_METHOD = CELL_ANGLE_METHODS[0]  # set by the method loop below

# ── Config grid (each combination gets its own output subdir) ────────
# MIN_CELL_R:  cell-quality filter on argmax r  (None = keep all cells)
# TOP_K:       number of top-tuned lags counted as a match
#              (3 = the "top-3" figure; 1 = argmax-only, no top-K padding)
MIN_CELL_R_OPTIONS = [0.30, None]
TOP_K_OPTIONS      = [3, 1]
if os.environ.get('CELL_LAG_MIN_R_OPTIONS'):
    MIN_CELL_R_OPTIONS = [
        None if x.strip().lower() == 'none' else float(x)
        for x in os.environ['CELL_LAG_MIN_R_OPTIONS'].split(',')
    ]
if os.environ.get('CELL_LAG_TOP_K_OPTIONS'):
    TOP_K_OPTIONS = [
        int(x) for x in os.environ['CELL_LAG_TOP_K_OPTIONS'].split(',')
    ]
# Set at run-time by the config loop; the rest of the script reads these.
MIN_CELL_R = MIN_CELL_R_OPTIONS[0]
TOP_K      = TOP_K_OPTIONS[0]

# ── Filters ──────────────────────────────────────────────────────────
# Alternatively, gate by the permutation p at the argmax lag (uses the
# CSV's p_lag###_ctrl columns). Set to None to disable.
MAX_CELL_P      = None      # e.g. 0.10 for a mild filter, 0.05 for stricter
#
# (2) fMRI-voxel filter: drop cells whose fMRI voxel has weak harmonic
#     signal/agreement (small amplitude → unreliable arctan2 direction).
#     In unit-vector mode amplitude is between-subject directional agreement;
#     otherwise it is the raw group-vector magnitude.
#     Percentile is computed WITHIN the amplitude sampled at cell
#     coords, so it adapts to whatever cell set survives filter (1).
MIN_FMRI_AMP_PCT = None     # keep cells above this percentile of cell-voxel amp

# (3) Spatial smoothing of the fMRI prediction. When > 0, the predicted
#     angle at each cell is the amp-weighted circular mean of all voxels
#     within `SPHERE_RADIUS_MM` mm of the cell's MNI coord. This helps
#     when a single voxel is noisy but the local neighbourhood is
#     consistent.  0 = point-sample (original behaviour).
SPHERE_RADIUS_MM = 6.0

# ── Match harmonic_maps_brain_overlay.py's display processing ─────────
# When either of these is on, we load cos_group.nii.gz and sin_group.nii.gz
# from the harmonic-angle-maps folder, apply the SAME symmetrisation +
# volumetric smoothing that produces the surface figure, and recompute
# per-cell angles from THAT processed (cos, sin) pair.  Setting both to
# their off-values (False / 0.0) uses the raw angle_deg.nii.gz path.
FMRI_SYMMETRISE          = True     # x-flip cos/sin volumes and average
FMRI_PRE_SMOOTH_FWHM_MM  = 3.0      # Gaussian on cos/sin volumes (0 = off)

# Circular colour wheel (same anchors as harmonic_maps_brain_overlay.py)
CIRCULAR_ANCHORS_HEX = [
    '#1E88E5',   # -180°  → blue
    '#43A047',   #  -90°  → green
    '#FCE300',   #    0°  → yellow
    '#E53935',   #  +90°  → red
    '#1E88E5',   # +180°  → blue
]
CIRC_CMAP = LinearSegmentedColormap.from_list(
    'circular_wheel', CIRCULAR_ANCHORS_HEX)


# ── Helpers ──────────────────────────────────────────────────────────
def signed_circ_diff(a_deg, b_deg):
    """Signed circular difference (a − b) wrapped into (−180°, +180°]."""
    return (np.asarray(a_deg) - np.asarray(b_deg) + 180.0) % 360.0 - 180.0


def circular_linear_r(theta_rad_1, theta_rad_2):
    """Fisher–Lee circular–circular correlation (r ∈ [0, 1]).

    Robust to per-axis wrapping.  Returns (r, chi², p) with df=4 under
    an asymptotic χ² null.
    """
    a = np.asarray(theta_rad_1)
    b = np.asarray(theta_rad_2)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = a.size
    if n < 5:
        return np.nan, np.nan, np.nan
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    num  = 4 * (np.sum(ca * cb) * np.sum(sa * sb)
                 - np.sum(ca * sb) * np.sum(sa * cb))
    den  = np.sqrt((n ** 2 - np.sum(np.cos(2 * a)) ** 2
                          - np.sum(np.sin(2 * a)) ** 2)
                    * (n ** 2 - np.sum(np.cos(2 * b)) ** 2
                              - np.sum(np.sin(2 * b)) ** 2))
    if den == 0:
        return np.nan, np.nan, np.nan
    r = num / den
    # Fisher & Lee (1983) asymptotic test: n · r² ~ χ²(4).
    chi2 = n * r * r
    p = 1.0 - stats.chi2.cdf(chi2, df=4)
    return float(r), float(chi2), float(p)


def sample_map_at_mni(img, mni_coords):
    inv = np.linalg.inv(img.affine)
    data = img.get_fdata()
    shape = np.array(data.shape)
    out = np.empty(len(mni_coords))
    for i, mni in enumerate(mni_coords):
        ijk = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
        if (ijk >= 0).all() and (ijk < shape).all():
            v = data[tuple(ijk)]
            out[i] = v if np.isfinite(v) else np.nan
        else:
            out[i] = np.nan
    return out


def snap_to_lag_bin(angles_deg):
    """Round angles to the nearest of the 12 lag bins (0, 30, ..., 330)."""
    a = np.asarray(angles_deg, dtype=float) % 360.0
    out = np.full(a.shape, -1, dtype=int)
    finite = np.isfinite(a)
    out[finite] = np.round(a[finite] / 30.0).astype(int) % 12
    return out


def _bilaterally_symmetrise(img):
    """Return `img` averaged with its own left-right mirror.
    Assumes standard MNI orientation."""
    data = img.get_fdata()
    from nilearn.image import new_img_like
    return new_img_like(img, (data + data[::-1, ...]) / 2.0)


def _smooth_cos_sin(cos_img, sin_img, fwhm_mm):
    from nilearn.image import smooth_img
    return smooth_img(cos_img, fwhm_mm), smooth_img(sin_img, fwhm_mm)


def sample_cos_sin_at_cells(cos_img, sin_img, mni_coords, radius_mm):
    """Sample cos and sin at each MNI coord (point-sample if radius=0,
    otherwise mean over sphere), then compute angle & amp per cell.

    Returns (angle_360, amplitude)."""
    inv = np.linalg.inv(cos_img.affine)
    cos_data = cos_img.get_fdata()
    sin_data = sin_img.get_fdata()
    shape = np.array(cos_data.shape)
    n = len(mni_coords)
    out_ang = np.full(n, np.nan)
    out_amp = np.full(n, np.nan)
    if radius_mm and radius_mm > 0:
        vox_mm = np.sqrt((cos_img.affine[:3, :3] ** 2).sum(axis=0)).mean()
        r_vox = int(np.ceil(radius_mm / vox_mm))
        for i, mni in enumerate(mni_coords):
            centre = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
            lo = np.clip(centre - r_vox, 0, shape - 1)
            hi = np.clip(centre + r_vox + 1, 0, shape)
            block_c = cos_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            block_s = sin_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            xg, yg, zg = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            d = np.sqrt((xg - centre[0]) ** 2
                        + (yg - centre[1]) ** 2
                        + (zg - centre[2]) ** 2) * vox_mm
            amp_here = np.sqrt(block_c ** 2 + block_s ** 2)
            sig = (d <= radius_mm) & (amp_here > 1e-6) \
                  & np.isfinite(block_c) & np.isfinite(block_s)
            if not sig.any():
                continue
            C_bar = np.mean(block_c[sig])
            S_bar = np.mean(block_s[sig])
            out_ang[i] = np.degrees(np.arctan2(S_bar, C_bar)) % 360.0
            out_amp[i] = float(np.sqrt(C_bar ** 2 + S_bar ** 2))
    else:
        for i, mni in enumerate(mni_coords):
            ijk = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
            if (ijk >= 0).all() and (ijk < shape).all():
                c = float(cos_data[tuple(ijk)])
                s = float(sin_data[tuple(ijk)])
                if np.isfinite(c) and np.isfinite(s):
                    out_ang[i] = np.degrees(np.arctan2(s, c)) % 360.0
                    out_amp[i] = float(np.sqrt(c * c + s * s))
    return out_ang, out_amp


def sample_angle_sphere(ang_img, amp_img, mni_coords, radius_mm):
    """Amp-weighted circular mean of fMRI angles in a sphere per MNI coord.

    Returns (mean_angle [0, 360), mean_amplitude).  Cells whose sphere has
    no signal → NaN.
    """
    ang_data = ang_img.get_fdata()
    amp_data = amp_img.get_fdata()
    inv = np.linalg.inv(ang_img.affine)
    shape = np.array(ang_data.shape)
    vox_mm = np.sqrt((ang_img.affine[:3, :3] ** 2).sum(axis=0)).mean()
    r_vox = int(np.ceil(radius_mm / vox_mm))
    n = len(mni_coords)
    out_ang = np.full(n, np.nan)
    out_amp = np.full(n, np.nan)
    for i, mni in enumerate(mni_coords):
        centre = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
        lo = np.clip(centre - r_vox, 0, shape - 1)
        hi = np.clip(centre + r_vox + 1, 0, shape)
        block_a = ang_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        block_m = amp_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        xg, yg, zg = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = np.sqrt((xg - centre[0]) ** 2
                    + (yg - centre[1]) ** 2
                    + (zg - centre[2]) ** 2) * vox_mm
        sig = (d <= radius_mm) & (block_m > 1e-6) & np.isfinite(block_a)
        if not sig.any():
            continue
        w = block_m[sig]
        theta = np.radians(block_a[sig])
        C = np.sum(w * np.cos(theta))
        S = np.sum(w * np.sin(theta))
        out_ang[i] = np.degrees(np.arctan2(S, C)) % 360.0
        out_amp[i] = float(w.mean())
    return out_ang, out_amp


def circular_mean_deg(angles_deg, weights=None):
    """Weighted circular mean, in degrees; returns NaN if all weights zero."""
    a = np.asarray(angles_deg, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    m = np.isfinite(a) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    theta = np.radians(a[m])
    C = np.sum(w[m] * np.cos(theta))
    S = np.sum(w[m] * np.sin(theta))
    return float(np.degrees(np.arctan2(S, C)))


def rayleigh_test(angles_deg):
    """One-sample Rayleigh test: H0 = uniform on the circle.
    Returns (R̄, Z, p) where R̄ ∈ [0, 1] is the mean resultant length."""
    a = np.asarray(angles_deg, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 4:
        return np.nan, np.nan, np.nan
    theta = np.radians(a)
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C * C + S * S)
    Z = n * R * R
    # Zar 2010 asymptotic approximation
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n * n - Z * Z)) - (1 + 2 * n))
    return float(R), float(Z), float(p)


def watson_u2_permutation_test(angles_a_deg, angles_b_deg, n_perm=5000,
                                seed=42):
    """Non-parametric two-sample circular test: are the two angle
    distributions the same? Uses Watson's U² statistic and a
    label-shuffle null (no need for pycircstat / distribution tables).

    Returns (u2_obs, p_perm).
    """
    a = np.asarray(angles_a_deg, dtype=float)
    b = np.asarray(angles_b_deg, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 < 5 or n2 < 5:
        return np.nan, np.nan
    N = n1 + n2

    def _u2(x, y):
        # Watson U² (Watson 1961). Rank-based on the pooled sample.
        pooled = np.concatenate([x, y]) % 360.0
        order = np.argsort(pooled)
        origin = pooled[order[0]]
        # Cumulative counts along the circle from the smallest angle.
        rot = (pooled - origin) % 360.0
        idx = np.argsort(rot)
        labels = np.concatenate([np.zeros(n1, dtype=int),
                                  np.ones(n2, dtype=int)])[idx]
        # Empirical CDFs
        F1 = np.cumsum(labels == 0) / n1
        F2 = np.cumsum(labels == 1) / n2
        d = F1 - F2
        u2 = (n1 * n2 / N ** 2) * (np.sum(d * d) - (np.sum(d) ** 2) / N)
        return float(u2)

    u2_obs = _u2(a, b)
    rng = np.random.default_rng(seed)
    pool = np.concatenate([a, b])
    null = np.empty(n_perm)
    for p in range(n_perm):
        rng.shuffle(pool)
        null[p] = _u2(pool[:n1], pool[n1:])
    p_perm = float(np.mean(null >= u2_obs))
    return u2_obs, p_perm


def angles_to_colours(angles_360):
    """Map angles in [0, 360) → circular colour via [-180, 180] anchors.

    The colormap anchors are placed at −180°, −90°, 0°, +90°, +180°, so
    we need to convert [0, 360) into a signed representation before
    calling the norm, otherwise 0° would render blue (=−180 anchor).
    """
    a = np.asarray(angles_360, dtype=float)
    signed = ((a + 180.0) % 360.0) - 180.0        # → (−180, 180]
    return CIRC_CMAP(Normalize(vmin=-180, vmax=180)(signed))


def savefig(fig, stem):
    """Save a figure — PNG always, PDF only if SAVE_PDF is True."""
    fig.savefig(f'{stem}.png', dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(f'{stem}.pdf', dpi=300, bbox_inches='tight')


def best_of_topk_match(fmri_snap_idx, cell_top_idx):
    """Per cell: pick the top-K lag closest circularly to the fMRI snap.

    Returns (target_bin_index, exact_hit_bool). `target_bin_index` is
    always one of the cell's top-K lags.
    """
    n = len(fmri_snap_idx)
    tgt = np.zeros(n, dtype=int)
    hit = np.zeros(n, dtype=bool)
    for i in range(n):
        top = np.asarray(cell_top_idx[i])
        f = fmri_snap_idx[i]
        d = np.minimum((top - f) % 12, (f - top) % 12)
        j = int(np.argmin(d))
        tgt[i] = top[j]
        hit[i] = (d[j] == 0)
    return tgt, hit


def harmonic_cell_preference(r_values, p_values=None):
    """Continuous preferred angle from each cell's 12-lag profile.

    The first circular harmonic is computed at the actual lag angles:

        C_i = weighted_mean_k(r_ik * cos(theta_k))
        S_i = weighted_mean_k(r_ik * sin(theta_k))
        angle_i = atan2(S_i, C_i) mod 360 degrees

    With ``p_values=None``, every finite correlation receives equal weight.
    Otherwise the exploratory confidence weight is ``1 - p_perm``. Signed
    correlations are retained: this is a Fourier projection, not a circular
    mean requiring non-negative probability weights.

    Returns
    -------
    angle_deg : (n_cells,) continuous angles in [0, 360)
    amplitude : first-harmonic amplitude in correlation units
    relative_strength : amplitude divided by the mean absolute weighted
        correlation contribution; useful as a scale-free QC measure.
    """
    r = np.asarray(r_values, dtype=float)
    if r.ndim != 2 or r.shape[1] != len(LAG_ANGLES):
        raise ValueError(
            f"r_values must have shape (n_cells, {len(LAG_ANGLES)}); "
            f"got {r.shape}")

    finite = np.isfinite(r)
    if p_values is None:
        confidence = finite.astype(float)
    else:
        p = np.asarray(p_values, dtype=float)
        if p.shape != r.shape:
            raise ValueError(f"p_values shape {p.shape} != r_values {r.shape}")
        confidence = np.where(
            finite & np.isfinite(p), 1.0 - np.clip(p, 0.0, 1.0), 0.0)

    weighted_r = np.where(finite, r, 0.0) * confidence
    denom = confidence.sum(axis=1)
    theta = np.radians(LAG_ANGLES)
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.sum(weighted_r * np.cos(theta), axis=1) / denom
        S = np.sum(weighted_r * np.sin(theta), axis=1) / denom
        mean_abs = np.sum(np.abs(weighted_r), axis=1) / denom
    amplitude = np.hypot(C, S)
    angle_deg = np.degrees(np.arctan2(S, C)) % 360.0
    valid = (denom > 0) & np.isfinite(amplitude) & (amplitude > 1e-12)
    angle_deg = np.where(valid, angle_deg, np.nan)
    relative_strength = np.divide(
        amplitude, mean_abs, out=np.full_like(amplitude, np.nan),
        where=valid & np.isfinite(mean_abs) & (mean_abs > 0))
    relative_strength = np.clip(relative_strength, 0.0, 1.0)
    return angle_deg, amplitude, relative_strength


# ── Load per-cell CSV (dataset-independent, config-independent) ──────
df_all = read_per_cell_table(CSV_PATH)
mpfc = df_all[df_all['roi'] == CELL_ROI].copy()
print(f"Loaded {len(mpfc)} '{CELL_ROI}' cells from {CSV_PATH.name}")

lag_cols = [f'r_lag{a:03d}_{LAG_COL_KIND}' for a in LAG_ANGLES]
r_matrix = mpfc[lag_cols].to_numpy(dtype=float)          # (n_cells, 12)
# Drop cells with all-NaN rows (defensive; shouldn't happen).
usable = np.isfinite(r_matrix).any(axis=1)
mpfc   = mpfc.iloc[usable].reset_index(drop=True)
r_matrix = r_matrix[usable]
print(f"  {len(mpfc)} cells with valid tuning data")

# Original discrete cell preference (retained as a comparison method and as
# the common cell-quality filter for a fair comparison across methods).
cell_argmax_idx    = np.nanargmax(r_matrix, axis=1)
cell_argmax_angle  = LAG_ANGLES[cell_argmax_idx].astype(float)

# Per-cell significance of the argmax lag (config-independent).
p_cols = [f'p_lag{a:03d}_{LAG_COL_KIND}' for a in LAG_ANGLES]
p_matrix = mpfc[p_cols].to_numpy(dtype=float)
cell_argmax_p = p_matrix[np.arange(len(mpfc)), cell_argmax_idx]
sig_argmax    = cell_argmax_p <  SIG_P_THRESHOLD
print(f"  Cells with sig argmax   (p<{SIG_P_THRESHOLD}): "
      f"{int(sig_argmax.sum())}/{len(mpfc)}")

cell_argmax_r = r_matrix[np.arange(len(r_matrix)), cell_argmax_idx]
coords = mpfc[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(dtype=float)

# Continuous cell-angle definitions. The p-weighted version uses the same
# per-lag permutation-p columns as the significance annotations below.
cell_harmonic_angle, cell_harmonic_amplitude, \
    cell_harmonic_relative_strength = harmonic_cell_preference(r_matrix)
cell_harmonic_p_angle, cell_harmonic_p_amplitude, \
    cell_harmonic_p_relative_strength = harmonic_cell_preference(
        r_matrix, p_matrix)

CELL_ANGLE_RESULTS = {
    'argmax': {
        'angle': cell_argmax_angle,
        'amplitude': np.full(len(mpfc), np.nan),
        'relative_strength': np.full(len(mpfc), np.nan),
        'description': 'discrete lag with maximum correlation',
    },
    'harmonic_r': {
        'angle': cell_harmonic_angle,
        'amplitude': cell_harmonic_amplitude,
        'relative_strength': cell_harmonic_relative_strength,
        'description': 'continuous first harmonic of signed correlations',
    },
    'harmonic_r_pweighted': {
        'angle': cell_harmonic_p_angle,
        'amplitude': cell_harmonic_p_amplitude,
        'relative_strength': cell_harmonic_p_relative_strength,
        'description': ('continuous first harmonic of signed correlations, '
                        'weighted by 1 - permutation p'),
    },
}

# Globals selected by the method loop; initialise them for static analysis and
# interactive execution of individual helpers.
cell_pref_angle = CELL_ANGLE_RESULTS[CELL_ANGLE_METHOD]['angle']
cell_pref_snap_idx = snap_to_lag_bin(cell_pref_angle)
cell_pref_amplitude = CELL_ANGLE_RESULTS[CELL_ANGLE_METHOD]['amplitude']
cell_pref_relative_strength = CELL_ANGLE_RESULTS[
    CELL_ANGLE_METHOD]['relative_strength']

rng = np.random.default_rng(SEED)


def build_config_arrays(min_cell_r, top_k):
    """Compute the per-config arrays: top-K lag indices/angles, top-K
    significance, and cell-quality mask (`cell_ok`).  Returns a dict.
    """
    top_idx      = np.argsort(-r_matrix, axis=1)[:, :top_k]
    top_angles   = LAG_ANGLES[top_idx].astype(float)
    topk_p       = np.array([p_matrix[i, top_idx[i]].min()
                             for i in range(len(mpfc))])
    sig_topk_any = topk_p < SIG_P_THRESHOLD
    print(f"  Cells with ≥1 sig top-{top_k} lag: "
          f"{int(sig_topk_any.sum())}/{len(mpfc)}")

    cell_ok = np.ones(len(mpfc), dtype=bool)
    if min_cell_r is not None:
        cell_ok &= (cell_argmax_r >= min_cell_r)
        print(f"  [filter 1a] MIN_CELL_R = {min_cell_r}  →  "
              f"{int(cell_ok.sum())}/{len(mpfc)} cells pass")
    else:
        print(f"  [filter 1a] MIN_CELL_R = None  →  "
              f"{int(cell_ok.sum())}/{len(mpfc)} cells pass (no r-filter)")
    if MAX_CELL_P is not None:
        cell_ok &= (cell_argmax_p <= MAX_CELL_P)
        print(f"  [filter 1b] MAX_CELL_P = {MAX_CELL_P}  →  "
              f"{int(cell_ok.sum())}/{len(mpfc)} cells pass "
              f"(after combining with 1a)")

    return dict(cell_top_idx=top_idx, cell_top_angles=top_angles,
                sig_topk_any=sig_topk_any, cell_ok=cell_ok)


# ── For each dataset: sample fMRI, compare, plot ─────────────────────
def run_dataset(ds):
    ang_path = BASE_HARMONIC / ds / 'angle_deg.nii.gz'
    amp_path = BASE_HARMONIC / ds / 'amplitude.nii.gz'
    if not ang_path.exists():
        print(f"[skip] {ds}: {ang_path} not found")
        return
    # NEW: match the brain-overlay's display processing when either
    # symmetrisation or smoothing is on — load cos/sin, process, sample.
    use_cos_sin_path = (FMRI_SYMMETRISE or
                        (FMRI_PRE_SMOOTH_FWHM_MM and FMRI_PRE_SMOOTH_FWHM_MM > 0))
    if use_cos_sin_path:
        cos_p = BASE_HARMONIC / ds / 'cos_group.nii.gz'
        sin_p = BASE_HARMONIC / ds / 'sin_group.nii.gz'
        cos_img = nib.load(str(cos_p))
        sin_img = nib.load(str(sin_p))
        proc_notes = []
        if FMRI_SYMMETRISE:
            cos_img = _bilaterally_symmetrise(cos_img)
            sin_img = _bilaterally_symmetrise(sin_img)
            proc_notes.append('bilaterally symmetrised')
        if FMRI_PRE_SMOOTH_FWHM_MM and FMRI_PRE_SMOOTH_FWHM_MM > 0:
            cos_img, sin_img = _smooth_cos_sin(
                cos_img, sin_img, FMRI_PRE_SMOOTH_FWHM_MM)
            proc_notes.append(f'{FMRI_PRE_SMOOTH_FWHM_MM} mm Gaussian on cos/sin')
        fmri_ang_360, fmri_amp = sample_cos_sin_at_cells(
            cos_img, sin_img, coords, SPHERE_RADIUS_MM)
        print(f"  [cos/sin] {', '.join(proc_notes)}; "
              f"per-cell sample in {SPHERE_RADIUS_MM:.1f} mm sphere")
    elif SPHERE_RADIUS_MM and SPHERE_RADIUS_MM > 0:
        ang_img = nib.load(str(ang_path))
        amp_img = nib.load(str(amp_path))
        fmri_ang_360, fmri_amp = sample_angle_sphere(
            ang_img, amp_img, coords, SPHERE_RADIUS_MM)
        print(f"  [sphere] amp-weighted circular mean in "
              f"{SPHERE_RADIUS_MM:.1f} mm sphere around each cell")
    else:
        ang_img = nib.load(str(ang_path))
        amp_img = nib.load(str(amp_path))
        fmri_ang_raw = sample_map_at_mni(ang_img, coords)
        fmri_amp     = sample_map_at_mni(amp_img, coords)
        fmri_ang_360 = fmri_ang_raw % 360.0
    fmri_snap_idx = snap_to_lag_bin(fmri_ang_360)
    fmri_snap_angle = np.full(len(fmri_snap_idx), np.nan)
    finite_snap = fmri_snap_idx >= 0
    fmri_snap_angle[finite_snap] = LAG_ANGLES[
        fmri_snap_idx[finite_snap]].astype(float)

    have_signal = np.isfinite(fmri_ang_360) & np.isfinite(cell_pref_angle) \
                   & (fmri_amp > 1e-6)

    # Filter 2: keep only cells whose fMRI voxel amplitude is above the
    # requested percentile of the amp distribution across the cells that
    # passed filter 1 (i.e. adaptive per-dataset, per-cohort).
    amp_ok = np.ones_like(have_signal)
    if MIN_FMRI_AMP_PCT is not None:
        pool_mask = have_signal & cell_ok
        if pool_mask.any():
            amp_thr = float(np.percentile(fmri_amp[pool_mask],
                                            MIN_FMRI_AMP_PCT))
            amp_ok = fmri_amp > amp_thr
        else:
            amp_thr = np.nan

    valid = have_signal & cell_ok & amp_ok
    n_valid = int(valid.sum())
    print(f"\n=== dataset: {ds}  ({n_valid}/{len(mpfc)} cells survive all filters) ===")
    if MIN_FMRI_AMP_PCT is not None and np.isfinite(amp_thr):
        print(f"  [filter 2] fMRI amp > {amp_thr:.5f} "
              f"(={MIN_FMRI_AMP_PCT}th pct of {int((have_signal & cell_ok).sum())} qualifying cells)  "
              f"→  {int(amp_ok.sum())} cells pass on amp alone")

    # --- Circular correlation ---
    r_circ, chi2_circ, p_circ = circular_linear_r(
        np.radians(cell_pref_angle[valid]),
        np.radians(fmri_ang_360[valid]))
    print(f"  Fisher-Lee circ-circ r = {r_circ:+.3f}  "
          f"χ²(4)={chi2_circ:.1f}  p = {p_circ:.4g}")

    # --- Angular errors ---
    signed_err = signed_circ_diff(fmri_ang_360, cell_pref_angle)
    abs_err    = np.abs(signed_err)
    mean_abs   = float(np.nanmean(abs_err[valid]))
    med_abs    = float(np.nanmedian(abs_err[valid]))
    baseline_mean_abs = 90.0
    print(f"  Mean   |Δangle| = {mean_abs:5.1f}°   (baseline 90°)")
    print(f"  Median |Δangle| = {med_abs:5.1f}°")

    # --- Exact and top-K match rates ---
    exact = np.array([f == c
                      for f, c in zip(fmri_snap_idx, cell_pref_snap_idx)])
    topk  = np.array([f in top for f, top
                      in zip(fmri_snap_idx, cell_top_idx)])
    exact_rate = float(exact[valid].mean())
    topk_rate  = float(topk[valid].mean())
    base_exact = 1.0 / 12.0
    base_topk  = TOP_K / 12.0
    print(f"  Exact-lag match  : {int(exact[valid].sum())}/{n_valid} "
          f"= {exact_rate*100:5.1f}%   (baseline {base_exact*100:.1f}%)")
    print(f"  Within top-{TOP_K}    : {int(topk[valid].sum())}/{n_valid} "
          f"= {topk_rate*100:5.1f}%   (baseline {base_topk*100:.1f}%)")

    # --- Permutation null (shuffle fMRI predictions across cells) ---
    v_idx = np.where(valid)[0]
    fmri_snap_v = fmri_snap_idx[v_idx]
    fmri_ang_v  = fmri_ang_360[v_idx]
    cell_snap_v = cell_pref_snap_idx[v_idx]
    cell_pref_v = cell_pref_angle[v_idx]
    cell_top_v  = cell_top_idx[v_idx]

    null_exact = np.empty(N_PERM)
    null_topk  = np.empty(N_PERM)
    null_abs   = np.empty(N_PERM)
    null_r     = np.empty(N_PERM)
    for p in range(N_PERM):
        order = rng.permutation(len(v_idx))
        fs = fmri_snap_v[order]
        fa = fmri_ang_v[order]
        null_exact[p] = np.mean(fs == cell_snap_v)
        null_topk[p]  = np.mean([f in t for f, t in zip(fs, cell_top_v)])
        null_abs[p]   = np.mean(np.abs(signed_circ_diff(fa, cell_pref_v)))
        r_p, _, _ = circular_linear_r(
            np.radians(cell_pref_v), np.radians(fa))
        null_r[p] = r_p if np.isfinite(r_p) else np.nan
    p_exact_perm = float(np.mean(null_exact >= exact_rate))
    p_topk_perm  = float(np.mean(null_topk  >= topk_rate))
    p_abs_perm   = float(np.mean(null_abs   <= mean_abs))
    p_r_perm     = float(np.mean(null_r     >= r_circ))
    print("\n  Null A1 — shuffle pairing WITHIN mPFC (strictest):")
    print(f"    exact  p = {p_exact_perm:.4f}   (null mean {null_exact.mean()*100:.1f}%)")
    print(f"    topk   p = {p_topk_perm:.4f}    (null mean {null_topk.mean()*100:.1f}%)")
    print(f"    mean|Δ| p = {p_abs_perm:.4f}   (null mean {null_abs.mean():.1f}°)")
    print(f"    circ r  p = {p_r_perm:.4f}     (null mean {np.nanmean(null_r):+.3f})")

    # ── Null A2: random WHOLE-BRAIN voxel per cell ──────────────
    # Tests: "does the anatomical mPFC-cell ↔ mPFC-voxel pairing carry
    # information beyond what a random brain voxel would?"
    # Pool of angle values = every finite voxel with amplitude > 1e-6.
    if use_cos_sin_path:
        # (cos_img, sin_img are the processed volumes; recover amp+angle)
        cos_data = cos_img.get_fdata(); sin_data = sin_img.get_fdata()
        wb_amp_full  = np.sqrt(cos_data ** 2 + sin_data ** 2)
        wb_ang_full  = np.degrees(np.arctan2(sin_data, cos_data)) % 360.0
    else:
        wb_ang_full = ang_img.get_fdata() % 360.0
        wb_amp_full = amp_img.get_fdata()
    wb_ok = np.isfinite(wb_ang_full) & np.isfinite(wb_amp_full) \
            & (wb_amp_full > 1e-6)
    wb_ang_pool = wb_ang_full[wb_ok].astype(float)
    wb_snap_pool = snap_to_lag_bin(wb_ang_pool)
    print(f"\n  [Null A2] whole-brain angle pool: {wb_ang_pool.size} voxels")

    null_wb_exact = np.empty(N_PERM)
    null_wb_topk  = np.empty(N_PERM)
    null_wb_abs   = np.empty(N_PERM)
    null_wb_r     = np.empty(N_PERM)
    for p in range(N_PERM):
        idx = rng.integers(0, wb_ang_pool.size, size=len(v_idx))
        fs_wb = wb_snap_pool[idx]
        fa_wb = wb_ang_pool[idx]
        null_wb_exact[p] = np.mean(fs_wb == cell_snap_v)
        null_wb_topk[p]  = np.mean([f in t for f, t in zip(fs_wb, cell_top_v)])
        null_wb_abs[p]   = np.mean(np.abs(signed_circ_diff(fa_wb, cell_pref_v)))
        r_p, _, _ = circular_linear_r(np.radians(cell_pref_v), np.radians(fa_wb))
        null_wb_r[p] = r_p if np.isfinite(r_p) else np.nan
    p_wb_exact = float(np.mean(null_wb_exact >= exact_rate))
    p_wb_topk  = float(np.mean(null_wb_topk  >= topk_rate))
    p_wb_abs   = float(np.mean(null_wb_abs   <= mean_abs))
    p_wb_r     = float(np.mean(null_wb_r     >= r_circ))
    print("  Null A2 — random WHOLE-BRAIN voxel per cell "
          "(tests anatomical specificity):")
    print(f"    exact  p = {p_wb_exact:.4f}   (null mean {null_wb_exact.mean()*100:.1f}%)")
    print(f"    topk   p = {p_wb_topk:.4f}    (null mean {null_wb_topk.mean()*100:.1f}%)")
    print(f"    mean|Δ| p = {p_wb_abs:.4f}   (null mean {null_wb_abs.mean():.1f}°)")
    print(f"    circ r  p = {p_wb_r:.4f}     (null mean {np.nanmean(null_wb_r):+.3f})")

    # ── Null B2: circular-shift each cell's 12-lag tuning ──────────
    # Rolls the r-values by a random {0..11} shift → new argmax/top-K.
    # fMRI at each cell's actual voxel stays fixed.  Tests: "does the
    # cell's ACTUAL argmax matter, given its tuning noise structure?"
    # For match rates this collapses to ≈ uniform baseline (1/12, 3/12).
    null_sh_exact = np.empty(N_PERM)
    null_sh_topk  = np.empty(N_PERM)
    for p in range(N_PERM):
        shifts = rng.integers(0, 12, size=len(v_idx))
        sh_argmax = (cell_snap_v + shifts) % 12
        sh_top    = (cell_top_v + shifts[:, None]) % 12
        null_sh_exact[p] = np.mean(fmri_snap_v == sh_argmax)
        null_sh_topk[p]  = np.mean(
            [fmri_snap_v[i] in sh_top[i] for i in range(len(v_idx))])
    p_sh_exact = float(np.mean(null_sh_exact >= exact_rate))
    p_sh_topk  = float(np.mean(null_sh_topk  >= topk_rate))
    print("  Null B2 — circular-shift cell tuning (fMRI fixed):")
    print(f"    exact  p = {p_sh_exact:.4f}   (null mean {null_sh_exact.mean()*100:.1f}%)")
    print(f"    topk   p = {p_sh_topk:.4f}    (null mean {null_sh_topk.mean()*100:.1f}%)")

    # --- Confusion v1 (rows = selected cell-angle bin, cols = fMRI snap) ---
    cm = np.zeros((12, 12), dtype=int)
    for c, f in zip(cell_snap_v, fmri_snap_v):
        cm[c, f] += 1

    # --- Best-of-top-K target: for each cell, which of its top-K lags
    #     sits closest circularly to the fMRI snap.
    bo_tgt_v, bo_hit_v = best_of_topk_match(fmri_snap_v, cell_top_v)
    bo_tgt_angle_v = LAG_ANGLES[bo_tgt_v].astype(float)
    cm_top = np.zeros((12, 12), dtype=int)
    for c, f in zip(bo_tgt_v, fmri_snap_v):
        cm_top[c, f] += 1

    # --- |Δangle| under best-of-top-K target (computed on valid subset) ---
    bo_abs_err = np.abs(signed_circ_diff(fmri_ang_360[valid], bo_tgt_angle_v))
    bo_mean = float(np.nanmean(bo_abs_err))
    bo_med  = float(np.nanmedian(bo_abs_err))

    # --- Broader display mask: use every cell with a finite fMRI +
    #     cell prediction, IGNORING the MIN_CELL_R filter, so histograms
    #     show the true sig-vs-non-sig split rather than just the
    #     post-filter tail.  Metrics/perm-p keep using the `valid` mask.
    show_mask = np.isfinite(fmri_ang_360) & np.isfinite(cell_pref_angle) \
                & (fmri_amp > 1e-6)
    # Best-of-top-K for the full show set too (no filter):
    show_fmri_snap = fmri_snap_idx[show_mask]
    show_top_idx   = cell_top_idx[show_mask]
    show_bo_tgt, _ = best_of_topk_match(show_fmri_snap, show_top_idx)
    show_bo_tgt_angle = LAG_ANGLES[show_bo_tgt].astype(float)
    show_bo_abs = np.abs(signed_circ_diff(fmri_ang_360[show_mask],
                                            show_bo_tgt_angle))

    # ── Plots ────────────────────────────────────────────────────────
    _plot_confusion(cm, ds, suffix=CELL_ANGLE_METHOD,
                     title=f'Confusion — {ds} '
                           f'(rows = cell {CELL_ANGLE_METHOD})  '
                           f'n_cells = {int(cm.sum())}')
    _plot_confusion(cm_top, ds, suffix=f'best_of_top{TOP_K}',
                     title=f'Confusion — {ds} '
                           f'(rows = cell best-of-top-{TOP_K})  '
                           f'n_cells = {int(cm_top.sum())}')
    _plot_angular_error(abs_err[show_mask], sig_argmax[show_mask],
                         float(np.nanmean(abs_err[show_mask])),
                         float(np.nanmedian(abs_err[show_mask])), ds,
                         null_abs=null_abs, mode=CELL_ANGLE_METHOD)
    _plot_angular_error(show_bo_abs, sig_topk_any[show_mask],
                         float(np.nanmean(show_bo_abs)),
                         float(np.nanmedian(show_bo_abs)), ds,
                         null_abs=None, mode=f'best_of_top{TOP_K}')
    _plot_scatter(cell_pref_angle[show_mask], fmri_ang_360[show_mask], ds,
                   r=r_circ, p=p_r_perm, mode=CELL_ANGLE_METHOD,
                   sig_mask=sig_argmax[show_mask])
    _plot_scatter(show_bo_tgt_angle, fmri_ang_360[show_mask], ds,
                   r=r_circ, p=p_r_perm, mode=f'best_of_top{TOP_K}',
                   sig_mask=sig_topk_any[show_mask])
    if RUN_TOPK_GLASS_BRAINS:
        _plot_glass_brain_best_of_topk(coords[valid], bo_tgt_angle_v, ds)
    # MNE surface plot (fsaverage medial view): coloured fMRI angle backdrop
    # plus cells coloured by the selected cell-angle method. This is the
    # direct spatial map used to assess whether smoothing cell preferences
    # improves the correspondence.
    if RUN_MNE_SURFACE_PLOTS:
        plot_mne_surface_cells(
            coords[show_mask], cell_pref_angle[show_mask], ds,
            cell_angle_label=CELL_ANGLE_METHOD,
            cos_img=cos_img if use_cos_sin_path else None,
            sin_img=sin_img if use_cos_sin_path else None)
    # Side-by-side histograms of cell preference vs fMRI-predicted
    # distributions, so the narrow fMRI concentration is visible.
    plot_lag_distribution_comparison(
        cell_pref_angle[show_mask], fmri_ang_360[show_mask], ds)

    # --- Per-cell CSV ---
    per_cell = pd.DataFrame({
        'neuron':            mpfc['neuron'],
        'MNI_x':             coords[:, 0],
        'MNI_y':             coords[:, 1],
        'MNI_z':             coords[:, 2],
        'cell_angle_method': CELL_ANGLE_METHOD,
        'cell_preferred_angle_deg': cell_pref_angle,
        'cell_preferred_angle_bin_deg': np.where(
            cell_pref_snap_idx >= 0,
            LAG_ANGLES[np.clip(cell_pref_snap_idx, 0, 11)], np.nan),
        'cell_angle_harmonic_amplitude': cell_pref_amplitude,
        'cell_angle_relative_strength': cell_pref_relative_strength,
        'cell_argmax_lag_deg': cell_argmax_angle,
        'cell_argmax_r':       cell_argmax_r,
        'cell_argmax_p':       cell_argmax_p,
        'cell_top_lags_deg':   [','.join(map(str, t)) for t in cell_top_angles],
        'fmri_pred_angle':   fmri_ang_360,
        'fmri_snap_lag':     fmri_snap_angle,
        'fmri_amplitude':    fmri_amp,
        'abs_angle_error':   abs_err,
        'exact_match':       exact,
        f'within_top{TOP_K}': topk,
        'valid':             valid,
    })
    csv_out = OUT_DIR / f'per_cell_summary_{ds}.csv'
    per_cell.to_csv(csv_out, index=False)
    print(f"  → {csv_out.name}")

    # ── Population-level distribution tests ─────────────────────────
    # H0-A (chi²): the cell-argmax distribution and the fMRI-predicted
    # distribution across mPFC voxels are the same categorical distribution
    # over the 12 lag bins. Uses a 2×12 contingency table.
    cell_bin_counts = np.bincount(cell_snap_v, minlength=12)
    fmri_bin_counts = np.bincount(fmri_snap_v, minlength=12)
    contingency = np.vstack([cell_bin_counts, fmri_bin_counts])
    # Drop bins that are zero in BOTH rows (chi² undefined on empty cols).
    keep = contingency.sum(axis=0) > 0
    if keep.sum() >= 2:
        chi2_stat, chi2_p, chi2_df, _ = stats.chi2_contingency(
            contingency[:, keep])
    else:
        chi2_stat, chi2_p, chi2_df = np.nan, np.nan, np.nan

    # H0-B (Watson U², perm): same distribution question but treating the
    # data as circular samples rather than nominal categories.
    u2_stat, u2_p = watson_u2_permutation_test(
        cell_pref_v, fmri_ang_v, n_perm=2000, seed=SEED + 7)

    # H0-C (circular means): are the two amp-weighted / uniform circular
    # means the same? Compute both means + Rayleigh concentration.
    cell_mean = circular_mean_deg(cell_pref_v)
    fmri_mean = circular_mean_deg(fmri_ang_v)
    delta_mean = signed_circ_diff(cell_mean, fmri_mean)
    cell_R, cell_Z, cell_p_ray = rayleigh_test(cell_pref_v)
    fmri_R, fmri_Z, fmri_p_ray = rayleigh_test(fmri_ang_v)
    print("\n  === Population-level tests ===")
    print(f"  Chi²(12-bin cell vs fMRI counts): "
          f"χ²({chi2_df})={chi2_stat:.2f}, p = {chi2_p:.4g}")
    print(f"  Watson U² (perm, n=2000):  U² = {u2_stat:.4f}, p = {u2_p:.4g}")
    print(f"  Circular mean — cells: {cell_mean:+7.2f}°   "
          f"(R̄={cell_R:.3f}, Rayleigh p={cell_p_ray:.2e})")
    print(f"                — fMRI:  {fmri_mean:+7.2f}°   "
          f"(R̄={fmri_R:.3f}, Rayleigh p={fmri_p_ray:.2e})")
    print(f"  Δ mean angle (cell − fMRI): {delta_mean:+6.1f}°")

    return {
        'ds':          ds,
        'cell_angle_method': CELL_ANGLE_METHOD,
        'min_cell_r':  MIN_CELL_R,
        'top_k':       TOP_K,
        'n_valid':     n_valid,
        'r_circ':      r_circ,
        # Null A1 (shuffle mPFC pairing) — perm p's
        'p_circ_perm_A1': p_r_perm,
        'exact_rate':  exact_rate,
        'p_exact_A1':  p_exact_perm,
        'topk_rate':   topk_rate,
        'p_topk_A1':   p_topk_perm,
        'mean_abs':    mean_abs,
        'med_abs':     med_abs,
        'p_abs_A1':    p_abs_perm,
        # Null A2 (whole-brain voxel) — perm p's
        'p_exact_A2':  p_wb_exact,
        'p_topk_A2':   p_wb_topk,
        'p_abs_A2':    p_wb_abs,
        'p_circ_A2':   p_wb_r,
        'A2_null_exact_mean':  float(null_wb_exact.mean()),
        'A2_null_topk_mean':   float(null_wb_topk.mean()),
        'A2_null_abs_mean':    float(null_wb_abs.mean()),
        'A2_null_r_mean':      float(np.nanmean(null_wb_r)),
        # Null B2 (shift cell tuning) — perm p's (match metrics only)
        'p_exact_B2':  p_sh_exact,
        'p_topk_B2':   p_sh_topk,
        'B2_null_exact_mean':  float(null_sh_exact.mean()),
        'B2_null_topk_mean':   float(null_sh_topk.mean()),
        # Legacy aliases so old readers still work
        'p_circ_perm': p_r_perm,
        'p_exact_perm': p_exact_perm,
        'p_topk_perm':  p_topk_perm,
        'p_abs_perm':  p_abs_perm,
        # Population-level distribution tests
        'chi2_stat':   chi2_stat,
        'chi2_df':     chi2_df,
        'chi2_p':      chi2_p,
        'watson_u2':   u2_stat,
        'watson_p':    u2_p,
        'cell_mean_deg': cell_mean,
        'fmri_mean_deg': fmri_mean,
        'delta_mean_deg': delta_mean,
        'cell_rayleigh_R': cell_R,
        'cell_rayleigh_p': cell_p_ray,
        'fmri_rayleigh_R': fmri_R,
        'fmri_rayleigh_p': fmri_p_ray,
        # Also: bin counts for the .md
        'cell_bin_counts': cell_bin_counts.tolist(),
        'fmri_bin_counts': fmri_bin_counts.tolist(),
    }


def _plot_confusion(cm, ds, suffix='argmax', title=None):
    fig, ax = plt.subplots(figsize=(5.0, 4.6), constrained_layout=True)
    im = ax.imshow(cm, cmap='magma_r', aspect='equal', origin='lower')
    ax.set_xticks(range(12)); ax.set_xticklabels(LAG_ANGLES, fontsize=8)
    ax.set_yticks(range(12)); ax.set_yticklabels(LAG_ANGLES, fontsize=8)
    ax.set_xlabel('fMRI-predicted lag (° snapped)', fontsize=10)
    if suffix.startswith('best_of_top'):
        ylab = f'Cell best-of-top-{TOP_K} lag (°)'
    else:
        ylab = f'Cell preferred-angle bin ({CELL_ANGLE_METHOD}; °)'
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title or f'Confusion — {ds}  (n_cells = {int(cm.sum())})',
                 fontsize=11)
    diag = np.arange(12)
    ax.plot(diag, diag, color='#0e3d3a', lw=1.0, alpha=0.4)
    fig.colorbar(im, ax=ax, shrink=0.7, label='# cells')
    for i in range(12):
        for j in range(12):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=6.5, color='white' if cm[i, j] > cm.max() * .55 else 'black')
    savefig(fig, OUT_DIR / f'confusion_matrix_{suffix}_{ds}')
    plt.close(fig)


def _plot_angular_error(abs_err, sig_mask, mean_abs, med_abs, ds,
                         null_abs=None, mode='argmax'):
    """Stacked-bar histogram: sig cells (dark green) on top of non-sig
    (light green) per bin. Reader tip: a "good prediction" would peak on
    the LEFT (|Δangle| near 0°); bad = flat / peak near 90°."""
    fig, ax = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    bins = np.arange(0, 181, 15)
    err = np.asarray(abs_err)
    sig = np.asarray(sig_mask, dtype=bool)
    counts_sig, _   = np.histogram(err[sig],  bins=bins)
    counts_nsig, _  = np.histogram(err[~sig], bins=bins)
    centres = (bins[:-1] + bins[1:]) / 2
    width = np.diff(bins).mean() * 0.95
    ax.bar(centres, counts_nsig,       width=width,
           color='#c8e6c9', edgecolor='k', lw=0.4,
           label=f'not sig (p_argmax≥{SIG_P_THRESHOLD})  '
                 f'n={int((~sig).sum())}')
    ax.bar(centres, counts_sig, bottom=counts_nsig, width=width,
           color='#1b5e20', edgecolor='k', lw=0.4,
           label=f'sig (p<{SIG_P_THRESHOLD})  n={int(sig.sum())}')
    ax.axvline(mean_abs, color='#8B0000', lw=1.6,
                label=f'mean = {mean_abs:.1f}°')
    ax.axvline(med_abs, color='#0e3d3a', lw=1.6, ls='--',
                label=f'median = {med_abs:.1f}°')
    ax.axvline(90, color='#888', lw=1.0, ls=':', label='chance = 90°')
    if null_abs is not None:
        null_mean_lo = float(np.percentile(null_abs, 2.5))
        null_mean_hi = float(np.percentile(null_abs, 97.5))
        ax.axvspan(null_mean_lo, null_mean_hi, color='#888',
                    alpha=0.12,
                    label=f'null mean 95% CI [{null_mean_lo:.1f}°, {null_mean_hi:.1f}°]')
    ax.text(0.03, 0.97, '← perfect prediction',
             transform=ax.transAxes, fontsize=8, va='top', color='#333')
    label = (f'cell best-of-top-{TOP_K}' if mode.startswith('best_of_top')
             else f'cell preference ({mode})')
    ax.set_xlabel(f'|Δangle|  (fMRI predicted vs {label})  [°]',
                  fontsize=10)
    ax.set_ylabel('# mPFC cells', fontsize=10)
    ax.set_title(f'Angular error — {ds}  ({mode})', fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, OUT_DIR / f'angular_error_hist_{mode}_{ds}')
    plt.close(fig)


def _plot_scatter(cell_deg, fmri_deg, ds, r, p, mode='argmax',
                    sig_mask=None):
    fmri_360 = np.asarray(fmri_deg) % 360.0
    cell_360 = np.asarray(cell_deg) % 360.0
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    colours = angles_to_colours(cell_360)
    if sig_mask is not None:
        sig = np.asarray(sig_mask, dtype=bool)
        # non-sig cells → same colour but with black cross marker
        ax.scatter(cell_360[~sig], fmri_360[~sig], s=28,
                    facecolors='none', edgecolors=colours[~sig], lw=0.8,
                    marker='x', label=f'not sig  n={int((~sig).sum())}')
        ax.scatter(cell_360[sig], fmri_360[sig], s=42,
                    c=colours[sig], edgecolor='k', lw=0.4,
                    label=f'sig (p<{SIG_P_THRESHOLD})  n={int(sig.sum())}')
    else:
        ax.scatter(cell_360, fmri_360, s=32, c=colours,
                    edgecolor='k', lw=0.4)
    ax.plot([0, 360], [0, 360], color='#888', ls='--', lw=1, label='y = x')
    ax.fill_between([0, 360], [-30, 330], [30, 390],
                     color='#888', alpha=0.10,
                     label='y = x ± 30°')
    ax.set_xlim(-10, 370); ax.set_ylim(-10, 370)
    ax.set_xticks(LAG_ANGLES); ax.set_yticks(LAG_ANGLES)
    ax.tick_params(labelsize=8)
    xlab = (f'Cell best-of-top-{TOP_K} lag (°)'
            if mode.startswith('best_of_top')
            else f'Cell preferred angle ({mode}; °)')
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel('fMRI predicted angle (°) — [0, 360)', fontsize=10)
    ax.set_title(f'{ds} — {mode}  |  circ-circ r = {r:+.3f}, '
                  f'perm p = {p:.3g}', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, OUT_DIR / f'scatter_cell_vs_fmri_{mode}_{ds}')
    plt.close(fig)


def _plot_glass_brain_best_of_topk(coords, target_angle, ds):
    """Glass brain: cells coloured by their best-matched top-K lag,
    using the correct signed-angle → circular-cmap mapping."""
    try:
        from nilearn.plotting import plot_glass_brain
    except Exception as exc:
        print(f"  [glass_brain] nilearn unavailable: {exc}")
        return
    fig = plt.figure(figsize=(9.0, 3.6), dpi=300, constrained_layout=True)
    display = plot_glass_brain(
        None, display_mode='ortho', figure=fig, annotate=True,
        black_bg=False, colorbar=False, alpha=0.35,
    )
    colours = angles_to_colours(target_angle)
    display.add_markers(coords, marker_color=colours,
                         marker_size=28, alpha=0.9)
    # Signed-angle colourbar (matches the actual plot colours).
    cbar_ax = fig.add_axes([0.10, 0.02, 0.35, 0.03])
    sm = ScalarMappable(cmap=CIRC_CMAP, norm=Normalize(vmin=-180, vmax=180))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbar.set_label(f"cell's best-matched top-{TOP_K} lag (°) — {ds}",
                    fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    savepng = str(OUT_DIR / f'glass_brain_best_of_top{TOP_K}_{ds}.png')
    display.savefig(savepng, dpi=300)
    if SAVE_PDF:
        display.savefig(savepng.replace('.png', '.pdf'), dpi=300)
    plt.close(fig)


def _plot_lag_diagnostic(r_matrix, argmax_idx):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4),
                              constrained_layout=True)
    ax = axes[0]
    counts = np.bincount(argmax_idx, minlength=12)
    ax.bar(LAG_ANGLES, counts, width=25, color='#43A047',
            edgecolor='k', lw=0.5)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(LAG_ANGLES[i], c + 0.3, str(c),
                    ha='center', va='bottom', fontsize=7)
    ax.set_xlabel('Cell argmax lag (°)', fontsize=10)
    ax.set_ylabel('# mPFC cells', fontsize=10)
    ax.set_title(f'Argmax-lag distribution ({len(argmax_idx)} cells)',
                  fontsize=10)
    ax.set_xticks(LAG_ANGLES); ax.tick_params(axis='x', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    means = np.nanmean(r_matrix, axis=0)
    sems  = np.nanstd(r_matrix, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(r_matrix), axis=0))
    ax.errorbar(LAG_ANGLES, means, yerr=sems, marker='o', ms=5,
                 lw=1.4, capsize=3, color='#0e3d3a')
    ax.axhline(0, color='#888', lw=0.8, ls='--')
    peak_i = int(np.nanargmax(means))
    ax.axvline(LAG_ANGLES[peak_i], color='#E53935', lw=1.0, ls=':',
                label=f'peak at {LAG_ANGLES[peak_i]}° '
                      f'(mean r = {means[peak_i]:+.4f})')
    ax.set_xlabel('Lag (°)', fontsize=10)
    ax.set_ylabel('Population mean r (± SEM)', fontsize=10)
    ax.set_title('Mean tuning per lag', fontsize=10)
    ax.set_xticks(LAG_ANGLES); ax.tick_params(axis='x', labelsize=7)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)

    savefig(fig, OUT_DIR / 'diagnostic_lag_distribution')
    plt.close(fig)


def plot_mne_surface_cells(coords, target_angle_360, ds, cell_angle_label,
                           cos_img=None, sin_img=None):
    """MNE Brain (fsaverage pial, medial view) with the *fMRI angle
    prediction* as a coloured backdrop and cells plotted as per-cell-
    coloured foci on top. Each cell's colour is its selected preferred angle
    on the same cyclic wheel; the backdrop is the group-mean
    (symmetrised + smoothed) fMRI angle at each vertex, restricted to
    an ROI mask.

    Two backdrop variants are produced per dataset/hemisphere:
      * masked by the mPFC anatomical mask (`mask_PFC_LR_smoothed`)
      * masked by the DSR-gradient mask (`gradient_thr_1.5`)

    `cos_img` / `sin_img` must be the same processed (symmetrised +
    smoothed) volumes used to sample cells, so backdrop and cell
    colours share a common angle convention.  If either is None, the
    function falls back to the old grey-mask backdrop.
    """
    try:
        from mne.datasets import fetch_fsaverage
        from mne.viz import Brain
        from nilearn import surface as _surface
    except Exception as exc:
        print(f"  [mne surface] backend unavailable: {exc}")
        return
    subjects_dir = os.path.dirname(fetch_fsaverage())

    # Assemble mask options actually available on disk.
    mask_options = []
    if MPFC_MASK_PATH.exists():
        mask_options.append(('mPFC', nib.load(str(MPFC_MASK_PATH))))
    else:
        print(f"  [mne surface] mPFC mask missing: {MPFC_MASK_PATH}")
    if GRAD15_MASK_PATH.exists():
        mask_options.append(('gradient', nib.load(str(GRAD15_MASK_PATH))))
    else:
        print(f"  [mne surface] gradient mask missing: {GRAD15_MASK_PATH}")
    if not mask_options:
        print("  [mne surface] no masks available, skipping")
        return

    have_fmri_backdrop = (cos_img is not None) and (sin_img is not None)

    for mask_name, mask_img in mask_options:
        for hemi in ('lh', 'rh'):
            try:
                brain = Brain(subject='fsaverage', hemi=hemi, surf='pial',
                              background='white', size=(1000, 900),
                              subjects_dir=subjects_dir, alpha=0.3,
                              title=None)
            except TypeError:
                brain = Brain(subject='fsaverage', hemi=hemi, surf='pial',
                              background='white', size=(1000, 900),
                              subjects_dir=subjects_dir, title=None)

            surf_path = os.path.join(subjects_dir, 'fsaverage', 'surf',
                                     f'{hemi}.pial')

            # ── Project mask to the surface ──
            mask_txt = _surface.vol_to_surf(
                mask_img, surf_path,
                interpolation='nearest').astype(float)
            in_mask = mask_txt >= 0.5

            if have_fmri_backdrop and np.any(in_mask):
                # ── fMRI angle backdrop: project cos and sin, arctan2
                #    per vertex, mask, shift into [1, 361] so MNE's
                #    fmin stays positive (mirrors harmonic_maps_brain_
                #    overlay.py's `shifted = texture + 181`).
                cos_txt = _surface.vol_to_surf(
                    cos_img, surf_path,
                    interpolation='linear').astype(float)
                sin_txt = _surface.vol_to_surf(
                    sin_img, surf_path,
                    interpolation='linear').astype(float)
                amp_txt = np.sqrt(cos_txt ** 2 + sin_txt ** 2)
                angle_signed = np.degrees(np.arctan2(sin_txt, cos_txt))
                has_signal = in_mask & (amp_txt > 1e-6) \
                              & np.isfinite(angle_signed)
                shifted = np.where(has_signal, angle_signed + 181.0, np.nan)
                if np.any(np.isfinite(shifted)):
                    add_kwargs = dict(hemi=hemi, fmin=1.0, fmid=181.0,
                                       fmax=361.0, colormap=CIRC_CMAP,
                                       alpha=0.85, colorbar=False)
                    try:
                        brain.add_data(shifted,
                                        smoothing_steps=5, **add_kwargs)
                    except TypeError:
                        brain.add_data(shifted, **add_kwargs)
            else:
                # Fallback: faint grey mask patch.
                backdrop = np.where(in_mask, 1.0, np.nan)
                if np.any(np.isfinite(backdrop)):
                    try:
                        brain.add_data(backdrop, hemi=hemi, fmin=0.5,
                                        fmid=1.0, fmax=1.5,
                                        colormap='Greys', alpha=0.35,
                                        colorbar=False, smoothing_steps=3)
                    except TypeError:
                        brain.add_data(backdrop, hemi=hemi, fmin=0.5,
                                        fmid=1.0, fmax=1.5,
                                        colormap='Greys', alpha=0.35,
                                        colorbar=False)

            # ── Cells on this hemisphere, one add_foci per cell so
            #    each keeps its own colour.
            keep = (coords[:, 0] <= 0) if hemi == 'lh' \
                   else (coords[:, 0] >= 0)
            coords_h = coords[keep]
            angles_h = np.asarray(target_angle_360)[keep]
            colours_h = angles_to_colours(angles_h)
            rng_local = np.random.default_rng(hash(hemi) & 0xFFFF)
            jitter = rng_local.uniform(-2.5, 2.5, size=coords_h.shape)
            coords_h_j = coords_h + jitter
            for c, col in zip(coords_h_j, colours_h):
                try:
                    brain.add_foci(c[np.newaxis, :], coords_as_verts=False,
                                    hemi=hemi, color=tuple(col[:3]),
                                    scale_factor=0.35)
                except Exception as exc:
                    print(f"  [mne surface] add_foci failed: {exc}")

            try:
                brain.show_view('medial')
            except Exception:
                pass

            stem = (f'mne_surface_cells_by_{cell_angle_label}_{ds}'
                    f'_backdrop_{mask_name}_{hemi}_medial')
            png = str(OUT_DIR / f'{stem}.png')
            try:
                brain.save_image(png)
            except Exception as exc:
                print(f"  [mne surface] save failed: {exc}")

            # Append a matplotlib colourbar with the raw (−180..+180°)
            # angle labels, re-save.
            try:
                img = plt.imread(png)
                fig = plt.figure(figsize=(img.shape[1] / 300,
                                           img.shape[0] / 300 + 0.6),
                                  dpi=300)
                gs = fig.add_gridspec(2, 1,
                                       height_ratios=[img.shape[0], 60],
                                       hspace=0.02, left=0, right=1,
                                       bottom=0, top=1)
                ax_img = fig.add_subplot(gs[0])
                ax_img.imshow(img); ax_img.axis('off')
                ax_cb = fig.add_subplot(gs[1])
                cb_pos = ax_cb.get_position()
                ax_cb.set_position([0.225, cb_pos.y0,
                                     0.55, cb_pos.height * 0.4])
                norm = Normalize(vmin=-180, vmax=180)
                sm = ScalarMappable(cmap=CIRC_CMAP, norm=norm)
                cbar = fig.colorbar(sm, cax=ax_cb,
                                     orientation='horizontal')
                cbar.set_ticks([-180, -90, 0, 90, 180])
                cbar.set_label(
                    f"angle (°)  — backdrop: fMRI ({mask_name}-masked);  "
                    f"cells: {cell_angle_label} preference  — {ds}",
                    fontsize=8)
                cbar.ax.tick_params(labelsize=8)
                fig.savefig(png, dpi=300, bbox_inches='tight')
                if SAVE_PDF:
                    fig.savefig(png.replace('.png', '.pdf'),
                                 dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as exc:
                print(f"  [mne surface] colourbar overlay failed: {exc}")
            try:
                brain.close()
            except Exception:
                pass
        print(f"  → {stem.replace(f'_{hemi}_medial', '')}_[lh,rh]_medial.png")


def plot_lag_distribution_comparison(cell_angle_deg, fmri_pred_deg, ds):
    """Side-by-side histograms of the selected cell-angle distribution and the
    fMRI-predicted-angle distribution at cell coordinates.  Makes the
    "fMRI is narrower than cells" story visible."""
    cell_snap = snap_to_lag_bin(cell_angle_deg)
    fmri_snap = snap_to_lag_bin(fmri_pred_deg)
    cell_counts = np.bincount(cell_snap, minlength=12)
    fmri_counts = np.bincount(fmri_snap, minlength=12)

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    x = np.arange(12)
    w = 0.4
    bars_c = ax.bar(x - w/2, cell_counts, width=w, color='#43A047',
                     edgecolor='k', lw=0.5,
                     label=(f'cell {CELL_ANGLE_METHOD} angle '
                            f'(n={int(cell_counts.sum())})'))
    bars_f = ax.bar(x + w/2, fmri_counts, width=w, color='#E53935',
                     edgecolor='k', lw=0.5,
                     label=f'fMRI predicted lag  (n={int(fmri_counts.sum())})')
    for bars, counts in [(bars_c, cell_counts), (bars_f, fmri_counts)]:
        for b, c in zip(bars, counts):
            if c > 0:
                ax.text(b.get_x() + b.get_width() / 2, c + 0.5, str(c),
                        ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(LAG_ANGLES, fontsize=8)
    ax.set_xlabel('lag (°)', fontsize=10)
    ax.set_ylabel('# cells', fontsize=10)
    ax.set_title(f'Lag distribution comparison — {ds}\n'
                 f'(cells vs. fMRI-predicted at cell coordinates)',
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, OUT_DIR / f'lag_distribution_comparison_{ds}')
    plt.close(fig)


def plot_glass_brain_cells_by_pref_lag(coords, pref_angle, sig_mask=None,
                                        output_dir=None):
    """Glass brain: mPFC cells coloured by their own most-tuned lag.
    Non-significant cells (if `sig_mask` given) drawn hollow, sig drawn filled."""
    try:
        from nilearn.plotting import plot_glass_brain
    except Exception as exc:
        print(f"  [glass_brain] nilearn unavailable: {exc}")
        return
    fig = plt.figure(figsize=(9.0, 3.6), dpi=300, constrained_layout=True)
    display = plot_glass_brain(
        None, display_mode='ortho', figure=fig, annotate=True,
        black_bg=False, colorbar=False, alpha=0.35,
    )
    colours = angles_to_colours(pref_angle)
    if sig_mask is None:
        display.add_markers(coords, marker_color=colours,
                             marker_size=28, alpha=0.85)
    else:
        sig = np.asarray(sig_mask, dtype=bool)
        if (~sig).any():
            display.add_markers(coords[~sig], marker_color=colours[~sig],
                                 marker_size=14, alpha=0.35)
        if sig.any():
            display.add_markers(coords[sig], marker_color=colours[sig],
                                 marker_size=34, alpha=0.95)
    cbar_ax = fig.add_axes([0.10, 0.02, 0.35, 0.03])
    sm = ScalarMappable(cmap=CIRC_CMAP, norm=Normalize(vmin=-180, vmax=180))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbar.set_label("cell's preferred lag (°)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    target_dir = OUT_DIR if output_dir is None else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = str(target_dir / 'glass_brain_cells_by_pref_lag.png')
    display.savefig(stem, dpi=300)
    if SAVE_PDF:
        display.savefig(stem.replace('.png', '.pdf'), dpi=300)
    plt.close(fig)
    print(f"  → {stem}")


# ── Write a comprehensive Markdown summary ───────────────────────────
def _write_md(summary_rows):
    md = OUT_DIR / 'RESULTS.md'
    L = LAG_ANGLES.tolist()
    def _f(x, n=4):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return 'n/a'
        return f'{x:.{n}g}'
    def _pct(x): return _f(100*x, 2) + '%'
    lines = [
        "# fMRI → mPFC-cell preferred-lag prediction — results",
        "",
        f"Generated by `scripts/cell_fmri_lag_prediction.py`. "
        f"Outputs live in `{OUT_DIR}`.",
        "",
        "## Question",
        "Does the fMRI harmonic-angle map at each mPFC cell's MNI voxel",
        "predict that cell's preferred spatial-tuning lag?",
        "",
        "## Settings",
        f"- **harmonic vector mode**: "
            f"`{'unit_vector_derived' if USE_UNIT_VECTOR_MAPS else 'magnitude_weighted'}`",
        f"- **harmonic input root**: `{BASE_HARMONIC}`",
        f"- **cell ROI**: `{CELL_ROI}`",
        f"- **lag-column kind**: `{LAG_COL_KIND}`",
        f"- **cell preferred-angle method**: `{CELL_ANGLE_METHOD}` — "
            f"{CELL_ANGLE_RESULTS[CELL_ANGLE_METHOD]['description']}",
        f"- **cells loaded**: {len(mpfc)} (with valid tuning data)",
        f"- **filter 1 (MIN_CELL_R)**: `{MIN_CELL_R}` — cells passing: "
            f"{int(cell_ok.sum())}",
        f"- **filter 2 (MIN_FMRI_AMP_PCT)**: `{MIN_FMRI_AMP_PCT}`",
        f"- **spatial smoothing (SPHERE_RADIUS_MM)**: `{SPHERE_RADIUS_MM}` mm "
            "(amp-weighted circular mean of fMRI angles in that sphere)",
        f"- **bilateral symmetrise cos/sin (FMRI_SYMMETRISE)**: "
            f"`{FMRI_SYMMETRISE}`",
        f"- **volumetric Gaussian on cos/sin (FMRI_PRE_SMOOTH_FWHM_MM)**: "
            f"`{FMRI_PRE_SMOOTH_FWHM_MM}` mm",
        f"- **top-K**: `{TOP_K}`",
        f"- **N_PERM**: `{N_PERM}`",
        f"- **significance threshold** (per-lag p): `{SIG_P_THRESHOLD}`",
        f"- **cells with sig argmax (p<{SIG_P_THRESHOLD})**: "
            f"{int(sig_argmax.sum())}/{len(mpfc)}",
        f"- **cells with ≥1 sig top-{TOP_K} lag**: "
            f"{int(sig_topk_any.sum())}/{len(mpfc)}",
        "",
        "## Per-cell metrics — observed vs. three null distributions",
        "",
        "Each row shows the observed value and permutation p-values against",
        "**three** different nulls that answer different questions.",
        "",
        "- **A1 (shuffle mPFC pairing)** — permute fMRI predictions across mPFC",
        "  cells.  Preserves both marginals; answers *\"is the specific per-cell",
        "  pairing informative *beyond* what the two marginal distributions",
        "  already give?\"*.  Strictest.",
        "- **A2 (random whole-brain voxel)** — replace each cell's fMRI",
        "  prediction with the angle at a random brain voxel drawn from anywhere",
        "  with signal.  Answers *\"does the ANATOMICAL cell↔mPFC-voxel pairing",
        "  matter?\"*.  Best for the marginal-alignment claim.",
        "- **B2 (circular-shift cell tuning)** — rotate each cell's selected",
        "  preferred angle by a random multiple of 30°; fMRI stays fixed.",
        "  Answers *\"does the cell's ACTUAL preference (not a shifted version of",
        "  its noise structure) match the fMRI?\"*.  For match metrics this",
        "  collapses to ≈ the uniform baseline.",
        "",
        "| dataset | n cells | metric | observed | p (A1: shuffle mPFC) "
        "| p (A2: whole-brain) | p (B2: shift cell) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        n = r['n_valid']; ds = r['ds']
        lines += [
            f"| {ds} | {n} | Fisher–Lee r | {_f(r['r_circ'])} | "
                f"{_f(r['p_circ_perm_A1'])} | {_f(r['p_circ_A2'])} | — |",
            f"| {ds} | {n} | exact match | {_pct(r['exact_rate'])} | "
                f"{_f(r['p_exact_A1'])} | {_f(r['p_exact_A2'])} | "
                f"{_f(r['p_exact_B2'])} |",
            f"| {ds} | {n} | top-{TOP_K} match | {_pct(r['topk_rate'])} | "
                f"{_f(r['p_topk_A1'])} | {_f(r['p_topk_A2'])} | "
                f"{_f(r['p_topk_B2'])} |",
            f"| {ds} | {n} | mean \\|Δ\\| | {_f(r['mean_abs'], 3)}° | "
                f"{_f(r['p_abs_A1'])} | {_f(r['p_abs_A2'])} | — |",
        ]
    lines += [
        "",
        "**Null means** for reference (what each null gives on average):",
        "",
        "| dataset | A1 exact | A1 top-K | A2 exact | A2 top-K | B2 exact | B2 top-K |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['ds']} | "
            f"— | — | "
            f"{_pct(r['A2_null_exact_mean'])} | {_pct(r['A2_null_topk_mean'])} | "
            f"{_pct(r['B2_null_exact_mean'])} | {_pct(r['B2_null_topk_mean'])} |")
    lines += [
        "",
        f"Baselines under uniform random matching: exact = "
        f"{100/12:.1f}%, top-{TOP_K} = {100*TOP_K/12:.1f}%, "
        f"mean |Δ| = 90°.",
        "",
        "## Population-level distribution tests",
        "",
        "These answer *\"do the fMRI-predicted-lag distribution and the",
        "selected cell-angle distribution look like the same distribution?\"* —",
        "the marginal-alignment claim you asked about.",
        "",
        "### Chi-square test (nominal, 12-bin contingency table)",
        "",
        "H0: cell preferred-angle bin counts and fMRI-predicted bin counts come from",
        "the same categorical distribution over the 12 lag bins.",
        "",
        "| dataset | χ² | df | p |",
        "|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['ds']} | {_f(r['chi2_stat'], 3)} | "
            f"{r['chi2_df']} | {_f(r['chi2_p'])} |")
    lines += [
        "",
        "### Watson U² (circular, permutation p)",
        "",
        "Non-parametric two-sample circular test; label-shuffle null.",
        "",
        "| dataset | U² | perm p |",
        "|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['ds']} | {_f(r['watson_u2'])} | "
            f"{_f(r['watson_p'])} |")
    lines += [
        "",
        "### Circular means + Rayleigh concentration",
        "",
        "R̄ is the mean resultant length (0 = uniform, 1 = perfect ",
        "concentration).  Rayleigh p tests H0 = uniform on the circle.",
        "",
        "| dataset | cell mean (Rayleigh R̄, p) | fMRI mean (Rayleigh R̄, p)"
        " | Δ mean (cell − fMRI) |",
        "|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['ds']} | {_f(r['cell_mean_deg'], 4)}° "
            f"(R̄={_f(r['cell_rayleigh_R'])}, p={_f(r['cell_rayleigh_p'])}) "
            f"| {_f(r['fmri_mean_deg'], 4)}° "
            f"(R̄={_f(r['fmri_rayleigh_R'])}, p={_f(r['fmri_rayleigh_p'])}) "
            f"| {_f(r['delta_mean_deg'], 4)}° |")
    # Per-bin counts table
    lines += ["", "### Per-bin counts (both distributions)", ""]
    for r in summary_rows:
        lines += [
            f"**{r['ds']}**",
            "",
            "| lag (°) | " + " | ".join(f"{a}°" for a in L) + " |",
            "|---" * (len(L) + 1) + "|",
            f"| cell {CELL_ANGLE_METHOD} | "
                + " | ".join(str(c) for c in r['cell_bin_counts']) + " |",
            f"| fMRI predicted | "
                + " | ".join(str(c) for c in r['fmri_bin_counts']) + " |",
            "",
        ]
    lines += [
        "## Interpretation guide",
        "",
        "1. **Per-cell prediction**: compare Fisher–Lee r, mean |\u0394|, and",
        "   their permutation p-values across the three method folders.",
        "   A lower mean |\u0394| is only convincing if it also improves relative",
        "   to the pairing-shuffle null.",
        "",
        "2. **Population-level alignment**: chi² and Watson U² test whether",
        "   the two angle distributions match; they do not establish that the",
        "   correct fMRI voxel predicts the correct individual neuron.",
        "",
        "3. **Harmonic strength**: for continuous methods, inspect",
        "   `cell_angle_harmonic_amplitude` and",
        "   `cell_angle_relative_strength` in the per-cell CSV. A smooth angle",
        "   is poorly determined when its first-harmonic strength is near zero.",
        "",
        "## Files",
        "",
    ]
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix in ('.png', '.pdf', '.csv'):
            lines.append(f"- `{f.name}`")
    lines.append("")
    md.write_text("\n".join(lines))
    print(f"\nWrote {md}")


# ── Main ─────────────────────────────────────────────────────────────
def _config_tag(min_cell_r, top_k):
    r_tag = f'minR{min_cell_r:g}' if min_cell_r is not None else 'anyR'
    k_tag = 'argmax' if top_k == 1 else f'top{top_k}'
    return f'{r_tag}_{k_tag}'


print("Harmonic vector mode: "
      f"{'unit-vector derived' if USE_UNIT_VECTOR_MAPS else 'magnitude-weighted'}")
print(f"Harmonic input root: {BASE_HARMONIC}")
unknown_methods = sorted(set(CELL_ANGLE_METHODS) - set(CELL_ANGLE_RESULTS))
if unknown_methods:
    raise ValueError(f"Unknown CELL_ANGLE_METHODS: {unknown_methods}")

all_method_summary_rows = []
for _cell_method in CELL_ANGLE_METHODS:
    # Select the cell preference used by all direct cell–fMRI comparisons.
    CELL_ANGLE_METHOD = _cell_method
    selected = CELL_ANGLE_RESULTS[_cell_method]
    cell_pref_angle = selected['angle']
    cell_pref_snap_idx = snap_to_lag_bin(cell_pref_angle)
    cell_pref_amplitude = selected['amplitude']
    cell_pref_relative_strength = selected['relative_strength']

    method_dir = OUT_BASE_DIR / _cell_method
    print("\n" + "#" * 72)
    print(f"CELL ANGLE METHOD: {_cell_method}")
    print(f"  {selected['description']}")
    print("#" * 72)

    for _min_r in MIN_CELL_R_OPTIONS:
        for _top_k in TOP_K_OPTIONS:
            # Set module globals so run_dataset() + plot helpers see them.
            MIN_CELL_R = _min_r
            TOP_K      = _top_k
            tag        = _config_tag(_min_r, _top_k)
            OUT_DIR    = method_dir / tag
            OUT_DIR.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 72)
            print(f"CONFIG  MIN_CELL_R = {_min_r}   TOP_K = {_top_k}   "
                  f"→  {_cell_method}/{tag}")
            print("=" * 72)

            cfg = build_config_arrays(_min_r, _top_k)
            cell_top_idx    = cfg['cell_top_idx']
            cell_top_angles = cfg['cell_top_angles']
            sig_topk_any    = cfg['sig_topk_any']
            cell_ok         = cfg['cell_ok']

            # Re-seed so methods/configs see identical random draws (fair).
            rng = np.random.default_rng(SEED)

            _plot_lag_diagnostic(r_matrix, cell_argmax_idx)
            print("Wrote diagnostic_lag_distribution.pdf/png  "
                  "(argmax histogram + population mean r per lag)")

            summary_rows = []
            for ds in DATASETS:
                row = run_dataset(ds)
                if row is not None:
                    summary_rows.append(row)
                    all_method_summary_rows.append(row)

            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(
                    OUT_DIR / 'summary_metrics.csv', index=False)
                print(f"\nSummary metrics → "
                      f"{OUT_DIR / 'summary_metrics.csv'}")

            _write_md(summary_rows)

    # This cell-only map does not depend on MIN_CELL_R or TOP_K, so render it
    # exactly once per angle method rather than once per configuration.
    print("\n=== Glass-brain overlay of mPFC cells by "
          f"{_cell_method} preferred angle ===")
    plot_glass_brain_cells_by_pref_lag(
        coords, cell_pref_angle, sig_mask=sig_argmax,
        output_dir=method_dir)

if all_method_summary_rows:
    comparison_path = OUT_BASE_DIR / 'cell_angle_method_comparison.csv'
    pd.DataFrame(all_method_summary_rows).to_csv(comparison_path, index=False)
    print(f"\nCross-method comparison → {comparison_path}")

print(f"\nAll outputs in: {OUT_BASE_DIR}")
