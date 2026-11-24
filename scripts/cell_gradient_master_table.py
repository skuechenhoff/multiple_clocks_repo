#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master table for the cell <-> fMRI future-lag gradient.

Idea
----
Per-cell argmax of the 12-lag spatial-tuning profile is unreliable at this
SNR.  The robust unit is the *group-pooled mean profile*: pool the 12-lag
profiles of a set of cells, average them, and only then read off a preferred
lag (argmax AND continuous first-harmonic).  Averaging happens in profile
space, which denoises, before any argmax/angle is taken.

This script writes two tables so that *grouping is a downstream choice*:

1. ``per_cell_master.csv`` - one row per mPFC cell, carrying:
     * the raw 12-lag profile (noctrl + ctrl),
     * per-cell preference (argmax lag+r, harmonic angle+vector length),
     * fMRI preferred angle sampled at the cell (quarters + eighths), using
       the same symmetrise + smooth + 6 mm sphere pipeline as the surface
       display,
     * ordering axes: MNI_z, the gradient-mask principal-axis projection
       (dorsoventral, fMRI-independent), gradient-mask membership, distance
       to the mask centroid.

2. ``group_pooled_preference.csv`` - for several grouping schemes and two
   weightings (cell-weighted, subject-balanced), the pooled mean profile's
   preferred lag with a bootstrap CI, n_cells, n_subjects, and the group's
   mean fMRI angle / anatomical position.

Grouping schemes:
   * ``subject``        - one group per subject (the nice subject-mean view).
   * ``anat_tercile``   - terciles of the dorsoventral gradient axis
                          (ventral -> present, dorsal -> future). HEADLINE,
                          fMRI-independent.
   * ``fmri_angle_bin`` - quartile bins of the fMRI-predicted angle at each
                          cell (direct cross-modal correspondence).
   * ``gradient_tier``  - the algorithmic version of the "3 circles":
                          outside-mask / ventral-in-mask / dorsal-in-mask.

Diagnostic figures visualise the headline (cell lag vs anatomy), the
cross-modal version (cell lag vs fMRI angle), the 3-tier pooled profiles,
and a sanity check that the fMRI gradient itself is present across the cell
locations.

@author: Svenja Kuechenhoff (with Claude)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from mc.analyse.roi_labels import read_per_cell_table
from nilearn.image import new_img_like, smooth_img


# ---------------------------------------------------------------------
# Settings & paths
# ---------------------------------------------------------------------
# `CELL_TABLE` supplies the per-cell LAG STATISTICS only. Its MNI columns are
# frozen at the per-lag *base* run, because `relabel_per_cell` refreshes the
# `roi` column and nothing else -- so on 2026-08-28 they were up to 43 mm stale
# (session 52 still carried the retired placeholder coordinate). Coordinates are
# therefore taken from ROI_TABLE, the canonical anatomy table written by
# scripts/cell_to_roi_july26.py.
# >>> RERUN-CHECK: hardcoded upstream run dir -- update after re-running per_lag_encoding.py
CELL_TABLE = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/ephys_humans/derivatives/group/per_lag_encoding'
    '/2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled'
    '/per_cell_ALL_ROIs.csv'
)
ROI_TABLE = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/ephys_humans/derivatives/neurons_with_ROI_labels.csv'
)
HARMONIC_ROOT = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps'
    '/unit_vector_derived'
)
GRADIENT_MASK = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/masks/gradient_thr_1.5.nii.gz'
)
OUT_ROOT = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/ephys_humans/derivatives/group/cell_gradient_master'
)

CELL_ROI = 'mPFC'
LAGS_DEG = np.arange(0, 360, 30)            # 12 lags, 30 deg apart
FMRI_DATASETS = ('quarters', 'eighths')
FMRI_PRIMARY = 'quarters'                   # used for fmri_angle_bin grouping

# fMRI sampling pipeline (matches harmonic_maps_brain_overlay display).
FMRI_SYMMETRISE = False #True
FMRI_SMOOTH_FWHM_MM = 0 #3.0
SPHERE_RADIUS_MM = 0 #6.0

N_BOOT = 2000
RANDOM_SEED = 42

# 3-shade pink ramp for the ventral->dorsal (present->future) tiers.
TIER_COLOURS = ['#FCDDE3', '#D7657F', '#5C1027']


# ---------------------------------------------------------------------
# Cell-preference definitions
# ---------------------------------------------------------------------
def harmonic_angle(profiles):
    """First-harmonic angle (deg, [0,360)) and vector length per row.

    C = mean_k r_k cos(theta_k),  S = mean_k r_k sin(theta_k) over finite lags;
    angle = atan2(S, C), length = hypot(C, S).  Signed r retained (Fourier
    projection, not a probability-weighted circular mean).
    """
    P = np.atleast_2d(np.asarray(profiles, float))
    theta = np.radians(LAGS_DEG)
    finite = np.isfinite(P)
    n = finite.sum(1)
    safe = np.where(finite, P, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.sum(safe * np.cos(theta), 1) / n
        S = np.sum(safe * np.sin(theta), 1) / n
    length = np.hypot(C, S)
    angle = np.degrees(np.arctan2(S, C)) % 360.0
    valid = (n > 0) & np.isfinite(length)
    return np.where(valid, angle, np.nan), np.where(valid, length, np.nan)


def argmax_lag(profiles):
    """Discrete preferred lag (deg) and its correlation per row."""
    P = np.atleast_2d(np.asarray(profiles, float))
    idx = np.nanargmax(np.where(np.isfinite(P), P, -np.inf), axis=1)
    rows = np.arange(len(P))
    return LAGS_DEG[idx].astype(float), P[rows, idx]


# ---------------------------------------------------------------------
# fMRI sampling (symmetrise + smooth cos/sin, 6 mm sphere per cell)
# ---------------------------------------------------------------------
def _symmetrise(img):
    d = img.get_fdata()
    return new_img_like(img, (d + d[::-1, ...]) / 2.0)


def load_processed_cos_sin(dataset):
    cos_img = nib.load(str(HARMONIC_ROOT / dataset / 'cos_group.nii.gz'))
    sin_img = nib.load(str(HARMONIC_ROOT / dataset / 'sin_group.nii.gz'))
    if FMRI_SYMMETRISE:
        cos_img, sin_img = _symmetrise(cos_img), _symmetrise(sin_img)
    if FMRI_SMOOTH_FWHM_MM:
        cos_img = smooth_img(cos_img, FMRI_SMOOTH_FWHM_MM)
        sin_img = smooth_img(sin_img, FMRI_SMOOTH_FWHM_MM)
    return cos_img, sin_img


def sample_sphere(cos_img, sin_img, coords, radius_mm):
    """Amplitude-weighted circular mean angle (deg) and mean amp per coord."""
    cos_d, sin_d = cos_img.get_fdata(), sin_img.get_fdata()
    inv = np.linalg.inv(cos_img.affine)
    shape = np.array(cos_d.shape)
    vox_mm = np.sqrt((cos_img.affine[:3, :3] ** 2).sum(0)).mean()
    r_vox = int(np.ceil(radius_mm / vox_mm))
    ang = np.full(len(coords), np.nan)
    amp = np.full(len(coords), np.nan)
    for i, mni in enumerate(coords):
        c = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
        lo = np.clip(c - r_vox, 0, shape - 1)
        hi = np.clip(c + r_vox + 1, 0, shape)
        bc = cos_d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        bs = sin_d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        xg, yg, zg = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = np.sqrt((xg - c[0]) ** 2 + (yg - c[1]) ** 2
                    + (zg - c[2]) ** 2) * vox_mm
        a = np.sqrt(bc ** 2 + bs ** 2)
        sig = (d <= radius_mm) & (a > 1e-6) & np.isfinite(bc) & np.isfinite(bs)
        if not sig.any():
            continue
        C, S = np.mean(bc[sig]), np.mean(bs[sig])
        ang[i] = np.degrees(np.arctan2(S, C)) % 360.0
        amp[i] = float(np.hypot(C, S))
    return ang, amp


# ---------------------------------------------------------------------
# Gradient-mask anatomical axis (dorsoventral, fMRI-independent)
# ---------------------------------------------------------------------
def gradient_axis(mask_img):
    """Return (centroid_mni, pc1_unit) of the gradient mask on folded-x coords.

    PC1 is oriented so that the +z (dorsal) direction is positive, matching
    the expected present(ventral) -> future(dorsal) progression.
    """
    xyz = np.argwhere(mask_img.get_fdata() > 0)
    mni = nib.affines.apply_affine(mask_img.affine, xyz)
    mni[:, 0] = np.abs(mni[:, 0])           # fold LR (bilateral, symmetric)
    centroid = mni.mean(0)
    _, _, vt = np.linalg.svd(mni - centroid, full_matrices=False)
    pc1 = vt[0]
    if pc1[2] < 0:
        pc1 = -pc1
    return centroid, pc1


def sample_mask(mask_img, coords):
    inv = np.linalg.inv(mask_img.affine)
    d = mask_img.get_fdata()
    shape = np.array(d.shape)
    out = np.zeros(len(coords), bool)
    for i, mni in enumerate(coords):
        ijk = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
        if (ijk >= 0).all() and (ijk < shape).all():
            out[i] = d[tuple(ijk)] > 0
    return out


# ---------------------------------------------------------------------
# Pooled-group preference + bootstrap CI
# ---------------------------------------------------------------------
def circular_ci(angles_deg):
    """Circular mean and 2.5/97.5 pct interval of a bootstrap angle sample."""
    a = np.asarray(angles_deg, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    theta = np.radians(a)
    centre = np.degrees(np.arctan2(np.mean(np.sin(theta)),
                                   np.mean(np.cos(theta)))) % 360.0
    dev = (a - centre + 180.0) % 360.0 - 180.0
    lo, hi = np.percentile(dev, [2.5, 97.5])
    return centre, (centre + lo) % 360.0, (centre + hi) % 360.0


def pooled_profile(profiles, subjects, weighting):
    """Mean 12-lag profile of a cell set.

    weighting='cell'    : plain nanmean over cells.
    weighting='subject' : nanmean within subject, then nanmean over subjects.
    """
    P = np.asarray(profiles, float)
    if weighting == 'cell':
        return np.nanmean(P, axis=0)
    per_subj = [np.nanmean(P[subjects == s], 0) for s in np.unique(subjects)]
    return np.nanmean(np.vstack(per_subj), axis=0)


def group_preference(profiles, subjects, weighting, rng):
    """Pooled preferred lag (argmax + harmonic) with a bootstrap CI."""
    prof = pooled_profile(profiles, subjects, weighting)
    arg = argmax_lag(prof)[0][0]
    harm = harmonic_angle(prof)[0][0]

    boot = np.empty(N_BOOT)
    n = len(profiles)
    subj_u = np.unique(subjects)
    for b in range(N_BOOT):
        if weighting == 'cell':
            idx = rng.integers(0, n, n)
            bp = pooled_profile(profiles[idx], subjects[idx], 'cell')
        else:                                # resample subjects
            take = rng.choice(subj_u, len(subj_u), replace=True)
            rows = np.concatenate([np.where(subjects == s)[0] for s in take])
            bsub = np.concatenate([np.full((subjects == s).sum(), i)
                                   for i, s in enumerate(take)])
            bp = pooled_profile(profiles[rows], bsub, 'subject')
        boot[b] = harmonic_angle(bp)[0][0]
    centre, lo, hi = circular_ci(boot)
    return dict(pref_argmax_deg=arg, pref_harmonic_deg=harm,
                harmonic_boot_mean_deg=centre,
                harmonic_ci_lo_deg=lo, harmonic_ci_hi_deg=hi)


# ---------------------------------------------------------------------
# Build the master per-cell table
# ---------------------------------------------------------------------
def refresh_coordinates(cells, roi_table=ROI_TABLE, verbose=True):
    """Replace the per-cell CSV's MNI columns with the canonical ones.

    Joined on (subject, cell idx) -- the same keys
    `mc.analyse.roi_relabel.relabel_per_cell` uses. Fails loudly rather than
    silently keeping stale coordinates: the gradient analysis is entirely about
    where a cell sits, so a stale coordinate is not a cosmetic problem.
    """
    t = pd.read_csv(roi_table)[['subject', 'cell idx',
                                'MNI_x_final', 'MNI_y_final', 'MNI_z_final']]
    t.columns = ['subject_id', 'cell_idx', '_x', '_y', '_z']
    t['subject_id'] = t['subject_id'].astype(str)
    t['cell_idx'] = pd.to_numeric(t['cell_idx'], errors='coerce')
    if t.duplicated(['subject_id', 'cell_idx']).any():
        raise ValueError(f'{roi_table} has duplicate (subject, cell idx) keys')

    out = cells.copy()
    out['subject_id'] = out['subject_id'].astype(str)
    out['cell_idx'] = pd.to_numeric(out['cell_idx'], errors='coerce')
    out = out.merge(t, on=['subject_id', 'cell_idx'], how='left')

    missing = int(out['_x'].isna().sum())
    if missing:
        raise ValueError(
            f'{missing} of {len(out)} cells have no row in {roi_table.name}; '
            'the per-cell table and the anatomy table are out of sync.')

    shift = np.linalg.norm(
        out[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
        - out[['_x', '_y', '_z']].to_numpy(float), axis=1)
    for c in 'xyz':
        out[f'MNI_{c}'] = out[f'_{c}']
    out = out.drop(columns=['_x', '_y', '_z'])
    if verbose:
        print(f'  coordinates refreshed from {roi_table.name}: '
              f'{len(out)} cells, median shift {np.nanmedian(shift):.2f} mm, '
              f'max {np.nanmax(shift):.1f} mm, '
              f'{int((shift > 1).sum())} cells moved > 1 mm')
    return out


def build_master(cell_table=CELL_TABLE):
    src = read_per_cell_table(cell_table)
    cells = src[src.roi == CELL_ROI].copy()
    cells = refresh_coordinates(cells)
    prof_cols = [f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]
    R = cells[prof_cols].to_numpy(float)
    keep = np.isfinite(R).any(1)
    cells, R = cells[keep].reset_index(drop=True), R[keep]
    # per-repeat tables have no _ctrl variant (NaN here; Fig 3d never uses it)
    R_ctrl = cells.reindex(
        columns=[f'r_lag{a:03d}_ctrl' for a in LAGS_DEG]).to_numpy(float)
    coords = cells[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)

    out = cells[['neuron', 'subject_id', 'MNI_x', 'MNI_y', 'MNI_z']].copy()
    out['abs_x'] = np.abs(coords[:, 0])

    # per-cell preference
    a_lag, a_r = argmax_lag(R)
    h_ang, h_len = harmonic_angle(R)
    out['argmax_lag_deg'] = a_lag
    out['argmax_r'] = a_r
    out['harmonic_angle_deg'] = h_ang
    out['harmonic_vector_len'] = h_len
    out['argmax_lag_deg_ctrl'] = argmax_lag(R_ctrl)[0]
    out['harmonic_angle_deg_ctrl'] = harmonic_angle(R_ctrl)[0]
    for j, a in enumerate(LAGS_DEG):
        out[f'r_lag{a:03d}_noctrl'] = R[:, j]

    # fMRI angle at cell
    for ds in FMRI_DATASETS:
        cos_img, sin_img = load_processed_cos_sin(ds)
        ang, amp = sample_sphere(cos_img, sin_img, coords, SPHERE_RADIUS_MM)
        out[f'fmri_angle_{ds}_deg'] = ang
        out[f'fmri_amp_{ds}'] = amp

    # anatomical gradient axis (dorsoventral, fMRI-independent)
    mask_img = nib.load(str(GRADIENT_MASK))
    centroid, pc1 = gradient_axis(mask_img)
    fold = coords.copy()
    fold[:, 0] = np.abs(fold[:, 0])
    out['grad_axis_coord'] = (fold - centroid) @ pc1
    out['in_gradient_mask'] = sample_mask(mask_img, coords)
    out['dist_to_grad_centroid_mm'] = np.linalg.norm(fold - centroid, axis=1)

    return out, R


# ---------------------------------------------------------------------
# Build the group-pooled preference table
# ---------------------------------------------------------------------
def assign_groups(master):
    """Return a dict: scheme -> Series of group labels (NaN = excluded)."""
    proj = master['grad_axis_coord'].to_numpy()
    inm = master['in_gradient_mask'].to_numpy(bool)
    fmri = master[f'fmri_angle_{FMRI_PRIMARY}_deg'].to_numpy()

    schemes = {}
    schemes['subject'] = master['subject_id'].astype(str)

    # anatomical terciles (ventral / mid / dorsal)
    q = np.quantile(proj, [1/3, 2/3])
    anat = np.where(proj <= q[0], 'ventral',
                    np.where(proj <= q[1], 'mid', 'dorsal'))
    schemes['anat_tercile'] = pd.Series(anat, index=master.index)

    # fMRI-angle quartile bins (only cells with a finite fMRI angle)
    fbin = pd.Series(index=master.index, dtype=object)
    ok = np.isfinite(fmri)
    edges = np.quantile(fmri[ok], [0.25, 0.5, 0.75])
    lab = np.digitize(fmri[ok], edges)
    names = ['Q1_low', 'Q2', 'Q3', 'Q4_high']
    fbin.loc[master.index[ok]] = [names[i] for i in lab]
    schemes['fmri_angle_bin'] = fbin

    # gradient tier: outside / ventral-in / dorsal-in (median split within mask)
    med_in = np.median(proj[inm])
    tier = np.where(~inm, 'outside_mask',
                    np.where(proj <= med_in, 'ventral_in_mask',
                             'dorsal_in_mask'))
    schemes['gradient_tier'] = pd.Series(tier, index=master.index)
    return schemes


def build_group_table(master, R):
    subjects_all = master['subject_id'].astype(str).to_numpy()
    proj = master['grad_axis_coord'].to_numpy()
    schemes = assign_groups(master)
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    for scheme, labels in schemes.items():
        weightings = ['cell'] if scheme == 'subject' else ['cell', 'subject']
        for group in pd.unique(labels.dropna()):
            sel = (labels == group).to_numpy()
            if sel.sum() < 2:
                continue
            for w in weightings:
                pref = group_preference(R[sel], subjects_all[sel], w, rng)
                rows.append(dict(
                    scheme=scheme, group=str(group), weighting=w,
                    n_cells=int(sel.sum()),
                    n_subjects=int(pd.unique(subjects_all[sel]).size),
                    mean_grad_axis=float(np.nanmean(proj[sel])),
                    mean_fmri_angle_quarters=float(np.nanmean(
                        master.loc[sel, 'fmri_angle_quarters_deg'])),
                    mean_fmri_angle_eighths=float(np.nanmean(
                        master.loc[sel, 'fmri_angle_eighths_deg'])),
                    **pref,
                ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------
def _errbars(sub):
    """Lower/upper positive error magnitudes for a circular CI (wrapped)."""
    c = sub['pref_harmonic_deg'].to_numpy()
    lo = (c - sub['harmonic_ci_lo_deg'].to_numpy() + 180) % 360 - 180
    hi = (sub['harmonic_ci_hi_deg'].to_numpy() - c + 180) % 360 - 180
    return np.abs(np.vstack([lo, hi]))


def make_plots(master, group_tbl, out_dir):
    R = master[[f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]].to_numpy(float)

    # A. Headline: pooled cell lag vs anatomical position (anat_tercile)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(master['grad_axis_coord'], master['harmonic_angle_deg'],
               s=14, c='#bbb', alpha=0.6, label='individual cells')
    sub = group_tbl[(group_tbl.scheme == 'anat_tercile')
                    & (group_tbl.weighting == 'cell')]
    order = {'ventral': 0, 'mid': 1, 'dorsal': 2}
    sub = sub.sort_values('group', key=lambda s: s.map(order))
    ax.errorbar(sub['mean_grad_axis'], sub['pref_harmonic_deg'],
                yerr=_errbars(sub), fmt='o-', ms=9, lw=2, capsize=4,
                color='#5C1027', label='pooled tercile (harmonic + 95% CI)')
    ax.set_xlabel('dorsoventral gradient axis  (ventral -> dorsal)')
    ax.set_ylabel('preferred future lag (deg)')
    ax.set_title('HEADLINE: pooled cell lag vs anatomy')
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out_dir / 'A_headline_lag_vs_anatomy.png', dpi=200)
    plt.close(fig)

    # B. 3-tier pooled mean profiles
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    tiers = ['outside_mask', 'ventral_in_mask', 'dorsal_in_mask']
    schemes = assign_groups(master)
    lab = schemes['gradient_tier']
    for tier, col in zip(tiers, TIER_COLOURS):
        sel = (lab == tier).to_numpy()
        prof = np.nanmean(R[sel], 0)
        ax.plot(LAGS_DEG, prof, '-o', color=col, ms=4,
                label=f'{tier} (n={int(sel.sum())})')
        ax.axvline(LAGS_DEG[np.argmax(prof)], color=col, ls=':', lw=1)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('lag (deg)')
    ax.set_ylabel('pooled mean CV r')
    ax.set_title('3 algorithmic gradient tiers: pooled profiles')
    ax.set_xticks(LAGS_DEG)
    ax.tick_params(axis='x', labelsize=7)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out_dir / 'B_gradient_tier_profiles.png', dpi=200)
    plt.close(fig)

    # C. Cross-modal: pooled cell lag vs fMRI angle bin
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sub = group_tbl[(group_tbl.scheme == 'fmri_angle_bin')
                    & (group_tbl.weighting == 'cell')].sort_values(
                        'mean_fmri_angle_quarters')
    ax.errorbar(sub['mean_fmri_angle_quarters'], sub['pref_harmonic_deg'],
                yerr=_errbars(sub), fmt='o-', ms=9, lw=2, capsize=4,
                color='#23677E')
    ax.plot([0, 360], [0, 360], ls='--', color='#888', lw=1, label='y = x')
    ax.set_xlabel('mean fMRI predicted angle in bin (deg)')
    ax.set_ylabel('pooled cell preferred lag (deg)')
    ax.set_title('Cross-modal: cell lag vs fMRI angle')
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out_dir / 'C_crossmodal_lag_vs_fmri.png', dpi=200)
    plt.close(fig)

    # D. Sanity: does the fMRI gradient exist across cell locations?
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sc = ax.scatter(master['grad_axis_coord'],
                    master[f'fmri_angle_{FMRI_PRIMARY}_deg'],
                    c=master['in_gradient_mask'].map({True: '#5C1027',
                                                      False: '#bbb'}),
                    s=18)
    ax.set_xlabel('dorsoventral gradient axis')
    ax.set_ylabel('fMRI predicted angle (deg)')
    ax.set_title('Sanity: fMRI gradient across cell locations\n'
                 '(dark = inside gradient mask)')
    fig.savefig(out_dir / 'D_sanity_fmri_vs_anatomy.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--cell-table', type=Path, default=CELL_TABLE,
                   help='per_cell_ALL_ROIs.csv from a per-lag run')
    p.add_argument('--label', default='',
                   help='appended to the output folder name')
    args = p.parse_args()
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = OUT_ROOT / (f'{stamp}_{args.label}' if args.label else stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    master, R = build_master(args.cell_table)
    master.to_csv(out_dir / 'per_cell_master.csv', index=False)
    print(f'Wrote per_cell_master.csv  ({len(master)} cells)')

    group_tbl = build_group_table(master, R)
    group_tbl.to_csv(out_dir / 'group_pooled_preference.csv', index=False)
    print(f'Wrote group_pooled_preference.csv  ({len(group_tbl)} rows)')

    make_plots(master, group_tbl, out_dir)
    print('Wrote diagnostic figures A-D')

    config = dict(
        cell_table=str(args.cell_table), harmonic_root=str(HARMONIC_ROOT),
        gradient_mask=str(GRADIENT_MASK), cell_roi=CELL_ROI,
        lags_deg=LAGS_DEG.tolist(), fmri_datasets=list(FMRI_DATASETS),
        fmri_symmetrise=FMRI_SYMMETRISE, fmri_smooth_fwhm_mm=FMRI_SMOOTH_FWHM_MM,
        sphere_radius_mm=SPHERE_RADIUS_MM, n_boot=N_BOOT, seed=RANDOM_SEED,
        note='Pooled mean profile per group -> preferred lag (argmax + '
             'harmonic). Anatomical axis = gradient-mask PC1 (folded x), '
             'fMRI-independent.',
    )
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'\nAll outputs in {out_dir}')


if __name__ == '__main__':
    main()
