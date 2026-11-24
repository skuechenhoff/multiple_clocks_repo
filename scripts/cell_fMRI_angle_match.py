#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do human mPFC single units show the same dorsoventral future-lag
gradient that the 7T fMRI shows?

This is the reported cell <-> fMRI correspondence analysis. Cells are
ordered along an anatomical axis derived from the fMRI gradient mask,
split into groups, and each group's pooled spatial-tuning profile is read
out for its preferred future lag. If the cell population carries the same
gradient, the preferred lag should increase from ventral to dorsal.

PIPELINE POSITION
-----------------
  per_lag_encoding.py        -> per-cell 12-lag spatial-tuning profiles
                                (`r_lag{000..330}_noctrl`: leave-one-
                                configuration-out correlation between
                                lag-shifted 9-location rate maps)
  cell_gradient_master_table.py -> per_cell_master.csv: adds the gradient
                                axis coordinate, gradient-mask membership,
                                and the fMRI preferred angle sampled at
                                each cell
  THIS SCRIPT                -> the splits, their pooled profiles, the
                                brain overlays, and the two CSVs

This script is self-contained: it imports no other script in this repo.

USAGE
-----
    python cell_fMRI_angle_match.py [--master-csv per_cell_master.csv]
                                    [--out DIR] [--brains | --no-brains]

Set `PLOT_BRAINS` at the top of this file to turn the fsaverage brain
overlays on or off (--brains / --no-brains override it for one run).

Everything numeric — the splits, the pooled profiles, both CSVs and the
2D profile plots — is reproduced from `per_cell_master.csv` alone (~80 KB)
on numpy + pandas + matplotlib. nibabel, nilearn and mne are imported
lazily and ONLY when the brains are drawn, which additionally needs the
fMRI group volumes, the gradient mask, a working 3D backend and an
fsaverage download.

THE BACKDROP IS NOT CONFIGURABLE HERE — AND THAT IS DELIBERATE
--------------------------------------------------------------
The brain figure shows cells coloured by their group's preferred lag on
top of the fMRI preferred-angle map. Those two things are only comparable
if the map is processed exactly as the per-cell angles were. So the
symmetrisation, the pre-projection smoothing, the harmonic volumes and
the gradient mask are all READ FROM the `config.json` that
cell_gradient_master_table.py wrote next to `per_cell_master.csv`
(`load_master_provenance`), and the script refuses to render if that file
is missing rather than guessing. To see a raw, unsmoothed map, change the
settings in cell_gradient_master_table.py and regenerate the master table
— then the cells and the backdrop move together and the figure still
means something. The only rendering knob left here is
`SURFACE_SMOOTHING_STEPS`, which is display-side mesh smoothing with no
counterpart in the per-cell sampling and cannot change a reported number.

THE ANATOMICAL AXIS
-------------------
Defined in cell_gradient_master_table.py as PC1 of the MNI coordinates of
the fMRI gradient-mask voxels, with x folded to |x| so the axis is
bilaterally symmetric, oriented +z = dorsal. It is essentially
dorsoventral (r = 0.98 with MNI z) with a slight posterior tilt. Note it
comes from the GEOMETRY of the mask, not from the fMRI angle values, so
the anatomical ordering of cells is independent of the fMRI effect it is
compared against.

SPLITS (in-mask cells only; outside-mask cells shown grey, never used to
define the gradient)
  1. z_tercile_consist : MNI z, 3 terciles, consistency-weighted
                         -> ventral 30deg, mid 60deg, dorsal 60deg
                            (fMRI at those cells: 34 / 59 / 85deg)
  2. pc1_ventral_dorsal: PC1 axis, median split, unweighted
                         -> ventral 30deg (n=48) / dorsal 60deg (n=39)
                            *** the split reported in the manuscript ***
  3. y_antpost_subj    : MNI y, median split, subject-weighted
                         -> posterior 60deg / anterior 30deg

Profiles are POOLED FIRST, then the peak is read off — never an average
of per-cell argmaxes, which are unreliable at this SNR. Three weightings
are available: unweighted, consistency-weighted (by each cell's peak r),
and subject-weighted (mean within subject, then across subjects).
Uncertainty: +-SEM across cells for the unweighted split; the 16-84th
percentile of 1000 bootstrap resamples for the consistency-weighted
(cells resampled) and subject-weighted (subjects resampled) splits.

TWO CAVEATS THAT BELONG WITH ANY REPORT OF THIS RESULT
------------------------------------------------------
* Effective N is recording SITES, not cells. The 87 in-mask cells (of 158
  mPFC cells) come from 19 distinct coordinates; cells on one microwire
  bundle share identical coordinates. Six cells sit exactly ON the pc1
  median (-13.51), which is why that split is 48/39 rather than 43/44
  (ties go to the `<= median` ventral side), and the two groups draw on
  12 and 7 sites. The per-cell circular correlation between cell lag and
  fMRI angle is null (approx -0.15 to -0.21); the agreement here is
  coarse and directional, at group level.
* The axis/n_groups/weighting combinations reported above were picked
  from a much larger sweep for the cleanest gradient-direction match.
  Those exploratory scripts now live in
  `scripts/old/cell_fMRI_gradient_exploration/` (see its README). The
  splits are therefore descriptive, not an inferential test, and should
  be presented as such.

FIGURES
-------
Brain (BRAIN_SCHEMES only; the manuscript shows pc1_ventral_dorsal): the
medial fsaverage surface with a semi-transparent fMRI-gradient backdrop,
rendered with an ORTHOGRAPHIC camera looking along x. The cells, the split
axis (white) and the median split (black) are then drawn on top in MNI mm,
so every cell sits at its exact (y, z) -- the same projection as the
sagittal glass-brain views of Fig 2c -- and none can hide behind the
cortex. Cells are never jittered; cells on one bundle share a coordinate
and are drawn as one marker. Cells are coloured by their group's
preferred angle on the backdrop's cyclic wheel, outside-mask cells grey.
Profiles: 4.5 x 3 cm, Arial >= 9 pt, grey outside-mask line. Bands are
+-SEM across cells for the unweighted split (pc1_ventral_dorsal, as Fig 3c;
changed 2026-11-23), bootstrap 16-84% for the weighted ones.

OUTPUTS (in <master>/final_splits/)
  {scheme}_{lh,rh}_medial.{png,pdf}       -- one hemisphere, print size
  {scheme}_medial_pair.{png,pdf}          -- lh | rh side by side
  {scheme}_profiles.{png,pdf}             -- profile overlay
  final_splits_per_cell.csv               -- per-cell group + angle
  final_splits_summary.csv                -- per-group summary
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator

# NOTE: nibabel / nilearn / mne are imported LAZILY inside the figure code
# only. The numeric result (splits, pooled profiles, both CSVs) runs on
# numpy + pandas + matplotlib alone, so a broken 3D backend or a missing
# fsaverage download cannot block reproducing the reported numbers.

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 9
mpl.rcParams['pdf.fonttype'] = 42

# ── Paths (defaults; override with --master-csv / --out) ─────────────
# >>> RERUN-CHECK: hardcoded upstream run dir -- update after re-running cell_gradient_master_table.py
MASTER_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/cell_gradient_master/2026-11-23_14-22-23_per_repeat_execute'
    # per-repeat rate maps (2026-11-23); published mode version: 2026-08-28_15-19-35
)
DEFAULT_MASTER_CSV = MASTER_DIR / 'per_cell_master.csv'
DEFAULT_OUT = MASTER_DIR / 'final_splits'

# ── Render the fsaverage brain overlays? ─────────────────────────────
# True  -> also write the medial brain panels (BRAIN_SCHEMES). Needs mne +
#          nilearn, a working 3D backend, and an fsaverage download.
# False -> CSVs and 2D profile plots only (all the numeric results).
PLOT_BRAINS = True

# The column of per_cell_master.csv carrying the fMRI angle at each cell.
# The backdrop is built from the SAME dataset this column came from.
FMRI_ANGLE_COLUMN = 'fmri_angle_quarters_deg'

L = np.arange(0, 360, 30)
CM, FS = 1 / 2.54, 9
GREY = (0.7, 0.7, 0.7)

AMP_EPS = 1e-6

# Display-only: MNE mesh-neighbour smoothing of the projected backdrop.
# This has NO counterpart in the per-cell sampling (which reads the volume
# inside a sphere and never touches a surface), so it is a rendering
# choice, not a data-processing one, and cannot change any reported number.
SURFACE_SMOOTHING_STEPS = 5

# ── Provenance: everything that determines the ANGLE VALUES ──────────
# Symmetrisation, pre-projection smoothing, the harmonic volumes and the
# gradient mask are deliberately NOT settable here. They are read from the
# config.json that cell_gradient_master_table.py wrote next to
# per_cell_master.csv, so the backdrop is always built exactly the way the
# per-cell angles were. Choosing them independently would put the cells and
# the map underneath them on two different conventions and make any
# apparent cell-vs-backdrop mismatch uninterpretable.
def load_master_provenance(master_csv):
    """Return the fMRI-sampling settings used to build `master_csv`.

    Raises if the sibling config.json is missing rather than falling back
    to defaults — a guessed convention is exactly what this prevents.
    """
    import json
    cfg_path = Path(master_csv).parent / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config.json next to {master_csv}. The brain overlay must be "
            f"built with the same fMRI sampling settings as the per-cell "
            f"angles, and those settings live in the master run's "
            f"config.json. Re-run cell_gradient_master_table.py, or point "
            f"--master-csv at a run directory that has one.")
    cfg = json.load(open(cfg_path))
    missing = [k for k in ('harmonic_root', 'gradient_mask',
                           'fmri_symmetrise', 'fmri_smooth_fwhm_mm')
               if k not in cfg]
    if missing:
        raise KeyError(
            f"{cfg_path} is missing {missing} — it predates the provenance "
            f"fields. Re-run cell_gradient_master_table.py.")
    dataset = FMRI_ANGLE_COLUMN.replace('fmri_angle_', '').replace('_deg', '')
    prov = {
        'harmonic_root': Path(cfg['harmonic_root']),
        'gradient_mask': Path(cfg['gradient_mask']),
        'dataset':       dataset,
        'symmetrise':    bool(cfg['fmri_symmetrise']),
        'smooth_fwhm':   float(cfg['fmri_smooth_fwhm_mm']),
        'sphere_radius': cfg.get('sphere_radius_mm'),
        'config_path':   str(cfg_path),
    }
    print(f"  [provenance] {cfg_path.parent.name}/config.json -> "
          f"dataset={prov['dataset']}, symmetrise={prov['symmetrise']}, "
          f"smooth_fwhm={prov['smooth_fwhm']} mm, "
          f"per-cell sphere={prov['sphere_radius']} mm")
    return prov

# ── Brain panel: medial surface, orthographic camera, exact positions ──
# The camera looks along x from the midline with a PARALLEL projection, so
# where a cell lands on the image is exactly its MNI (y, z), whatever its
# depth -- the projection of the sagittal glass-brain views in Fig 2c.
# Checked 2026-10-06: test spheres rendered at known (y, z) land within
# 0.07 mm of where the extent below puts them.
# Sizes are for the panel AT PRINT SIZE (Fig 3d: ~2.1 x 3.0 cm per
# hemisphere), so do not rescale the PDF when placing it.
BRAIN_SCHEMES = ('pc1_ventral_dorsal',)   # the split shown in the manuscript
SURF_ALPHA = 0.25           # see-through cortex, as in panel b
CORTEX = 'classic'          # mne.viz.Brain curvature shading
BACKDROP_ALPHA = 0.65       # fMRI gradient (1 = opaque)
CELL_STYLE = 'halo'         # 'edge' (black rim) | 'halo' (+ white ring)
CELL_DIAM_PT = 4.0          # marker diameter
CELL_EDGE_PT = 0.25         # black rim
CELL_HALO_PT = 0.4          # white ring outside the rim
VIEW_FOCAL_YZ = (20.0, 10.0)    # mm, centre of the rendered view
VIEW_HALF_HEIGHT_MM = 60.0      # camera parallel_scale
RENDER_SIZE = (1400, 1000)      # px, render window
CROP_Y_MM = (10.0, 72.0)        # the frontal medial wall shown in the panel
CROP_Z_MM = (-27.0, 62.0)
PANEL_W_CM, PANEL_H_CM = 2.1, 3.0   # one hemisphere at print size
AXIS_LINE_COLOR = 'w'       # split axis (PC1 of the gradient mask)
BOUND_LINE_COLOR = 'k'      # median split
AXIS_LINE_PT, BOUND_LINE_PT = 1.0, 0.9

# Cyclic wheel: -180 blue, -90 green, 0 yellow, +90 red, +180 blue.
CIRCULAR_ANCHORS_HEX = ['#1E88E5', '#43A047', '#FCE300', '#E53935', '#1E88E5']
CIRC_CMAP = LinearSegmentedColormap.from_list('circular_wheel',
                                              CIRCULAR_ANCHORS_HEX)


def angles_to_colours(angles_360):
    """[0,360) angle -> signed (-180,180] -> cyclic wheel RGBA."""
    a = np.asarray(angles_360, float)
    signed = ((a + 180.0) % 360.0) - 180.0
    return CIRC_CMAP(Normalize(vmin=-180, vmax=180)(signed))


def backdrop_texture(hemi, subjects_dir, prov):
    """+181-shifted fMRI preferred-angle texture for the gradient-masked
    backdrop on this hemisphere (NaN outside mask / no signal).

    Angle is circular, so the angle volume is NOT projected directly (that
    would smear across the +-180 wrap): the cos and sin group volumes are
    projected and the angle recomputed on the surface.

    `prov` comes from `load_master_provenance` — the settings the per-cell
    angles were sampled with. It is not user-settable, so the backdrop and
    the cells always share one angle convention.
    """
    import nibabel as nib
    from nilearn import surface as _surface
    from nilearn.image import new_img_like, smooth_img

    symmetrise, smooth_fwhm = prov['symmetrise'], prov['smooth_fwhm']

    def _symmetrise(img):
        d = img.get_fdata()
        return new_img_like(img, (d + d[::-1, ...]) / 2.0)

    surf = os.path.join(subjects_dir, 'fsaverage', 'surf', f'{hemi}.pial')
    cos_img = nib.load(str(prov['harmonic_root'] / prov['dataset'] / 'cos_group.nii.gz'))
    sin_img = nib.load(str(prov['harmonic_root'] / prov['dataset'] / 'sin_group.nii.gz'))
    if symmetrise:
        cos_img, sin_img = _symmetrise(cos_img), _symmetrise(sin_img)
    if smooth_fwhm:
        cos_img = smooth_img(cos_img, smooth_fwhm)
        sin_img = smooth_img(sin_img, smooth_fwhm)
    cos_txt = _surface.vol_to_surf(cos_img, surf, interpolation='linear')
    sin_txt = _surface.vol_to_surf(sin_img, surf, interpolation='linear')
    angle = np.degrees(np.arctan2(sin_txt, cos_txt))
    amp = np.hypot(cos_txt, sin_txt)
    mask_txt = _surface.vol_to_surf(
        nib.load(str(prov['gradient_mask'])), surf, interpolation='nearest')
    keep = (amp > AMP_EPS) & (mask_txt >= 0.5) & np.isfinite(angle)
    return np.where(keep, angle + 181.0, np.nan)

# scheme: (tag, axis_column, n_groups, weighting, group_names ventral->dorsal)
SCHEMES = [
    ('z_tercile_consist', 'MNI_z', 3, 'consist', ['ventral', 'mid', 'dorsal']),
    ('pc1_ventral_dorsal', 'grad_axis_coord', 2, 'unw', ['ventral', 'dorsal']),
    ('y_antpost_subj', 'MNI_y', 2, 'subj', ['posterior', 'anterior']),
]


# ── shared computation ───────────────────────────────────────────────
def cmean(a):
    a = np.radians(np.asarray(a, float)); a = a[np.isfinite(a)]
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360 \
        if a.size else np.nan


def _prof_from(Rg, sg, cg, weighting):
    if weighting == 'unw':
        return np.nanmean(Rg, 0)
    if weighting == 'consist':
        w = np.clip(cg, 0, None)
        return np.nansum(w[:, None] * Rg, 0) / (w.sum() + 1e-12)
    return np.nanmean(np.vstack([np.nanmean(Rg[sg == s], 0)
                                 for s in np.unique(sg)]), 0)


def profile(R, sel, subj, consist, weighting):
    return _prof_from(R[sel], subj[sel], consist[sel], weighting)


def group_angle(R, sel, subj, consist, weighting):
    p = profile(R, sel, subj, consist, weighting)
    return int(L[np.nanargmax(p)]) if np.isfinite(p).any() else np.nan


def boot_band(R, sel, subj, consist, weighting, nboot=1000):
    Rg, sg, cg = R[sel], subj[sel], consist[sel]
    rng = np.random.default_rng(0)
    boots = np.empty((nboot, len(L)))
    subs = np.unique(sg)
    for b in range(nboot):
        if weighting == 'subj':
            take = rng.choice(subs, subs.size, replace=True)
            rows = np.concatenate([np.where(sg == s)[0] for s in take])
            sb = np.concatenate([np.full((sg == s).sum(), i)
                                 for i, s in enumerate(take)])
            boots[b] = _prof_from(Rg[rows], sb, cg[rows], 'subj')
        else:
            i = rng.integers(0, len(Rg), len(Rg))
            boots[b] = _prof_from(Rg[i], sg[i], cg[i], weighting)
    return np.nanpercentile(boots, 16, 0), np.nanpercentile(boots, 84, 0)


def band(R, sel, subj, consist, weighting):
    """Unweighted profile: mean +- SEM across cells (as Fig 3c). The
    weighted schemes keep the 16-84th bootstrap percentile."""
    if weighting != 'unw':
        return boot_band(R, sel, subj, consist, weighting)
    Rg = R[sel]
    m = np.nanmean(Rg, 0)
    e = np.nanstd(Rg, 0, ddof=1) / np.sqrt(np.isfinite(Rg).sum(0))
    return m - e, m + e


def make_labels(x, inm, n):
    lab = np.array(['outside'] * len(x), dtype=object)
    if n == 2:
        med = np.median(x[inm])
        lab[inm & (x <= med)] = 'g0'; lab[inm & (x > med)] = 'g1'
    else:
        e = np.quantile(x[inm], [1/3, 2/3])
        names = ['g0', 'g1', 'g2']
        lab[inm] = [names[i] for i in np.digitize(x[inm], e)]
    return lab


def build_scheme(c, R, subj, consist, fmri, axis, n, weighting):
    inm = c['in_gradient_mask'].to_numpy(bool)
    lab = make_labels(axis, inm, n)
    order = sorted([g for g in set(lab) if g != 'outside'],
                   key=lambda g: np.nanmean(axis[lab == g]))
    ang = {g: group_angle(R, lab == g, subj, consist, weighting) for g in order}
    fm = {g: cmean(fmri[lab == g]) for g in order}
    return lab, order, ang, fm


def boundary_rule(axis_col, axis, inm, n):
    unit = 'PC1 mm' if axis_col == 'grad_axis_coord' else f'{axis_col} mm'
    direction = {'MNI_z': 'ventral->dorsal', 'MNI_y': 'posterior->anterior',
                 'grad_axis_coord': 'ventral->dorsal'}[axis_col]
    if n == 2:
        med = np.median(axis[inm])
        return (f'{unit} median split at {med:.2f} ({direction}); '
                f'low = axis<=median, high = axis>median')
    e = np.quantile(axis[inm], [1/3, 2/3])
    return (f'{unit} terciles at {e[0]:.2f} and {e[1]:.2f} ({direction})')


def summarize_scheme(tag, axis_col, axis, lab, order, rename, weighting, n,
                     R, subj, consist, fmri, inm):
    rule = boundary_rule(axis_col, axis, inm, n)
    rows = []
    for g in list(order) + ['outside']:
        sel = lab == g
        if sel.sum() == 0:
            continue
        prof = profile(R, sel, subj, consist, weighting)
        fv = fmri[sel]; av = axis[sel]
        rows.append(dict(
            scheme=tag, axis=axis_col, weighting=weighting,
            group=rename.get(g, 'outside'), in_mask=g != 'outside',
            n_in_group=int(sel.sum()), n_in_mask_total=int(inm.sum()),
            n_cells_total=int(len(lab)), boundary_rule=rule,
            group_axis_min=round(float(np.nanmin(av)), 2),
            group_axis_max=round(float(np.nanmax(av)), 2),
            group_axis_mean=round(float(np.nanmean(av)), 2),
            pooled_lag_deg=int(L[np.nanargmax(prof)]),
            pooled_peak_r=round(float(np.nanmax(prof)), 3),
            fmri_mean_deg=round(cmean(fv), 0),
            fmri_min_deg=round(float(np.nanmin(fv)), 0),
            fmri_max_deg=round(float(np.nanmax(fv)), 0)))
    return rows


# ── split-axis + boundary "t-bar" geometry ───────────────────────────
def gradient_pc1(prov):
    """Gradient-mask centroid + PC1 unit vector (folded x), oriented +z.

    Recomputed here only to draw the split-axis t-bars; the per-cell
    projection itself (`grad_axis_coord`) already comes from
    cell_gradient_master_table.py using the identical definition."""
    import nibabel as nib
    m = nib.load(str(prov['gradient_mask']))
    mni = nib.affines.apply_affine(m.affine, np.argwhere(m.get_fdata() > 0))
    mni[:, 0] = np.abs(mni[:, 0])
    centroid = mni.mean(0)
    _, _, vt = np.linalg.svd(mni - centroid, full_matrices=False)
    pc1 = vt[0]
    return centroid, (-pc1 if pc1[2] < 0 else pc1)


def compute_split_lines(c, axis_col, axis, inm, n, prov):
    """Return (axis_line, [boundary_lines]) as folded-(x,y,z) point arrays:
    a grey line along the split axis and a black perpendicular bar at each
    division boundary (median for n=2; terciles for n=3)."""
    coords = c[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    cx = np.abs(coords[inm, 0]).mean()
    cy, cz = coords[inm, 1].mean(), coords[inm, 2].mean()
    yv, zv = coords[inm, 1], coords[inm, 2]
    edges = ([np.median(axis[inm])] if n == 2
             else list(np.quantile(axis[inm], [1/3, 2/3])))
    if axis_col == 'MNI_z':
        zr = np.linspace(axis[inm].min() - 2, axis[inm].max() + 2, 40)
        axis_line = np.column_stack([np.full(40, cx), np.full(40, cy), zr])
        yw = np.linspace(yv.min() - 1, yv.max() + 1, 24)
        bounds = [np.column_stack([np.full(24, cx), yw, np.full(24, b)])
                  for b in edges]
    elif axis_col == 'MNI_y':
        yr = np.linspace(axis[inm].min() - 2, axis[inm].max() + 2, 40)
        axis_line = np.column_stack([np.full(40, cx), yr, np.full(40, cz)])
        zw = np.linspace(zv.min() - 1, zv.max() + 1, 24)
        bounds = [np.column_stack([np.full(24, cx), np.full(24, b), zw])
                  for b in edges]
    else:  # grad_axis_coord (PC1)
        centroid, pc1 = gradient_pc1(prov)
        tr = np.linspace(axis[inm].min() - 1, axis[inm].max() + 1, 40)
        axis_line = centroid[None] + tr[:, None] * pc1[None]
        perp = np.array([0.0, -pc1[2], pc1[1]]); perp /= np.linalg.norm(perp)
        fold = coords.copy(); fold[:, 0] = np.abs(fold[:, 0])
        hw = np.abs((fold[inm] - centroid) @ perp).max()
        uw = np.linspace(-hw, hw, 24)
        bounds = [(centroid + b * pc1)[None] + uw[:, None] * perp[None]
                  for b in edges]
    return axis_line, bounds


# ── brain panel (orthographic medial render + exact cell positions) ──
def render_medial_surface(hemi, subjects_dir, prov):
    """fsaverage pial, medial view, orthographic camera, gradient backdrop.

    Returns the screenshot and its extent in MNI mm (left, right, bottom,
    top) for imshow. lh medial has anterior on the right, rh on the left.
    """
    from mne.viz import Brain
    sign = -1 if hemi == 'lh' else 1
    brain = Brain('fsaverage', hemi, 'pial', background='white',
                  size=RENDER_SIZE, subjects_dir=subjects_dir,
                  alpha=SURF_ALPHA, cortex=CORTEX)
    brain.add_data(backdrop_texture(hemi, subjects_dir, prov), hemi=hemi,
                   fmin=1, fmid=181, fmax=361, colormap=CIRC_CMAP,
                   alpha=BACKDROP_ALPHA, colorbar=False,
                   smoothing_steps=(SURFACE_SMOOTHING_STEPS or None))
    plotter = brain._renderer.plotter
    cam = plotter.camera
    cam.focal_point = (0.0, *VIEW_FOCAL_YZ)
    cam.position = (-sign * 600.0, *VIEW_FOCAL_YZ)   # from the midline outward
    cam.up = (0, 0, 1)
    cam.parallel_projection = True
    cam.parallel_scale = VIEW_HALF_HEIGHT_MM
    plotter.renderer.ResetCameraClippingRange()
    img = brain.screenshot()
    brain.close()
    half_w = VIEW_HALF_HEIGHT_MM * img.shape[1] / img.shape[0]
    y_lo, y_hi = VIEW_FOCAL_YZ[0] - half_w, VIEW_FOCAL_YZ[0] + half_w
    z_lo, z_hi = (VIEW_FOCAL_YZ[1] - VIEW_HALF_HEIGHT_MM,
                  VIEW_FOCAL_YZ[1] + VIEW_HALF_HEIGHT_MM)
    y_ext = (y_lo, y_hi) if hemi == 'lh' else (y_hi, y_lo)
    return img, (*y_ext, z_lo, z_hi)


def draw_cells(ax, yz, colour):
    """One marker per recording site at its exact (y, z) in mm. Sizes are
    in points at print size (CELL_DIAM_PT, CELL_EDGE_PT, CELL_HALO_PT)."""
    if CELL_STYLE not in ('edge', 'halo'):
        raise ValueError(f'Unknown CELL_STYLE {CELL_STYLE!r}')
    if CELL_STYLE == 'halo':          # white disc behind = ring outside the rim
        halo_diam = CELL_DIAM_PT + CELL_EDGE_PT + 2 * CELL_HALO_PT
        ax.scatter(yz[:, 0], yz[:, 1], s=halo_diam ** 2, c='w',
                   linewidths=0, zorder=4)
    ax.scatter(yz[:, 0], yz[:, 1], s=CELL_DIAM_PT ** 2, c=[colour[:3]],
               edgecolors='k', linewidths=CELL_EDGE_PT, zorder=5)


def draw_brain_panel(ax, img, extent, hemi, coords, lab, order, ang,
                     axis_line, boundary_lines):
    ax.imshow(img, extent=extent, interpolation='lanczos')
    ax.plot(axis_line[:, 1], axis_line[:, 2], color=AXIS_LINE_COLOR,
            lw=AXIS_LINE_PT, solid_capstyle='round', zorder=8,
            path_effects=[patheffects.Stroke(linewidth=AXIS_LINE_PT + 0.5,
                                             foreground='0.35'),
                          patheffects.Normal()])
    for bl in boundary_lines:
        ax.plot(bl[:, 1], bl[:, 2], color=BOUND_LINE_COLOR, lw=BOUND_LINE_PT,
                solid_capstyle='round', zorder=8)
    in_hemi = coords[:, 0] < 0 if hemi == 'lh' else coords[:, 0] > 0
    colours = {g: angles_to_colours([ang[g]])[0] for g in order}
    colours['outside'] = GREY
    for g in ['outside'] + list(order):            # groups on top of grey
        # cells on one bundle share a coordinate exactly: draw each site once
        yz = np.unique(coords[in_hemi & (lab == g)][:, 1:3], axis=0)
        draw_cells(ax, yz, colours[g])
    ax.set_xlim(CROP_Y_MM if hemi == 'lh' else CROP_Y_MM[::-1])
    ax.set_ylim(CROP_Z_MM)
    ax.set_aspect('equal')
    ax.axis('off')


def render_brain(c, lab, order, ang, tag, subjects_dir,
                 axis_line, boundary_lines, out_dir, prov):
    """Backdrop built from `prov` (the master run's own sampling settings),
    so cells and map always share one convention."""
    coords = c[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    renders = {hemi: render_medial_surface(hemi, subjects_dir, prov)
               for hemi in ('lh', 'rh')}
    for hemi, (img, extent) in renders.items():
        fig, ax = plt.subplots(figsize=(PANEL_W_CM * CM, PANEL_H_CM * CM))
        fig.subplots_adjust(0, 0, 1, 1)
        draw_brain_panel(ax, img, extent, hemi, coords, lab, order, ang,
                         axis_line, boundary_lines)
        for ext in ('pdf', 'png'):
            fig.savefig(out_dir / f'{tag}_{hemi}_medial.{ext}', dpi=600)
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(2 * PANEL_W_CM * CM, PANEL_H_CM * CM))
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    for ax, hemi in zip(axs, ('lh', 'rh')):
        draw_brain_panel(ax, *renders[hemi], hemi, coords, lab, order, ang,
                         axis_line, boundary_lines)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'{tag}_medial_pair.{ext}', dpi=600)
    plt.close(fig)
    print(f'  wrote {tag}_{{lh,rh}}_medial + {tag}_medial_pair (pdf/png)')


# ── profile overlay (bootstrap bands + grey outside line) ────────────
def render_profiles(c, R, subj, consist, lab, order, tag, weighting, out_dir):
    fig, ax = plt.subplots(figsize=(4.5 * CM, 3 * CM), constrained_layout=True)
    out = lab == 'outside'
    po = profile(R, out, subj, consist, weighting)
    lo, hi = band(R, out, subj, consist, weighting)
    ax.fill_between(L, lo, hi, color='0.6', alpha=0.18, linewidth=0)
    ax.plot(L, po, '-', color='0.55', lw=1.0)
    for g in order:
        sel = lab == g
        p = profile(R, sel, subj, consist, weighting)
        lo, hi = band(R, sel, subj, consist, weighting)
        col = angles_to_colours([int(L[np.nanargmax(p)])])[0]
        ax.fill_between(L, lo, hi, color=col, alpha=0.2, linewidth=0)
        ax.plot(L, p, '-', color=col, lw=1.2)
        ax.plot(L[np.nanargmax(p)], np.nanmax(p), 'o', color=col, ms=2.6)
    ax.axhline(0, color='0.6', lw=0.5, ls='--')
    ax.set_xticks([0, 120, 240])
    ax.set_xlabel('lag (deg)', fontsize=FS, labelpad=1)
    ax.set_ylabel('CV r', fontsize=FS, labelpad=1)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=FS, length=2, pad=1, width=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.spines[['top', 'right']].set_visible(False)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'{tag}_profiles.{ext}', dpi=600)
    plt.close(fig)
    print(f'  wrote {tag}_profiles.pdf/png')


# ── main ─────────────────────────────────────────────────────────────
# =============================================================================
# STATISTICS
# =============================================================================
# Two read-outs are written alongside the split figures, so the manuscript
# numbers for this panel come from the same run that draws it.
#
#   lagwise_vs_zero.csv   one-sided t vs zero at every lag, at CELL and SUBJECT
#                         level. Cells on one microwire bundle share a
#                         coordinate exactly, so the subject-level row is the
#                         conservative one; the cell-level row matches the
#                         shading in the profile plots.
#   fmri_z_readout.csv    where each group sits on the fMRI gradient, read from
#                         the map's own z-profile rather than from single-voxel
#                         lookups at cell coordinates (SPHERE_RADIUS_MM = 0
#                         upstream): a single voxel inherits x/y variation and
#                         noise, which is not the quantity a dorsoventral
#                         gradient claim is about.
#   fmri_angle_z_profile.csv  that profile itself, for plotting.

FMRI_Z_SLAB_MM = 3.0      # half-width of the z window per profile step
FMRI_Z_MIN_VOX = 20


def bh_fdr(p):
    """Benjamini-Hochberg q-values, returned in the input order."""
    p = np.asarray(p, float)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return q


def _one_sided_t(values):
    """One-sided (greater than zero) t-test on Fisher-z transformed r."""
    from scipy import stats
    v = pd.Series(values).dropna()
    if len(v) < 2:
        return np.nan, np.nan, len(v)
    z = np.arctanh(np.clip(v.to_numpy(float), -0.999, 0.999))
    t, p_two = stats.ttest_1samp(z, 0.0)
    return float(t), float(p_two / 2 if t > 0 else 1 - p_two / 2), int(len(v))


def lagwise_vs_zero(c, R, lab, order, rename, tag):
    """One-sided tests against zero at every lag, cell and subject level.

    FDR is Benjamini-Hochberg across the groups of this scheme within each
    (unit, lag) -- the same family structure used for ROIs elsewhere in the
    project.
    """
    subj = c['subject_id'].to_numpy()
    rows = []
    for unit in ('cell', 'subject'):
        for g in order:
            m = np.array([x == g for x in lab])
            if not m.any():
                continue
            for k, lag in enumerate(L):
                vals = (R[m, k] if unit == 'cell'
                        else pd.Series(R[m, k]).groupby(subj[m]).mean())
                t, p, n = _one_sided_t(vals)
                rows.append(dict(
                    scheme=tag, unit=unit, group=rename.get(g, g),
                    lag_deg=int(lag), n=n,
                    mean_r=float(pd.Series(vals).dropna().mean()),
                    t=t, p_one_sided=p))
    out = pd.DataFrame(rows)
    if len(out):
        out['p_fdr_across_groups'] = np.nan
        for _, idx in out.groupby(['unit', 'lag_deg']).groups.items():
            out.loc[idx, 'p_fdr_across_groups'] = bh_fdr(
                out.loc[idx, 'p_one_sided'])
    return out


def fmri_z_profile(prov, dataset='quarters'):
    """Vector-mean fMRI preferred angle per 1 mm step of MNI z.

    Averaged over every gradient-mask voxel in a +-FMRI_Z_SLAB_MM slab, honouring
    the symmetrise / smoothing settings recorded by the master run so the profile
    matches the per-cell angles.
    """
    import nibabel as nib
    root = Path(prov['harmonic_root'])
    cos_i = nib.load(str(root / dataset / 'cos_group.nii.gz'))
    sin_i = nib.load(str(root / dataset / 'sin_group.nii.gz'))
    if prov.get('fmri_symmetrise'):
        def _sym(img):
            d = img.get_fdata()
            return nib.Nifti1Image((d + d[::-1]) / 2.0, img.affine, img.header)
        cos_i, sin_i = _sym(cos_i), _sym(sin_i)
    if prov.get('fmri_smooth_fwhm_mm'):
        from nilearn.image import smooth_img
        cos_i = smooth_img(cos_i, prov['fmri_smooth_fwhm_mm'])
        sin_i = smooth_img(sin_i, prov['fmri_smooth_fwhm_mm'])

    C, S = cos_i.get_fdata(), sin_i.get_fdata()
    msk = nib.load(str(prov['gradient_mask'])).get_fdata() > 0
    idx = np.argwhere(msk)
    z = nib.affines.apply_affine(cos_i.affine, idx)[:, 2]
    cc, ss = C[tuple(idx.T)], S[tuple(idx.T)]

    rows = []
    for zz in np.arange(np.floor(z.min()), np.ceil(z.max()) + 1e-9, 1.0):
        m = (z >= zz - FMRI_Z_SLAB_MM) & (z < zz + FMRI_Z_SLAB_MM)
        if m.sum() < FMRI_Z_MIN_VOX:
            continue
        rows.append(dict(
            z_mm=float(zz), n_voxels=int(m.sum()),
            angle_deg=float(np.rad2deg(
                np.arctan2(np.mean(ss[m]), np.mean(cc[m]))) % 360)))
    return pd.DataFrame(rows)


def _circ_median_deg(angles_deg):
    """Circular median: the angle minimising total circular distance.

    Reported alongside the vector mean because the fMRI angle at the recording
    sites is broadly distributed (roughly 0-120 deg), and a vector mean of a
    broad circular sample collapses towards the middle -- it read 58 vs 61 deg
    for two groups whose medians are 32 and 83 deg. The median is the honest
    summary here; the mean is kept only for continuity.
    """
    r = np.deg2rad(np.asarray(angles_deg, float))
    r = r[np.isfinite(r)]
    if not len(r):
        return np.nan
    d = np.abs(np.angle(np.exp(1j * (r[None, :] - r[:, None])))).sum(1)
    return float(np.rad2deg(r[int(np.argmin(d))]) % 360)


def fmri_z_readout(c, lab, order, rename, tag, prof):
    """Per group: z extent, the fMRI angle AT THE CELLS, and the z-profile span.

    Two different read-outs, deliberately both stored:

      *_at_cells_*   the fMRI angle sampled at each cell's own voxel, summarised
                     over the group. This is what "where do these cells sit on
                     the gradient" means, and it is the number to quote.
      *_zprofile_*   the whole-mask angle profile evaluated at the group's mean
                     z. Useful as context for the gradient as a whole, but it
                     averages over the full y and x extent of the mask (y 16-70),
                     far beyond the electrodes, so it does NOT describe the cells.
    """
    # unwrap so interpolation does not jump across 0/360
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(prof.angle_deg.to_numpy())))
    at = lambda zz: float(np.interp(zz, prof.z_mm, unwrapped))
    rows = []
    for g in order:
        m = np.array([x == g for x in lab])
        if not m.any():
            continue
        sub = c[m]
        zs = sub['MNI_z'].to_numpy(float)
        a_lo, a_hi = at(zs.min()), at(zs.max())
        fa = sub[FMRI_ANGLE_COLUMN].to_numpy(float)
        fa = fa[np.isfinite(fa)]
        vec_mean = (np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(fa))))) % 360
                    if len(fa) else np.nan)
        rows.append(dict(
            scheme=tag, group=rename.get(g, g), n_cells=int(m.sum()),
            n_sites=len(sub[['MNI_x', 'MNI_y', 'MNI_z']].round(2)
                        .drop_duplicates()),
            z_min=float(zs.min()), z_max=float(zs.max()),
            z_mean=float(zs.mean()),
            # --- at the cells' own voxels (quote these) ---
            fmri_at_cells_median_deg=_circ_median_deg(fa),
            fmri_at_cells_vec_mean_deg=vec_mean,
            fmri_at_cells_min_deg=float(fa.min()) if len(fa) else np.nan,
            fmri_at_cells_max_deg=float(fa.max()) if len(fa) else np.nan,
            # --- whole-mask z-profile (context only) ---
            fmri_zprofile_at_z_mean_deg=at(zs.mean()),
            fmri_zprofile_at_z_min_deg=a_lo,
            fmri_zprofile_at_z_max_deg=a_hi))
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--master-csv', type=Path, default=DEFAULT_MASTER_CSV,
                   help='per_cell_master.csv from cell_gradient_master_table.py. '
                        'This one file is all the numeric result needs.')
    p.add_argument('--out', type=Path, default=None,
                   help='output directory (default: <master-csv dir>/final_splits)')
    p.add_argument('--brains', dest='plot_brains', action='store_true',
                   default=PLOT_BRAINS,
                   help=f'render the fsaverage brain overlays '
                        f'(PLOT_BRAINS={PLOT_BRAINS} at the top of this file)')
    p.add_argument('--no-brains', dest='plot_brains', action='store_false',
                   help='skip the brain overlays; write CSVs + profile plots '
                        'only. Needs no mne / 3D backend / fsaverage.')
    # NOTE: symmetrisation, pre-projection smoothing, the harmonic volumes
    # and the gradient mask are intentionally NOT exposed here — they are
    # read from the master run's config.json (see load_master_provenance).
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out or args.master_csv.parent / 'final_splits'
    out.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(args.master_csv)
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    subj = c['subject_id'].to_numpy()
    consist = c['argmax_r'].to_numpy(float)
    if FMRI_ANGLE_COLUMN not in c.columns:
        raise KeyError(f"{args.master_csv} has no column {FMRI_ANGLE_COLUMN!r}")
    fmri = c[FMRI_ANGLE_COLUMN].to_numpy(float)

    # provenance is needed for the fMRI z-profile too, not only for the brains
    prov = load_master_provenance(args.master_csv)
    subjects_dir = None
    if args.plot_brains:
        from mne.datasets import fetch_fsaverage
        subjects_dir = os.path.dirname(fetch_fsaverage())
    else:
        print('(brains off: writing CSVs + profile plots only. Set '
              'PLOT_BRAINS = True at the top, or pass --brains.)')

    inm = c['in_gradient_mask'].to_numpy(bool)
    zprof = fmri_z_profile(prov)
    stat_rows, zread_rows = [], []
    per_cell = c[['neuron', 'subject_id', 'MNI_x', 'MNI_y', 'MNI_z',
                  'in_gradient_mask']].copy()
    per_cell['fmri_angle'] = fmri
    summary_rows = []

    for tag, col, n, w, names in SCHEMES:
        axis = c[col].to_numpy(float)
        lab, order, ang, fm = build_scheme(c, R, subj, consist, fmri, axis, n, w)
        rename = {g: nm for g, nm in zip(order, names)}
        sub = ', '.join(f'{rename[g]}={ang[g]:.0f}deg(fMRI{fm[g]:.0f})'
                        for g in order)
        print(f'\n=== {tag} ({w}) ===  ' + sub)
        per_cell[f'{tag}_group'] = [rename.get(g, 'outside') for g in lab]
        per_cell[f'{tag}_angle'] = [ang.get(g, np.nan) for g in lab]
        summary_rows += summarize_scheme(tag, col, axis, lab, order, rename, w,
                                         n, R, subj, consist, fmri, inm)
        if args.plot_brains and tag in BRAIN_SCHEMES:
            axis_line, boundary_lines = compute_split_lines(
                c, col, axis, inm, n, prov)
            render_brain(c, lab, order, ang, tag, subjects_dir,
                         axis_line, boundary_lines, out, prov)
        render_profiles(c, R, subj, consist, lab, order, tag, w, out)
        stat_rows.append(lagwise_vs_zero(c, R, lab, order, rename, tag))
        zread_rows.append(fmri_z_readout(c, lab, order, rename, tag, zprof))

    per_cell.to_csv(out / 'final_splits_per_cell.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / 'final_splits_summary.csv', index=False)

    stats_df = pd.concat(stat_rows, ignore_index=True)
    stats_df.to_csv(out / 'lagwise_vs_zero.csv', index=False)
    zread = pd.concat(zread_rows, ignore_index=True)
    zread.to_csv(out / 'fmri_z_readout.csv', index=False)
    zprof.to_csv(out / 'fmri_angle_z_profile.csv', index=False)
    pd.set_option('display.width', 240); pd.set_option('display.max_columns', 20)
    print('\n=== GROUP SUMMARY ===')
    print(summary.drop(columns=['boundary_rule', 'n_cells_total']).to_string(
        index=False))
    rep = stats_df[(stats_df.scheme == 'pc1_ventral_dorsal')
                   & (stats_df.lag_deg.isin([30, 60]))]
    if len(rep):
        print('\n=== vs zero at 30 / 60 deg (all lags in lagwise_vs_zero.csv) ===')
        print(rep[['unit', 'group', 'lag_deg', 'n', 'mean_r', 't',
                   'p_one_sided', 'p_fdr_across_groups']].round(4)
              .to_string(index=False))
    zr = zread[zread.scheme == 'pc1_ventral_dorsal']
    if len(zr):
        print('\n=== fMRI angle AT THE CELLS (quote these) ===')
        print(zr[['group', 'n_cells', 'n_sites', 'z_min', 'z_max', 'z_mean',
                  'fmri_at_cells_median_deg', 'fmri_at_cells_vec_mean_deg',
                  'fmri_at_cells_min_deg', 'fmri_at_cells_max_deg']]
              .round(1).to_string(index=False))
        print('\n=== whole-mask z-profile at the same z (context only) ===')
        print(zr[['group', 'fmri_zprofile_at_z_mean_deg',
                  'fmri_zprofile_at_z_min_deg', 'fmri_zprofile_at_z_max_deg']]
              .round(1).to_string(index=False))

    print(f'\nWrote {out / "final_splits_per_cell.csv"}')
    print(f'Wrote {out / "final_splits_summary.csv"}')
    print(f'Wrote {out / "lagwise_vs_zero.csv"}')
    print(f'Wrote {out / "fmri_z_readout.csv"}')
    print(f'Wrote {out / "fmri_angle_z_profile.csv"}')
    print(f'All outputs in {out}')


if __name__ == '__main__':
    main()
