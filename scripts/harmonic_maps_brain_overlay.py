#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface overlays of the group-level harmonic maps with mPFC single-unit
cell coordinates plotted on top — mirrors the fsaverage / mne.viz.Brain
rendering used in ``scripts/cell_mask_overlap.py``.

For each dataset in DATASETS × each map in MAPS:
  * angle_deg / angle_deg_mPFC    → transformed to |angle| ∈ [0°, 180°]
    and rendered with a bright-yellow → dark-red ramp (0° = yellow;
    ±90° = orange; ±180° = dark red).  Sign is dropped by design so the
    colour is symmetric around 0°.
  * cos_group / sin_group         → RdBu_r diverging colormap centred
    on 0 (blue = negative, white = 0, red = positive).

Every map is gated by the dataset's ``amplitude.nii.gz`` so voxels with
no harmonic signal don't fake a 0° reading in yellow. The first-bin centre
is 45° for quarters and 22.5° for eighths.

The ``circular_alpha`` mode uses hue for preferred angle and per-vertex
opacity for the unit-vector mean resultant length R. R=0 is transparent
(subject directions cancel), while R=1 is maximally opaque (perfect
directional agreement). Its alpha source always comes from the unit-vector
results branch, even when the displayed angle is magnitude-weighted.

Cells: filtered to ``alt_final_roi == 'mPFC'`` in
``neurons_with_ROI_labels.csv`` (canonical MNI coords from
``MNI_{x,y,z}_final``), rendered as dark-green foci on the lh + rh
medial views of the fsaverage pial surface, hemisphere-split by
sign of ``MNI_x`` and jittered by a few mm so co-located cells separate.

Set ``USE_UNIT_VECTOR_MAPS`` to select either the original magnitude-weighted
harmonic maps or the separately stored unit-vector-derived maps. Outputs are
written beneath the selected branch.

Outputs (PDF + PNG per (dataset × map × hemi)):
  <selected_harmonic_maps>/brain_overlays_with_mPFC_cells/<dataset>/…

@author: Svenja Küchenhoff
"""
#import pdb; pdb.set_trace()
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

try:
    from mne.datasets import fetch_fsaverage
    from mne.viz import Brain
    from mne import Label
    HAS_MNE_BRAIN = True
except Exception as _exc:
    print(f"[warn] mne not available: {_exc}")
    HAS_MNE_BRAIN = False

try:
    from nilearn import surface as _surface
    from nilearn.image import resample_to_img, smooth_img, new_img_like
except Exception as _exc:
    print(f"[error] nilearn required: {_exc}")
    sys.exit(1)

# Import mask helpers from cell_mask_overlap so we don't duplicate the
# DSR-main-effect / gradient-union construction logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import cell_mask_overlap as cmo
    HAS_CMO = True
except Exception as _exc:
    print(f"[warn] cell_mask_overlap unavailable: {_exc}")
    HAS_CMO = False

# Binary masks (the ROI gate and the DSR outline) are projected with the same
# ribbon "projfrac-max" recipe used for the standalone main-effect figure, so
# the outline drawn here is pixel-identical to the filled main-effect panel.
import dsr_main_effect_surface as dsurf


# ── Settings ─────────────────────────────────────────────────────────
# Keep this flag aligned with ``harmonic_angle_maps.py``. True reads maps in
# ``unit_vector_derived/``; False reads the original magnitude-weighted maps.
USE_UNIT_VECTOR_MAPS = False
UNIT_VECTOR_RESULTS_DIRNAME = 'unit_vector_derived'

HARMONIC_RESULTS_ROOT = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps')
BASE_HARMONIC = (HARMONIC_RESULTS_ROOT / UNIT_VECTOR_RESULTS_DIRNAME
                 if USE_UNIT_VECTOR_MAPS else HARMONIC_RESULTS_ROOT)
CELL_TABLE = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                  '/ephys_humans/derivatives/neurons_with_ROI_labels.csv')
ROI_LABEL_COLUMN = 'alt_final_roi'
CELL_ROI_TO_PLOT = 'mPFC'

MPFC_MASK_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                      '/masks/mask_PFC_LR_smoothed_resampled.nii.gz')
GRAD15_MASK_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                        '/masks/gradient_thr_1.5.nii.gz')

# Optional black contour over preferred-angle maps.  This is deliberately
# separate from the ROI used to gate a map: it lets the DSR main-effect
# boundary be shown on every preferred-angle plot, including whole-brain and
# gradient plots.  Set to True when the contour is needed.
SHOW_DSR_MAIN_EFFECT_OUTLINE = True
DSR_MAIN_EFFECT_OUTLINE_PATH = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks/'
    'DSR_main_effect_mask.nii.gz')
DSR_MAIN_EFFECT_OUTLINE_COLOR = 'black'
# Number of cortical-mesh rings included in the contour.  One is a thin line.
DSR_MAIN_EFFECT_OUTLINE_WIDTH = 2

# ── Binary-mask surface projection (ROI gate + DSR outline) ──────────
# The old code used nilearn's default vol_to_surf, which averages the volume
# along a +-3 mm normal and then got thresholded at 0.5.  That eroded the
# elongated medial-wall cluster into disconnected gyral patches ("two
# islands").  Sampling the strongest value inside the cortical ribbon instead
# (FreeSurfer's --projfrac-max), then closing a couple of mesh rings,
# recovers the shape seen in the volumetric FSLeyes view.
MASK_RIBBON_DEPTHS = dsurf.RIBBON_DEPTHS
MASK_CLOSE_RINGS = 2
MASK_MIN_PATCH_VERTICES = 40

OUT_ROOT = BASE_HARMONIC / 'brain_overlays_with_mPFC_cells'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = ['quarters', 'eighths', 'quarters_state']

# Each map file: short_name → filename
MAP_FILES = {
    'angle_deg': 'angle_deg.nii.gz',
    'cos_group': 'cos_group.nii.gz',
    'sin_group': 'sin_group.nii.gz',
}

# Amplitude threshold — voxels below this are treated as background.
# In unit-vector mode amplitude is the mean resultant length (directional
# agreement); otherwise it is the raw group-vector magnitude.
AMP_EPS = 1e-6

# For the amp-gated circular mode: percentile of the in-signal +
# in-mask amplitude distribution below which voxels are hidden.
# Milder than the Hotelling / Rayleigh p<0.05 test but strong enough
# to suppress the small-magnitude arctan2 noise (bottom ~15% of voxels
# are where the "angle jumps" between neighbours come from).
AMP_GATE_PERCENTILE = 0

# Opacity mapping for ``circular_alpha``.  The highest-agreement vertices
# are fully opaque; only the remaining lower-agreement vertices are faded.
# Set AGREEMENT_ALPHA_OPAQUE_TOP_PCT to None for the original continuously
# transparent mapping across every vertex.
AGREEMENT_ALPHA_OPAQUE_TOP_PCT = 100 #40.0
# Gamma controls fading below the opaque cutoff: 0.5 uses sqrt(R), 1.0 is
# linear, and values above 1 fade low-agreement vertices more strongly.
AGREEMENT_ALPHA_GAMMA = 1.2
# Used only when AGREEMENT_ALPHA_OPAQUE_TOP_PCT is None (the legacy,
# fully-continuous alpha mapping).
AGREEMENT_ALPHA_MAX = 0.95
ALPHA_OPACITY_TAG = (
    'continuous' if AGREEMENT_ALPHA_OPAQUE_TOP_PCT is None
    else f'top{float(AGREEMENT_ALPHA_OPAQUE_TOP_PCT):g}_opaque')

# ── Angle-projection smoothing knobs (visualisation only) ─────────────
# The `circular` / `circular_gated` modes can either project the angle
# volume directly (default, may show ±180° wrap artefacts at hemisphere
# boundaries) OR project cos and sin separately and compute the angle
# per surface vertex (recommended, kills wrap discontinuities).
PROJECT_VIA_COS_SIN      = True     # A — project cos & sin, arctan2 at vertex
PRE_PROJ_SMOOTH_FWHM_MM  = 3.0      # B — Gaussian on cos/sin volumes (mm)
SURFACE_SMOOTHING_STEPS  = 5        # C — MNE mesh-neighbour iterations
BILATERAL_SYMMETRISE     = False     # D — x-flip cos/sin and average
MASK_AMP_TOP_PCT         = None     # E — keep top X% of amp within mask

# Save PDF alongside PNG?  Off by default (only PNG saved).
SAVE_PDF                 = True

# What to render.  (map, mask_key, mode)
#   mask_key = 'whole' | 'mPFC' | 'DSR_main' | 'gradient'  (looked up in MASKS)
#   mode     = 'abs_yellow_red' (yellow=0°, red=±180°) | 'diverging' (RdBu_r)
COMBINATIONS = [
    # Symmetric |angle| (yellow=0°, red=±180°, sign lost)
    ('angle_deg', 'whole',            'abs_yellow_red'),
    ('angle_deg', 'mPFC',             'abs_yellow_red'),
    ('angle_deg', 'DSR_main',         'abs_yellow_red'),
    ('angle_deg', 'gradient',         'abs_yellow_red'),
    ('angle_deg', 'gradient_thr1.5',  'abs_yellow_red'),
    # Cyclic wheel (yellow=0°, red=+90°, blue=±180°, green=-90°)
    ('angle_deg', 'whole',            'circular'),
    ('angle_deg', 'mPFC',             'circular'),
    ('angle_deg', 'DSR_main',         'circular'),
    ('angle_deg', 'gradient',         'circular'),
    ('angle_deg', 'gradient_thr1.5',  'circular'),
    # Cyclic wheel + amp gate (hides bottom AMP_GATE_PERCENTILE% of amp
    # within the mask to suppress arctan2 jitter at low-signal voxels)
    ('angle_deg', 'whole',            'circular_gated'),
    ('angle_deg', 'mPFC',             'circular_gated'),
    ('angle_deg', 'DSR_main',         'circular_gated'),
    ('angle_deg', 'gradient',         'circular_gated'),
    ('angle_deg', 'gradient_thr1.5',  'circular_gated'),
    # Cyclic hue + opacity given by between-subject directional agreement.
    ('angle_deg', 'mPFC',             'circular_alpha'),
    ('angle_deg', 'gradient',         'circular_alpha'),
    ('angle_deg', 'gradient_thr1.5',  'circular_alpha'),
    # Cos / sin (diverging RdBu_r).  Bilaterally symmetrised in the
    # 'diverging' branch when BILATERAL_SYMMETRISE = True (default),
    # so what you see is the same processed volume the cyclic angle
    # map is projected from.
    ('cos_group', 'whole',            'diverging'),
    ('sin_group', 'whole',            'diverging'),
    ('cos_group', 'mPFC',             'diverging'),
    ('sin_group', 'mPFC',             'diverging'),
    ('cos_group', 'gradient_thr1.5',  'diverging'),
    ('sin_group', 'gradient_thr1.5',  'diverging'),
]

# Short tag per mode for output filenames so all modes coexist.
MODE_TAG = {
    'abs_yellow_red': 'abs',
    'circular':       'circ',
    'circular_gated': f'circg{int(AMP_GATE_PERCENTILE)}',
    'circular_alpha': (f'circ_alphaR_{ALPHA_OPACITY_TAG}'
                       f'_g{AGREEMENT_ALPHA_GAMMA:g}'),
    'diverging':      'div',
}

# Custom yellow → dark-red ramp (same palette used for the gradient
# figure in cell_mask_overlap.py). Bright yellow at low end = 0°;
# dark red at top end = ±180°.  Sign of the angle is discarded.
YELLOW_RED_HEX = ['#FCE300', '#FF8C00', '#D2691E', '#8B0000', '#3D0000']

# Cyclic colour wheel: matches the reference wheel — anchor colours at
# angles -180°, -90°, 0°, +90°, +180° so opposite angles are opposite
# colours and neighbouring angles blend smoothly.  Linear interpolation
# yields teal between blue/green and orange between yellow/red.
CIRCULAR_ANCHORS_HEX = [
    '#1E88E5',   # -180°  → blue  (wraps to +180°)
    '#43A047',   #  -90°  → green
    '#FCE300',   #    0°  → yellow
    '#E53935',   #  +90°  → red
    '#1E88E5',   # +180°  → blue  (same as -180°)
]

# (AMP_EPS and AMP_GATE_PERCENTILE are defined above MODE_TAG, near
#  the head of the settings block, because MODE_TAG references
#  AMP_GATE_PERCENTILE in its f-string.)

CELL_COLOR = '#0e3d3a'   # CLAUDE.md "observed value" dark green
CELL_SCALE = 0.35
JITTER_MM  = 2.5
# Set False to render the surface maps without single-unit cell markers.
PLOT_CELLS = False

HEMIS = ('lh', 'rh')
VIEWS = ('medial', 'lateral')

# Fast-render filter — set to None to render every combination × hemi × view
# in the lists above.  Set to a dict to restrict to a subset (useful when
# iterating on a single plot; a full run is ~50 min).  The keys are optional
# — omit any of them to keep the full list for that axis.
RENDER_FILTER = {
    'combinations': [
        # ('angle_deg', 'mPFC',            'circular_gated'),
        # ('angle_deg', 'whole',           'circular_gated'),
        # ('angle_deg', 'gradient', 'circular_gated'),
        # ('angle_deg', 'gradient_thr1.5', 'circular_gated'),
        # ('angle_deg', 'mPFC',            'circular_alpha'),
        # ('angle_deg', 'gradient',        'circular_alpha'),
        ('angle_deg', 'gradient_thr1.5', 'circular_alpha')
        # ('cos_group', 'whole',           'diverging'),
        # ('sin_group', 'whole',           'diverging'),
        # ('cos_group', 'mPFC',            'diverging'),
        # ('sin_group', 'mPFC',            'diverging'),
        # ('cos_group', 'gradient_thr1.5', 'diverging'),
        # ('sin_group', 'gradient_thr1.5', 'diverging'),
    ],
    'views':        ('medial',),
    # 'hemis':      ('lh',),          # example: uncomment to restrict hemis
    # 'datasets':   ('quarters',),    # example: uncomment to restrict datasets
}
# RENDER_FILTER = None
# Surface geometry/appearance is shared with the DSR main-effect figure so
# the two panels are directly comparable — change it in one place only.
BRAIN_SIZE    = dsurf.BRAIN_SIZE
SURF_ALPHA    = dsurf.SURF_ALPHA
OVERLAY_ALPHA = 0.75


# ── Cached fsaverage path ────────────────────────────────────────────
_FSAVERAGE_PATH = None
def _ensure_fsaverage():
    global _FSAVERAGE_PATH
    if _FSAVERAGE_PATH is not None:
        return _FSAVERAGE_PATH
    fs_dir = fetch_fsaverage()
    _FSAVERAGE_PATH = os.path.dirname(fs_dir)
    return _FSAVERAGE_PATH

def _jitter(coords, mm, seed=0):
    if coords.size == 0 or mm <= 0:
        return coords
    rng = np.random.default_rng(seed)
    return coords + rng.uniform(-mm, mm, size=coords.shape)


def _resample_mask(mask_img, ref_img):
    """Nearest-neighbour resample `mask_img` onto `ref_img`'s grid if needed."""
    if (mask_img.shape[:3] == ref_img.shape[:3]
            and np.allclose(mask_img.affine, ref_img.affine, atol=1e-3)):
        return mask_img
    print(f"  [resample] {mask_img.shape[:3]} → {ref_img.shape[:3]}")
    return resample_to_img(mask_img, ref_img, interpolation='nearest')


def load_masks(ref_img):
    """Return dict {mask_key: nifti_img or None} resampled to `ref_img`.
    Silently drops any mask that fails to build/load."""
    masks = {'whole': None}
    # mPFC
    if MPFC_MASK_PATH.exists():
        masks['mPFC'] = _resample_mask(nib.load(str(MPFC_MASK_PATH)), ref_img)
        print(f"  mPFC mask: {int((masks['mPFC'].get_fdata() > 0.5).sum())} vox")
    else:
        print(f"  [warn] mPFC mask missing: {MPFC_MASK_PATH}")

    # Gradient_thr_1.5: union of 4 per-quarter t-maps thresholded at t>1.5
    # (built by hand — see gradient_thr_1.5.nii.gz in data/masks/).
    if GRAD15_MASK_PATH.exists():
        masks['gradient_thr1.5'] = _resample_mask(
            nib.load(str(GRAD15_MASK_PATH)), ref_img)
        print(f"  gradient_thr1.5 mask: "
              f"{int((masks['gradient_thr1.5'].get_fdata() > 0.5).sum())} vox")
    else:
        print(f"  [warn] gradient_thr1.5 mask missing: {GRAD15_MASK_PATH}")

    if not HAS_CMO:
        print("  [warn] cell_mask_overlap unavailable — no DSR/gradient masks.")
        return masks

    # DSR main effect (built on demand from cluster-mass PALM output).
    try:
        dsr = cmo.load_input_mask('DSR_main_effect',
                                   cmo.DSR_MAIN_EFFECT_MASK_PATH)
        masks['DSR_main'] = _resample_mask(dsr, ref_img)
        print(f"  DSR_main mask: "
              f"{int((masks['DSR_main'].get_fdata() > 0.5).sum())} vox")
    except Exception as exc:
        print(f"  [warn] DSR_main mask unavailable: {exc}")

    # Gradient (rebuild union from per-lag tstat maps).
    try:
        grad = cmo.build_gradient_union_mask(
            cmo.GRADIENT_TSTAT_DIR, cmo.GRADIENT_TSTAT_MAPS,
            cmo.GRADIENT_TSTAT_THRESHOLDS,
            prebuilt_path=cmo.GRADIENT_PREBUILT_MASK,
        )
        if grad is not None:
            masks['gradient'] = _resample_mask(grad, ref_img)
            print(f"  gradient mask: "
                  f"{int((masks['gradient'].get_fdata() > 0.5).sum())} vox")
    except Exception as exc:
        print(f"  [warn] gradient mask unavailable: {exc}")

    return masks


def load_dsr_main_effect_outline(ref_img):
    """Load the optional DSR mask contour on the harmonic-map grid."""
    if not SHOW_DSR_MAIN_EFFECT_OUTLINE:
        return None
    if not DSR_MAIN_EFFECT_OUTLINE_PATH.exists():
        print("  [warn] DSR main-effect outline mask missing: "
              f"{DSR_MAIN_EFFECT_OUTLINE_PATH}")
        return None
    outline_img = _resample_mask(
        nib.load(str(DSR_MAIN_EFFECT_OUTLINE_PATH)), ref_img)
    print("  DSR main-effect outline mask: "
          f"{int((outline_img.get_fdata() > 0.5).sum())} vox")
    return outline_img


def load_cells():
    df = pd.read_csv(CELL_TABLE)
    for ax in ('x', 'y', 'z'):
        if f'MNI_{ax}_final' in df.columns:
            df[f'MNI_{ax}'] = df[f'MNI_{ax}_final']
    df = df.dropna(subset=[ROI_LABEL_COLUMN, 'MNI_x', 'MNI_y', 'MNI_z'])
    df = df[df[ROI_LABEL_COLUMN] == CELL_ROI_TO_PLOT].copy()
    print(f"Loaded {len(df)} '{CELL_ROI_TO_PLOT}' cells "
          f"across {df['subject'].nunique()} subjects")
    return df


def _make_brain(hemi, subjects_dir, title=None):
    """Translucent pial glass brain — the SAME surface, size, background and
    cortex shading as the DSR main-effect panel, so the gradient overlay and
    the main-effect figure can sit side by side without looking like two
    different renderings.  Style lives in ``dsr_main_effect_surface`` only.
    """
    return dsurf.make_brain(hemi, 'pial', subjects_dir)


def _bilaterally_symmetrise(img):
    """Return `img` averaged with its own left-right mirror.
    Assumes standard MNI orientation where x-flipping the voxel array
    reflects the volume about the mid-sagittal plane."""
    data = img.get_fdata()
    flipped = data[::-1, ...]
    return new_img_like(img, (data + flipped) / 2.0)


def _unit_agreement_path(nii_path):
    """Return the matching unit-vector mean-resultant-length map.

    This deliberately does not use the magnitude-weighted ``amplitude``
    map: only the length of the mean subject-wise UNIT vectors has the
    interpretation "between-subject directional agreement".
    """
    dataset = Path(nii_path).parent.name
    return (HARMONIC_RESULTS_ROOT / UNIT_VECTOR_RESULTS_DIRNAME / dataset
            / 'amplitude.nii.gz')


_MESH_CACHE = {}


def _hemi_meshes(hemi, subjects_dir):
    """Cached (white, pial, faces) for one fsaverage hemisphere."""
    if hemi not in _MESH_CACHE:
        _MESH_CACHE[hemi] = dsurf.load_hemi_meshes(hemi, subjects_dir)
    return _MESH_CACHE[hemi]


def _project_binary_mask(mask_img, hemi, subjects_dir):
    """Boolean surface texture for a binary volume mask (ribbon projfrac-max).

    Returns ``None`` when the mask does not reach this hemisphere's ribbon.
    """
    if mask_img is None:
        return None
    white, pial, faces = _hemi_meshes(hemi, subjects_dir)
    texture = dsurf.project_ribbon(mask_img, white, pial, agg='max',
                                   depths=MASK_RIBBON_DEPTHS)
    mask = texture >= 0.5
    if not mask.any():
        return None
    mask = dsurf.close_mesh(mask, faces, MASK_CLOSE_RINGS)
    mask, n_drop, n_pat = dsurf.drop_small_patches(
        mask, faces, MASK_MIN_PATCH_VERTICES)
    if n_drop:
        print(f"    [{hemi}] dropped {n_drop} speckle vertices "
              f"in {n_pat} patches (<{MASK_MIN_PATCH_VERTICES} vertices)")
    return mask


def _add_mask_outline(brain, mask_img, hemi, subjects_dir):
    """Project a binary volume mask and add its boundary as a black line.

    ``Brain.add_label(..., borders=...)`` finds the boundary on the cortical
    mesh, so only the DSR mask's perimeter is drawn; its interior remains
    fully transparent and does not cover the preferred-angle colours.
    """
    if mask_img is None:
        return
    try:
        mask = _project_binary_mask(mask_img, hemi, subjects_dir)
        if mask is None:
            print(f"  [outline] DSR mask has no surface vertices on {hemi}")
            return
        outline = Label(vertices=np.flatnonzero(mask), hemi=hemi,
                        name='DSR_main_effect_outline')
        brain.add_label(outline, color=DSR_MAIN_EFFECT_OUTLINE_COLOR,
                        alpha=1.0,
                        borders=int(DSR_MAIN_EFFECT_OUTLINE_WIDTH))
    except Exception as exc:
        print(f"  [outline] DSR main-effect contour failed: {exc}")


def _project_and_transform(nii_path, amp_path, mode, hemi, subjects_dir,
                            roi_mask_img=None):
    """Load the map + amplitude (+ optional ROI mask), project each to
    `hemi` pial surface, apply the mode-specific transform, and return
    (data, cmap, fmin, fmid, fmax) ready to hand to brain.add_data.

    When `roi_mask_img` is given, vertices outside the mask become NaN
    so MNE renders them transparent (the underlying pial surface shows
    through as light grey)."""
    vertex_alpha = None
    img = nib.load(str(nii_path))
    amp_img = nib.load(str(amp_path))
    surf_path = os.path.join(subjects_dir, 'fsaverage', 'surf', f'{hemi}.pial')

    # For the diverging (cos/sin) branch we want the SAME processing that
    # the cyclic-angle branch applies (bilateral symmetrisation + volumetric
    # Gaussian on the cos/sin volume) so what you see for cos_group /
    # sin_group is the exact volume the angle map is projected from.
    if mode == 'diverging':
        if BILATERAL_SYMMETRISE:
            img = _bilaterally_symmetrise(img)
        if PRE_PROJ_SMOOTH_FWHM_MM and PRE_PROJ_SMOOTH_FWHM_MM > 0:
            img = smooth_img(img, PRE_PROJ_SMOOTH_FWHM_MM)
        texture = _surface.vol_to_surf(
            img, surf_path, interpolation='linear').astype(float)
    else:
        texture = _surface.vol_to_surf(
            img, surf_path, interpolation='nearest').astype(float)
    amp_txt = _surface.vol_to_surf(amp_img, surf_path,
                                    interpolation='nearest').astype(float)
    has_signal = amp_txt > AMP_EPS
    roi_mask_surf = None
    if roi_mask_img is not None:
        roi_mask_surf = _project_binary_mask(roi_mask_img, hemi, subjects_dir)
        if roi_mask_surf is None:
            roi_mask_surf = np.zeros_like(has_signal)
        has_signal = has_signal & roi_mask_surf

    if mode == 'abs_yellow_red':
        # Symmetric |angle| in degrees, 0 at yellow, 180 at dark red.
        data = np.where(has_signal, np.abs(texture), np.nan)
        cmap = LinearSegmentedColormap.from_list('yellow_to_darkred',
                                                  YELLOW_RED_HEX)
        fmin, fmid, fmax = 0.0, 90.0, 180.0
        cbar_info = {'vmin': 0.0, 'vmax': 180.0,
                     'ticks': [0, 45, 90, 135, 180],
                     'label': '|preferred angle| (°)'}
    elif mode in ('circular', 'circular_gated', 'circular_alpha'):
        # Cyclic wheel: keep the sign of the angle. Shift into [1, 361]
        # so MNE's fmin can stay strictly positive.  If PROJECT_VIA_COS_SIN,
        # project cos and sin group maps separately (optionally smoothed)
        # then compute arctan2 per vertex — kills ±180° wrap discontinuities.
        if PROJECT_VIA_COS_SIN:
            cos_path = nii_path.parent / 'cos_group.nii.gz'
            sin_path = nii_path.parent / 'sin_group.nii.gz'
            if cos_path.exists() and sin_path.exists():
                cos_img = nib.load(str(cos_path))
                sin_img = nib.load(str(sin_path))
                # (D) Bilateral symmetrisation on cos & sin volumes
                if BILATERAL_SYMMETRISE:
                    cos_img = _bilaterally_symmetrise(cos_img)
                    sin_img = _bilaterally_symmetrise(sin_img)
                # (B) Volumetric Gaussian on cos & sin (cos/sin are
                # continuous → smoothing is well-defined; smoothing the
                # angle directly would be a wrap-around disaster).
                if PRE_PROJ_SMOOTH_FWHM_MM and PRE_PROJ_SMOOTH_FWHM_MM > 0:
                    cos_img = smooth_img(cos_img, PRE_PROJ_SMOOTH_FWHM_MM)
                    sin_img = smooth_img(sin_img, PRE_PROJ_SMOOTH_FWHM_MM)
                cos_txt = _surface.vol_to_surf(
                    cos_img, surf_path, interpolation='linear').astype(float)
                sin_txt = _surface.vol_to_surf(
                    sin_img, surf_path, interpolation='linear').astype(float)
                # Recompute angle & amp on the surface (per vertex).
                texture = np.degrees(np.arctan2(sin_txt, cos_txt))
                amp_surf = np.sqrt(cos_txt ** 2 + sin_txt ** 2)
                # Refresh has_signal with the surface amplitude so the
                # ROI mask still applies.
                has_signal = (amp_surf > AMP_EPS)
                amp_txt = amp_surf   # for the gated-mode percentile below
                if roi_mask_surf is not None:
                    has_signal = has_signal & roi_mask_surf
            # else: fall through to the direct-angle path
        # (E) Optional: keep only the top X% of amp within the mask
        if MASK_AMP_TOP_PCT is not None and np.any(has_signal):
            drop_below = float(np.percentile(amp_txt[has_signal],
                                              MASK_AMP_TOP_PCT))
            has_signal = has_signal & (amp_txt >= drop_below)
        # circular_gated: additional in-mask amp cutoff
        if mode == 'circular_gated' and np.any(has_signal):
            amp_thr = float(np.percentile(amp_txt[has_signal],
                                           AMP_GATE_PERCENTILE))
            has_signal = has_signal & (amp_txt > amp_thr)
        # The harmonic generator already models angular-bin centres (45°
        # for quarters; 22.5° for eighths), so no display-only correction
        # belongs here.
        data = np.where(has_signal, texture + 181.0, np.nan)
        cmap = LinearSegmentedColormap.from_list('circular_wheel',
                                                  CIRCULAR_ANCHORS_HEX)
        fmin, fmid, fmax = 1.0, 181.0, 361.0
        cbar_info = {'vmin': -180.0, 'vmax': 180.0,
                     'ticks': [-180, -90, 0, 90, 180],
                     'label': 'preferred angle (°)'}
        if mode == 'circular_alpha':
            agreement_path = _unit_agreement_path(nii_path)
            if not agreement_path.exists():
                raise FileNotFoundError(
                    "circular_alpha needs the unit-vector agreement map: "
                    f"{agreement_path}")
            agreement_img = nib.load(str(agreement_path))
            if (agreement_img.shape[:3] != img.shape[:3]
                    or not np.allclose(agreement_img.affine, img.affine,
                                       atol=1e-3)):
                agreement_img = resample_to_img(
                    agreement_img, img, interpolation='linear')
            agreement_txt = _surface.vol_to_surf(
                agreement_img, surf_path,
                interpolation='linear').astype(float)
            agreement_txt = np.clip(
                np.nan_to_num(agreement_txt, nan=0.0), 0.0, 1.0)
            opaque_cutoff = None
            if AGREEMENT_ALPHA_OPAQUE_TOP_PCT is None:
                vertex_alpha = AGREEMENT_ALPHA_MAX * np.power(
                    agreement_txt, AGREEMENT_ALPHA_GAMMA)
                alpha_label = (
                    'subject agreement R\n'
                    f'opacity = {AGREEMENT_ALPHA_MAX:g} × '
                    f'R^{AGREEMENT_ALPHA_GAMMA:g}')
            else:
                opaque_top_pct = float(AGREEMENT_ALPHA_OPAQUE_TOP_PCT)
                if not 0.0 < opaque_top_pct <= 100.0:
                    raise ValueError(
                        'AGREEMENT_ALPHA_OPAQUE_TOP_PCT must be in '
                        '(0, 100], or None')
                if np.any(has_signal):
                    # E.g. 70% opaque means the cutoff is the 30th
                    # percentile: all vertices at/above it get alpha=1.
                    opaque_cutoff = float(np.percentile(
                        agreement_txt[has_signal], 100.0 - opaque_top_pct))
                else:
                    opaque_cutoff = 1.0
                if opaque_cutoff > 0:
                    fade_fraction = np.clip(
                        agreement_txt / opaque_cutoff, 0.0, 1.0)
                    vertex_alpha = np.power(
                        fade_fraction, AGREEMENT_ALPHA_GAMMA)
                else:
                    # A zero percentile cutoff means that many vertices have
                    # R=0. Keep those transparent and make non-zero R opaque
                    # rather than accidentally turning the entire map opaque.
                    vertex_alpha = (agreement_txt > 0).astype(float)
                alpha_label = (
                    'subject agreement R\n'
                    f'top {opaque_top_pct:g}% opaque; lower R fades '
                    f'(γ={AGREEMENT_ALPHA_GAMMA:g})')
            vertex_alpha = np.where(has_signal, vertex_alpha, 0.0)
            cbar_info['alpha_label'] = alpha_label
            cbar_info['alpha_opaque_cutoff'] = opaque_cutoff
            cbar_info['alpha_source'] = str(agreement_path)
    elif mode == 'diverging':
        raw = np.where(has_signal, texture, np.nan)
        vmax = float(np.nanmax(np.abs(raw))) if np.any(has_signal) else 1.0
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        # MNE's add_data can crash on negative fmin; shift the data
        # into [0, 2·vmax] so raw −vmax → 0, raw 0 → vmax (colormap
        # centre = white), raw +vmax → 2·vmax. RdBu_r goes blue→white
        # →red across [0, 1], so this preserves the diverging look.
        data = raw + vmax
        cmap = plt.get_cmap('RdBu_r')
        fmin, fmid, fmax = 0.0, vmax, 2.0 * vmax
        cbar_info = {'vmin': -vmax, 'vmax': vmax,
                     'ticks': [-vmax, 0, vmax],
                     'label': 'group mean'}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return data, cmap, fmin, fmid, fmax, cbar_info, vertex_alpha


def _save_brain_figure(brain, stem, cmap, cbar_info, vertex_alpha):
    """Save one rendered Brain as PNG (+PDF) with the matplotlib
    colourbar — and the R-opacity legend — underneath. Shared by
    `render_one` and `render_cell_gradient_panels` so every panel
    carries an identical colourbar."""
    png_path = str(stem) + '.png'
    pdf_path = str(stem) + '.pdf'
    try:
        brain.save_image(png_path)
        # Embed PNG in a PDF page with a matplotlib colourbar underneath
        # so the labels show raw values (not the shifted numbers MNE
        # would otherwise print).
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        img = plt.imread(png_path)
        h_in = img.shape[0] / 300; w_in = img.shape[1] / 300
        fig = plt.figure(figsize=(w_in, h_in + 0.55), dpi=300)
        gs = fig.add_gridspec(
            2, 1, height_ratios=[img.shape[0], 55],
            hspace=0.02, left=0.0, right=1.0, bottom=0.0, top=1.0,
        )
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(img); ax_img.axis('off')
        ax_cb = fig.add_subplot(gs[1])
        # Shrink the colourbar horizontally so it doesn't look absurd.
        cb_pos = ax_cb.get_position()
        cb_width_frac = 0.50 if vertex_alpha is not None else 0.55
        cb_left = 0.08 if vertex_alpha is not None else (1 - cb_width_frac) / 2
        ax_cb.set_position([cb_left, cb_pos.y0,
                             cb_width_frac, cb_pos.height * 0.4])
        norm = Normalize(vmin=cbar_info['vmin'], vmax=cbar_info['vmax'])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
        cbar.set_ticks(cbar_info['ticks'])
        cbar.set_ticklabels([f'{t:g}' for t in cbar_info['ticks']])
        cbar.set_label(cbar_info['label'], fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        if vertex_alpha is not None:
            # A compact second legend: black is composited over the same
            # light-grey background with alpha increasing from R=0 to R=1.
            ax_alpha = fig.add_axes(
                [0.68, cb_pos.y0, 0.24, cb_pos.height * 0.4])
            ramp = np.linspace(0.0, 1.0, 256)
            rgba = np.zeros((1, 256, 4), dtype=float)
            rgba[..., :3] = 0.05
            opaque_cutoff = cbar_info['alpha_opaque_cutoff']
            if opaque_cutoff is None:
                rgba[..., 3] = (AGREEMENT_ALPHA_MAX
                                * ramp ** AGREEMENT_ALPHA_GAMMA)
            elif opaque_cutoff > 0:
                rgba[..., 3] = np.clip(
                    ramp / opaque_cutoff, 0.0, 1.0) ** AGREEMENT_ALPHA_GAMMA
            else:
                rgba[..., 3] = (ramp > 0).astype(float)
            ax_alpha.set_facecolor('#d9d9d9')
            ax_alpha.imshow(rgba, aspect='auto', origin='lower',
                            extent=[0, 1, 0, 1])
            ax_alpha.set_yticks([])
            ax_alpha.set_xticks([0, 0.5, 1])
            ax_alpha.tick_params(axis='x', labelsize=8)
            ax_alpha.set_xlabel(cbar_info['alpha_label'], fontsize=8)
        # PNG always; PDF only if SAVE_PDF (default off).
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        if SAVE_PDF:
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote {png_path if not SAVE_PDF else pdf_path}")
    except Exception as exc:
        print(f"  [mne] save failed: {exc}")
    finally:
        try:
            brain.close()
        except Exception:
            pass


def render_one(ds, map_name, mask_name, nii_path, amp_path, mode,
                cells_df, subjects_dir, hemi, view,
                roi_mask_img=None, outline_mask_img=None):
    """Build one Brain, overlay data + cells, save PDF+PNG.

    `mask_name` is used only for output filenames / titles; the actual
    gating is via `roi_mask_img` (pass None to render whole brain)."""
    if not HAS_MNE_BRAIN:
        print("  [mne] not installed — skipping.")
        return
    if not nii_path.exists():
        print(f"  [skip] map missing: {nii_path}")
        return
    if not amp_path.exists():
        print(f"  [skip] amplitude missing: {amp_path}")
        return

    data, cmap, fmin, fmid, fmax, cbar_info, vertex_alpha = \
        _project_and_transform(
        nii_path, amp_path, mode, hemi, subjects_dir,
        roi_mask_img=roi_mask_img)
    if not np.any(np.isfinite(data)):
        print(f"  [skip] no supra-threshold voxels on {hemi} for "
              f"{map_name}×{mask_name}")
        return

    brain = _make_brain(hemi, subjects_dir, title=None)

    # Disable MNE's built-in colourbar — we add a matplotlib one during
    # PDF embedding so the labels can show raw (unshifted) values.
    add_kwargs = dict(hemi=hemi, fmin=fmin, fmid=fmid, fmax=fmax,
                      colormap=cmap, alpha=OVERLAY_ALPHA, colorbar=False)
    try:
        brain.add_data(data, smoothing_steps=int(SURFACE_SMOOTHING_STEPS),
                        **add_kwargs)
    except TypeError:
        brain.add_data(data, **add_kwargs)
    if vertex_alpha is not None:
        # MNE's public ``alpha`` argument is scalar-only, but its layered
        # mesh supports an opacity value per vertex. Replace the data
        # overlay's scalar opacity with R-derived opacity after add_data.
        mesh = brain._layered_meshes[hemi]
        mesh.update_overlay(name='data', opacity=vertex_alpha)
        brain._renderer._update()
        in_mask_alpha = vertex_alpha[vertex_alpha > 0]
        if in_mask_alpha.size:
            print("    alpha from unit-vector R: "
                  f"median={np.median(in_mask_alpha):.3f}, "
                  f"range={in_mask_alpha.min():.3f}.."
                  f"{in_mask_alpha.max():.3f}")

    # The outline is independent of the plot's gating mask and is limited to
    # preferred-angle maps (not cos/sin plots).
    if SHOW_DSR_MAIN_EFFECT_OUTLINE and map_name == 'angle_deg':
        _add_mask_outline(brain, outline_mask_img, hemi, subjects_dir)

    # Cells (hemisphere-split).  The flag also avoids requiring the cell table
    # for a surface-only rendering pass.
    if PLOT_CELLS and cells_df is not None:
        coords = cells_df[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(dtype=float)
        keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
        coords_h = _jitter(coords[keep], JITTER_MM,
                            seed=hash((hemi, map_name)) & 0xFFFF)
        if coords_h.shape[0]:
            try:
                brain.add_foci(coords_h, coords_as_verts=False, hemi=hemi,
                                color=CELL_COLOR, scale_factor=CELL_SCALE,
                                name=f'{CELL_ROI_TO_PLOT}_cells')
            except Exception as exc:
                print(f"  [mne] add_foci failed: {exc}")

    try:
        brain.show_view(view)
    except Exception as exc:
        print(f"  [mne] show_view({view!r}) failed: {exc}")

    out_dir = OUT_ROOT / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = MODE_TAG.get(mode, mode)
    # Append a small suffix when smoothing/symmetrisation is stronger
    # than default, so different visualisation passes don't overwrite.
    if BILATERAL_SYMMETRISE:
        mode_tag = f'{mode_tag}_bil'
    if SHOW_DSR_MAIN_EFFECT_OUTLINE and map_name == 'angle_deg':
        mode_tag = f'{mode_tag}_dsr_outline'
    # Marks the ribbon projfrac-max mask projection so the new renderings do
    # not overwrite the older vol_to_surf-projected figures.
    mode_tag = f'{mode_tag}_ribbonmax'
    if not PLOT_CELLS:
        mode_tag = f'{mode_tag}_no_cells'
    stem = out_dir / f'{map_name}__{mask_name}__{mode_tag}__{hemi}_{view}'
    _save_brain_figure(brain, stem, cmap, cbar_info, vertex_alpha)


# ── Cell-gradient figure panels (one map, four overlays) ─────────────
# The mPFC cell panels and the fMRI gradient figure have to show the SAME
# map, so they are rendered here, from this script's projection pipeline,
# rather than rebuilt next to the cell analysis. Only the cell markers and
# the split geometry come from cell_fMRI_angle_match.
CELL_PANEL_SPLITS_CSV = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/cell_gradient_master/2026-08-28_15-19-35'
    '/final_splits/final_splits_per_cell.csv')
CELL_PANEL_SCHEME = 'pc1_ventral_dorsal'   # the split reported in the paper
CELL_PANEL_DATASET = 'quarters'
# Unit-vector maps (one equal vote per subject) rather than the
# magnitude-weighted group mean. Amplitude weighting silently suppresses a
# weak-but-consistent direction — here the -90 deg territory — because the
# subjects and voxels with the largest effects dominate the group vector.
# That is a display bias against the smallest true effect, so these panels
# use equal subject weight. Opacity still comes from the unit-vector mean
# resultant length (`_unit_agreement_path`), which is unchanged either way.
CELL_PANEL_UNIT_VECTOR = True


def render_cell_gradient_panels(dataset=CELL_PANEL_DATASET,
                                splits_csv=CELL_PANEL_SPLITS_CSV,
                                out_dir=None, hemis=HEMIS, view='medial',
                                use_unit_vector=CELL_PANEL_UNIT_VECTOR):
    """Four figure panels that share ONE gradient map, per hemisphere:

      A  gradient only
      B  gradient + black DSR main-effect outline
      C  gradient + all mPFC cells in grey
      D  gradient + cells coloured by their group's preferred lag,
         with the split axis (grey) and boundary (black) on top

    The map is projected ONCE per hemisphere with this module's own
    `_project_and_transform` (mode `circular_alpha`, gradient_thr1.5 gate),
    so all four panels are pixel-identical in the map they show. NOTE the
    default map source here is the UNIT-VECTOR branch, while `render_one`
    follows the module-level `USE_UNIT_VECTOR_MAPS`; set both the same way
    for these panels and the standalone fMRI gradient figure to agree. Cell
    coordinates, group colours and the split geometry come from
    cell_fMRI_angle_match / its `final_splits_per_cell.csv`.
    """
    import cell_fMRI_angle_match as cfa

    subjects_dir = _ensure_fsaverage()
    root = (HARMONIC_RESULTS_ROOT / UNIT_VECTOR_RESULTS_DIRNAME
            if use_unit_vector else HARMONIC_RESULTS_ROOT)
    src_tag = 'unitvec' if use_unit_vector else 'magw'
    nii_path = root / dataset / 'angle_deg.nii.gz'
    amp_path = root / dataset / 'amplitude.nii.gz'
    ref_img = nib.load(str(nii_path))
    grad_img = _resample_mask(nib.load(str(GRAD15_MASK_PATH)), ref_img)
    outline_img = _resample_mask(
        nib.load(str(DSR_MAIN_EFFECT_OUTLINE_PATH)), ref_img)

    splits_csv = Path(splits_csv)
    out_dir = Path(out_dir) if out_dir else splits_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(splits_csv)
    coords = cells[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    grey = tuple(np.clip(np.array(cfa.GREY) * cfa.CELL_DARKEN, 0, 1))
    split_cols = np.tile(np.array(grey + (1.0,)), (len(cells), 1))
    grp = cells[f'{CELL_PANEL_SCHEME}_group'].to_numpy()
    ang = cells[f'{CELL_PANEL_SCHEME}_angle'].to_numpy(float)
    inm = grp != 'outside'
    split_cols[inm, :3] = np.clip(
        cfa.angles_to_colours(ang[inm])[:, :3] * cfa.CELL_DARKEN, 0, 1)

    # split axis + boundary, from the master table the splits were made on
    master_csv = splits_csv.parent.parent / 'per_cell_master.csv'
    master = pd.read_csv(master_csv)
    prov = cfa.load_master_provenance(master_csv)
    axis_line, boundary_lines = cfa.compute_split_lines(
        master, 'grad_axis_coord', master['grad_axis_coord'].to_numpy(float),
        master['in_gradient_mask'].to_numpy(bool), 2, prov)

    print(f"Cell-gradient panels: {dataset}, {src_tag} "
          f"({'equal subject weight' if use_unit_vector else 'magnitude-weighted'}), "
          f"{PRE_PROJ_SMOOTH_FWHM_MM:g} mm, gate=gradient_thr1.5, "
          f"mode=circular_alpha -> {out_dir}")
    for hemi in hemis:
        data, cmap, fmin, fmid, fmax, cbar_info, vertex_alpha = \
            _project_and_transform(nii_path, amp_path, 'circular_alpha',
                                   hemi, subjects_dir, roi_mask_img=grad_img)
        keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
        rng = np.random.default_rng(abs(hash(hemi)) % (2 ** 32))
        cj = coords[keep] + rng.uniform(-cfa.JITTER_MM, cfa.JITTER_MM,
                                        coords[keep].shape)
        sign = -1 if hemi == 'lh' else 1

        def _to_hemi(pts):
            q = np.asarray(pts, float).copy()
            q[:, 0] = sign * cfa.BAR_X
            return q

        for panel, cell_mode, outline, bars in (
                ('A_gradient_only',            None,    False, False),
                ('B_gradient_dsr_outline',     None,    True,  False),
                ('C_gradient_cells_grey',      'grey',  False, False),
                ('D_gradient_cells_ventral_dorsal', 'split', False, True)):
            brain = _make_brain(hemi, subjects_dir)
            add_kwargs = dict(hemi=hemi, fmin=fmin, fmid=fmid, fmax=fmax,
                              colormap=cmap, alpha=OVERLAY_ALPHA,
                              colorbar=False)
            try:
                brain.add_data(data, smoothing_steps=int(
                    SURFACE_SMOOTHING_STEPS), **add_kwargs)
            except TypeError:
                brain.add_data(data, **add_kwargs)
            if vertex_alpha is not None:
                brain._layered_meshes[hemi].update_overlay(
                    name='data', opacity=vertex_alpha)
                brain._renderer._update()
            if outline:
                _add_mask_outline(brain, outline_img, hemi, subjects_dir)
            if cell_mode is not None:
                cols = (split_cols[keep] if cell_mode == 'split'
                        else np.tile(np.array(grey + (1.0,)),
                                     (int(keep.sum()), 1)))
                for cc, col in zip(cj, cols):
                    brain.add_foci(cc[None], coords_as_verts=False, hemi=hemi,
                                   color=tuple(col[:3]),
                                   scale_factor=cfa.CELL_SCALE)
            if bars:
                brain.add_foci(_to_hemi(axis_line), coords_as_verts=False,
                               hemi=hemi, color=cfa.AXIS_BAR_COLOR,
                               scale_factor=cfa.BAR_SCALE)
                for bl in boundary_lines:
                    brain.add_foci(_to_hemi(bl), coords_as_verts=False,
                                   hemi=hemi, color=cfa.BOUND_BAR_COLOR,
                                   scale_factor=cfa.BAR_SCALE)
            try:
                brain.show_view(view)
            except Exception as exc:
                print(f"  [mne] show_view({view!r}) failed: {exc}")
            n = int(keep.sum()) if cell_mode else 0
            stem = out_dir / (f'cellpanel_{panel}__{dataset}_{src_tag}_'
                              f'gradient_thr1.5__{hemi}_{view}')
            _save_brain_figure(brain, stem, cmap, cbar_info, vertex_alpha)
            print(f'  {hemi} {panel}: {n} cells')


def main():
    vector_mode = ('unit-vector derived' if USE_UNIT_VECTOR_MAPS
                   else 'magnitude-weighted')
    print(f"Harmonic vector mode: {vector_mode}")
    print(f"Harmonic input root: {BASE_HARMONIC}")
    print(f"Overlay output root: {OUT_ROOT}")
    cells_df = load_cells() if PLOT_CELLS else None
    if not PLOT_CELLS:
        print("Cell markers: disabled (PLOT_CELLS = False)")
    subjects_dir = _ensure_fsaverage()
    if subjects_dir is None:
        print("[error] fsaverage unavailable — aborting.")
        return

    # Reference image: any harmonic output (they share the fMRI grid).
    ref_img = None
    for ds in DATASETS:
        cand = BASE_HARMONIC / ds / 'amplitude.nii.gz'
        if cand.exists():
            ref_img = nib.load(str(cand))
            break
    if ref_img is None:
        print("[error] no harmonic outputs found — aborting.")
        return

    print("\n=== Loading masks ===")
    masks = load_masks(ref_img)
    print(f"Masks available: {sorted(masks.keys())}")
    outline_mask_img = load_dsr_main_effect_outline(ref_img)

    # Apply optional render filter (see RENDER_FILTER at the top).
    rf = RENDER_FILTER or {}
    combos_to_run = rf.get('combinations', COMBINATIONS)
    views_to_run  = rf.get('views',        VIEWS)
    hemis_to_run  = rf.get('hemis',        HEMIS)
    ds_to_run     = rf.get('datasets',     DATASETS)
    if RENDER_FILTER:
        print(f"\n[RENDER_FILTER] active — restricting to:")
        print(f"  combinations: {combos_to_run}")
        print(f"  views:        {views_to_run}")
        print(f"  hemis:        {hemis_to_run}")
        print(f"  datasets:     {ds_to_run}")

    for ds in ds_to_run:
        ds_dir = BASE_HARMONIC / ds
        if not ds_dir.exists():
            print(f"[skip] {ds}: {ds_dir} not found")
            continue
        amp_path = ds_dir / 'amplitude.nii.gz'
        print(f"\n{'='*70}\n=== {ds} ===\n{'='*70}")
        for map_name, mask_name, mode in combos_to_run:
            if mask_name not in masks:
                print(f"\n[{ds}]  {map_name} × {mask_name}  — mask unavailable, skipping")
                continue
            fname = MAP_FILES[map_name]
            nii_path = ds_dir / fname
            roi_mask_img = masks[mask_name]
            print(f"\n[{ds}]  {map_name} × {mask_name}  ({mode})  ← {fname}")
            for hemi in hemis_to_run:
                #import pdb; pdb.set_trace()
                for view in views_to_run:
                    render_one(ds, map_name, mask_name, nii_path, amp_path,
                                mode, cells_df, subjects_dir, hemi, view,
                                roi_mask_img=roi_mask_img,
                                outline_mask_img=outline_mask_img)

    print(f"\nAll overlays under: {OUT_ROOT}")


if __name__ == '__main__':
    main()
