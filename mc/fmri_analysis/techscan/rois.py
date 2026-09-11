"""
Where tSNR is measured: ROI masks in MNI, in each run's native EPI space, and
restricted to the anatomy a comparison's arms all reached.

Three things this fixes relative to the 2023 masks:

  * the frontal ROI. Harvard-Oxford "Frontal Medial Cortex" spans both the
    ventral strip that loses its signal to the frontal sinus and the dorsal
    cortex that keeps it, so a mean over it averages two opposite regimes.
    Splitting into mOFC (MNI z -30..-2) and mPFC (z -10..+48) separates them.
  * threshold. `-thr 10` gives an 11.1 cm3 hippocampus against a true ~3.5-4
    per hemisphere, bleeding into ventricle and white matter.
  * resampling. `-subsamp2 -bin` on a 1 mm binary mask dilates it by about half
    again. Masks are resampled through their affines here, never by index.

tSNR is computed in native EPI space, because resampling a 1.2 mm acquisition
onto a 2 mm grid averages ~4.6 native voxels together and inflates its tSNR for
reasons that have nothing to do with the sequence. The masks therefore travel
backwards, MNI -> native, through the inverse of FEAT's own registration.
"""
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

from .config import (ATLAS_FILES, ATLAS_THRESHOLD, COMPARISONS,
                     DERIVATIVES_DIR, FEAT_DIR_NAME, GARVERT_MTL_FILE,
                     MNI_2MM_BRAIN, OUTPUT_DIR, PRIMARY_ROIS,
                     PROJECT_MASK_DIR, ROI_DEFINITIONS, RUNS)

MNI_REFERENCE = nib.load(MNI_2MM_BRAIN)


def fsl(*command):
    subprocess.run([str(part) for part in command], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def resample_to_mni(image, order=1):
    """Resample any image onto the MNI152 2 mm grid through its affine."""
    return np.asarray(resample_from_to(
        image, (MNI_REFERENCE.shape, MNI_REFERENCE.affine),
        order=order, cval=0).dataobj)


def atlas_probability(atlas_key, volume_indices):
    """Summed probability map across one or more atlas volumes (e.g. L + R)."""
    atlas = nib.load(ATLAS_FILES[atlas_key])
    assert np.allclose(atlas.affine, MNI_REFERENCE.affine), \
        f'{atlas_key} is not on the MNI152 2 mm grid'
    probability = np.zeros(atlas.shape[:3], dtype=np.float32)
    for index in np.atleast_1d(volume_indices):
        probability += np.asarray(atlas.dataobj[..., int(index)],
                                  dtype=np.float32)
    return probability


def build_mni_masks(verbose=True):
    """Write every ROI mask onto the MNI152 2 mm grid."""
    garvert_mtl = resample_to_mni(
        nib.load(PROJECT_MASK_DIR / GARVERT_MTL_FILE)) > 0.5
    hippocampus_prob = atlas_probability('HO_subcortical', (8, 18))
    entorhinal_prob = atlas_probability('Juelich', (18, 19))

    mni_dir = OUTPUT_DIR / 'rois' / 'MNI'
    mni_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for roi, (kind, spec, display, priority) in ROI_DEFINITIONS.items():
        if kind == 'project_file':
            mask = resample_to_mni(nib.load(PROJECT_MASK_DIR / spec)) > 0.5
            source = spec
        elif kind == 'atlas':
            atlas_key, indices = spec
            mask = atlas_probability(atlas_key, indices) > ATLAS_THRESHOLD
            source = f'{atlas_key} volumes {indices} > {ATLAS_THRESHOLD}%'
        elif kind == 'garvert':
            # split the project's MTL boundary by whichever atlas structure
            # claims each voxel more strongly, so the EC/HC division follows the
            # atlas while the outer boundary stays Garvert's
            if spec == 'whole':
                mask = garvert_mtl
            elif spec == 'HC':
                mask = garvert_mtl & (hippocampus_prob > entorhinal_prob) \
                       & (hippocampus_prob > 0)
            else:
                mask = garvert_mtl & (entorhinal_prob >= hippocampus_prob) \
                       & (entorhinal_prob > 0)
            source = f'{GARVERT_MTL_FILE} ({spec})'
        else:
            raise ValueError(f'unknown ROI source kind: {kind}')

        nib.save(nib.Nifti1Image(mask.astype(np.uint8), MNI_REFERENCE.affine,
                                 MNI_REFERENCE.header),
                 mni_dir / f'{roi}.nii.gz')

        voxel_coords = np.array(np.where(mask)).T
        mni_coords = nib.affines.apply_affine(MNI_REFERENCE.affine, voxel_coords)
        rows.append(dict(
            roi=roi, display=display, priority=priority, source=source,
            primary=roi in PRIMARY_ROIS, n_voxels=int(mask.sum()),
            volume_cm3=round(mask.sum() * 8 / 1000, 2),
            x_min=int(mni_coords[:, 0].min()), x_max=int(mni_coords[:, 0].max()),
            y_min=int(mni_coords[:, 1].min()), y_max=int(mni_coords[:, 1].max()),
            z_min=int(mni_coords[:, 2].min()), z_max=int(mni_coords[:, 2].max())))

    volumes = pd.DataFrame(rows).sort_values('priority')
    (OUTPUT_DIR / 'tables').mkdir(parents=True, exist_ok=True)
    volumes.to_csv(OUTPUT_DIR / 'tables' / 'roi_volumes.csv', index=False)
    if verbose:
        print(volumes[['roi', 'display', 'priority', 'primary', 'volume_cm3',
                       'z_min', 'z_max', 'source']].to_string(index=False))
    return volumes


def warp_masks_to_native(subject, run):
    """Push every MNI mask into this run's EPI space, through the inverse of
    FEAT's own registration."""
    reg_dir = DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME / 'reg'
    example_func = reg_dir / 'example_func.nii.gz'
    inverse_warp = reg_dir / 'standard2example_func_warp.nii.gz'
    if not inverse_warp.exists():
        fsl('invwarp', '-w', reg_dir / 'example_func2standard_warp.nii.gz',
            '-o', inverse_warp, '-r', example_func)

    native_dir = OUTPUT_DIR / 'rois' / 'native'
    native_dir.mkdir(parents=True, exist_ok=True)
    for roi in ROI_DEFINITIONS:
        out = native_dir / f'{subject}_{run}_{roi}.nii.gz'
        if out.exists():
            continue
        fsl('applywarp', '-i', OUTPUT_DIR / 'rois' / 'MNI' / f'{roi}.nii.gz',
            '-r', example_func, '-w', inverse_warp, '-o', out,
            '--interp=trilinear')
        fsl('fslmaths', out, '-thr', 0.5, '-bin', out)


def acquired_slab_in_mni(subject, run):
    """The acquired field of view in MNI. The native array *is* the FOV, so a
    volume of ones warped forward marks exactly what was imaged. This is purely
    geometric -- it says nothing about whether there was signal there."""
    feat_dir = DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME
    out = OUTPUT_DIR / 'rois' / 'slab' / f'{subject}_{run}_slab_MNI.nii.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        example_func = nib.load(feat_dir / 'reg' / 'example_func.nii.gz')
        ones = out.parent / f'{subject}_{run}_ones.nii.gz'
        nib.save(nib.Nifti1Image(np.ones(example_func.shape, np.uint8),
                                 example_func.affine), ones)
        fsl('applywarp', '-i', ones, '-r', feat_dir / 'reg' / 'standard.nii.gz',
            '-w', feat_dir / 'reg' / 'example_func2standard_warp.nii.gz',
            '-o', out, '--interp=trilinear')
        fsl('fslmaths', out, '-thr', 0.5, '-bin', out)
    return nib.load(out).get_fdata() > 0


def signal_extent_in_mni(subject, run):
    """Where the run actually has usable signal: FEAT's brain mask, warped into
    MNI. That mask is BET on the mean functional at 10% of the robust intensity
    range, so it is the acquired slab minus whatever dropped out."""
    feat_dir = DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME
    out = OUTPUT_DIR / 'rois' / 'signal' / f'{subject}_{run}_signal_MNI.nii.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        fsl('applywarp', '-i', feat_dir / 'mask.nii.gz',
            '-r', feat_dir / 'reg' / 'standard.nii.gz',
            '-w', feat_dir / 'reg' / 'example_func2standard_warp.nii.gz',
            '-o', out, '--interp=trilinear')
        fsl('fslmaths', out, '-thr', 0.5, '-bin', out)
    return nib.load(out).get_fdata() > 0


def build_common_coverage_masks(verbose=True):
    """Per comparison, the part of each ROI that every arm reached.

    Arms were prescribed with slightly different slabs, so they cover different
    fractions of an ROI. Comparing raw means then confounds how good the
    sequence is with which part of the ROI the operator included -- and the
    parts that differ are the dropout-prone edges, so the confound is neither
    small nor random. It also cuts both ways: an arm that covered *more* of a
    region is penalised for having included its worst voxels.

    Returns the fraction of each ROI surviving the intersection, per comparison.
    """
    kept_fractions = {}
    for name, comparison in COMPARISONS.items():
        common_extent = None
        for subject, run in comparison['runs']:
            extent = signal_extent_in_mni(subject, run)
            common_extent = extent if common_extent is None \
                else (common_extent & extent)

        group_dir = OUTPUT_DIR / 'rois' / 'common' / name
        group_dir.mkdir(parents=True, exist_ok=True)
        kept = {}
        for roi in ROI_DEFINITIONS:
            full = nib.load(OUTPUT_DIR / 'rois' / 'MNI' / f'{roi}.nii.gz'
                            ).get_fdata() > 0
            common = (full & common_extent).astype(np.uint8)
            kept[roi] = float(common.sum() / full.sum()) if full.sum() else np.nan
            nib.save(nib.Nifti1Image(common, MNI_REFERENCE.affine,
                                     MNI_REFERENCE.header),
                     group_dir / f'{roi}.nii.gz')
        kept_fractions[name] = kept

        for subject, run in comparison['runs']:
            feat_dir = DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME
            for roi in ROI_DEFINITIONS:
                out = (OUTPUT_DIR / 'rois' / 'common_native' / name /
                       f'{subject}_{run}_{roi}.nii.gz')
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.exists():
                    continue
                fsl('applywarp', '-i', group_dir / f'{roi}.nii.gz',
                    '-r', feat_dir / 'reg' / 'example_func.nii.gz',
                    '-w', feat_dir / 'reg' / 'standard2example_func_warp.nii.gz',
                    '-o', out, '--interp=trilinear')
                fsl('fslmaths', out, '-thr', 0.5, '-bin', out)

        if verbose:
            summary = '  '.join(f'{roi}={kept[roi]:.2f}' for roi in PRIMARY_ROIS)
            print(f'  {name:<28} ROI fraction reached by every arm: {summary}')
    return kept_fractions


def run(verbose=True):
    build_mni_masks(verbose=verbose)
    if verbose:
        print()
    for subject, run_name, _, _ in RUNS:
        warp_masks_to_native(subject, run_name)
    if verbose:
        print(f'  warped masks into {len(RUNS)} native EPI spaces')
    return build_common_coverage_masks(verbose=verbose)
