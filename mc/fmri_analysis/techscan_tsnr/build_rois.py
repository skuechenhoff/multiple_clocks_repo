"""
Stage 1: build ROI masks and put them into every run's native EPI space.

Masks come from two places: the project's own mask library (mPFC, vOFC, the
Garvert MTL boundary) and the FSL atlases (entorhinal, hippocampus). All of
them are resampled onto the MNI152_T1_2mm reference grid **through their
affines**, never by array index -- one of the project masks is stored with
the opposite x-orientation to the FSL standard, so index-wise masking would
silently mirror it left-right.

Three things this fixes relative to the 2023 masks:

  * the frontal ROI. Harvard-Oxford "Frontal Medial Cortex" spans both the
    ventral strip that loses most of its signal to the frontal sinus and the
    dorsal cortex that keeps it, so its mean averages two opposite regimes.
    Splitting into vOFC (MNI z -30..-2) and mPFC (z -10..+48) separates them.
  * threshold. `-thr 10` gives an 11.1 cm3 hippocampus against a true ~3.5-4
    per hemisphere; it bleeds into ventricle and white matter.
  * resampling. `-subsamp2 -bin` on a 1 mm binary mask dilates it by roughly
    half again.

tSNR itself is computed in each run's NATIVE space, because resampling a
1.2 mm acquisition onto a 2 mm grid averages ~4.6 native voxels together and
inflates its tSNR for reasons that have nothing to do with the sequence. The
masks are therefore warped backwards, MNI -> native EPI, through the inverse
of FEAT's own registration.

Writes:
    rois/MNI/<roi>.nii.gz                 masks on the MNI152_T1_2mm grid
    rois/native/<sub>_<run>_<roi>.nii.gz  same masks in each run's EPI space
    qc/roi_on_meanEPI_<sub>_<run>.jpeg    ROI outlines on the mean functional
    tables/roi_volumes.csv                volume and MNI extent per ROI

Run:  python -m mc.fmri_analysis.techscan_tsnr.build_rois
"""
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import (DERIV, OUT, MNI2MM, ATLAS, ROIS, ROI_THR, RUNS, FEAT,
                     PROJECT_MASKS, GARVERT_MASK, PRIMARY_ROIS)

REF = nib.load(MNI2MM)


def sh(*cmd):
    subprocess.run([str(c) for c in cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def onto_ref(img, order=1):
    """Resample any image onto the MNI152_T1_2mm grid via its affine."""
    return np.asarray(
        resample_from_to(img, (REF.shape, REF.affine), order=order,
                         cval=0).dataobj)


def atlas_prob(key, indices):
    """Summed probability map for one or more atlas volumes (L + R)."""
    img = nib.load(ATLAS[key])
    p = np.zeros(img.shape[:3], dtype=np.float32)
    for i in np.atleast_1d(indices):
        p += np.asarray(img.dataobj[..., int(i)], dtype=np.float32)
    assert np.allclose(img.affine, REF.affine), f'{key} off the reference grid'
    return p


def build_mni_masks():
    garvert = onto_ref(nib.load(PROJECT_MASKS / GARVERT_MASK)) > 0.5
    hc_p = atlas_prob('HO_sub', (8, 18))
    ec_p = atlas_prob('Juelich', (18, 19))

    mni_dir = OUT / 'rois' / 'MNI'
    mni_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for roi, (kind, spec, disp, rank) in ROIS.items():
        if kind == 'file':
            mask = onto_ref(nib.load(PROJECT_MASKS / spec)) > 0.5
            src = spec
        elif kind == 'atlas':
            key, idx = spec
            mask = atlas_prob(key, idx) > ROI_THR
            src = f'{key} vol {idx} > {ROI_THR}%'
        elif kind == 'garvert':
            # split the project's MTL boundary by whichever atlas structure
            # claims each voxel more strongly, so the EC/HC division follows
            # the atlas while the outer boundary stays Garvert's
            if spec == 'whole':
                mask = garvert
            elif spec == 'HC':
                mask = garvert & (hc_p > ec_p) & (hc_p > 0)
            else:
                mask = garvert & (ec_p >= hc_p) & (ec_p > 0)
            src = f'{GARVERT_MASK} ({spec})'
        else:
            raise ValueError(kind)

        nib.save(nib.Nifti1Image(mask.astype(np.uint8), REF.affine, REF.header),
                 mni_dir / f'{roi}.nii.gz')

        ijk = np.array(np.where(mask)).T
        xyz = nib.affines.apply_affine(REF.affine, ijk)
        rows.append(dict(
            roi=roi, display=disp, priority=rank, source=src,
            primary=roi in PRIMARY_ROIS,
            n_voxels=int(mask.sum()), volume_cm3=round(mask.sum() * 8 / 1000, 2),
            x_min=int(xyz[:, 0].min()), x_max=int(xyz[:, 0].max()),
            y_min=int(xyz[:, 1].min()), y_max=int(xyz[:, 1].max()),
            z_min=int(xyz[:, 2].min()), z_max=int(xyz[:, 2].max()),
        ))

    df = pd.DataFrame(rows).sort_values('priority')
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'tables' / 'roi_volumes.csv', index=False)
    return df


def warp_to_native(sub, run):
    feat = DERIV / sub / run / 'func' / FEAT
    reg = feat / 'reg'
    ref = reg / 'example_func.nii.gz'
    inv = reg / 'standard2example_func_warp.nii.gz'
    if not inv.exists():
        sh('invwarp', '-w', reg / 'example_func2standard_warp.nii.gz',
           '-o', inv, '-r', ref)

    nat_dir = OUT / 'rois' / 'native'
    nat_dir.mkdir(parents=True, exist_ok=True)
    for roi in ROIS:
        out = nat_dir / f'{sub}_{run}_{roi}.nii.gz'
        if out.exists():
            continue
        sh('applywarp', '-i', OUT / 'rois' / 'MNI' / f'{roi}.nii.gz',
           '-r', ref, '-w', inv, '-o', out, '--interp=trilinear')
        sh('fslmaths', out, '-thr', 0.5, '-bin', out)


def qc_overlay(sub, run):
    """ROI outlines on the mean functional -- the project's own rule is to
    check ROIs against the EPI, never against the T1."""
    from . import plotstyle as ps
    feat = DERIV / sub / run / 'func' / FEAT
    epi = nib.load(feat / 'mean_func.nii.gz').get_fdata()
    masks = {r: nib.load(OUT / 'rois' / 'native' / f'{sub}_{run}_{r}.nii.gz'
                         ).get_fdata() > 0 for r in PRIMARY_ROIS}

    fig, ax = plt.subplots(1, len(PRIMARY_ROIS), figsize=(11, 3.2))
    for a, roi in zip(ax, PRIMARY_ROIS):
        m = masks[roi]
        z = int(np.argmax(m.sum(axis=(0, 1)))) if m.sum() else epi.shape[2] // 2
        sl = np.rot90(epi[:, :, z])
        a.imshow(sl, cmap='gray', vmax=np.percentile(sl, 99.5))
        for r2, m2 in masks.items():
            if m2[:, :, z].sum():
                a.contour(np.rot90(m2[:, :, z]), levels=[0.5],
                          colors=ps.ROI_COLOUR[r2], linewidths=0.9)
        cover = 100 * (m & (epi > 0)).sum() / m.sum() if m.sum() else 0
        a.set_title(f'{roi}  z={z}   {cover:.0f}% in FOV', fontsize=8)
        a.axis('off')
    fig.suptitle(f'{sub} / {run} -- ROIs on mean EPI, native space', fontsize=9)
    fig.tight_layout()
    (OUT / 'qc').mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'qc' / f'roi_on_meanEPI_{sub}_{run}.jpeg', dpi=110)
    plt.close(fig)


def main():
    df = build_mni_masks()
    pd.set_option('display.width', 200)
    print(df[['roi', 'display', 'priority', 'primary', 'volume_cm3',
              'z_min', 'z_max', 'source']].to_string(index=False))
    print()
    for sub, run, _, _ in RUNS:
        warp_to_native(sub, run)
        qc_overlay(sub, run)
        print(f'  {sub}/{run:<24} warped + QC')
    print('\nwrote', OUT / 'rois', 'and', OUT / 'qc')


if __name__ == '__main__':
    main()
