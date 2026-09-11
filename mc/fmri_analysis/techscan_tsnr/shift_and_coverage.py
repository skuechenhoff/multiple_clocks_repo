"""
Stage 3b: measured geometric distortion, and coverage split into its two
separate causes.

GEOMETRIC DISTORTION. An EPI readout encodes the phase-encode direction
slowly, so wherever the B0 field is off-resonance -- above the frontal sinus,
beside the petrous bone -- signal is written to the wrong place along that
axis. The displacement is

    shift [voxels] = off-resonance [Hz] x total readout time [s]
    shift [mm]     = shift [voxels] x voxel size along phase encode

The off-resonance term is measured: it is the acquired B0 fieldmap, which
`fsl_prepare_fieldmap` stores in rad/s. So the only per-protocol quantity is
the readout duration, and the map of *where* distortion happens comes from
this participant's own field, not from a model.

(FEAT's `reg/unwarp/EF_UD_shift+mag.nii.gz` looks like the obvious source but
is not: the "+mag" files are display composites built for the HTML report,
packing the shift map together with a magnitude image, and their voxel values
are not the shift in voxels. Reading them directly gives shifts several times
too large.)

COVERAGE. Two quite different things were being conflated in the earlier
`roi_fov_coverage`, which used FEAT's brain mask:

  * slab coverage -- was this anatomy inside the prescribed imaging volume at
    all? Purely a function of how the slab was positioned and how many slices
    were run. Missing here means the data were never acquired.
  * signal -- was there usable BOLD signal there? Missing here means dropout.

FEAT's mask is the intersection of both, so it cannot distinguish them. Here
the geometric slab is measured separately, by warping a volume of ones from
each run's native space into MNI: the native array *is* the field of view, so
its footprint in MNI is exactly the acquired slab. Dropout is then whatever
the slab reached but the brain mask did not.

Writes tables/shiftmaps.csv and tables/coverage_decomposed.csv.

Run:  python -m mc.fmri_analysis.techscan_tsnr.shift_and_coverage
"""
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib

from .config import DERIV, OUT, RUNS, FEAT, ROIS, PRIMARY_ROIS


def sh(*cmd):
    subprocess.run([str(c) for c in cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def fieldmap_hz(sub, run):
    """The acquired B0 fieldmap in Hz, in this run's own EPI space.

    FEAT names this file after whichever image drove the registration, so the
    prefix varies between runs; glob rather than assume. Runs that used an
    initial-highres step write a file with the same suffix in a different
    space, so the shape is checked against the EPI."""
    feat = DERIV / sub / run / 'func' / FEAT
    ref = nib.load(feat / 'reg' / 'example_func.nii.gz')
    for f in sorted((feat / 'reg').glob('*fieldmaprads2epi.nii.gz')):
        img = nib.load(f)
        if img.shape[:3] == ref.shape[:3]:
            return np.asarray(img.dataobj) / (2 * np.pi), ref, run
    return None, ref, None


def fieldmap_hz_or_sibling(sub, run, session):
    """Runs that registered through an initial-highres image write their
    fieldmap in that image's space instead of the EPI's. Fall back to a
    sibling run from the same session, which is legitimate here because a
    single fieldmap acquisition was applied to every run in the session --
    the off-resonance field is a property of that head in that shim, and only
    the readout duration differs between runs."""
    hz, ref, src = fieldmap_hz(sub, run)
    if hz is not None:
        return hz, ref, src
    for s2, r2, sess2, _ in RUNS:
        if s2 == sub and sess2 == session and r2 != run:
            hz, ref, src = fieldmap_hz(s2, r2)
            if hz is not None:
                return hz, ref, r2
    return None, ref, None


def shift_map_mm(sub, run, readout_s, session):
    """Per-voxel phase-encode displacement in mm for this run's readout."""
    hz, ref, src = fieldmap_hz_or_sibling(sub, run, session)
    if hz is None:
        return None, None, None
    mask = nib.load(DERIV / sub / src / 'func' / FEAT / 'mask.nii.gz'
                    ).get_fdata() > 0
    # phase encode is j (A>>P) throughout, so the second voxel dimension
    mm_per_vox = float(ref.header.get_zooms()[1])
    return np.abs(hz) * readout_s * mm_per_vox, mask, src


def slab_in_mni(sub, run):
    """The acquired field of view, in MNI. The native array is the FOV, so a
    volume of ones warped forward marks exactly what was imaged."""
    feat = DERIV / sub / run / 'func' / FEAT
    ones = OUT / 'rois' / 'slab' / f'{sub}_{run}_ones.nii.gz'
    out = OUT / 'rois' / 'slab' / f'{sub}_{run}_slab_MNI.nii.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        ef = nib.load(feat / 'reg' / 'example_func.nii.gz')
        nib.save(nib.Nifti1Image(np.ones(ef.shape, np.uint8), ef.affine),
                 ones)
        sh('applywarp', '-i', ones, '-r', feat / 'reg' / 'standard.nii.gz',
           '-w', feat / 'reg' / 'example_func2standard_warp.nii.gz',
           '-o', out, '--interp=trilinear')
        sh('fslmaths', out, '-thr', 0.5, '-bin', out)
    return nib.load(out).get_fdata() > 0


def signal_in_mni(sub, run):
    """Where the run actually has usable signal: FEAT's brain mask, which is
    the acquired slab minus whatever dropped out, warped into MNI."""
    feat = DERIV / sub / run / 'func' / FEAT
    out = OUT / 'rois' / 'coverage' / f'{sub}_{run}_cov_MNI.nii.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        sh('applywarp', '-i', feat / 'mask.nii.gz',
           '-r', feat / 'reg' / 'standard.nii.gz',
           '-w', feat / 'reg' / 'example_func2standard_warp.nii.gz',
           '-o', out, '--interp=trilinear')
        sh('fslmaths', out, '-thr', 0.5, '-bin', out)
    return nib.load(out).get_fdata() > 0


def main():
    shift_rows, cov_rows = [], []
    # run folder names repeat across participants, so key on both
    dist = pd.read_csv(OUT / 'tables' / 'distortion.csv'
                       ).set_index(['sub', 'run'])

    for sub, run, session, label in RUNS:
        # ---- distortion, from this run's own fieldmap -------------------
        readout_s = float(dist.loc[(sub, run), 'total_readout_ms']) / 1e3
        mm, bmask, fsrc = shift_map_mm(sub, run, readout_s, session)
        if mm is None:
            print(f'{sub}/{run:<24} no EPI-space fieldmap, skipped')
            shift_rows.append(dict(sub=sub, run=run, session=session,
                                   label=label, readout_ms=readout_s * 1e3))
        else:
            v = mm[bmask]
            shift_rows.append(dict(
                sub=sub, run=run, session=session, label=label,
                readout_ms=round(readout_s * 1e3, 2),
                fieldmap_from=fsrc,
                median_shift_mm=round(float(np.median(v)), 2),
                p95_shift_mm=round(float(np.percentile(v, 95)), 2),
                p99_shift_mm=round(float(np.percentile(v, 99)), 2),
                max_shift_mm=round(float(v.max()), 2),
                frac_over_2mm=round(float((v > 2).mean()), 4),
            ))

        # ---- coverage, decomposed --------------------------------------
        slab = slab_in_mni(sub, run)
        signal = signal_in_mni(sub, run)
        for roi in ROIS:
            m = nib.load(OUT / 'rois' / 'MNI' / f'{roi}.nii.gz').get_fdata() > 0
            n = m.sum()
            in_slab = (m & slab).sum()
            with_signal = (m & signal).sum()
            cov_rows.append(dict(
                sub=sub, run=run, session=session, label=label, roi=roi,
                n_roi=int(n),
                slab_coverage=round(in_slab / n, 4),
                signal_coverage=round(with_signal / n, 4),
                # acquired but no usable signal: this is dropout, and it is
                # the part a better sequence can actually fix
                dropout_within_slab=round((in_slab - with_signal) / max(in_slab, 1), 4),
            ))

        r_ = shift_rows[-1]
        if 'median_shift_mm' in r_:
            print(f'{sub}/{run:<24} readout {r_["readout_ms"]:5.2f} ms  '
                  f'median shift {r_["median_shift_mm"]:5.2f} mm  '
                  f'p99 {r_["p99_shift_mm"]:5.2f} mm  '
                  f'{100*r_["frac_over_2mm"]:4.1f}% of brain >2 mm')

    pd.DataFrame(shift_rows).to_csv(OUT / 'tables' / 'shiftmaps.csv', index=False)
    pd.DataFrame(cov_rows).to_csv(OUT / 'tables' / 'coverage_decomposed.csv',
                                  index=False)
    print('\nwrote', OUT / 'tables' / 'shiftmaps.csv')
    print('wrote', OUT / 'tables' / 'coverage_decomposed.csv')

    cov = pd.DataFrame(cov_rows)
    print('\nCoverage for the acceleration sweep (sub-03), primary ROIs:')
    runs = ['run1_3Trep_7T2', 'run3_mb4_fast_7T2', 'run4_mb3ipat3_7T2',
            'run5_ipat4_7T2']
    for metric in ['slab_coverage', 'signal_coverage', 'dropout_within_slab']:
        print(f'  -- {metric}')
        print(cov[cov.roi.isin(PRIMARY_ROIS)]
              .pivot_table(index='run', columns='roi', values=metric)
              .loc[runs][PRIMARY_ROIS].round(2).to_string())


if __name__ == '__main__':
    main()
