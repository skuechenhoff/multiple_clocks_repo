"""
Stage 2: duration-matched tSNR, in native EPI space and in MNI space.

tSNR = mean / SD over time, voxelwise (Triantafyllou et al. 2005), on FEAT's
preprocessed 4D: motion-corrected, slice-time corrected, fieldmap-unwarped,
NOT spatially smoothed, 100 s highpass. FEAT's grand-mean scaling is a single
constant per run and cancels in mean/SD.

Two choices worth defending:

  * NATIVE space is primary. Resampling to MNI 2 mm averages neighbouring
    native voxels together -- ~4.6 of them for the 1.2 mm acquisition -- which
    inflates tSNR for the high-resolution sequences by an amount set by the
    interpolation, not the sequence. MNI-space tSNR is computed too, because
    the map figures need a common space and because it reproduces the 2023
    numbers, but the voxel-size comparison is only honest in native space.

  * DURATION matching, not volume matching. Scanner time is the binding
    constraint on the real experiment, so the question is what a fixed-length
    task sample buys. Every run is truncated to the same MATCH_SECONDS; N per
    run therefore varies with TR, and the efficiency metric below is what
    actually answers the question.

    tsnr_efficiency = tSNR / sqrt(TR)

    proportional to tSNR * sqrt(N/T): the detection power a fixed-duration
    scan delivers. A short TR partly repays its own per-volume tSNR penalty by
    collecting more samples in the same minutes.

Writes tsnr/*.nii.gz (maps) and tables/tsnr_roi.csv (one row per run x ROI).

Run:  python -m mc.fmri_analysis.techscan_tsnr.compute_tsnr
"""
import numpy as np
import pandas as pd
import nibabel as nib

from .config import (DERIV, OUT, RUNS, FEAT, ROIS, MATCH_SECONDS,
                     TSNR_USABLE, SEED)

np.random.seed(SEED)


def tsnr_from_4d(path, n_vols):
    """Voxelwise mean/SD over the first n_vols volumes."""
    img = nib.load(path)
    data = np.asarray(img.dataobj[..., :n_vols], dtype=np.float32)
    mean = data.mean(axis=-1)
    sd = data.std(axis=-1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tsnr = np.where(sd > 0, mean / sd, 0.0)
    return mean, sd, np.nan_to_num(tsnr, nan=0.0, posinf=0.0), img.affine


def mean_fd(feat, n_vols):
    """Mean relative RMS displacement over the matched window, from MCFLIRT."""
    p = feat / 'mc' / 'prefiltered_func_data_mcf_rel.rms'
    if not p.exists():
        return np.nan
    rms = np.loadtxt(p)
    return float(np.mean(rms[:max(n_vols - 1, 1)]))


def main():
    seq = pd.read_csv(OUT / 'tables' / 'sequences.csv')
    (OUT / 'tsnr').mkdir(parents=True, exist_ok=True)
    rows = []

    for sub, run, session, label in RUNS:
        s = seq[(seq['sub'] == sub) & (seq.run == run)].iloc[0]
        TR = float(s.TR_s)
        n_vols = int(np.floor(MATCH_SECONDS / TR))
        assert n_vols <= s.n_vols, f'{sub}/{run}: needs {n_vols}, has {s.n_vols}'

        feat = DERIV / sub / run / 'func' / FEAT
        brain = nib.load(feat / 'mask.nii.gz').get_fdata() > 0

        # ---------- native space (primary) ------------------------------
        mean_n, sd_n, tsnr_n, aff_n = tsnr_from_4d(
            feat / 'filtered_func_data.nii.gz', n_vols)
        nib.save(nib.Nifti1Image(tsnr_n, aff_n),
                 OUT / 'tsnr' / f'{sub}_{run}_tsnr_native.nii.gz')

        # ---------- MNI space (for maps + comparison with 2023) ---------
        _, _, tsnr_s, aff_s = tsnr_from_4d(
            feat / 'filtered_func_data_standard.nii.gz', n_vols)
        nib.save(nib.Nifti1Image(tsnr_s, aff_s),
                 OUT / 'tsnr' / f'{sub}_{run}_tsnr_MNI.nii.gz')

        fd = mean_fd(feat, n_vols)
        wb = tsnr_n[brain]

        for roi in ROIS:
            m_nat = nib.load(OUT / 'rois' / 'native' /
                             f'{sub}_{run}_{roi}.nii.gz').get_fdata() > 0
            m_mni = nib.load(OUT / 'rois' / 'MNI' /
                             f'{roi}.nii.gz').get_fdata() > 0

            # what fraction of the ROI the acquired slab actually reached.
            # anything outside is a coverage failure, not a tSNR failure, and
            # the two must not be averaged together.
            inside = m_nat & brain
            fov_cov = inside.sum() / m_nat.sum() if m_nat.sum() else np.nan

            v_nat = tsnr_n[inside]
            v_mni = tsnr_s[m_mni & (tsnr_s > 0)]

            rows.append(dict(
                sub=sub, run=run, session=session, label=label, roi=roi,
                TR_s=TR, TE_ms=s.TE_ms, MB=s.MB, iPAT=s.iPAT,
                vox_mm3=s.vox_mm3, field_T=s.field_T, n_vols_used=n_vols,
                window_s=round(n_vols * TR, 1), mean_fd_mm=round(fd, 4),
                n_roi_voxels=int(m_nat.sum()),
                n_roi_in_fov=int(inside.sum()),
                roi_fov_coverage=round(float(fov_cov), 4),
                tsnr_native_mean=round(float(v_nat.mean()), 2) if v_nat.size else np.nan,
                tsnr_native_median=round(float(np.median(v_nat)), 2) if v_nat.size else np.nan,
                tsnr_native_sd=round(float(v_nat.std(ddof=1)), 2) if v_nat.size > 1 else np.nan,
                tsnr_efficiency=round(float(v_nat.mean() / np.sqrt(TR)), 2) if v_nat.size else np.nan,
                usable_frac=round(float((v_nat > TSNR_USABLE).mean()), 4) if v_nat.size else np.nan,
                tsnr_MNI_mean=round(float(v_mni.mean()), 2) if v_mni.size else np.nan,
                tsnr_wholebrain_mean=round(float(wb.mean()), 2),
            ))

        print(f'{sub}/{run:<24} N={n_vols:>3} ({n_vols*TR:6.1f}s)  '
              f'WB tSNR={wb.mean():6.2f}  FD={fd:.3f}mm')

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'tables' / 'tsnr_roi.csv', index=False)
    print('\nwrote', OUT / 'tables' / 'tsnr_roi.csv')
    return df


if __name__ == '__main__':
    main()
