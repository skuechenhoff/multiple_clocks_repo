"""
Stage 2b: score every comparison on the anatomy all its runs actually reached.

Runs were prescribed with slightly different slabs, so they cover different
fractions of an ROI -- mPFC especially, which sits at the top edge of a
restricted slab. Comparing raw ROI means then confounds two things: how good
the sequence is, and which part of the ROI the operator happened to include.
The parts that differ are exactly the dropout-prone edges, so the confound is
neither small nor random. It also runs both ways: a run that covered MORE of
mPFC is penalised for having included the worst voxels.

Per comparison group (config.COMPARISONS):

    1. warp every run's native brain mask forward into MNI
    2. intersect them -- the anatomy every run in the group reached
    3. intersect with each ROI
    4. warp that common ROI back into every run's native space
    5. recompute tSNR there

Every run in a group is then scored on identical anatomy, so any remaining
difference is the sequence. Coverage is still reported, separately, from
`roi_fov_coverage` in tsnr_roi.csv -- it matters on its own, it just must not
be silently averaged into a tSNR mean.

`roi_frac_common` says how much of the ROI survived the intersection. Where it
is low the comparison is only about that surviving fragment, and the figures
mark it.

Writes tables/tsnr_roi_common.csv (one row per group x run x ROI).

Run:  python -m mc.fmri_analysis.techscan_tsnr.common_coverage
"""
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib

from .config import (DERIV, OUT, RUNS, FEAT, ROIS, MATCH_SECONDS,
                     TSNR_USABLE, COMPARISONS)

LABEL = {(s, r): lab for s, r, _, lab in RUNS}
SESSION = {(s, r): sess for s, r, sess, _ in RUNS}


def sh(*cmd):
    subprocess.run([str(c) for c in cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def coverage_in_mni(sub, run):
    """This run's acquired-and-brain-extracted extent, pushed into MNI 2mm."""
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
    seq = pd.read_csv(OUT / 'tables' / 'sequences.csv')
    rows = []

    for gname, g in COMPARISONS.items():
        # ---- 1-2: anatomy every run in this group reached ----------------
        common = None
        for sub, run in g['runs']:
            cov = coverage_in_mni(sub, run)
            common = cov if common is None else (common & cov)
        ref = nib.load(OUT / 'rois' / 'MNI' / 'HC.nii.gz')

        # ---- 3: common part of each ROI ----------------------------------
        gdir = OUT / 'rois' / 'common' / gname
        gdir.mkdir(parents=True, exist_ok=True)
        kept = {}
        for roi in ROIS:
            m = nib.load(OUT / 'rois' / 'MNI' / f'{roi}.nii.gz'
                         ).get_fdata() > 0
            c = (m & common).astype(np.uint8)
            kept[roi] = float(c.sum() / m.sum()) if m.sum() else np.nan
            nib.save(nib.Nifti1Image(c, ref.affine, ref.header),
                     gdir / f'{roi}.nii.gz')

        print(f'\n=== {gname} ({len(g["runs"])} runs) ===')
        print('    ROI fraction reached by every run: ' +
              '  '.join(f'{k}={v:.2f}' for k, v in kept.items()))

        # ---- 4-5: back to native, recompute ------------------------------
        for sub, run, arm in g['arms']:
            s = seq[(seq['sub'] == sub) & (seq.run == run)].iloc[0]
            TR = float(s.TR_s)
            n_vols = int(np.floor(MATCH_SECONDS / TR))
            feat = DERIV / sub / run / 'func' / FEAT
            tsnr = nib.load(OUT / 'tsnr' /
                            f'{sub}_{run}_tsnr_native.nii.gz').get_fdata()

            for roi in ROIS:
                nat = (OUT / 'rois' / 'common_native' / gname /
                       f'{sub}_{run}_{roi}.nii.gz')
                nat.parent.mkdir(parents=True, exist_ok=True)
                if not nat.exists():
                    sh('applywarp', '-i', gdir / f'{roi}.nii.gz',
                       '-r', feat / 'reg' / 'example_func.nii.gz',
                       '-w', feat / 'reg' / 'standard2example_func_warp.nii.gz',
                       '-o', nat, '--interp=trilinear')
                    sh('fslmaths', nat, '-thr', 0.5, '-bin', nat)
                m = nib.load(nat).get_fdata() > 0
                v = tsnr[m & (tsnr > 0)]
                rows.append(dict(
                    comparison=gname, sub=sub, run=run, arm=arm,
                    session=SESSION[(sub, run)], label=LABEL[(sub, run)],
                    roi=roi, TR_s=TR, TE_ms=s.TE_ms, MB=s.MB, iPAT=s.iPAT,
                    vox_mm3=s.vox_mm3, field_T=s.field_T,
                    n_vols_used=n_vols,
                    roi_frac_common=round(kept[roi], 4),
                    n_voxels=int(v.size),
                    tsnr_mean=round(float(v.mean()), 2) if v.size else np.nan,
                    tsnr_median=round(float(np.median(v)), 2) if v.size else np.nan,
                    tsnr_sem=round(float(v.std(ddof=1) / np.sqrt(v.size)), 3) if v.size > 1 else np.nan,
                    efficiency=round(float(v.mean() / np.sqrt(TR)), 2) if v.size else np.nan,
                    usable_frac=round(float((v > TSNR_USABLE).mean()), 4) if v.size else np.nan,
                ))
            print(f'    {sub}/{run:<24} N={n_vols:>3}  done')

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'tables' / 'tsnr_roi_common.csv', index=False)
    print('\nwrote', OUT / 'tables' / 'tsnr_roi_common.csv')
    return df


if __name__ == '__main__':
    main()
