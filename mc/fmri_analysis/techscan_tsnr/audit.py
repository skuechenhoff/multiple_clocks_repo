"""
Stage 6: re-derive every number quoted in the write-up and check it.

Nothing here recomputes the analysis; it reads the result tables and the raw
headers and asserts that each claim in the report matches. Anything that does
not match prints FAIL, so a claim cannot quietly drift out of date as the
pipeline changes.

Run:  python -m mc.fmri_analysis.techscan_tsnr.audit
"""
import re
import numpy as np
import pandas as pd
import nibabel as nib

from .config import (OUT, DERIV, RUNS, FEAT, PRIMARY_ROIS, COMPARISONS,
                     ADOPTED_TILT_DEG, PHASE_ENCODE, PROJECT_MASKS)

PASS, FAIL = [], []


def check(label, got, expected, tol=0.51, unit=''):
    """Numeric claims are quoted rounded, so agreement to the last quoted
    digit is what counts."""
    ok = (abs(got - expected) <= tol) if expected is not None else False
    (PASS if ok else FAIL).append(
        f'{"OK  " if ok else "FAIL"}  {label:<58} report={expected}{unit}  '
        f'recomputed={got:.4g}{unit}')
    return ok


def check_eq(label, got, expected):
    ok = (got == expected)
    (PASS if ok else FAIL).append(
        f'{"OK  " if ok else "FAIL"}  {label:<58} report={expected}  got={got}')
    return ok


def pct(piv, roi, a, b):
    return 100 * (piv.loc[b, roi] / piv.loc[a, roi] - 1)


def main():
    common = pd.read_csv(OUT / 'tables' / 'tsnr_roi_common.csv')
    tsnr = pd.read_csv(OUT / 'tables' / 'tsnr_roi.csv')
    seq = pd.read_csv(OUT / 'tables' / 'sequences.csv')
    dist = pd.read_csv(OUT / 'tables' / 'distortion.csv')
    shift = pd.read_csv(OUT / 'tables' / 'shiftmaps.csv')
    cov = pd.read_csv(OUT / 'tables' / 'coverage_decomposed.csv')
    grad = pd.read_csv(OUT / 'tables' / 'tilt_gradient_projection.csv')

    def arms(comp):
        d = common[(common.comparison == comp) & (common.roi.isin(PRIMARY_ROIS))]
        return (d.groupby(['arm', 'roi']).tsnr_mean.mean().unstack()
                .reindex(COMPARISONS[comp]['order'])[PRIMARY_ROIS])

    # ---------------- step 1: field strength -------------------------
    p = arms('field_3T_vs_7T')
    for roi, claim in [('HC', 53), ('EC', 41), ('mPFC', -11), ('vOFC', -47)]:
        check(f'step1 3T->7T {roi} change (%)', pct(p, roi, '3T', '7T'), claim)

    # ---------------- step 2: voxel size -----------------------------
    v2 = arms('voxel_7T_s2')
    vol_ratio = 15.771 / 8.0
    check('step2 2.5/2.0 mm voxel-volume ratio', vol_ratio, 1.97, tol=0.005)
    ratios = (v2.loc['2.5 mm'] / v2.loc['2.0 mm'])
    check('step2 smallest tSNR ratio 2.5/2.0', ratios.min(), 1.24, tol=0.005)
    check('step2 largest  tSNR ratio 2.5/2.0', ratios.max(), 1.36, tol=0.005)
    v1 = arms('voxel_7T_s1')
    check('step2 HC 1.5mm / 2.0mm ratio',
          v1.loc['1.5 mm', 'HC'] / v1.loc['2.0 mm', 'HC'], 0.69, tol=0.005)
    check('step2 HC 1.2mm / 2.0mm ratio',
          v1.loc['1.2 mm*', 'HC'] / v1.loc['2.0 mm', 'HC'], 0.34, tol=0.005)
    hc12 = tsnr[(tsnr['sub'] == 'sub-02') & (tsnr.run == 'run6_T7_12mm_3iPat_3MB')
                & (tsnr.roi == 'HC')].tsnr_native_mean.iloc[0]
    check('step2 HC tSNR at 1.2 mm (all in-FOV voxels)', hc12, 14.6, tol=0.06)

    # ---------------- step 3: slice angle ----------------------------
    al = arms('angle_Leip_3T')
    check('step3 Leip EC, +30 vs transverse (%)',
          pct(al, 'EC', 'transverse', '+30 deg'), 19, tol=1.0)
    check('step3 Leip vOFC, -30 vs transverse (%)',
          pct(al, 'vOFC', 'transverse', '-30 deg'), 7, tol=1.0)
    # the adopted tilt, from every acquired 7T run and from the protocol file
    tilts = {}
    for sub, run, sess, _ in RUNS:
        a = nib.load(DERIV / sub / run / 'func' / FEAT / 'reg' /
                     'example_func.nii.gz').affine
        n = a[:3, 2] / np.linalg.norm(a[:3, 2])
        tilts[(sub, run)] = round(float(np.degrees(np.arctan2(n[1], n[2]))))
    seven = {v for k, v in tilts.items() if k[0] == 'sub-03' or 'T7' in k[1]}
    check_eq('step3 every 7T run acquired at one tilt', seven, {ADOPTED_TILT_DEG})
    proto = (PROJECT_MASKS.parent / 'techscan' / 'protocols' /
             '2023_017 Music Box - Protocol v1.md')
    ptxt = proto.read_text()
    i = [m.start() for m in
         re.finditer(r'cmrr_mbep2d_bold_3Trep_2mm_mb2p4(?!_)', ptxt)][-1]
    blk = ptxt[i:i + 9000]
    check_eq('step3 protocol file slab orientation',
             re.search(r'Orientation\s+(T > C[\d.]+)', blk).group(1), 'T > C30.0')
    check_eq('step3 protocol file phase-encode direction',
             re.search(r'Phase enc\. dir\.\s+(\S+ >> \S+)', blk).group(1).replace(' ', ''),
             PHASE_ENCODE.replace(' ', ''))
    # the field-based mechanism: which tilt minimises the PE gradient
    for roi in PRIMARY_ROIS:
        g = grad[grad.roi == roi].set_index('tilt_deg')
        check_eq(f'step3 tilt minimising PE gradient in {roi}',
                 int(g.g_phase_encode_Hz_per_mm.idxmin()), ADOPTED_TILT_DEG)

    # ---------------- step 4: multiband ------------------------------
    mb = arms('multiband_7T_s1')
    losses = [-pct(mb, r, 'MB3', 'MB6') for r in PRIMARY_ROIS]
    check('step4 smallest MB6 tSNR loss (%)', min(losses), 29, tol=1.0)
    check('step4 largest  MB6 tSNR loss (%)', max(losses), 44, tol=1.0)

    # ---------------- step 5: where to spend acceleration ------------
    sp = arms('split_matched_7T_s2')
    a0, a1 = COMPARISONS['split_matched_7T_s2']['order']
    for roi, claim in [('mPFC', 6), ('EC', -5), ('HC', -4), ('vOFC', 39)]:
        check(f'step5 matched split {roi} change (%)', pct(sp, roi, a0, a1), claim)
    s3 = seq[seq['sub'] == 'sub-03'].set_index('run')
    check_eq('step5 both arms have MB x iPAT = 8',
             {int(s3.loc[r, 'MB'] * s3.loc[r, 'iPAT'])
              for r in ('run3_mb4_fast_7T2', 'run5_ipat4_7T2')}, {8})
    check('step5 TE through-plane arm (ms)', s3.loc['run3_mb4_fast_7T2', 'TE_ms'], 20.6, tol=.05)
    check('step5 TE in-plane arm (ms)', s3.loc['run5_ipat4_7T2', 'TE_ms'], 12.8, tol=.05)
    d3 = dist[dist['sub'] == 'sub-03'].set_index('run')
    check('step5 readout through-plane arm (ms)', d3.loc['run3_mb4_fast_7T2', 'total_readout_ms'], 27.8, tol=.06)
    check('step5 readout in-plane arm (ms)', d3.loc['run5_ipat4_7T2', 'total_readout_ms'], 14.2, tol=.06)
    sh3 = shift[shift['sub'] == 'sub-03'].set_index('run')
    check('step5 displacement through-plane arm (mm)', sh3.loc['run3_mb4_fast_7T2', 'p99_shift_mm'], 5.6, tol=.06)
    check('step5 displacement in-plane arm (mm)', sh3.loc['run5_ipat4_7T2', 'p99_shift_mm'], 2.9, tol=.06)
    c3 = cov[(cov['sub'] == 'sub-03') & (cov.roi == 'vOFC')].set_index('run')
    check('step5 vOFC dropout through-plane arm (%)', 100 * c3.loc['run3_mb4_fast_7T2', 'dropout_within_slab'], 30, tol=1.0)
    check('step5 vOFC dropout in-plane arm (%)', 100 * c3.loc['run5_ipat4_7T2', 'dropout_within_slab'], 5, tol=1.0)
    check_eq('step5 total acceleration across the full sweep',
             [int(s3.loc[r, 'MB'] * s3.loc[r, 'iPAT']) for r in
              ('run1_3Trep_7T2', 'run3_mb4_fast_7T2', 'run4_mb3ipat3_7T2',
               'run5_ipat4_7T2')], [6, 8, 9, 8])

    # ---------------- step 6: against 3T -----------------------------
    f = arms('field_3T_vs_7T_final')
    b0, b1 = COMPARISONS['field_3T_vs_7T_final']['order']
    for roi, claim in [('mPFC', 16), ('EC', 18), ('HC', 24), ('vOFC', 6)]:
        check(f'step6 3T -> adopted {roi} change (%)', pct(f, roi, b0, b1), claim)

    # ---------------- coverage and provenance ------------------------
    # CORRECTED CLAIM. The earlier write-up said the slab contained every ROI
    # in every run. It does not: in 7T session 1 the slab was prescribed lower
    # and clipped the dorsal part of mPFC in five of six runs. It does hold for
    # every 3T run and for the whole of session 2, which is the session the
    # adopted protocol was piloted in -- which is the claim that matters.
    d = cov[cov.roi.isin(PRIMARY_ROIS)]
    ok_sessions = d[d.session != '7T-s1']
    check_eq('slab contains every primary ROI in 3T and 7T session 2',
             bool((ok_sessions.slab_coverage >= 0.999).all()), True)
    s1 = d[d.session == '7T-s1']
    clipped = s1[s1.slab_coverage < 0.999]
    check_eq('7T session 1: only mPFC is clipped',
             set(clipped.roi.unique()), {'mPFC'})
    check_eq('7T session 1: number of runs with mPFC clipped',
             int(clipped.run.nunique()), 5)
    check('7T session 1: worst mPFC slab coverage (%)',
          100 * clipped.slab_coverage.min(), 71, tol=1.0)
    adopted = d[(d['sub'] == 'sub-03') & (d.run == 'run5_ipat4_7T2')]
    check_eq('adopted protocol pilot: all four ROIs fully inside the slab',
             bool((adopted.slab_coverage >= 0.999).all()), True)
    s2 = seq[seq['sub'] == 'sub-02'].set_index('run')
    check('MB6 corroboration: run5 ms per excitation',
          s2.loc['run5_T7_superfast', 'ms_per_excitation'], 59.8, tol=.06)
    check('MB6 corroboration: run4 ms per excitation',
          s2.loc['run4_T7_15mm_2iPat_3MB', 'ms_per_excitation'], 59.1, tol=.06)

    # ---------------- the adopted protocol itself --------------------
    for field, patt, claim in [
            ('TR (ms)', r'TR\s+(\d+) ms', '1078'),
            ('TE (ms)', r'TE\s+([\d.]+) ms', '12.80'),
            ('flip angle (deg)', r'Flip angle\s+(\d+) deg', '54'),
            ('slices', r'Slices\s+(\d+)', '64'),
            ('multiband', r'Multi-band accel\. factor\s+(\d+)', '2'),
            ('iPAT', r'Accel\. factor PE\s+(\d+)', '4'),
            ('base resolution', r'Base resolution\s+(\d+)', '108'),
            ('FoV read (mm)', r'FoV read\s+(\d+) mm', '216'),
            ('slice thickness (mm)', r'Slice thickness\s+([\d.]+) mm', '2.00')]:
        m = re.search(patt, blk)
        check_eq(f'adopted protocol {field}', m.group(1) if m else None, claim)

    # ---------------- report ----------------------------------------
    for line in PASS + FAIL:
        print(line)
    print(f'\n{len(PASS)} checks passed, {len(FAIL)} failed')
    return len(FAIL) == 0


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
