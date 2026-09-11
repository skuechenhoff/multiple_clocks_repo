"""
Runner for the 7T EPI protocol-selection re-analysis.

    conda activate env_multiple_clocks
    python -m mc.fmri_analysis.techscan.run_analysis

Stages, in order:

    1 inventory   sequence parameters from headers, and filename conflicts
    2 rois        ROI masks in MNI, in native EPI space, and common-coverage
    3 tsnr        duration-matched tSNR maps and per-ROI tables
    4 distortion  readout duration, measured displacement, coverage split
    5 tilt        slab-tilt field projection and predicted penalties
    6 figures     the vector panels
    7 audit       re-derive every number quoted in the write-up and check it

Masks and warps are cached, so a re-run after the first is fast. Pass --stage
to run one stage, --fresh to rebuild the cached masks and warps.

Nothing is written into the repository. All output goes to config.OUTPUT_DIR.
"""
import argparse
import re
import shutil
import sys

import nibabel as nib
import numpy as np
import pandas as pd

from . import figures, inventory, metrics, rois
from .config import (ADOPTED_TILT_DEG, COMPARISONS, FINAL_PROTOCOL_FILE,
                     FINAL_PROTOCOL_NAME, OUTPUT_DIR, PHASE_ENCODE_DIRECTION,
                     PRIMARY_ROIS)

STAGES = ['inventory', 'rois', 'tsnr', 'distortion', 'tilt', 'figures', 'audit']


def banner(text):
    print(f'\n{"=" * 74}\n{text}\n{"=" * 74}')


# ------------------------------------------------------------------ audit
class Audit:
    """Re-derives each quoted number from the result tables and the raw headers
    and asserts it matches the write-up, so a claim cannot quietly drift out of
    date as the pipeline changes."""

    def __init__(self):
        self.passed, self.failed = [], []

    def close_to(self, label, recomputed, reported, tolerance=0.51):
        ok = abs(recomputed - reported) <= tolerance
        self._record(ok, label, reported, f'{recomputed:.4g}')

    def equals(self, label, recomputed, reported):
        self._record(recomputed == reported, label, reported, recomputed)

    def _record(self, ok, label, reported, recomputed):
        line = (f'{"OK  " if ok else "FAIL"}  {label:<58} '
                f'report={reported}  recomputed={recomputed}')
        (self.passed if ok else self.failed).append(line)

    def report(self):
        for line in self.passed + self.failed:
            print(line)
        print(f'\n{len(self.passed)} checks passed, {len(self.failed)} failed')
        return not self.failed


def percent_change(table, roi, from_arm, to_arm):
    return 100 * (table.loc[to_arm, roi] / table.loc[from_arm, roi] - 1)


def run_audit(sequences, common, tsnr_table, displacement, coverage, projection):
    audit = Audit()
    arms = lambda name: metrics.tsnr_by_arm(common, name)

    # step 1: field strength
    field = arms('field_strength')
    for roi, reported in [('HC', 53), ('EC', 41), ('mPFC', -11), ('mOFC', -47)]:
        audit.close_to(f'step1 3T->7T {roi} change (%)',
                       percent_change(field, roi, '3T', '7T'), reported)

    # step 2: voxel size
    voxel = arms('voxel_size')
    audit.close_to('step2 2.5/2.0 mm voxel-volume ratio', 15.771 / 8.0, 1.97, 0.005)
    ratios = voxel.loc['2.5 mm'] / voxel.loc['2.0 mm']
    audit.close_to('step2 smallest tSNR ratio 2.5/2.0', ratios.min(), 1.24, 0.005)
    audit.close_to('step2 largest tSNR ratio 2.5/2.0', ratios.max(), 1.36, 0.005)
    fine = arms('voxel_size_fine')
    audit.close_to('step2 HC 1.5/2.0 mm ratio',
                   fine.loc['1.5 mm', 'HC'] / fine.loc['2.0 mm', 'HC'], 0.69, 0.005)
    audit.close_to('step2 HC 1.2/2.0 mm ratio',
                   fine.loc['1.2 mm*', 'HC'] / fine.loc['2.0 mm', 'HC'], 0.34, 0.005)
    hc_at_1p2 = tsnr_table[(tsnr_table.subject == 'sub-02')
                           & (tsnr_table.run == 'run6_T7_12mm_3iPat_3MB')
                           & (tsnr_table.roi == 'HC')].tsnr_native_mean.iloc[0]
    audit.close_to('step2 HC tSNR at 1.2 mm', hc_at_1p2, 14.6, 0.06)

    # step 3: slice angle
    unaccelerated = arms('slice_angle_unaccelerated')
    audit.close_to('step3 EC +30 vs transverse, long readout (%)',
                   percent_change(unaccelerated, 'EC', 'transverse', '+30 deg'), 19, 1.0)
    audit.close_to('step3 mOFC -30 vs transverse, long readout (%)',
                   percent_change(unaccelerated, 'mOFC', 'transverse', '-30 deg'), 7, 1.0)
    seven_tesla_tilts = set(sequences[sequences.field_strength_t == 7].slab_tilt_deg)
    audit.equals('step3 every 7T run acquired at one tilt',
                 seven_tesla_tilts, {ADOPTED_TILT_DEG})
    protocol_text = FINAL_PROTOCOL_FILE.read_text()
    start = [m.start() for m in re.finditer(
        FINAL_PROTOCOL_NAME + r'(?!_)', protocol_text)][-1]
    protocol_block = protocol_text[start:start + 9000]
    audit.equals('step3 protocol file slab orientation',
                 re.search(r'Orientation\s+(T > C[\d.]+)', protocol_block).group(1),
                 'T > C30.0')
    audit.equals('step3 protocol file phase-encode direction',
                 re.search(r'Phase enc\. dir\.\s+(\S+ >> \S+)',
                           protocol_block).group(1).replace(' ', ''),
                 PHASE_ENCODE_DIRECTION.replace(' ', ''))
    for roi in PRIMARY_ROIS:
        by_tilt = projection[projection.roi == roi].set_index('tilt_deg')
        audit.equals(f'step3 tilt minimising PE gradient in {roi}',
                     int(by_tilt.g_phase_encode_hz_per_mm.idxmin()),
                     ADOPTED_TILT_DEG)

    # step 4: multiband
    multiband = arms('multiband')
    losses = [-percent_change(multiband, roi, 'MB3', 'MB6') for roi in PRIMARY_ROIS]
    audit.close_to('step4 smallest MB6 tSNR loss (%)', min(losses), 29, 1.0)
    audit.close_to('step4 largest MB6 tSNR loss (%)', max(losses), 44, 1.0)

    # step 5: where to spend the acceleration
    split = arms('acceleration_split')
    first, second = COMPARISONS['acceleration_split']['order']
    for roi, reported in [('mPFC', 6), ('EC', -5), ('HC', -4), ('mOFC', 39)]:
        audit.close_to(f'step5 matched split {roi} change (%)',
                       percent_change(split, roi, first, second), reported)
    sub03 = sequences[sequences.subject == 'sub-03'].set_index('run')
    audit.equals('step5 both arms have MB x iPAT = 8',
                 {int(sub03.loc[r, 'total_acceleration'])
                  for r in ('run3_mb4_fast_7T2', 'run5_ipat4_7T2')}, {8})
    audit.close_to('step5 TE through-plane arm (ms)',
                   sub03.loc['run3_mb4_fast_7T2', 'te_ms'], 20.6, .05)
    audit.close_to('step5 TE in-plane arm (ms)',
                   sub03.loc['run5_ipat4_7T2', 'te_ms'], 12.8, .05)
    readout = pd.read_csv(OUTPUT_DIR / 'tables' / 'readout_duration.csv'
                          ).set_index(['subject', 'run'])
    audit.close_to('step5 readout through-plane arm (ms)',
                   readout.loc[('sub-03', 'run3_mb4_fast_7T2'), 'readout_ms'], 27.8, .06)
    audit.close_to('step5 readout in-plane arm (ms)',
                   readout.loc[('sub-03', 'run5_ipat4_7T2'), 'readout_ms'], 14.2, .06)
    shift = displacement.set_index(['subject', 'run'])
    audit.close_to('step5 displacement through-plane arm (mm)',
                   shift.loc[('sub-03', 'run3_mb4_fast_7T2'), 'p99_shift_mm'], 5.6, .06)
    audit.close_to('step5 displacement in-plane arm (mm)',
                   shift.loc[('sub-03', 'run5_ipat4_7T2'), 'p99_shift_mm'], 2.9, .06)
    dropout = coverage[coverage.roi == 'mOFC'].set_index(['subject', 'run'])
    audit.close_to('step5 mOFC dropout through-plane arm (%)',
                   100 * dropout.loc[('sub-03', 'run3_mb4_fast_7T2'), 'dropout_within_slab'], 30, 1.0)
    audit.close_to('step5 mOFC dropout in-plane arm (%)',
                   100 * dropout.loc[('sub-03', 'run5_ipat4_7T2'), 'dropout_within_slab'], 5, 1.0)
    audit.equals('step5 total acceleration across the sweep',
                 [int(sub03.loc[r, 'total_acceleration']) for r in
                  ('run1_3Trep_7T2', 'run3_mb4_fast_7T2', 'run4_mb3ipat3_7T2',
                   'run5_ipat4_7T2')], [6, 8, 9, 8])

    # step 6: against 3T
    versus_3t = arms('field_vs_adopted')
    first, second = COMPARISONS['field_vs_adopted']['order']
    for roi, reported in [('mPFC', 16), ('EC', 18), ('HC', 24), ('mOFC', 6)]:
        audit.close_to(f'step6 3T -> adopted {roi} change (%)',
                       percent_change(versus_3t, roi, first, second), reported)

    # coverage: the slab did NOT contain every ROI in every run
    primary = coverage[coverage.roi.isin(PRIMARY_ROIS)]
    audit.equals('slab contains every primary ROI in 3T and 7T session 2',
                 bool((primary[primary.session != '7T-s1'].slab_coverage >= 0.999).all()),
                 True)
    session1 = primary[primary.session == '7T-s1']
    clipped = session1[session1.slab_coverage < 0.999]
    audit.equals('7T session 1: only mPFC is clipped',
                 set(clipped.roi.unique()), {'mPFC'})
    audit.equals('7T session 1: runs with mPFC clipped',
                 int(clipped.run.nunique()), 5)
    audit.close_to('7T session 1: worst mPFC slab coverage (%)',
                   100 * clipped.slab_coverage.min(), 71, 1.0)
    adopted = primary[(primary.subject == 'sub-03')
                      & (primary.run == 'run5_ipat4_7T2')]
    audit.equals('adopted pilot: all four ROIs fully inside the slab',
                 bool((adopted.slab_coverage >= 0.999).all()), True)

    # provenance: the MB6 filename conflict, corroborated by the readout
    sub02 = sequences[sequences.subject == 'sub-02'].set_index('run')
    audit.close_to('MB6 corroboration: run5 ms per excitation',
                   sub02.loc['run5_T7_superfast', 'ms_per_excitation'], 59.8, .06)
    audit.close_to('MB6 corroboration: run4 ms per excitation',
                   sub02.loc['run4_T7_15mm_2iPat_3MB', 'ms_per_excitation'], 59.1, .06)

    # the adopted protocol itself, against the Siemens printout
    for field_name, pattern, reported in [
            ('TR (ms)', r'TR\s+(\d+) ms', '1078'),
            ('TE (ms)', r'TE\s+([\d.]+) ms', '12.80'),
            ('flip angle (deg)', r'Flip angle\s+(\d+) deg', '54'),
            ('slices', r'Slices\s+(\d+)', '64'),
            ('multiband', r'Multi-band accel\. factor\s+(\d+)', '2'),
            ('iPAT', r'Accel\. factor PE\s+(\d+)', '4'),
            ('base resolution', r'Base resolution\s+(\d+)', '108'),
            ('FoV read (mm)', r'FoV read\s+(\d+) mm', '216'),
            ('slice thickness (mm)', r'Slice thickness\s+([\d.]+) mm', '2.00')]:
        match = re.search(pattern, protocol_block)
        audit.equals(f'adopted protocol {field_name}',
                     match.group(1) if match else None, reported)

    return audit.report()


# ----------------------------------------------------------------- runner
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES,
                        help='run one stage only (earlier outputs must exist)')
    parser.add_argument('--fresh', action='store_true',
                        help='rebuild cached masks and warps from scratch')
    args = parser.parse_args(argv)

    if args.fresh:
        for cached in ('rois', 'tsnr'):
            shutil.rmtree(OUTPUT_DIR / cached, ignore_errors=True)
        print(f'cleared cached masks and maps under {OUTPUT_DIR}')

    stages = [args.stage] if args.stage else STAGES
    table_dir = OUTPUT_DIR / 'tables'

    def load(name):
        return pd.read_csv(table_dir / name)

    sequences = kept_fractions = tsnr_table = common = None
    readout = displacement = coverage = projection = penalties = None

    if 'inventory' in stages:
        banner('1  inventory: what was acquired')
        sequences = inventory.run()
    else:
        sequences = load('sequences.csv')

    if 'rois' in stages:
        banner('2  ROIs: where tSNR is measured')
        kept_fractions = rois.run()

    if 'tsnr' in stages:
        banner('3  tSNR: duration-matched, native space')
        tsnr_table = metrics.compute_tsnr_maps_and_roi_table(sequences)
        if kept_fractions is None:
            kept_fractions = rois.build_common_coverage_masks(verbose=False)
        common = metrics.compute_common_coverage_tsnr(
            sequences, kept_fractions, verbose=False)
        print(f'  wrote tSNR tables for {len(COMPARISONS)} comparisons')

    if 'distortion' in stages:
        banner('4  distortion and coverage')
        readout = metrics.readout_duration_table(sequences)
        displacement, coverage = metrics.measure_displacement_and_coverage(readout)

    if 'tilt' in stages:
        banner('5  slab tilt: the field-gradient mechanism')
        projection = metrics.tilt_field_projection()
        penalties = metrics.tilt_penalties(projection)

    if 'figures' in stages:
        banner('6  figures')
        common = common if common is not None else load('tsnr_by_comparison.csv')
        displacement = displacement if displacement is not None else load('displacement.csv')
        coverage = coverage if coverage is not None else load('coverage.csv')
        penalties = penalties if penalties is not None else load('tilt_penalties.csv')
        figures.run(common, displacement, coverage, penalties)
        tsnr_table = tsnr_table if tsnr_table is not None else load('tsnr_by_run_and_roi.csv')
        metrics.build_lookup_table(sequences, tsnr_table, displacement, coverage,
                                   figures.PANEL_INDEX, verbose=False)
        print(f'  wrote {table_dir / "sequence_lookup.csv"}')

    if 'audit' in stages:
        banner('7  audit: every quoted number, re-derived')
        common = common if common is not None else load('tsnr_by_comparison.csv')
        tsnr_table = tsnr_table if tsnr_table is not None else load('tsnr_by_run_and_roi.csv')
        displacement = displacement if displacement is not None else load('displacement.csv')
        coverage = coverage if coverage is not None else load('coverage.csv')
        projection = projection if projection is not None else load('tilt_field_projection.csv')
        if not run_audit(sequences, common, tsnr_table, displacement, coverage,
                         projection):
            return 1

    print(f'\nDone. Results in {OUTPUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
