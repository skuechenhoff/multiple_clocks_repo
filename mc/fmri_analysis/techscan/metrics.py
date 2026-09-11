"""
Everything that is measured: tSNR, coverage, geometric distortion, and the
slab-tilt field projection.

tSNR is computed voxelwise as mean/SD over time (Triantafyllou et al. 2005) on
FEAT's preprocessed 4D -- motion- and slice-time corrected, fieldmap-unwarped,
NOT spatially smoothed, 100 s highpass. FEAT's grand-mean scaling is one
constant per run and cancels in mean/SD. An ROI value is the arithmetic mean of
those voxelwise values.

Two choices worth defending:

  NATIVE SPACE is primary. Resampling to MNI averages neighbouring native
  voxels -- about 4.6 of them for the 1.2 mm acquisition -- inflating tSNR by
  an amount set by the interpolation rather than the sequence. MNI-space tSNR
  is computed too, because the map figures need a common space, but the
  voxel-size comparison is only honest in native space.

  DURATION MATCHING, not volume matching. Scanner time is the binding
  constraint, so every run is truncated to the same window; volumes per run
  then vary with TR, which is what the efficiency metric accounts for.
"""
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd

from .config import (ADOPTED_TILT_DEG, COMPARISONS, DERIVATIVES_DIR,
                     FEAT_DIR_NAME, MATCHED_WINDOW_SECONDS, OUTPUT_DIR,
                     PRIMARY_ROIS, RANDOM_SEED,
                     REFERENCE_OFFRESONANCE_HZ_AT_3T, ROI_DEFINITIONS, RUNS,
                     TILT_ANGLES_DEG, TILT_REFERENCE_SEQUENCES,
                     USABLE_TSNR_THRESHOLD)
from .rois import (MNI_REFERENCE, acquired_slab_in_mni, fsl,
                   signal_extent_in_mni)

np.random.seed(RANDOM_SEED)

SESSION_OF = {(sub, run): session for sub, run, session, _ in RUNS}
LABEL_OF = {(sub, run): label for sub, run, _, label in RUNS}


# --------------------------------------------------------------- helpers
def matched_volume_count(tr_s):
    """Volumes making up the shared analysis window at this TR."""
    return int(np.floor(MATCHED_WINDOW_SECONDS / tr_s))


def voxelwise_tsnr(path_4d, n_volumes):
    """tSNR per voxel over the first n_volumes."""
    image = nib.load(path_4d)
    timeseries = np.asarray(image.dataobj[..., :n_volumes], dtype=np.float32)
    mean_over_time = timeseries.mean(axis=-1)
    sd_over_time = timeseries.std(axis=-1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tsnr = np.where(sd_over_time > 0, mean_over_time / sd_over_time, 0.0)
    return np.nan_to_num(tsnr, nan=0.0, posinf=0.0), image.affine


def mean_framewise_displacement(feat_dir, n_volumes):
    """Mean relative RMS head displacement over the matched window."""
    path = feat_dir / 'mc' / 'prefiltered_func_data_mcf_rel.rms'
    if not path.exists():
        return np.nan
    return float(np.mean(np.loadtxt(path)[:max(n_volumes - 1, 1)]))


def _load_mask(path):
    return nib.load(path).get_fdata() > 0


# ------------------------------------------------------------------ tSNR
def compute_tsnr_maps_and_roi_table(sequences, verbose=True):
    """Write per-run tSNR maps and a per-run x ROI table."""
    (OUTPUT_DIR / 'tsnr').mkdir(parents=True, exist_ok=True)
    rows = []

    for subject, run, session, label in RUNS:
        sequence = sequences[(sequences.subject == subject)
                             & (sequences.run == run)].iloc[0]
        tr_s = float(sequence.tr_s)
        n_volumes = matched_volume_count(tr_s)
        assert n_volumes <= sequence.n_volumes, \
            f'{subject}/{run}: needs {n_volumes} volumes, has {sequence.n_volumes}'

        feat_dir = DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME
        brain_mask = _load_mask(feat_dir / 'mask.nii.gz')

        tsnr_native, affine_native = voxelwise_tsnr(
            feat_dir / 'filtered_func_data.nii.gz', n_volumes)
        nib.save(nib.Nifti1Image(tsnr_native, affine_native),
                 OUTPUT_DIR / 'tsnr' / f'{subject}_{run}_tsnr_native.nii.gz')

        tsnr_mni, affine_mni = voxelwise_tsnr(
            feat_dir / 'filtered_func_data_standard.nii.gz', n_volumes)
        nib.save(nib.Nifti1Image(tsnr_mni, affine_mni),
                 OUTPUT_DIR / 'tsnr' / f'{subject}_{run}_tsnr_MNI.nii.gz')

        head_motion_mm = mean_framewise_displacement(feat_dir, n_volumes)
        whole_brain_tsnr = tsnr_native[brain_mask]

        for roi in ROI_DEFINITIONS:
            roi_native = _load_mask(OUTPUT_DIR / 'rois' / 'native' /
                                    f'{subject}_{run}_{roi}.nii.gz')
            roi_mni = _load_mask(OUTPUT_DIR / 'rois' / 'MNI' / f'{roi}.nii.gz')
            measured = roi_native & brain_mask
            values = tsnr_native[measured]
            values_mni = tsnr_mni[roi_mni & (tsnr_mni > 0)]

            rows.append(dict(
                subject=subject, run=run, session=session, label=label, roi=roi,
                tr_s=tr_s, te_ms=sequence.te_ms, multiband=sequence.multiband,
                ipat=sequence.ipat, voxel_volume_mm3=sequence.voxel_volume_mm3,
                field_strength_t=sequence.field_strength_t,
                n_volumes_used=n_volumes,
                window_s=round(n_volumes * tr_s, 1),
                head_motion_mm=round(head_motion_mm, 4),
                n_roi_voxels=int(roi_native.sum()),
                n_roi_voxels_measured=int(measured.sum()),
                tsnr_native_mean=round(float(values.mean()), 2) if values.size else np.nan,
                tsnr_native_median=round(float(np.median(values)), 2) if values.size else np.nan,
                tsnr_efficiency=round(float(values.mean() / np.sqrt(tr_s)), 2) if values.size else np.nan,
                usable_fraction=round(float((values > USABLE_TSNR_THRESHOLD).mean()), 4) if values.size else np.nan,
                tsnr_mni_mean=round(float(values_mni.mean()), 2) if values_mni.size else np.nan,
                tsnr_whole_brain_mean=round(float(whole_brain_tsnr.mean()), 2)))

        if verbose:
            print(f'  {subject}/{run:<24} N={n_volumes:>3} '
                  f'({n_volumes * tr_s:6.1f} s)  whole-brain tSNR='
                  f'{whole_brain_tsnr.mean():6.2f}  motion={head_motion_mm:.3f} mm')

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / 'tables' / 'tsnr_by_run_and_roi.csv', index=False)
    return table


def compute_common_coverage_tsnr(sequences, kept_fractions, verbose=True):
    """Per comparison, tSNR restricted to voxels every arm reached."""
    rows = []
    for name, comparison in COMPARISONS.items():
        for subject, run, arm in comparison['arms']:
            sequence = sequences[(sequences.subject == subject)
                                 & (sequences.run == run)].iloc[0]
            tr_s = float(sequence.tr_s)
            tsnr = nib.load(OUTPUT_DIR / 'tsnr' /
                            f'{subject}_{run}_tsnr_native.nii.gz').get_fdata()

            for roi in ROI_DEFINITIONS:
                mask = _load_mask(OUTPUT_DIR / 'rois' / 'common_native' / name /
                                  f'{subject}_{run}_{roi}.nii.gz')
                values = tsnr[mask & (tsnr > 0)]
                rows.append(dict(
                    comparison=name, subject=subject, run=run, arm=arm,
                    session=SESSION_OF[(subject, run)],
                    label=LABEL_OF[(subject, run)], roi=roi, tr_s=tr_s,
                    te_ms=sequence.te_ms, multiband=sequence.multiband,
                    ipat=sequence.ipat,
                    voxel_volume_mm3=sequence.voxel_volume_mm3,
                    field_strength_t=sequence.field_strength_t,
                    n_volumes_used=matched_volume_count(tr_s),
                    roi_fraction_common=round(kept_fractions[name][roi], 4),
                    n_voxels=int(values.size),
                    tsnr_mean=round(float(values.mean()), 2) if values.size else np.nan,
                    tsnr_median=round(float(np.median(values)), 2) if values.size else np.nan,
                    tsnr_sem=round(float(values.std(ddof=1) / np.sqrt(values.size)), 3) if values.size > 1 else np.nan,
                    efficiency=round(float(values.mean() / np.sqrt(tr_s)), 2) if values.size else np.nan,
                    usable_fraction=round(float((values > USABLE_TSNR_THRESHOLD).mean()), 4) if values.size else np.nan))
        if verbose:
            print(f'  {name}')

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / 'tables' / 'tsnr_by_comparison.csv', index=False)
    return table


def tsnr_by_arm(common_table, comparison, metric='tsnr_mean', rois=None):
    """arm x ROI table for one comparison, averaged over participants where a
    comparison has more than one."""
    rois = rois or PRIMARY_ROIS
    subset = common_table[(common_table.comparison == comparison)
                          & (common_table.roi.isin(rois))]
    return (subset.groupby(['arm', 'roi'])[metric].mean().unstack()
            .reindex(COMPARISONS[comparison]['order'])[rois])


# ----------------------------------------------------- distortion, coverage
def readout_duration_table(sequences, verbose=True):
    """EPI readout duration and the displacement it implies.

    An EPI readout encodes the phase-encode direction slowly, so off-resonance
    displaces signal along it:

        shift [voxels] = off-resonance [Hz] x total readout time [s]
        shift [mm]     = shift [voxels] x voxel size along phase encode

    FEAT's `dwell` is the *effective* echo spacing, already divided by the
    in-plane acceleration, and FSL forms the total readout as
    dwell x (base_resolution - 1). That identity is checked here against the
    TotalReadoutTime the scanner wrote, for the runs that still have a sidecar;
    the two agree to about 1%, the residual being FEAT's rounding of dwell.
    """
    rows = []
    for subject, run, session, label in RUNS:
        sequence = sequences[(sequences.subject == subject)
                             & (sequences.run == run)].iloc[0]
        echo_spacing_s = sequence.echo_spacing_ms * 1e-3
        base_resolution = int(round(sequence.fov_mm / sequence.voxel_x_mm / 2) * 2)
        readout_from_feat_s = echo_spacing_s * (base_resolution - 1)

        readout_header_s = (sequence.readout_header_s
                            if not pd.isna(sequence.readout_header_s) else np.nan)
        if not np.isnan(readout_header_s):
            agrees = abs(readout_from_feat_s - readout_header_s) / readout_header_s < 0.02
            readout_s, source = readout_header_s, 'scanner header'
        else:
            agrees, readout_s, source = None, readout_from_feat_s, 'FEAT echo spacing'

        offresonance_hz = REFERENCE_OFFRESONANCE_HZ_AT_3T * sequence.field_strength_t / 3.0
        rows.append(dict(
            subject=subject, run=run, session=session, label=label,
            field_strength_t=sequence.field_strength_t, ipat=sequence.ipat,
            multiband=sequence.multiband, voxel_mm=sequence.voxel_y_mm,
            base_resolution=base_resolution,
            effective_echo_spacing_ms=sequence.echo_spacing_ms,
            readout_ms=round(readout_s * 1e3, 2), readout_source=source,
            readout_from_feat_ms=round(readout_from_feat_s * 1e3, 2),
            readout_header_ms=round(readout_header_s * 1e3, 2) if not np.isnan(readout_header_s) else np.nan,
            readout_matches_header=agrees,
            reference_offresonance_hz=round(offresonance_hz, 1),
            reference_shift_mm=round(offresonance_hz * readout_s * sequence.voxel_y_mm, 2)))

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / 'tables' / 'readout_duration.csv', index=False)
    if verbose:
        checked = table.dropna(subset=['readout_header_ms'])
        worst = (abs(checked.readout_from_feat_ms - checked.readout_header_ms)
                 / checked.readout_header_ms).max()
        print(f'  readout identity checked on {len(checked)}/{len(table)} runs: '
              f'{"all agree within 2%" if checked.readout_matches_header.all() else "MISMATCH"}'
              f' (largest deviation {100 * worst:.2f}%, from FEAT rounding)')
    return table


def fieldmap_hz_in_epi_space(subject, run):
    """The acquired B0 fieldmap in Hz, in this run's EPI space.

    FEAT names this file after whichever image drove the registration, so the
    prefix varies; runs that used an initial-highres step write a file with the
    same suffix in a different space, hence the shape check."""
    feat_dir = DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME
    example_func = nib.load(feat_dir / 'reg' / 'example_func.nii.gz')
    for path in sorted((feat_dir / 'reg').glob('*fieldmaprads2epi.nii.gz')):
        image = nib.load(path)
        if image.shape[:3] == example_func.shape[:3]:
            return np.asarray(image.dataobj) / (2 * np.pi), example_func, run
    return None, example_func, None


def fieldmap_hz_with_fallback(subject, run, session):
    """Fall back to a sibling run from the same session where a run's fieldmap
    was written in initial-highres space. Legitimate here because one fieldmap
    acquisition was applied to every run in a session: the off-resonance field
    belongs to that head in that shim, and only the readout differs."""
    field_hz, reference, source = fieldmap_hz_in_epi_space(subject, run)
    if field_hz is not None:
        return field_hz, reference, source
    for other_subject, other_run, other_session, _ in RUNS:
        if other_subject == subject and other_session == session and other_run != run:
            field_hz, reference, source = fieldmap_hz_in_epi_space(
                other_subject, other_run)
            if field_hz is not None:
                return field_hz, reference, other_run
    return None, reference, None


def displacement_map_mm(subject, run, readout_s, session):
    """Per-voxel phase-encode displacement in mm for this run's readout."""
    field_hz, reference, source_run = fieldmap_hz_with_fallback(
        subject, run, session)
    if field_hz is None:
        return None, None, None
    brain_mask = _load_mask(DERIVATIVES_DIR / subject / source_run / 'func' /
                            FEAT_DIR_NAME / 'mask.nii.gz')
    # phase encode is j (A>>P) throughout, so the second voxel dimension
    mm_per_voxel = float(reference.header.get_zooms()[1])
    return np.abs(field_hz) * readout_s * mm_per_voxel, brain_mask, source_run


def measure_displacement_and_coverage(readout_table, verbose=True):
    """Measured distortion per run, and coverage split into its two causes.

    The earlier single "coverage" number conflated two unrelated things:

      slab coverage   was this anatomy inside the prescribed imaging volume at
                      all? Purely geometry; missing here was never acquired.
      signal coverage was there usable signal there? Missing here is dropout.

    FEAT's brain mask is the intersection of both, so it cannot separate them.
    The geometric slab is measured independently in rois.acquired_slab_in_mni.
    """
    displacement_rows, coverage_rows = [], []
    readout_by_run = readout_table.set_index(['subject', 'run'])

    for subject, run, session, label in RUNS:
        readout_s = float(readout_by_run.loc[(subject, run), 'readout_ms']) / 1e3
        displacement_mm, brain_mask, source_run = displacement_map_mm(
            subject, run, readout_s, session)

        if displacement_mm is None:
            displacement_rows.append(dict(subject=subject, run=run,
                                          session=session, label=label,
                                          readout_ms=readout_s * 1e3))
        else:
            values = displacement_mm[brain_mask]
            displacement_rows.append(dict(
                subject=subject, run=run, session=session, label=label,
                readout_ms=round(readout_s * 1e3, 2),
                fieldmap_from_run=source_run,
                median_shift_mm=round(float(np.median(values)), 2),
                p95_shift_mm=round(float(np.percentile(values, 95)), 2),
                p99_shift_mm=round(float(np.percentile(values, 99)), 2),
                max_shift_mm=round(float(values.max()), 2),
                fraction_over_2mm=round(float((values > 2).mean()), 4)))

        slab = acquired_slab_in_mni(subject, run)
        signal = signal_extent_in_mni(subject, run)
        for roi in ROI_DEFINITIONS:
            roi_mask = _load_mask(OUTPUT_DIR / 'rois' / 'MNI' / f'{roi}.nii.gz')
            n_roi = roi_mask.sum()
            n_in_slab = (roi_mask & slab).sum()
            n_with_signal = (roi_mask & signal).sum()
            coverage_rows.append(dict(
                subject=subject, run=run, session=session, label=label, roi=roi,
                n_roi_voxels=int(n_roi),
                slab_coverage=round(n_in_slab / n_roi, 4),
                signal_coverage=round(n_with_signal / n_roi, 4),
                # acquired but no usable signal: dropout, the part a better
                # sequence can actually fix
                dropout_within_slab=round(
                    (n_in_slab - n_with_signal) / max(n_in_slab, 1), 4)))

        if verbose and displacement_mm is not None:
            row = displacement_rows[-1]
            print(f'  {subject}/{run:<24} readout {row["readout_ms"]:5.2f} ms  '
                  f'median shift {row["median_shift_mm"]:5.2f} mm  '
                  f'p99 {row["p99_shift_mm"]:5.2f} mm  '
                  f'{100 * row["fraction_over_2mm"]:4.1f}% of brain >2 mm')

    displacement = pd.DataFrame(displacement_rows)
    coverage = pd.DataFrame(coverage_rows)
    displacement.to_csv(OUTPUT_DIR / 'tables' / 'displacement.csv', index=False)
    coverage.to_csv(OUTPUT_DIR / 'tables' / 'coverage.csv', index=False)
    return displacement, coverage


# ------------------------------------------------------------- slab tilt
def fieldmap_hz_in_mni(subject='sub-03', run='run1_3Trep_7T2'):
    """A session's acquired fieldmap in Hz, pushed into MNI so it can be read
    against the MNI ROI masks."""
    feat_dir = DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME
    out = OUTPUT_DIR / 'tsnr' / 'fieldmap_hz_MNI.nii.gz'
    if not out.exists():
        field_hz, example_func, _ = fieldmap_hz_in_epi_space(subject, run)
        native = OUTPUT_DIR / 'tsnr' / 'fieldmap_hz_native.nii.gz'
        nib.save(nib.Nifti1Image(field_hz.astype(np.float32),
                                 example_func.affine), native)
        fsl('applywarp', '-i', native,
            '-r', feat_dir / 'reg' / 'standard.nii.gz',
            '-w', feat_dir / 'reg' / 'example_func2standard_warp.nii.gz',
            '-o', out, '--interp=trilinear')
    return nib.load(out).get_fdata()


def tilt_field_projection(verbose=True):
    """How each candidate slab tilt splits the measured field gradient.

    The slab tilt was only ever swept at 3T, on sequences with 2-3.8x longer
    readouts than the one adopted, so that sweep alone cannot license a claim
    about the adopted sequence. This can, because it starts from a quantity
    belonging to the head rather than the sequence: the acquired B0 fieldmap.

    Tilting does not change the field. It changes which direction counts as
    phase-encode and which as through-plane, redistributing a fixed gradient
    between the two mechanisms that destroy signal. For a tilt b about the
    left-right axis with A>>P encoding:

        phase-encode axis   e_pe = (0,  cos b, -sin b)
        slice normal        n    = (0,  sin b,  cos b)

    Weiskopf et al. (2006) note that tilt and PE polarity act primarily on the
    phase-encode component; z-shimming addresses the through-plane one.
    """
    field_hz = fieldmap_hz_in_mni()
    # gradient in Hz/mm; the MNI 2 mm grid has +2 spacing along y and z
    gradient_y = np.gradient(field_hz, 2.0, axis=1)
    gradient_z = np.gradient(field_hz, 2.0, axis=2)
    has_field = np.abs(field_hz) > 0

    rows = []
    for roi in PRIMARY_ROIS:
        roi_mask = _load_mask(OUTPUT_DIR / 'rois' / 'MNI' / f'{roi}.nii.gz') \
                   & has_field
        for tilt_deg in TILT_ANGLES_DEG:
            tilt = np.radians(tilt_deg)
            g_phase_encode = float(np.abs(
                gradient_y[roi_mask] * np.cos(tilt)
                - gradient_z[roi_mask] * np.sin(tilt)).mean())
            g_through_plane = float(np.abs(
                gradient_y[roi_mask] * np.sin(tilt)
                + gradient_z[roi_mask] * np.cos(tilt)).mean())
            rows.append(dict(roi=roi, tilt_deg=tilt_deg,
                             g_phase_encode_hz_per_mm=round(g_phase_encode, 3),
                             g_through_plane_hz_per_mm=round(g_through_plane, 3)))

    projection = pd.DataFrame(rows)
    projection.to_csv(OUTPUT_DIR / 'tables' / 'tilt_field_projection.csv',
                      index=False)
    if verbose:
        print('  tilt minimising the phase-encode gradient, per ROI:')
        for roi in PRIMARY_ROIS:
            roi_rows = projection[projection.roi == roi]
            best = int(roi_rows.loc[roi_rows.g_phase_encode_hz_per_mm.idxmin(),
                                    'tilt_deg'])
            transverse = roi_rows[roi_rows.tilt_deg == 0
                                  ].g_phase_encode_hz_per_mm.iloc[0]
            reduction = 100 * (1 - roi_rows.g_phase_encode_hz_per_mm.min()
                               / transverse)
            print(f'    {roi:<6} {best:+d} deg  ({reduction:.0f}% below transverse)')
        print(f'  adopted tilt was {ADOPTED_TILT_DEG:+d} deg')
    return projection


def tilt_penalties(projection, verbose=True):
    """What each tilt actually costs, per sequence.

    The two penalties must be evaluated rather than compared as bare gradients,
    because they have different units and very different sizes:

        through-plane   signal kept      = |sinc(g_slice * dz * TE)|
        in-plane        voxel distortion = g_pe * voxel * readout

    That asymmetry is the result. Tilting to +30 deg raises the through-plane
    gradient, but at any realistic TE that costs under 4% of signal, while the
    in-plane distortion it removes is worth 3-20 percentage points. So +30 deg
    wins at every readout tested; a shorter readout shrinks its margin
    asymptotically towards zero but cannot reverse the sign.
    """
    rows = []
    for roi in PRIMARY_ROIS:
        by_tilt = projection[projection.roi == roi].set_index('tilt_deg')
        for name, params in TILT_REFERENCE_SEQUENCES.items():
            for tilt_deg in TILT_ANGLES_DEG:
                signal_kept = abs(np.sinc(
                    by_tilt.loc[tilt_deg, 'g_through_plane_hz_per_mm']
                    * params['slice_mm'] * params['te_s']))
                distortion = (by_tilt.loc[tilt_deg, 'g_phase_encode_hz_per_mm']
                              * params['slice_mm'] * params['readout_s'])
                rows.append(dict(
                    roi=roi, sequence=name, tilt_deg=tilt_deg,
                    signal_kept=round(float(signal_kept), 4),
                    voxel_distortion=round(float(distortion), 4),
                    net_signal=round(float(signal_kept * (1 - distortion)), 4)))

    penalties = pd.DataFrame(rows)
    penalties.to_csv(OUTPUT_DIR / 'tables' / 'tilt_penalties.csv', index=False)
    if verbose:
        print('  predicted signal retained after both penalties (%):')
        print((100 * penalties.pivot_table(index=['roi', 'sequence'],
                                           columns='tilt_deg',
                                           values='net_signal')).round(1)
              .to_string())
    return penalties


# ----------------------------------------------------------- lookup table
def build_lookup_table(sequences, tsnr_table, displacement, coverage,
                       panel_index, verbose=True):
    """One row per run: parameters, where it appears in the argument, results."""
    membership = {}
    for name, comparison in COMPARISONS.items():
        for subject, run in comparison['runs']:
            entry = membership.setdefault((subject, run),
                                          dict(steps=set(), comparisons=[],
                                               panels=set()))
            entry['steps'].add(comparison['step'])
            entry['comparisons'].append(name)
            entry['panels'].update(panel_index.get(name, []))

    rows = []
    displacement_by_run = displacement.set_index(['subject', 'run'])
    for subject, run, session, label in RUNS:
        sequence = sequences[(sequences.subject == subject)
                             & (sequences.run == run)].iloc[0]
        entry = membership.get((subject, run),
                               dict(steps=set(), comparisons=[], panels=set()))
        row = dict(
            subject=subject, run=run, session=session, protocol=sequence.protocol,
            field_strength_t=sequence.field_strength_t, tr_s=sequence.tr_s,
            te_ms=sequence.te_ms, flip_angle_deg=sequence.flip_angle_deg,
            multiband=sequence.multiband, ipat=sequence.ipat,
            total_acceleration=sequence.total_acceleration,
            voxel_mm=sequence.voxel_x_mm,
            voxel_volume_mm3=sequence.voxel_volume_mm3,
            n_slices=sequence.n_slices,
            slab_thickness_mm=sequence.slab_thickness_mm,
            slab_tilt_deg=sequence.slab_tilt_deg,
            n_volumes_acquired=sequence.n_volumes, duration_s=sequence.duration_s,
            decision_steps=','.join(str(s) for s in sorted(entry['steps'])) or '-',
            comparisons=';'.join(entry['comparisons']) or '-',
            figure_panels=';'.join(sorted(entry['panels'])) or '-')

        if (subject, run) in displacement_by_run.index:
            shift = displacement_by_run.loc[(subject, run)]
            row['readout_ms'] = shift.get('readout_ms', np.nan)
            row['displacement_p99_mm'] = shift.get('p99_shift_mm', np.nan)

        for roi in PRIMARY_ROIS:
            result = tsnr_table[(tsnr_table.subject == subject)
                                & (tsnr_table.run == run)
                                & (tsnr_table.roi == roi)]
            cover = coverage[(coverage.subject == subject)
                             & (coverage.run == run) & (coverage.roi == roi)]
            row[f'tsnr_{roi}'] = float(result.tsnr_native_mean.iloc[0]) if len(result) else np.nan
            row[f'efficiency_{roi}'] = float(result.tsnr_efficiency.iloc[0]) if len(result) else np.nan
            row[f'dropout_{roi}'] = round(float(cover.dropout_within_slab.iloc[0]), 3) if len(cover) else np.nan
        rows.append(row)

    lookup = pd.DataFrame(rows)
    lookup.to_csv(OUTPUT_DIR / 'tables' / 'sequence_lookup.csv', index=False)
    if verbose:
        print(lookup[['subject', 'run', 'field_strength_t', 'voxel_mm', 'tr_s',
                      'te_ms', 'multiband', 'ipat', 'total_acceleration',
                      'readout_ms', 'displacement_p99_mm', 'decision_steps']]
              .to_string(index=False))
    return lookup
