"""
What was actually acquired: sequence parameters read from headers, not names.

Protocol and folder names in this dataset are not reliable -- one run's
filename says MB3 where its DICOM header says MB6. Everything quoted in the
write-up therefore comes from this table, which reads the BOLD JSON sidecar,
the NIfTI header and the FEAT design file, and cross-checks them against each
other.
"""
import json
import re

import nibabel as nib
import numpy as np
import pandas as pd

from .config import (DERIVATIVES_DIR, FEAT_DIR_NAME, OUTPUT_DIR,
                     PROTOCOL_PRINTOUT_SUB03, RAW_DIR, RUNS)


def read_feat_design(design_path):
    """Preprocessing settings FEAT actually applied."""
    text = design_path.read_text()

    def field(key):
        match = re.search(r'set fmri\(%s\)\s+"?([^"\n]+)"?' % key, text)
        return match.group(1).strip() if match else None

    return dict(
        n_volumes_feat=int(field('npts')),
        tr_feat_s=float(field('tr')),
        echo_spacing_ms=float(field('dwell')),
        smoothing_mm=float(field('smooth')),
        highpass_s=float(field('paradigm_hp')),
        slice_timing=field('st'),
        fieldmap_unwarping=field('regunwarp_yn'),
        motion_correction=field('mc'),
        func_to_struct_dof=field('reghighres_dof'),
        nonlinear_to_standard=field('regstandard_nonlinear_yn'),
    )


def read_bold_sidecar(subject, run):
    """BOLD JSON if it survived conversion; sub-03's were not kept."""
    path = RAW_DIR / subject / run / 'func' / f'{subject}_bold.json'
    if not path.exists():
        return {}
    sidecar = json.loads(path.read_text())
    return dict(
        protocol=sidecar.get('ProtocolName'),
        field_strength_t=sidecar.get('MagneticFieldStrength'),
        tr_header_s=sidecar.get('RepetitionTime'),
        te_ms=1e3 * sidecar.get('EchoTime', np.nan),
        flip_angle_deg=sidecar.get('FlipAngle'),
        multiband=sidecar.get('MultibandAccelerationFactor'),
        ipat=sidecar.get('ParallelReductionFactorInPlane'),
        bandwidth_hz_px=sidecar.get('PixelBandwidth'),
        phase_encode_dir=sidecar.get('PhaseEncodingDirection'),
        readout_header_s=sidecar.get('TotalReadoutTime'),
        coil=sidecar.get('ReceiveCoilName'),
    )


def protocol_name_from_nifti(subject, run):
    """The scanner protocol name survives in the images_NNN_*.nii filename that
    dcm2niix wrote beside the BIDS copy, even where the sidecar is gone."""
    candidates = list((RAW_DIR / subject / run / 'func').glob('images_*.nii'))
    if not candidates:
        return None
    return re.sub(r'^images_\d+_', '', candidates[0].stem)


def slab_tilt_deg(subject, run):
    """Slab tilt about the left-right axis, from the image affine. Positive
    means the anterior edge of the slice is tilted towards the feet, matching
    the convention used in the slice-tilt literature."""
    affine = nib.load(DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME /
                      'reg' / 'example_func.nii.gz').affine
    slice_normal = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    return float(np.degrees(np.arctan2(slice_normal[1], slice_normal[2])))


def build_sequence_table():
    """One row per run: acquisition parameters, geometry, preprocessing."""
    rows = []
    for subject, run, session, label in RUNS:
        feat_dir = DERIVATIVES_DIR / subject / run / 'func' / FEAT_DIR_NAME
        row = dict(subject=subject, run=run, session=session, label=label)
        row.update(read_bold_sidecar(subject, run))
        row.update(read_feat_design(feat_dir / 'design.fsf'))

        if row.get('protocol') is None:
            row['protocol'] = protocol_name_from_nifti(subject, run)
            row['parameter_source'] = 'protocol printout + design.fsf (sidecar lost)'
            field_t, te, flip, mb, ipat, bandwidth = \
                PROTOCOL_PRINTOUT_SUB03[row['protocol']]
            row.update(field_strength_t=field_t, te_ms=te, flip_angle_deg=flip,
                       multiband=mb, ipat=ipat, bandwidth_hz_px=bandwidth)
            # TE is independently recorded by FEAT, so verify rather than trust
            te_from_feat = float(re.search(
                r'set fmri\(te\)\s+([\d.]+)',
                (feat_dir / 'design.fsf').read_text()).group(1))
            assert abs(te_from_feat - te) < 0.05, (run, te_from_feat, te)
        else:
            row['parameter_source'] = 'DICOM/JSON sidecar'

        image = nib.load(feat_dir / 'filtered_func_data.nii.gz')
        vox_x, vox_y, vox_z = image.header.get_zooms()[:3]
        row.update(
            voxel_x_mm=round(float(vox_x), 3),
            voxel_y_mm=round(float(vox_y), 3),
            voxel_z_mm=round(float(vox_z), 3),
            voxel_volume_mm3=round(float(vox_x * vox_y * vox_z), 3),
            n_slices=int(image.shape[2]),
            n_volumes=int(image.shape[3]),
            fov_mm=round(float(vox_x * image.shape[0]), 1),
            slab_thickness_mm=round(float(vox_z * image.shape[2]), 1),
            slab_tilt_deg=round(slab_tilt_deg(subject, run)),
        )

        tr_s = row.get('tr_header_s') or row['tr_feat_s']
        row['tr_s'] = tr_s
        row['duration_s'] = round(tr_s * row['n_volumes'], 1)
        row['total_acceleration'] = (row['multiband'] * row['ipat']
                                     if row.get('ipat') else np.nan)

        # Physical check on the multiband factor. With multiband MB the scanner
        # plays n_slices/MB excitations per TR, so TR*MB/n_slices is the time
        # per excitation -- set by the readout, not by MB. Two runs matched on
        # voxel size, slice count and iPAT must agree on it whatever their MB,
        # which is what makes it an independent test of any MB factor not read
        # from a sidecar.
        if row.get('multiband'):
            row['ms_per_excitation'] = round(
                1e3 * tr_s * row['multiband'] / row['n_slices'], 1)
        rows.append(row)

    return pd.DataFrame(rows)


def find_filename_header_conflicts(sequences):
    """Runs whose protocol name disagrees with the acquired header."""
    conflicts = []
    for _, row in sequences.iterrows():
        implied = re.search(r'[Mm][Bb](\d)', row.protocol or '')
        if implied and row.multiband is not None:
            if int(implied.group(1)) != int(row.multiband):
                conflicts.append(dict(
                    subject=row.subject, run=row.run, protocol=row.protocol,
                    implied_multiband=int(implied.group(1)),
                    header_multiband=int(row.multiband),
                    ipat=row.ipat, tr_s=row.tr_s,
                    ms_per_excitation=row.ms_per_excitation))
    return pd.DataFrame(conflicts)


def run(verbose=True):
    sequences = build_sequence_table()
    conflicts = find_filename_header_conflicts(sequences)

    columns = ['subject', 'run', 'session', 'label', 'protocol',
               'parameter_source', 'field_strength_t', 'tr_s', 'te_ms',
               'flip_angle_deg', 'multiband', 'ipat', 'total_acceleration',
               'ms_per_excitation', 'voxel_x_mm', 'voxel_y_mm', 'voxel_z_mm',
               'voxel_volume_mm3', 'n_slices', 'slab_thickness_mm',
               'slab_tilt_deg', 'fov_mm', 'bandwidth_hz_px',
               'readout_header_s', 'phase_encode_dir', 'coil', 'n_volumes',
               'duration_s', 'echo_spacing_ms', 'smoothing_mm', 'highpass_s',
               'slice_timing', 'fieldmap_unwarping', 'motion_correction',
               'func_to_struct_dof', 'nonlinear_to_standard']
    (OUTPUT_DIR / 'tables').mkdir(parents=True, exist_ok=True)
    sequences[columns].to_csv(OUTPUT_DIR / 'tables' / 'sequences.csv',
                              index=False)
    conflicts.to_csv(OUTPUT_DIR / 'tables' / 'filename_header_conflicts.csv',
                     index=False)

    if verbose:
        print(sequences[['subject', 'run', 'session', 'field_strength_t',
                         'tr_s', 'te_ms', 'multiband', 'ipat',
                         'total_acceleration', 'voxel_volume_mm3',
                         'slab_tilt_deg', 'n_volumes', 'duration_s']]
              .to_string(index=False))
        if len(conflicts):
            print('\nFilename/header conflicts:')
            print(conflicts.to_string(index=False))
        print(f'\nShortest run: {sequences.duration_s.min():.1f} s '
              f'(the duration match must not exceed this)')
    return sequences
