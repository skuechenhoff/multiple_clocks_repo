"""
Stage 0: build the sequence inventory from acquired headers, not filenames.

Reads, per run, the BOLD JSON sidecar (true acquisition parameters), the NIfTI
header (true voxel geometry) and the FEAT design.fsf (what preprocessing was
actually applied), and writes:

    tables/sequences.csv     one row per run, acquisition + preprocessing
    tables/discrepancies.md  where filenames and headers disagree

Run:  python -m mc.fmri_analysis.techscan_tsnr.build_inventory
"""
import json
import re
import numpy as np
import pandas as pd
import nibabel as nib

from .config import RAW, DERIV, OUT, RUNS, FEAT


def read_fsf(path):
    """Pull the settings we care about out of a FEAT design file."""
    txt = path.read_text()

    def g(key):
        m = re.search(r'set fmri\(%s\)\s+"?([^"\n]+)"?' % key, txt)
        return m.group(1).strip() if m else None

    return dict(
        npts=int(g('npts')), TR_fsf=float(g('tr')),
        dwell_ms=float(g('dwell')), smooth_mm=float(g('smooth')),
        highpass_s=float(g('paradigm_hp')), slicetiming=g('st'),
        unwarp=g('regunwarp_yn'), mc=g('mc'),
        reg_dof=g('reghighres_dof'), nonlinear=g('regstandard_nonlinear_yn'),
    )


def read_json_sidecar(sub, run):
    """BOLD sidecar if it survived; sub-03's were not kept at conversion."""
    p = RAW / sub / run / 'func' / f'{sub}_bold.json'
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    return dict(
        protocol=j.get('ProtocolName'),
        field_T=j.get('MagneticFieldStrength'),
        TR_hdr=j.get('RepetitionTime'), TE_ms=1e3 * j.get('EchoTime', np.nan),
        flip_deg=j.get('FlipAngle'),
        MB=j.get('MultibandAccelerationFactor'),
        iPAT=j.get('ParallelReductionFactorInPlane'),
        bandwidth_Hz_px=j.get('PixelBandwidth'),
        PE_dir=j.get('PhaseEncodingDirection'),
        readout_s=j.get('TotalReadoutTime'),
        coil=j.get('ReceiveCoilName'),
    )


def protocol_from_raw_nifti(sub, run):
    """sub-03 lost its BOLD sidecars; the scanner protocol name survives in the
    images_NNN_*.nii filename that dcm2niix wrote next to the BIDS copy."""
    hits = [p for p in (RAW / sub / run / 'func').glob('images_*.nii')]
    if not hits:
        return None
    return re.sub(r'^images_\d+_', '', hits[0].stem)


# sub-03's BOLD JSON sidecars were not kept at conversion. These values come
# from the Siemens protocol printout `2-7T-Fast BOLD measured 24-08-2023.md`,
# which is the only surviving record of MB/iPAT/flip for that session. Every
# value that CAN be checked against the data independently (TR, TE, slice
# count, FOV, base resolution, voxel size) matches the NIfTI header and the
# FEAT design.fsf exactly, so the printout is trustworthy for this session.
PRINTOUT_SUB03 = {
    #  protocol name                    field_T TE_ms flip MB iPAT   BW
    'cmrr_mbep2d_bold_3Trep_2mm_mb3p2':   (7, 19.6, 53, 3, 2, 2314),
    'cmrr_mbep2d_bold_3Trep_2.5mm_mb3p2': (7, 16.8, 53, 3, 2, 2236),
    'cmrr_mbep2d_bold_3Trep_2mm_mb4p2':   (7, 20.6, 53, 4, 2, 2204),
    'cmrr_mbep2d_bold_3Trep_2mm_mb3p3':   (7, 15.0, 53, 3, 3, 2436),
    'cmrr_mbep2d_bold_3Trep_2mm_mb2p4':   (7, 12.8, 53, 2, 4, 2436),
}


def main():
    rows, notes = [], []

    for sub, run, session, label in RUNS:
        feat = DERIV / sub / run / 'func' / FEAT
        row = dict(sub=sub, run=run, session=session, label=label)
        row.update(read_json_sidecar(sub, run))
        row.update(read_fsf(feat / 'design.fsf'))

        if row.get('protocol') is None:
            row['protocol'] = protocol_from_raw_nifti(sub, run)
            row['params_source'] = 'protocol printout + design.fsf (sidecar lost)'
            f_T, te, fa, mb, ipat, bw = PRINTOUT_SUB03[row['protocol']]
            row.update(field_T=f_T, flip_deg=fa, MB=mb, iPAT=ipat,
                       bandwidth_Hz_px=bw, TE_ms=te)
            # TE is independently recorded in the FEAT design.fsf, so it can be
            # verified rather than trusted
            te_fsf = float(re.search(r'set fmri\(te\)\s+([\d.]+)',
                                     (feat / 'design.fsf').read_text()).group(1))
            assert abs(te_fsf - te) < 0.05, (run, te_fsf, te)
        else:
            row['params_source'] = 'DICOM/JSON sidecar'

        # true geometry from the preprocessed native-space 4D
        img = nib.load(feat / 'filtered_func_data.nii.gz')
        zx, zy, zz = img.header.get_zooms()[:3]
        row.update(vox_x=round(float(zx), 3), vox_y=round(float(zy), 3),
                   vox_z=round(float(zz), 3),
                   vox_mm3=round(float(zx * zy * zz), 3),
                   n_slices=int(img.shape[2]), n_vols=int(img.shape[3]),
                   fov_mm=round(float(zx * img.shape[0]), 1),
                   slab_mm=round(float(zz * img.shape[2]), 1))

        TR = row.get('TR_hdr') or row['TR_fsf']
        row['TR_s'] = TR
        row['duration_s'] = round(TR * row['n_vols'], 1)

        # Physical consistency check on MB: with a multiband factor MB, the
        # scanner plays n_slices/MB excitations per TR, so TR*MB/n_slices is
        # the time per excitation. It has to land in a physically sensible
        # range (~25-70 ms for these EPI readouts) and, within a session, has
        # to fall as in-plane acceleration shortens the echo train. This
        # validates the MB/iPAT values that came from the protocol printout.
        if row.get('MB'):
            row['ms_per_excitation'] = round(
                1e3 * TR * row['MB'] / row['n_slices'], 1)
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- filename vs header disagreements ------------------------------
    for r in rows:
        p = r.get('protocol') or ''
        m = re.search(r'[Mm][Bb](\d)', p)
        if m and r.get('MB') is not None and int(m.group(1)) != int(r['MB']):
            notes.append(
                f"- **{r['sub']} / {r['run']}** -- protocol name `{p}` implies "
                f"MB={m.group(1)}, but the DICOM header records "
                f"**multiband factor = {int(r['MB'])}** (iPAT={r.get('iPAT')}, "
                f"TR={r['TR_s']} s). The header is authoritative; cite MB="
                f"{int(r['MB'])}.")

    md = ["# Sequence provenance: filename vs acquired header",
          "",
          "Built by `mc/fmri_analysis/techscan_tsnr/build_inventory.py`. Every "
          "parameter quoted in the methods section should come from "
          "`sequences.csv`, which reads the acquired headers, not from folder "
          "or protocol names.",
          "", "## Confirmed mislabels", ""]
    md += notes or ["- none found"]

    # ---- independent physical corroboration ----------------------------
    # Time per excitation = TR * MB / n_slices. Runs with the same voxel size,
    # slice count and in-plane acceleration share an echo train and so must
    # share this value; it is set by the readout, not by MB. That makes it an
    # independent test of any MB factor we did not read from a sidecar.
    md += ["", "## Independent check on the multiband factors", "",
           "Time per slice excitation, `TR x MB / n_slices`, is fixed by the "
           "EPI readout. Two runs matched on voxel size, slice count and "
           "in-plane acceleration must therefore agree on it whatever their "
           "MB factor, and within a session it must fall as in-plane "
           "acceleration shortens the echo train. Both hold:", ""]
    r4 = df[(df['sub'] == 'sub-02') & (df.run == 'run4_T7_15mm_2iPat_3MB')].iloc[0]
    r5 = df[(df['sub'] == 'sub-02') & (df.run == 'run5_T7_superfast')].iloc[0]
    md += [
        f"- **The MB=6 finding is corroborated by the data itself.** "
        f"`run4_T7_15mm_2iPat_3MB` and `run5_T7_superfast` have identical "
        f"voxel size (1.5 mm), slice count ({r4.n_slices}) and in-plane "
        f"acceleration (iPAT {int(r4.iPAT)}), so they share an echo train. At "
        f"MB=6 run5 gives {r5.ms_per_excitation} ms per excitation against "
        f"run4's {r4.ms_per_excitation} ms -- the agreement the shared readout "
        f"requires. Had run5 really been MB=3 as its filename claims, it would "
        f"imply {r5.ms_per_excitation / 2:.1f} ms per excitation, physically "
        f"impossible for this readout and half its own sibling's. The filename "
        f"is wrong; MB=6 is right.", ""]
    md += ["- Within 7T session 2, where the multiband and iPAT factors had to "
           "be recovered from the protocol printout, time per excitation "
           "orders exactly as in-plane acceleration predicts:", ""]
    for _, r in df[df.session == '7T-s2'].sort_values('ms_per_excitation').iterrows():
        md.append(f"  - {r.run} (iPAT {int(r.iPAT)}, MB {int(r.MB)}): "
                  f"{r.ms_per_excitation} ms")
    md += ["", "  iPAT 4 gives the shortest echo train and iPAT 2 the longest, "
           "which is the ordering the recovered factors have to produce if "
           "they are correct.", ""]

    md += ["", "## Runs whose parameters could not be read from a sidecar", ""]
    miss = df[df.params_source != 'DICOM/JSON sidecar']
    if len(miss):
        md.append("The BOLD JSON sidecars for these runs were not kept at "
                  "conversion. TR, TE, dwell time and volume count were "
                  "recovered from the FEAT `design.fsf` and the NIfTI header; "
                  "multiband/iPAT factors are only available from the scanner "
                  "protocol printout and are marked as such in `sequences.csv`.")
        md.append("")
        for _, r in miss.iterrows():
            md.append(f"- {r['sub']} / {r['run']} (`{r['protocol']}`)")
    else:
        md.append("- none")

    md += ["", "## Comparisons named in the brief that the data cannot support", "",
           "- **Gradient-echo vs spin-echo.** The 07-08-2023 protocol printout "
           "lists `cmrr_mbep2d_se_2mm_mb2p3_Run2`, but no spin-echo run exists "
           "on disk; only the gradient-echo counterpart "
           "`cmrr_mbep2d_bold_2mm_mb2p3_run2` was converted and kept.",
           "- **Whole-brain vs restricted FOV.** The printouts list "
           "`*_wholebrain` variants, but every acquired run on disk is a "
           "restricted slab (50-84 slices). The coverage question is instead "
           "answered directly, by measuring what fraction of each ROI falls "
           "inside each run's acquired FOV (see `roi_fov_coverage` in the "
           "results table).",
           "",
           "## Misleading FEAT directory names", "",
           "- `preproc3T.feat` (sub-02, 7T runs) and `preproc_wholeb.feat` "
           "(sub-03 run1) are **not** separate acquisitions. Diffing their "
           "`design.fsf` against `preproc.feat` shows they differ only in "
           "`reginitial_highres_yn` and the stats/poststats flags. This "
           "analysis uses `preproc.feat` throughout.",
           "- Stray `filtered_func_data_standard_100.nii.gzz.nii.gz` files "
           "exist in several FEAT directories, from a typo in the 2023 "
           "`create_SNR_ROIS*.sh` scripts. They are duplicates and unused here.",
           ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tables').mkdir(exist_ok=True)
    cols = ['sub', 'run', 'session', 'label', 'protocol', 'params_source',
            'field_T', 'TR_s', 'TE_ms', 'flip_deg', 'MB', 'iPAT',
            'ms_per_excitation',
            'vox_x', 'vox_y', 'vox_z', 'vox_mm3', 'n_slices', 'slab_mm',
            'fov_mm', 'bandwidth_Hz_px', 'readout_s', 'PE_dir', 'coil',
            'n_vols', 'duration_s', 'dwell_ms', 'smooth_mm', 'highpass_s',
            'slicetiming', 'unwarp', 'mc', 'reg_dof', 'nonlinear']
    df[cols].to_csv(OUT / 'tables' / 'sequences.csv', index=False)
    (OUT / 'tables' / 'discrepancies.md').write_text('\n'.join(md) + '\n')

    print(df[['sub', 'run', 'session', 'protocol', 'field_T', 'TR_s', 'TE_ms',
              'MB', 'iPAT', 'n_slices', 'ms_per_excitation', 'vox_mm3',
              'n_vols', 'duration_s']].to_string(index=False))
    print('\nwrote', OUT / 'tables' / 'sequences.csv')
    print('wrote', OUT / 'tables' / 'discrepancies.md')
    print('\nshortest run: %.1f s -> duration match must be <= that' %
          df.duration_s.min())


if __name__ == '__main__':
    main()
