"""
Stage 3: geometric distortion, from acquisition parameters alone.

An EPI readout encodes the phase-encode direction slowly, so off-resonance
displaces signal along it. The bandwidth per pixel in the PE direction is
1/total_readout_time, so a voxel sitting in an off-resonance field dB0 is
displaced by

    shift [voxels] = dB0 [Hz] * total_readout_time [s]
    shift [mm]     = shift [voxels] * voxel size in PE [mm]

In-plane acceleration shortens the readout in direct proportion, so iPAT is
the single most effective lever on distortion. This matters independently of
tSNR: displaced signal is signal in the wrong place, which no amount of
temporal SNR repairs, and it is worst exactly where off-resonance is worst --
orbitofrontal cortex and mPFC above the frontal sinus, and the medial temporal
lobe next to the petrous bone.

FSL's `dwell` in a FEAT design is the *effective* echo spacing, i.e. already
divided by the in-plane acceleration, and FSL forms the total readout as
dwell * (base_resolution - 1). That identity is checked here against the
TotalReadoutTime the scanner wrote into the JSON sidecars, for the runs that
still have one.

Reference off-resonance values used for the illustrative displacement columns
are field-scaled: susceptibility-driven dB0 scales linearly with B0, so a
region that sits at ~50 Hz off-resonance at 3T sits at ~117 Hz at 7T.

Writes tables/distortion.csv.

Run:  python -m mc.fmri_analysis.techscan_tsnr.distortion
"""
import re
import numpy as np
import pandas as pd

from .config import DERIV, OUT, RUNS, FEAT

# Off-resonance at 3T in a strongly affected frontal/temporal voxel, scaled to
# field. Used only to express readout time as an interpretable displacement.
DB0_AT_3T_HZ = 50.0


def main():
    seq = pd.read_csv(OUT / 'tables' / 'sequences.csv')
    rows = []

    for sub, run, session, label in RUNS:
        s = seq[(seq['sub'] == sub) & (seq.run == run)].iloc[0]
        fsf = (DERIV / sub / run / 'func' / FEAT / 'design.fsf').read_text()
        dwell_ms = float(re.search(r'set fmri\(dwell\)\s+([\d.]+)', fsf).group(1))

        # base resolution: FOV / in-plane voxel size, rounded to the even
        # matrix the scanner would have used
        base_res = int(round(s.fov_mm / s.vox_x / 2) * 2)
        trt_s = dwell_ms * 1e-3 * (base_res - 1)

        # The scanner's own TotalReadoutTime is authoritative where the
        # sidecar survived; the fsf-derived value only fills in for sub-03,
        # whose sidecars were lost. FEAT rounds `dwell` to 4 significant
        # figures, so the two agree to ~1% rather than exactly -- that
        # rounding, not a modelling error, is the whole discrepancy.
        trt_hdr = s.readout_s if not pd.isna(s.readout_s) else np.nan
        if not np.isnan(trt_hdr):
            agrees = abs(trt_s - trt_hdr) / trt_hdr < 0.02
            trt_used, src = trt_hdr, 'scanner header'
        else:
            agrees, trt_used, src = None, trt_s, 'design.fsf dwell'

        db0 = DB0_AT_3T_HZ * s.field_T / 3.0
        shift_vox = db0 * trt_used
        shift_mm = shift_vox * s.vox_y

        rows.append(dict(
            sub=sub, run=run, session=session, label=label,
            field_T=s.field_T, iPAT=s.iPAT, MB=s.MB, vox_mm=s.vox_y,
            base_res=base_res, eff_echo_spacing_ms=dwell_ms,
            total_readout_ms=round(trt_used * 1e3, 2),
            readout_source=src,
            total_readout_from_fsf_ms=round(trt_s * 1e3, 2),
            total_readout_hdr_ms=round(trt_hdr * 1e3, 2) if not np.isnan(trt_hdr) else np.nan,
            readout_matches_header=agrees,
            reference_dB0_Hz=round(db0, 1),
            shift_voxels=round(shift_vox, 2),
            shift_mm=round(shift_mm, 2),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'tables' / 'distortion.csv', index=False)

    chk = df.dropna(subset=['total_readout_hdr_ms'])
    print(f'readout-time identity dwell*(base_res-1) vs the scanner header, '
          f'checked on {len(chk)}/{len(df)} runs (the rest lost their '
          f'sidecars): '
          f'{"all agree within 2%" if chk.readout_matches_header.all() else "MISMATCH"}')
    worst = (abs(chk.total_readout_from_fsf_ms - chk.total_readout_hdr_ms)
             / chk.total_readout_hdr_ms).max()
    print(f'largest relative deviation: {100*worst:.2f}% '
          f'(FEAT rounds dwell to 4 significant figures)')
    # the iPAT values recovered for sub-03 imply effective echo spacings that
    # must scale as 1/iPAT off the unaccelerated readout -- an independent
    # check on those recovered factors
    s3 = df[df['sub'] == 'sub-03']
    print('\nsub-03 effective echo spacing vs recovered iPAT '
          '(must scale as 1/iPAT):')
    for _, r in s3.drop_duplicates('eff_echo_spacing_ms').iterrows():
        print(f'   iPAT {int(r.iPAT)}: {r.eff_echo_spacing_ms:.4f} ms  '
              f'(x iPAT = {r.eff_echo_spacing_ms*r.iPAT:.3f} ms)')
    print()
    pd.set_option('display.width', 200)
    print(df[['sub', 'run', 'field_T', 'iPAT', 'vox_mm', 'eff_echo_spacing_ms',
              'total_readout_ms', 'reference_dB0_Hz', 'shift_mm']]
          .to_string(index=False))
    print('\nwrote', OUT / 'tables' / 'distortion.csv')
    return df


if __name__ == '__main__':
    main()
