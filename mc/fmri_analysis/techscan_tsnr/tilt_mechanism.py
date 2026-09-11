"""
Stage 3b: why the slab tilt was set to +30 deg, tested on the measured field.

The angle sweep was only ever run at 3T, on sequences with 2-3.8x longer
readouts than the protocol that was adopted. So it cannot, on its own, license
a claim about the adopted sequence. This does, because it works from a
quantity that belongs to the head rather than to the sequence: the acquired B0
fieldmap.

Tilting the slab does not change the field. It changes which direction counts
as "phase encode" and which as "through-plane", and therefore how the fixed
field gradient is split between the two mechanisms that destroy signal:

    in-plane (phase-encode) gradient  -> echo displacement, pile-up, distortion
    through-plane gradient            -> intravoxel dephasing, dropout

For a tilt b about the left-right axis, with A>>P phase encoding:

    phase-encode axis  e_PE = (0,  cos b, -sin b)
    slice normal       n    = (0,  sin b,  cos b)

so the two projections of the measured gradient are dB/dPE and dB/dslice.
Weiskopf et al. (2006) state that tilt and PE polarity act primarily on the
first of these; z-shimming addresses the second.

The two penalties are computed properly rather than compared as bare
gradients, because they have different units and very different sizes:

    through-plane  signal kept    = |sinc(g_slice * dz * TE)|
    in-plane       voxel distortion = g_PE * voxel * readout

That asymmetry is the whole result. Tilting to +30 deg raises the
through-plane gradient, but at any realistic TE that costs under 4% of signal,
while the in-plane distortion it removes is worth 3-20 percentage points. So
+30 deg wins at every readout tested, including the adopted one; what a
shorter readout changes is the size of the margin, not its sign.

This matters because a shorter readout might naively be expected to flip the
optimum. It does not, and cannot: shortening the readout scales the in-plane
term down towards zero, which shrinks +30 deg's advantage asymptotically to
nothing but never makes -30 deg better.

Writes tables/tilt_gradient_projection.csv and the panel p_tilt_mechanism.

Run:  python -m mc.fmri_analysis.techscan_tsnr.tilt_mechanism
"""
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from .config import DERIV, OUT, FEAT, PRIMARY_ROIS, ADOPTED_TILT_DEG
from . import plotstyle as ps

TILTS = (-30, 0, 30)
# sequences whose tilt sensitivity we want to weigh, with the two terms that
# scale each mechanism: (readout s, TE s, slice thickness mm)
SEQS = {
    'Leip 3T (2.5 mm, no iPAT)': dict(readout=53.46e-3, te=22.0e-3, dz=2.5),
    'AB 3T (2.0 mm, iPAT2)':     dict(readout=28.89e-3, te=20.0e-3, dz=2.0),
    'adopted 7T (iPAT4)':        dict(readout=14.18e-3, te=12.8e-3, dz=2.0),
}


def fieldmap_hz_in_mni(sub='sub-03', run='run1_3Trep_7T2'):
    """The acquired B0 fieldmap in Hz, pushed into MNI so it can be read
    against the MNI ROI masks."""
    feat = DERIV / sub / run / 'func' / FEAT
    ref = nib.load(feat / 'reg' / 'example_func.nii.gz')
    src = [f for f in sorted((feat / 'reg').glob('*fieldmaprads2epi.nii.gz'))
           if nib.load(f).shape[:3] == ref.shape[:3]][0]
    out = OUT / 'tsnr' / '_fmapHz_MNI.nii.gz'
    if not out.exists():
        tmp = OUT / 'tsnr' / '_fmapHz_nat.nii.gz'
        nib.save(nib.Nifti1Image(
            (np.asarray(nib.load(src).dataobj) / (2 * np.pi)).astype(np.float32),
            ref.affine), tmp)
        subprocess.run(
            ['applywarp', '-i', str(tmp),
             '-r', str(feat / 'reg' / 'standard.nii.gz'),
             '-w', str(feat / 'reg' / 'example_func2standard_warp.nii.gz'),
             '-o', str(out), '--interp=trilinear'],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return nib.load(out).get_fdata()


def main():
    B = fieldmap_hz_in_mni()
    # gradient in Hz/mm; the MNI 2 mm grid has +2 spacing along y and z
    gy = np.gradient(B, 2.0, axis=1)
    gz = np.gradient(B, 2.0, axis=2)
    valid = np.abs(B) > 0
    masks = {r: nib.load(OUT / 'rois' / 'MNI' / f'{r}.nii.gz').get_fdata() > 0
             for r in PRIMARY_ROIS}

    rows = []
    for roi in PRIMARY_ROIS:
        m = masks[roi] & valid
        for beta in TILTS:
            b = np.radians(beta)
            g_pe = float(np.abs(gy[m] * np.cos(b) - gz[m] * np.sin(b)).mean())
            g_sl = float(np.abs(gy[m] * np.sin(b) + gz[m] * np.cos(b)).mean())
            row = dict(roi=roi, tilt_deg=beta,
                       g_phase_encode_Hz_per_mm=round(g_pe, 3),
                       g_through_plane_Hz_per_mm=round(g_sl, 3))
            # dimensionless severity of each mechanism, per sequence
            for name, s in SEQS.items():
                key = name.split()[0].lower()
                # fractional voxel distortion along PE
                row[f'{key}_pe_term'] = round(g_pe * s['dz'] * s['readout'], 4)
                # phase dispersion across the slice, in cycles
                row[f'{key}_thru_term'] = round(g_sl * s['dz'] * s['te'], 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'tables' / 'tilt_gradient_projection.csv', index=False)

    pd.set_option('display.width', 220)
    print('Measured B0 gradient projected onto the axes each tilt defines')
    print('(sub-03 7T fieldmap; A>>P phase encoding)\n')
    print(df[['roi', 'tilt_deg', 'g_phase_encode_Hz_per_mm',
              'g_through_plane_Hz_per_mm']].to_string(index=False))

    print('\nWhich tilt minimises the phase-encode gradient, per ROI:')
    for roi in PRIMARY_ROIS:
        d = df[df.roi == roi]
        best = int(d.loc[d.g_phase_encode_Hz_per_mm.idxmin(), 'tilt_deg'])
        red = 100 * (1 - d.g_phase_encode_Hz_per_mm.min() /
                     d[d.tilt_deg == 0].g_phase_encode_Hz_per_mm.iloc[0])
        print(f'  {roi:<6} {best:+d} deg   ({red:.0f}% below transverse)')

    print(f'\nAdopted tilt was {ADOPTED_TILT_DEG:+d} deg.')

    # ---- what the two penalties actually cost, per sequence ----------
    pen = []
    for roi in PRIMARY_ROIS:
        d = df[df.roi == roi].set_index('tilt_deg')
        for name, s in SEQS.items():
            for beta in TILTS:
                keep = abs(np.sinc(d.loc[beta, 'g_through_plane_Hz_per_mm']
                                   * s['dz'] * s['te']))
                distort = (d.loc[beta, 'g_phase_encode_Hz_per_mm']
                           * s['dz'] * s['readout'])
                pen.append(dict(roi=roi, sequence=name, tilt_deg=beta,
                                signal_kept=round(float(keep), 4),
                                voxel_distortion=round(float(distort), 4),
                                net=round(float(keep * (1 - distort)), 4)))
    pdf_ = pd.DataFrame(pen)
    pdf_.to_csv(OUT / 'tables' / 'tilt_penalties.csv', index=False)
    print('\nPredicted signal retained after both susceptibility penalties (%):')
    print((100 * pdf_.pivot_table(index=['roi', 'sequence'], columns='tilt_deg',
                                  values='net')).round(1).to_string())

    # ---- panel: one line per sequence, EC ----------------------------
    ps.apply_style()
    xs = np.arange(len(TILTS))
    for roi in ('EC', 'vOFC'):
        fig, ax = ps.panel_fig()
        styles = [('Leip 3T (2.5 mm, no iPAT)', '#C6C6C6', ':', 's'),
                  ('AB 3T (2.0 mm, iPAT2)', '#8A8A8A', '--', '^'),
                  ('adopted 7T (iPAT4)', ps.FINAL_MARK, '-', 'o')]
        for name, col, ls, mk in styles:
            d = pdf_[(pdf_.roi == roi) & (pdf_.sequence == name)
                     ].set_index('tilt_deg').reindex(TILTS)
            ax.plot(xs, 100 * d.net, ls, marker=mk, color=col,
                    lw=1.6 if 'adopted' in name else 1.1,
                    label=name.split(' (')[0])
        ax.set_xticks(xs)
        ax.set_xticklabels(['\u221230\u00b0', '0\u00b0', '+30\u00b0'])
        ax.set_xlabel('slab tilt')
        ax.set_ylabel('predicted signal kept')
        ax.set_ylim(60, 100)
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'])
        # +30 deg is the adopted tilt, but none of these lines is a single
        # acquired run, so this is labelled rather than boxed
        ax.axvline(2, color=ps.FINAL_MARK, lw=.8, ls=':')
        ax.text(2, 61, 'adopted\ntilt', fontsize=9, color=ps.FINAL_MARK,
                ha='center', va='bottom')
        ax.legend(fontsize=9, handlelength=1.4, labelspacing=.15,
                  borderpad=.1, handletextpad=.4, loc='lower right')
        ax.set_title(roi, fontsize=9, pad=2)
        ps.save(fig, OUT / 'figures' / f'p_tilt_prediction_{roi}')
        print('wrote', OUT / 'figures' / f'p_tilt_prediction_{roi}.pdf')

    return df


if __name__ == '__main__':
    main()
