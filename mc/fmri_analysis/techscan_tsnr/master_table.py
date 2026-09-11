"""
Stage 5: one lookup table per sequence.

For every pilot run: the acquired parameters, which decision step and which
figure panels it appears in, and its numeric results in all four ROIs. This is
the table to check any number quoted in the methods section against.

Two derived columns worth knowing about:

  total_accel = MB x iPAT
      The two acceleration factors are not independent knobs on tSNR: what
      costs signal is roughly their product, while how it is *split* between
      them sets TE and readout length. Sorting by this column shows that the
      sweep varied both at once, which is why the matched-total comparison
      exists.

  bold_sens = TE * exp(-TE / T2*)
      Relative BOLD signal change per unit activation, at a reference T2*.
      tSNR alone favours a short TE, but BOLD contrast peaks at TE = T2*, so a
      short TE only wins where T2* is short. Reported at T2* = 33 ms (cortex at
      7T) and 20 ms (a susceptibility-affected ventral region).

Writes tables/master_lookup.csv and tables/master_lookup.md.

Run:  python -m mc.fmri_analysis.techscan_tsnr.master_table
"""
import numpy as np
import pandas as pd

from .config import OUT, COMPARISONS, PRIMARY_ROIS, RUNS

# which panel files show which comparison
PANELS = {
    'field_3T_vs_7T':      ['p_field_bars', 'p_maps_field'],
    'voxel_7T_s2':         ['p_voxel_curve'],
    'voxel_7T_s1':         ['p_voxel_curve'],
    'angle_AB_3T':         ['p_angle'],
    'angle_Leip_3T':       ['(reported in text only)'],
    'multiband_7T_s1':     ['p_multiband'],
    'split_matched_7T_s2': ['p_split_matched', 'p_maps_split',
                            'p_distortion_maps'],
    'accel_7T_s2':         ['p_accel_tsnr', 'p_accel_change',
                            'p_distortion_measured', 'p_vofc_dropout',
                            'p_maps_accel'],
    'field_3T_vs_7T_final': ['(reported in text only)'],
}

T2_REF = {'cortex_33ms': 33.0, 'ventral_20ms': 20.0}


def main():
    seq = pd.read_csv(OUT / 'tables' / 'sequences.csv')
    tsnr = pd.read_csv(OUT / 'tables' / 'tsnr_roi.csv')
    common = pd.read_csv(OUT / 'tables' / 'tsnr_roi_common.csv')
    shift = pd.read_csv(OUT / 'tables' / 'shiftmaps.csv')
    cov = pd.read_csv(OUT / 'tables' / 'coverage_decomposed.csv')

    # which comparisons / steps / panels each run belongs to
    membership = {}
    for gname, g in COMPARISONS.items():
        for sub, run in g['runs']:
            m = membership.setdefault((sub, run), {'steps': set(),
                                                   'comparisons': [],
                                                   'panels': set()})
            m['steps'].add(g['step'])
            m['comparisons'].append(gname)
            m['panels'].update(PANELS.get(gname, []))

    rows = []
    for sub, run, session, label in RUNS:
        s = seq[(seq['sub'] == sub) & (seq.run == run)].iloc[0]
        sh = shift[(shift['sub'] == sub) & (shift.run == run)]
        m = membership.get((sub, run), {'steps': set(), 'comparisons': [],
                                        'panels': set()})

        row = dict(
            sub=sub, run=run, session=session,
            protocol=s.protocol,
            field_T=s.field_T, TR_s=s.TR_s, TE_ms=s.TE_ms, flip_deg=s.flip_deg,
            MB=s.MB, iPAT=s.iPAT,
            total_accel=(s.MB * s.iPAT) if not pd.isna(s.iPAT) else np.nan,
            voxel_mm=round(float(s.vox_x), 2), vox_mm3=s.vox_mm3,
            n_slices=s.n_slices, slab_mm=s.slab_mm,
            readout_ms=float(sh.readout_ms.iloc[0]) if len(sh) else np.nan,
            displacement_p99_mm=float(sh.p99_shift_mm.iloc[0]) if len(sh) else np.nan,
            n_vols_acquired=s.n_vols, duration_s=s.duration_s,
            decision_steps=','.join(str(x) for x in sorted(m['steps'])) or '-',
            comparisons=';'.join(m['comparisons']) or '-',
            figure_panels=';'.join(sorted(m['panels'])) or '-',
        )

        for t2name, t2 in T2_REF.items():
            row[f'bold_sens_{t2name}'] = round(
                float(s.TE_ms * np.exp(-s.TE_ms / t2)), 2)

        # results: native tSNR (all ROI voxels in FOV) and dropout
        for roi in PRIMARY_ROIS:
            r = tsnr[(tsnr['sub'] == sub) & (tsnr.run == run) & (tsnr.roi == roi)]
            c = cov[(cov['sub'] == sub) & (cov.run == run) & (cov.roi == roi)]
            row[f'tsnr_{roi}'] = float(r.tsnr_native_mean.iloc[0]) if len(r) else np.nan
            row[f'eff_{roi}'] = float(r.tsnr_efficiency.iloc[0]) if len(r) else np.nan
            row[f'dropout_{roi}'] = (round(float(c.dropout_within_slab.iloc[0]), 3)
                                     if len(c) else np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'tables' / 'master_lookup.csv', index=False)

    # --- readable markdown version ------------------------------------
    md = ['# Sequence lookup table', '',
          'Every pilot run: acquired parameters, where it appears in the '
          'argument, and its results. `tsnr_*` are native-space means over all '
          'ROI voxels inside that run\'s field of view; the figures use the '
          'common-coverage restriction instead, so they can differ slightly '
          'from these -- `tsnr_roi_common.csv` holds those.', '',
          '`total_accel` is MB x iPAT: roughly what sets the tSNR penalty, '
          'while the split between the two sets TE and readout length.', '']

    show = ['sub', 'run', 'field_T', 'voxel_mm', 'TR_s', 'TE_ms', 'MB', 'iPAT',
            'total_accel', 'readout_ms', 'displacement_p99_mm',
            'decision_steps', 'figure_panels']
    md.append('## Parameters and where each run is used')
    md.append('')
    md.append('| ' + ' | '.join(show) + ' |')
    md.append('|' + '|'.join(['---'] * len(show)) + '|')
    for _, r in df.iterrows():
        md.append('| ' + ' | '.join(str(r[c]) for c in show) + ' |')

    md += ['', '## Results, native-space tSNR by ROI', '']
    res = ['sub', 'run'] + [f'tsnr_{r}' for r in PRIMARY_ROIS] + \
          [f'dropout_{r}' for r in PRIMARY_ROIS]
    md.append('| ' + ' | '.join(res) + ' |')
    md.append('|' + '|'.join(['---'] * len(res)) + '|')
    for _, r in df.iterrows():
        md.append('| ' + ' | '.join(str(r[c]) for c in res) + ' |')

    md += ['', '## The decision sequence', '']
    for gname, g in sorted(COMPARISONS.items(), key=lambda kv: kv[1]['step']):
        md.append(f"**Step {g['step']} -- {g['title']}** (`{gname}`)  ")
        md.append(f"*{g['question']}*  ")
        md.append(f"Verdict: {g['verdict']}  ")
        md.append(f"Panels: {', '.join(PANELS.get(gname, ['-']))}  ")
        md.append(f"Caveat: {g['note']}")
        md.append('')

    (OUT / 'tables' / 'master_lookup.md').write_text('\n'.join(md) + '\n')

    pd.set_option('display.width', 250)
    print(df[show].to_string(index=False))
    print('\nwrote', OUT / 'tables' / 'master_lookup.csv')
    print('wrote', OUT / 'tables' / 'master_lookup.md')
    return df


if __name__ == '__main__':
    main()
