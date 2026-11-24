#!/usr/bin/env python3
"""Add ROI cell counts and RSA eligibility checks to an iEEG layout report."""
import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROIS = ['EC', 'HC_anterior', 'HC_mid', 'mPFC', 'mOFC', 'PCC']
MARKER = '\n\nCELL COVERAGE AND RSA SAMPLE AUDIT'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--coverage-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path(
        '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'))
    args = parser.parse_args()
    root, out = args.data_root, args.report_dir
    derivatives = root / 'derivatives'
    spec = importlib.util.spec_from_file_location(
        '_layout_cell_selection', Path(__file__).resolve().parents[1]
        / 'mc/analyse/cell_selection.py')
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    meta = pd.read_csv(out / 'recording_sessions.csv')
    meta['subject_int'] = meta.session.str[1:].astype(int)
    blocks = pd.read_csv(out / 'blocks.csv')
    report = json.loads((out / 'layout_summary.json').read_text())
    canonical = set(report['canonical_fixed_layouts'])
    rsa = {int(s) for s in cs.load_rsa_subjects(data_dir=str(derivatives))}
    canonical_sessions = set(meta.loc[meta.family.eq(
        'canonical_eight_ordered_layouts'), 'subject_int'])
    plotted = set(json.loads((args.coverage_dir / 'config.json').read_text())[
        'rsa_sessions'])
    assert rsa == canonical_sessions == plotted
    assert len(rsa) == 28
    grouping = json.loads((derivatives /
        'all_sessions_dsrRSA_grouping_summary.json').read_text())
    for sid, entries in grouping.items():
        actual = blocks[blocks.session.eq(f's{int(sid):02d}')]
        assert set(actual.layout) == {e['config'] for e in entries} == canonical
        for entry in entries:
            eligible_blocks = set(actual.loc[actual.layout.eq(entry['config']), 'grid'])
            assert set(entry['blocks']) <= eligible_blocks
            assert not set(entry['run1_blocks']) & set(entry['run2_blocks'])

    cells = cs.load_cells(data_dir=str(derivatives), include_unassigned=True)
    assert not cells.duplicated(['subject_int', 'cell_idx']).any()
    cells = cells.merge(meta, on='subject_int', validate='many_to_one')
    assert set(cells.roi.dropna()) == set(ROIS)
    in_rsa = cells.subject_int.isin(rsa)
    support = cells[in_rsa & cells.roi.notna()].groupby('roi').subject_int.nunique()
    retained_rois = support[support.ge(3)].index.tolist()
    analysed = cells[in_rsa & cells.roi.isin(retained_rois)]
    expected_ids = set(zip(analysed.roi, analysed.subject_int, analysed.cell_idx))
    saved_runs = {}
    for tag in ['2026-08-27_19-18-20', '2026-10-02_18-02-13']:
        directory = derivatives / 'group/DSR_RSA_simple_ROI' / tag
        coords = pd.read_csv(directory / 'roi_electrode_coords.csv')
        actual_ids = set(zip(coords.roi, coords.subject, coords.cell_idx))
        assert actual_ids == expected_ids
        results = pd.read_csv(directory / 'results_summary.csv')
        result_counts = results[['roi', 'n_neurons']].drop_duplicates().set_index(
            'roi').n_neurons.to_dict()
        assert result_counts == analysed.roi.value_counts().to_dict()
        saved_runs[tag] = {'cell_id_and_roi_memberships_match': True,
                           'n_cells': len(actual_ids)}

    sample_sessions = {
        'all_in_roi_table': set(meta.subject_int),
        'rsa': rsa,
        'not_in_rsa': set(meta.subject_int) - rsa,
    }
    for family, group in meta.groupby('family'):
        sample_sessions[family] = set(group.subject_int)
    sample_sessions['rsa_analysed_min3_sessions'] = rsa
    rows, totals = [], []
    recomputed_reference = {}
    for sample, sessions in sample_sessions.items():
        selected = cells[cells.subject_int.isin(sessions)]
        if sample == 'rsa_analysed_min3_sessions':
            selected = selected[selected.roi.isin(retained_rois)]
        assigned = selected[selected.roi.notna()]
        denominator = meta[meta.subject_int.isin(sessions)]
        totals.append({
            'sample': sample, 'eligible_patients': denominator.patient.nunique(),
            'eligible_sessions': len(denominator), 'cells_in_registry': len(selected),
            'assigned_cells': len(assigned),
            'assigned_sessions': assigned.subject_int.nunique(),
            'assigned_patients': assigned.patient.nunique(),
            'unassigned_cells': int(selected.roi.isna().sum()),
        })
        ref = {}
        for roi in ROIS + ['UNASSIGNED', 'TOTAL_ASSIGNED']:
            group = (selected[selected.roi.isna()] if roi == 'UNASSIGNED'
                     else assigned if roi == 'TOTAL_ASSIGNED'
                     else selected[selected.roi.eq(roi)])
            sites = group[['MNI_x', 'MNI_y', 'MNI_z']].dropna().drop_duplicates()
            row = {'sample': sample, 'roi': roi, 'cells': len(group),
                   'sessions': group.subject_int.nunique(),
                   'patients': group.patient.nunique(), 'sites': len(sites)}
            rows.append(row)
            if roi in ROIS:
                ref[roi] = {k: int(row[k]) for k in ['cells', 'sites', 'sessions']}
        ref['total'] = {'cells': len(assigned),
                        'sites': sum(v['sites'] for v in ref.values()),
                        'sessions': assigned.subject_int.nunique()}
        recomputed_reference[sample] = ref
    reference = json.loads((args.coverage_dir / 'coverage_summary.json').read_text())
    for sample in ['all_in_roi_table', 'rsa', 'not_in_rsa']:
        assert recomputed_reference[sample] == reference[sample], sample
    table = pd.DataFrame(rows)
    table.to_csv(out / 'cells_by_sample_and_roi.csv', index=False)
    pd.DataFrame(totals).to_csv(out / 'cell_sample_totals.csv', index=False)
    analysed.to_csv(out / 'verified_rsa_cells.csv', index=False)
    per_session = cells.assign(roi=cells.roi.fillna('UNASSIGNED')).groupby(
        ['session', 'patient', 'family', 'roi']).size().unstack('roi', fill_value=0)
    per_session['rsa_eligible'] = per_session.index.get_level_values('session').isin(
        {f's{s:02d}' for s in rsa})
    per_session.to_csv(out / 'cell_sample_session_allocation.csv')

    # Verify actual reward order in one successful attempt per alternative block.
    alternatives = meta[meta.family.eq('same_eight_location_sets_other_orders')]
    canonical_by_set = {tuple(sorted(map(int, c.split('-')))): c for c in canonical}
    order_rows, checks = [], []
    with h5py.File(root / 'abcd_data_08-Sep-2025.mat', 'r') as handle:
        for session in alternatives.itertuples():
            sid = session.subject_int
            raw = pd.read_csv(derivatives / session.session / 'cells_and_beh'
                              / f'all_trial_times_{sid:02d}.csv', header=None)
            trials = handle[handle['abcd_data']['trial_vars'][sid-1, 0]]
            for grid, group in raw.groupby(12, sort=False):
                good = group[group[13].eq(1) & group[4].notna()]
                if good.empty:
                    continue
                index = good.index[0]
                correct = np.asarray(handle[trials['correct_uncover_c'][index, 0]]).ravel()
                locations = np.asarray(handle[trials['visited_locations_c'][index, 0]]).ravel()
                actual_order = locations[correct == 1].astype(int).tolist()
                stored_order = good.loc[index, list(range(5, 9))].astype(int).tolist()
                assert actual_order == stored_order
                checks.append({'session': session.session, 'grid': int(grid),
                               'trial_index': int(index), 'raw_order_matches': True})
            for layout in blocks.loc[blocks.session.eq(session.session), 'layout'].unique():
                sequence = list(map(int, layout.split('-')))
                standard = canonical_by_set[tuple(sorted(sequence))]
                state_by_location = dict(zip(map(int, standard.split('-')), 'ABCD'))
                order_rows.append({
                    'session': session.session, 'patient': session.patient,
                    'canonical_order': standard, 'actual_order': layout,
                    'actual_order_in_canonical_states': '-'.join(
                        state_by_location[x] for x in sequence)})
    pd.DataFrame(order_rows).to_csv(out / 'alternative_reward_orders.csv', index=False)
    pd.DataFrame(checks).to_csv(out / 'alternative_order_raw_event_checks.csv', index=False)
    summary = {
        'roi_table': str(derivatives / 'neurons_with_ROI_labels.csv'),
        'roi_column': 'alt_final_roi', 'coordinate_columns': 'MNI_*_final',
        'reference_coverage_directory': str(args.coverage_dir),
        'reference_all_three_samples_reproduced_exactly': True,
        'rsa_sessions_match_canonical_layout_family': True,
        'rsa_eligible_sessions': sorted(rsa),
        'rsa_eligible_patients': int(meta[meta.subject_int.isin(rsa)].patient.nunique()),
        'rsa_retained_rois': retained_rois,
        'rsa_analysis_cells': len(analysed),
        'rsa_analysis_sessions': int(analysed.subject_int.nunique()),
        'rsa_analysis_patients': int(analysed.patient.nunique()),
        'eligible_sessions_without_assigned_roi_cells': sorted(
            rsa - set(cells.loc[in_rsa & cells.roi.notna(), 'subject_int'])),
        'actual_saved_rsa_results': saved_runs,
        'alternative_order_patients': int(alternatives.patient.nunique()),
        'alternative_order_sessions': alternatives.session.tolist(),
        'alternative_blocks_verified_against_raw_reward_events': len(checks),
        'sample_totals': totals,
    }
    report['cell_coverage'] = summary
    (out / 'layout_summary.json').write_text(json.dumps(report, indent=2))
    (out / 'cell_sample_audit.json').write_text(json.dumps(summary, indent=2))
    section = [MARKER.strip(), '',
        'The plotted RSA sample is exactly the canonical 18-patient / 28-session cohort.',
        'None of the alternative-order sessions is included. All three coverage-figure',
        'samples reproduce exactly, including cells, sites and session counts.', '',
        'Counts below are cells (recording sessions; patients).', '']
    display_samples = ['earlier_six_layout_version', 'canonical_eight_ordered_layouts',
                       'same_eight_location_sets_other_orders', 'all_in_roi_table',
                       'not_in_rsa', 'rsa_analysed_min3_sessions']
    for sample in display_samples:
        section.append(sample)
        for row in table[table['sample'].eq(sample)].itertuples():
            section.append(f'  {row.roi:16s} {row.cells:4d} ({row.sessions:2d}; {row.patients:2d})')
        section.append('')
    section.extend([
        'The RSA-eligible cohort contains 509 assigned cells, including 3 EC cells.',
        'The five ROIs supported by >=3 sessions contain 506 cells from 27 sessions',
        'and 17 patients. EC has only one supporting session and is excluded by',
        'MIN_RSA_SUBJECTS=3. Session s50 (the only session of patient UC0576) has',
        '16 registry cells but none assigned to a retained alt_final_roi, explaining',
        'the reduction from 28 eligible to 27 contributing sessions.',
        'The 506 cell identities and their ROI memberships exactly match saved RSA',
        'outputs dated 2026-08-27 and 2026-10-02; this is not only a table count.', '',
        'The six alternative-order patients are YEV, UT202314, UT202409, UT202413,',
        'UT202421 and UT202418 (sessions s26,s29,s30,s39,s41,s42,s52,s54).',
        'For example canonical 3-7-9-5 corresponds to actual 3-5-7-9 in seven of',
        'these sessions; s26 uses other permutations. They share physical target',
        'sets, not ordered A-D conditions. Original correct reward-uncover events',
        f'confirm the stored sequence order in {len(checks)} successful blocks.',
        'They remain outside the configuration-aligned RSA pseudo-populations.', '',
        'Suggested wording:',
        'Pseudo-populations were constructed from the 28 recording sessions of',
        '18 patients who performed the same eight ordered reward layouts. The',
        'five ROIs represented in at least three sessions contained 506 neurons',
        'from 27 sessions (17 patients): anterior hippocampus, 171 neurons from',
        '26 sessions; mid hippocampus, 145 from 19; medial orbitofrontal cortex,',
        '74 from 10; medial prefrontal cortex, 65 from 15; and posterior cingulate',
        'cortex, 51 from six. Entorhinal cortex contributed three neurons from one',
        'session and was excluded by the minimum-session criterion.', '',
        'Files: cells_by_sample_and_roi.csv, cell_sample_totals.csv,',
        'cell_sample_session_allocation.csv, verified_rsa_cells.csv,',
        'alternative_reward_orders.csv, alternative_order_raw_event_checks.csv,',
        'cell_sample_audit.json. Reproduce with scripts/audit_ieeg_cell_samples.py.',
    ])
    readme = out / 'README.txt'
    previous = readme.read_text().split(MARKER)[0] if readme.exists() else ''
    readme.write_text(previous.rstrip() + '\n\n' + '\n'.join(section) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
