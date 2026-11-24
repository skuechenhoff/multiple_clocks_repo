#!/usr/bin/env python3
"""Audit iEEG layouts and blocks from raw session metadata and behavioural CSVs.

A layout is the ordered (A, B, C, D) reward-location tuple. A block is one
(session entry, session_no, grid_no). We report presented blocks, blocks with
at least one complete correct repeat, and blocks reaching ten correct repeats.
Session entries s01-s63 can contain several behavioural runs (session_no).
"""
import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

COLUMNS = ['rep_correct', 't_A', 't_B', 't_C', 't_D', 'loc_A', 'loc_B',
           'loc_C', 'loc_D', 'rep_overall', 'new_grid_onset', 'session_no',
           'grid_no', 'correct']


def read_mat(handle, ref):
    array = np.asarray(handle[ref])
    if array.dtype == np.uint16:
        return ''.join(chr(int(x)) for x in array.ravel())
    return array.ravel().tolist()


def distribution(values):
    return {int(k): int(v) for k, v in values.value_counts().sort_index().items()}


def summarize(blocks):
    per_session = blocks.groupby('session').layout.nunique()
    per_patient = blocks.groupby('patient').layout.nunique()
    per_run = blocks.groupby(['session', 'run']).layout.nunique()
    repeat_counts = blocks.groupby(['session', 'layout']).size()
    run_counts = blocks.groupby(['session', 'run', 'layout']).size()
    return {
        'n_patients': int(blocks.patient.nunique()),
        'n_recording_sessions': int(blocks.session.nunique()),
        'n_behavioural_runs': int(blocks.groupby(['session', 'run']).ngroups),
        'n_blocks': int(len(blocks)),
        'n_distinct_ordered_layouts_global': int(blocks.layout.nunique()),
        'n_distinct_unordered_location_sets_global': len({
            tuple(sorted(map(int, value.split('-')))) for value in blocks.layout}),
        'layouts_per_recording_session': distribution(per_session),
        'layouts_per_patient_across_sessions': distribution(per_patient),
        'layouts_per_behavioural_run': distribution(per_run),
        'blocks_per_layout_per_recording_session': distribution(repeat_counts),
        'blocks_per_layout_per_behavioural_run': distribution(run_counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(
        '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'))
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        '_layout_swr_io', Path(__file__).resolve().parents[1] / 'mc/analyse/swr_io.py')
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    metadata, frames, settings = [], [], []
    with h5py.File(root / 'abcd_data_08-Sep-2025.mat', 'r') as handle:
        group = handle['abcd_data']
        for i in range(group['subject_ID'].shape[0]):
            session = f's{i+1:02d}'
            label = read_mat(handle, group['subject_ID'][i, 0])
            patient = io.normalise_subject_key(label)
            fixed = int(read_mat(handle, group['fixed_grids'][i, 0])[0])
            metadata.append({'session': session, 'patient': patient,
                             'subject_label': label, 'processed_fixed_flag': fixed})
            folder = root / 'derivatives' / session / 'cells_and_beh'
            data = pd.read_csv(folder / f'all_trial_times_{i+1:02d}.csv',
                               header=None, names=COLUMNS)
            records = []
            for (run, grid), block in data.groupby(['session_no', 'grid_no'], sort=False):
                configs = block[['loc_A', 'loc_B', 'loc_C', 'loc_D']].drop_duplicates()
                assert len(configs) == 1, (session, run, grid)
                locations = configs.iloc[0].astype(int).tolist()
                correct = block.correct.eq(1) & block.t_D.notna()
                records.append({
                    'session': session, 'patient': patient, 'run': int(run),
                    'grid': int(grid), 'layout': '-'.join(map(str, locations)),
                    'processed_fixed_flag': fixed,
                    'n_complete_correct_repeats': int(correct.sum()),
                    'solved': bool(correct.any()),
                    'completed': bool(correct.sum() >= 10),
                })
            saved = np.atleast_2d(np.genfromtxt(
                folder / f'all_configs_sub{i+1:02d}.csv', delimiter=',')).astype(int)
            actual = np.array([list(map(int, row['layout'].split('-'))) for row in records])
            np.testing.assert_array_equal(saved, actual)
            frames.append(pd.DataFrame(records))
            detail = handle[group['sessionDetails'][i, 0]]
            if isinstance(detail, h5py.Group):
                n_runs = len(data.session_no.unique())
                for key in ['num_grids', 'num_grid_reps', 'fixed_grids', 'randomize_states']:
                    if key not in detail:
                        continue
                    value = detail[key]
                    if h5py.check_dtype(ref=value.dtype):
                        values = [read_mat(handle, r) for r in value[:].ravel()]
                    else:
                        values = [np.asarray(value).ravel().tolist()]
                    settings.append({'session': session, 'field': key,
                                     'values': json.dumps(values),
                                     'n_behavioural_runs': n_runs})
    blocks = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(metadata)
    canonical = set(blocks.loc[blocks.processed_fixed_flag.eq(1), 'layout'])
    assert len(canonical) == 8
    canonical_sets = {tuple(sorted(map(int, x.split('-')))) for x in canonical}
    families = {}
    for session, block in blocks.groupby('session'):
        actual = set(block.layout)
        unordered = {tuple(sorted(map(int, x.split('-')))) for x in actual}
        if actual == canonical:
            family = 'canonical_eight_ordered_layouts'
        elif unordered == canonical_sets:
            family = 'same_eight_location_sets_other_orders'
        else:
            family = 'earlier_six_layout_version'
        families[session] = family
    blocks['family'] = blocks.session.map(families)
    meta['family'] = meta.session.map(families)
    assert meta.groupby('patient').family.nunique().eq(1).all()
    # Independent patient mapping already used by the demographics analysis.
    demo = pd.read_csv(root / 'derivatives/demographics_2026-10-10/patient_demographics.csv')
    expected = {s: row.patient for _, row in demo.iterrows() for s in row.sessions.split(';')}
    assert all(expected[row.session] == row.patient for row in meta.itertuples())
    report = {
        'sources': ['abcd_data_08-Sep-2025.mat',
                    'derivatives/s*/cells_and_beh/all_trial_times_*.csv',
                    'derivatives/s*/cells_and_beh/all_configs_sub*.csv'],
        'definitions': {
            'patient': 'normalise_subject_key from mc/analyse/swr_io.py',
            'recording_session': 'one entry of abcd_data (s01-s63)',
            'behavioural_run': 'session_no within a recording-session entry',
            'layout': 'ordered A-B-C-D location tuple',
            'solved_block': 'at least one correct repeat with a finite D timestamp',
            'completed_block': 'at least ten such correct repeats (task criterion)',
        },
        'selections': {}, 'groups': {},
        'canonical_fixed_layouts': sorted(canonical),
        'canonical_unique_locations_per_sequence_position': {
            state: len({int(x.split('-')[i]) for x in canonical})
            for i, state in enumerate('ABCD')},
    }
    for name, selected in [('presented', blocks),
                           ('at_least_one_correct_repeat', blocks[blocks.solved]),
                           ('ten_correct_repeats', blocks[blocks.completed])]:
        report['selections'][name] = summarize(selected)
    for family, selected in blocks.groupby('family'):
        report['groups'][family] = {
            'presented': summarize(selected),
            'ten_correct_repeats': summarize(selected[selected.completed]),
            'patients': sorted(selected.patient.unique().tolist()),
            'sessions': sorted(selected.session.unique().tolist()),
        }
    exceptions = (blocks[blocks.completed].groupby(['session', 'patient', 'layout'])
                  .size().rename('n_completed_blocks').reset_index())
    report['more_than_four_completed_blocks_per_layout'] = exceptions[
        exceptions.n_completed_blocks.gt(4)].to_dict('records')
    report['processed_fixed_flag_counts'] = meta.groupby('processed_fixed_flag').agg(
        patients=('patient', 'nunique'), sessions=('session', 'size')).to_dict('index')
    report['note'] = (
        'The processed fixed_grids flag identifies canonical ordered layouts; '
        'it is not equivalent to the original task setting. Eight sessions '
        'from six other patients used the same eight unordered location sets '
        'with different sequence orders. Earlier six-layout sessions also '
        'share layouts across patients, so non-canonical does not imply '
        'independently randomized layouts.')
    blocks.to_csv(out / 'blocks.csv', index=False)
    meta.to_csv(out / 'recording_sessions.csv', index=False)
    pd.DataFrame(settings).to_csv(out / 'task_settings.csv', index=False)
    patient_summary = blocks.groupby(['patient', 'family']).agg(
        recording_sessions=('session', 'nunique'),
        distinct_layouts=('layout', 'nunique'), blocks_presented=('grid', 'size'),
        blocks_solved=('solved', 'sum'), blocks_completed=('completed', 'sum'))
    patient_summary.to_csv(out / 'patients.csv')
    layout_counts = blocks.groupby(['family', 'layout']).agg(
        patients=('patient', 'nunique'), recording_sessions=('session', 'nunique'),
        blocks_presented=('grid', 'size'), blocks_completed=('completed', 'sum'))
    layout_counts.to_csv(out / 'layouts.csv')
    with open(out / 'layout_summary.json', 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
