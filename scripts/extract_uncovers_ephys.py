#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-attempt uncover counts for the human sEEG (cells) dataset.

``all_trial_times_<sub>.csv`` (written by ``extract_behaviour_for_LFP.m``)
carries only the binary ``trial_correct`` flag, which says whether an ABCD
attempt contained any incorrect uncover — but not how many.  The number of
incorrect uncovers is the most direct behavioural read-out of "the subject
knew where the reward was", so we pull it out of the raw session struct here.

Per move the raw data stores
    pressed_to_uncover : 1 if the subject uncovered this location
    correct_uncover    : 1 correct, 0 incorrect, NaN if nothing was uncovered
so an incorrect uncover is ``pressed_to_uncover == 1 & correct_uncover == 0``.

Also extracted is ``grid_id``, the ID of the *unique* layout.  ``grid_num``
counts runs, and the same layout can be run several times per session, so
``grid_id`` is what identifies a repeated layout.

Rows are written in the same order as ``all_trial_times_<sub>.csv``, and the
alignment is verified against it (grid, attempt index, correctness) before
anything is saved.

Output: data/ephys_humans/derivatives/group/ephys_uncovers_per_attempt.csv

@author: Svenja Kuchenhoff
"""

import os

import h5py
import numpy as np
import pandas as pd


DATA_ROOT   = '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
EPHYS_ROOT  = os.path.join(DATA_ROOT, 'ephys_humans')
RAW_MAT     = os.path.join(EPHYS_ROOT, 'abcd_data_08-Sep-2025.mat')
EPHYS_DERIV = os.path.join(EPHYS_ROOT, 'derivatives')
OUT_DIR     = os.path.join(EPHYS_DERIV, 'group')
OUT_CSV     = os.path.join(OUT_DIR, 'ephys_uncovers_per_attempt.csv')
FLAGGED_CSV = os.path.join(OUT_DIR, 'ephys_uncover_flag_discrepancies.csv')

# Same columns as in behaviour_summary.py.
BEH_COLS = ['rep_correct', 't_A', 't_B', 't_C', 't_D',
            'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall',
            'new_grid_onset', 'session_no', 'grid_no', 'correct']


def _scalar(handle, trials, name, index):
    """Read a per-trial scalar field of the MATLAB trial struct."""
    return float(np.asarray(handle[trials[name][index, 0]][()]).ravel()[0])


def _vector(handle, trials, name, index):
    """Read a per-move vector field of the MATLAB trial struct."""
    return np.asarray(handle[trials[name][index, 0]][()],
                      dtype=float).ravel()


def uncovers_for_subject(handle, trials):
    """One row per attempt: uncover counts, steps, layout identity."""
    rows = []
    for index in range(trials['grid_num'].shape[0]):
        pressed = _vector(handle, trials, 'pressed_to_uncover', index)
        correct_uncover = _vector(handle, trials, 'correct_uncover', index)
        is_uncover = pressed == 1
        rows.append({
            'session_no':  int(_scalar(handle, trials, 'session_num', index)),
            'grid_no':     int(_scalar(handle, trials, 'grid_num', index)),
            'grid_id':     int(_scalar(handle, trials, 'grid_id', index)),
            'rep_overall': int(_scalar(handle, trials,
                                       'trial_num_in_grid', index)),
            'correct':     int(_scalar(handle, trials,
                                       'trial_correct', index)),
            'n_steps_total':       int(_scalar(handle, trials,
                                               'num_steps', index)),
            'n_uncovers':          int(is_uncover.sum()),
            'n_correct_uncovers':  int(np.sum(is_uncover
                                              & (correct_uncover == 1))),
            'n_incorrect_uncovers': int(np.sum(is_uncover
                                               & (correct_uncover == 0))),
        })
    return pd.DataFrame(rows)


def check_alignment(uncovers, subject):
    """Verify row-by-row agreement with the already-extracted timing table."""
    path = os.path.join(EPHYS_DERIV, f's{subject}', 'cells_and_beh',
                        f'all_trial_times_{subject}.csv')
    if not os.path.isfile(path):
        return f'no all_trial_times file'
    timing = pd.read_csv(path, header=None)
    if timing.shape[1] != len(BEH_COLS):
        return f'unexpected n_cols={timing.shape[1]}'
    timing.columns = BEH_COLS
    if len(timing) != len(uncovers):
        return f'{len(timing)} timing rows vs {len(uncovers)} raw trials'
    for column in ('session_no', 'grid_no', 'rep_overall', 'correct'):
        mismatch = int((timing[column].to_numpy()
                        != uncovers[column].to_numpy()).sum())
        if mismatch:
            return f'{mismatch} rows disagree on {column}'
    return None


def flag_discrepancies(uncovers):
    """Attempts where the task's own flag disagrees with the move record.

    ``trial_correct`` should be equivalent to "no incorrect uncover in this
    attempt".  In 12 of 18228 attempts it is not: the attempt is flagged
    correct although the subject uncovered a wrong field.  Those attempts
    therefore also incremented the correct-repeat counter, which is what
    produces the ``rep_correct == 10`` overflow seen in the timing tables.
    We report them rather than silently re-labelling anything.
    """
    mismatch = ((uncovers['n_incorrect_uncovers'] == 0)
                != (uncovers['correct'] == 1))
    return uncovers.loc[mismatch]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    handle = h5py.File(RAW_MAT, 'r')
    trial_vars = handle['abcd_data']['trial_vars']
    frames, problems, discrepant = [], [], []
    for subject_index in range(trial_vars.shape[0]):
        subject = f'{subject_index + 1:02d}'
        trials = handle[trial_vars[subject_index, 0]]
        table = uncovers_for_subject(handle, trials)
        table.insert(0, 'subject', subject)
        problem = check_alignment(table, subject)
        if problem is not None:
            problems.append(f's{subject}: {problem}')
            continue
        frames.append(table)
        discrepant.append(flag_discrepancies(table))
    handle.close()

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f'wrote {len(combined)} attempts from '
          f'{combined["subject"].nunique()} sessions to {OUT_CSV}')

    flagged = pd.concat(discrepant, ignore_index=True)
    flagged.to_csv(FLAGGED_CSV, index=False)
    print(f'{len(flagged)} attempts flagged correct despite an incorrect '
          f'uncover (in {flagged["subject"].nunique()} sessions) -> '
          f'{FLAGGED_CSV}')
    if problems:
        print(f'skipped {len(problems)} sessions:')
        for problem in problems:
            print(f'  {problem}')


if __name__ == '__main__':
    main()
