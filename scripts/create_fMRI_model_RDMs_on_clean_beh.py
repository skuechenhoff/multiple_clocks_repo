#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:06:25 2025
Based on clean behavioural tables,
create regressors that I want to use for the fMRI.

I will store a standard set of models 
(currently "location", "curr_rew", "next_rew", "two_next_rew", "three_next_rew", "state"
 DSR, l2_norm, A_state)
in all possible regressors: both task halves, path x rewards x unique_tasks
You can choose later which regressors you want to use.


logic is as follows:
create the models based on the behaviour in time = 'steps'.
create regressors based on 'path' or 'reward' also in time = 'steps'
regress each model into the same binned dimension the fMRI is in.
I want to end with regressors that go like: '{model}_A1_backw_A_reward.txt'

note: needs clean_fmri_behaviour.py to have run first.

@author: Svenja Küchenhoff
"""

import pandas as pd
import numpy as np
import os
import pickle
import mc
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
from fnmatch import fnmatch

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

subjects = [f"sub-{subj_no}"]
subjects = subs_list = [f'sub-{i:02}' for i in range(22, 35)]

# --- Load configuration ---
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")

#config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_DSR_rew_vs_path_stepwise_combos.json"
config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_DSR_rew_stepwise_combos.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

#
# SETTINGS
#
#regression_version = '03-4' 
#RDM_version = '03-1'
# no_phase_neurons = 3
plot_RDMs = False 
save_RDMs = True
EV_string = config.get("load_EVs_from", "DSR_loc-fut-rews-state-dur-type")
plot_DSR_task_matrices = False
plot_DSR_tasks = [] # fill this with eg tasks[14]
plot_DSR_rotation_bins = None

coord_to_loc = {
    (-0.21,  0.29): 1, (0.0,  0.29): 2, (0.21,  0.29): 3,
    (-0.21,  0.0 ): 4, (0.0,  0.0 ): 5, (0.21,  0.0 ): 6,
    (-0.21, -0.29): 7, (0.0, -0.29): 8, (0.21, -0.29): 9,
}
loc_to_coord = {v:k for k,v in coord_to_loc.items()}

#models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)
   
# import pdb; pdb.set_trace()
        
for sub in subjects:
    # load the cleaned behavioural table.
    beh_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/beh"
    RDM_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/beh/modelled_EVs"
    if os.path.isdir(beh_dir):
        print(f"Running on laptop, now subject {sub}")
    else:
        beh_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/beh"
        RDM_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/beh/modelled_EVs"
        print(f"Running on Cluster, setting {beh_dir} as data directory")

    beh_df = pd.read_csv(f"{beh_dir}/{sub}_beh_fmri_clean.csv")
    tasks = beh_df['task_config_ex'].unique()
    states = beh_df['state'].unique()
    bin_type = beh_df['time_bin_type'].unique()

    locations = sorted(beh_df['curr_loc'].unique())
    coordinates = np.array([loc_to_coord[loc] for loc in locations])
    
    loc_to_row = {loc: i for i, loc in enumerate(locations)}
    
    
    # define regressors. unique_time_bin_type look like E1_forw_A_reward etc.
    regs = sorted(beh_df['unique_time_bin_type'].unique())
    regressors = {}
    for reg in regs:
        regressors[reg] = np.zeros(len(beh_df))
        regressors[reg][beh_df['unique_time_bin_type'] == reg] = 1

    # select which EVs are included in the RDM (same logic as fMRI_run_RSA_without_rsatoolbox_clean.py)
    conditions = config.get("EV_condition_selection", {})
    parts_to_use = conditions.get("parts")
    if parts_to_use:
        for _p in ("task", "direction", "state", "phase"):
            if _p not in parts_to_use:
                raise ValueError(f"Missing selection.parts['{_p}'] in config.")
        EV_keys = []
        for ev in sorted(regs):
            task, direction, state, phase = ev.split('_')
            for name, value in zip(
                ["task", "direction", "state", "phase"],
                [task, direction, state, phase],
            ):
                part = parts_to_use[name]
                includes = part.get("include", [])
                excludes = part.get("exclude", [])
                if any(fnmatch(value, pat) for pat in excludes):
                    break
                if includes and not any(fnmatch(value, pat) for pat in includes):
                    break
            else:
                EV_keys.append(ev)
    else:
        EV_keys = list(regs)


    # define models.
    models = {}
    # ['location', 'curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew', 'state', 'clocks']
    models['state'] = np.zeros((len(states), len(beh_df)))
    models['A-state'] = np.zeros((len(states), len(beh_df)))
    models['duration'] = np.expand_dims(beh_df['t_spent_at_curr_loc'].to_numpy(), axis=0)
    models['path_rew'] = np.expand_dims(beh_df['time_bin_type'].to_numpy(), axis=0)
    
    for s_i, state in enumerate(states):
        # import pdb; pdb.set_trace()
        if state == 'A':
            models['A-state'][s_i][(beh_df['state'] == state)& (beh_df['time_bin_type'] == 'reward')] = 1
        models['state'][s_i][beh_df['state'] == state] = 1
    
    for key in ["location", "curr_rew", "next_rew", "two_next_rew", "three_next_rew", "l2_norm", "curr_path"]:
        models[key] = np.zeros((len(locations), len(beh_df)), dtype=float)
    
    
    
    # I THINK I NEED TO DELETE THE FIRST TIMEPOINT IN SUBPATH A! that's reward D.
    for i_loc, loc in enumerate(locations):
        models['location'][i_loc][beh_df['curr_loc'] == loc] = 1
        models['curr_rew'][i_loc][beh_df['curr_rew'] == loc] = 1
        # models['path'][i_loc][(beh_df['curr_loc'] == loc) & (beh_df['time_bin_type'] == 'path')] = 1
        # models['rew'][i_loc][(beh_df['curr_rew'] == loc) & (beh_df['time_bin_type'] == 'reward')] = 1
        for idx_inner_loc, inner_loc in enumerate(locations):
            models['l2_norm'][idx_inner_loc][beh_df['curr_loc'] == loc] = -np.linalg.norm(coordinates[i_loc] - coordinates[idx_inner_loc])
   

    # this is for the future reward location models.
    # rotates the reward values by k, but keeps time-bin-length in place.
    def rotate_runs(arr, k):
        """Rotate the values of consecutive runs by k, preserving run lengths."""
        # Finds the points at which a new value starts and turn them into indices
        changes = np.r_[True, arr[1:] != arr[:-1]]
        starts  = np.flatnonzero(changes)
        # Count number of identical consecutive items
        lens    = np.diff(np.r_[starts, arr.size])
        # Find which values are repeated
        vals    = arr[starts]
        # rols, and then repeat and return
        rot_vals = np.roll(vals, - (k % len(vals)))   # left-roll so first run takes next run's value
        return np.repeat(rot_vals, lens)


    for task in tasks:
        idx  = (beh_df["task_config_ex"] == task)
        cols = np.flatnonzero(idx)
        rews = beh_df.loc[idx, "curr_rew"].to_numpy()
    
        fut1 = rotate_runs(rews, 1)  # +1 run
        fut2 = rotate_runs(rews, 2)  # +2 runs
        fut3 = rotate_runs(rews, 3)  # +3 runs
    
        for fut, name in [(fut1,"next_rew"), (fut2,"two_next_rew"), (fut3,"three_next_rew")]:
            rows = np.fromiter((loc_to_row[v] for v in fut), dtype=int, count=fut.size)
            models[name][rows, cols] = 1.0
    
    
    # create regressors.
    EVs = {}
    for model in models:
        EVs[model] = {}
        for reg in regressors:
            if model == 'path_rew':
                label = 'reward' if reg.endswith('reward') else 'path' if reg.endswith('path') else None
                EVs[model][reg] = np.full(len(models[model]), label, dtype=object)
            else:
                EVs[model][reg] = np.zeros((len(models[model])))  
                for index, row in enumerate(models[model]):
                    if model == 'duration':
                        # sum up the durations of each regressor and divide by how often they were 'on'
                        n_times_regressor_active = np.sum(np.diff(regressors[reg]) == 1) + (regressors[reg][0] == 1)
                        EVs[model][reg][index] = models[model].transpose()[regressors[reg].astype(bool)].sum()/n_times_regressor_active
                    else:
                        # Note I don't include an intercept by default.
                        # this is because the way I use ithem, the regressors would be a linear combination of the intercept ([11111] vector)
                        EVs[model][reg][index] = LinearRegression(fit_intercept=False).fit(regressors[reg].reshape(-1,1), row.reshape(-1,1)).coef_

    # additionally, add the simple musicbox: at each of the 8 timebins, the future is already encoded.
    # order inside a task
    temp_order = [
        "A_path", "A_reward",
        "B_path", "B_reward",
        "C_path", "C_reward",
        "D_path", "D_reward"
    ]
    
    

    models['DSR'] = np.zeros((len(temp_order)*len(locations)))
    EVs['DSR'] = {}
    for task in tasks:
        # build base matrix (8 x 9) in canonical order
        bins_curr_task = [f"{task}_{temp_bin}" for temp_bin in temp_order]
        try:
            # concatenate the 8 bins x 9-element vectors into a single 72-element vector
            # this will read: 0-8 = now. 9-18 = next subpath. 19-27 = subpath after, etc.
            # each EVs['location'][k] has 9 location 
            firing_for_subpath_A = np.concatenate([EVs['location'][k] for k in bins_curr_task], axis=0)  # shape (72,)
        except KeyError:
            continue
    
        n_bins = len(temp_order)  # 8 (4 x subpaths, 4x rewards)
        n_locations = firing_for_subpath_A.size // n_bins   # 9 locations
        assert n_locations * n_bins == firing_for_subpath_A.size
     
        # for each position, rotate by whole blocks of `block_len` so subpath-chunks move together
        for pos, temp_bin in enumerate(temp_order):
            bin_curr_task = f"{task}_{temp_bin}"
            # left-roll by pos blocks: multiply by block_len to rotate whole 9-element blocks
            rotated = np.roll(firing_for_subpath_A, -pos * n_locations).copy()  # shape (72,)
            # store as 1D vector of length 72; if you want shape (1,72) use rotated.reshape(1,-1)
            EVs['DSR'][bin_curr_task] = rotated

    if plot_DSR_task_matrices and plot_DSR_tasks:
        mc.plotting.results.plot_dsr_task_matrices(
            EVs,
            tasks=plot_DSR_tasks,
            temp_order=temp_order,
            rotation_bins=plot_DSR_rotation_bins,
        )

       
    if plot_RDMs == True:
        for model in models:
            if model == 'path_rew':
                continue
            #ev_array = np.zeros((int(len(EVs[model])/2), len(models[model])))
            # if model == 'DSR':
            #     ev_array = np.zeros((int(len(EVs[model])), len(EVs['DSR'][bin_curr_task])))
            # else:
            evs_for_model = [ev for ev in EV_keys if ev in EVs[model]]
            ev_array = np.zeros((int(len(evs_for_model)), len(models[model])))
            idx = -1
            y_labels = []
            for ev in evs_for_model:
                #if ev.endswith('reward'):
                idx = idx +1
                y_labels.append(ev)
                ev_array[idx] = EVs[model][ev]
                    
            # for ev in EVs[model]:
            #     if ev.endswith('reward'):
            #         idx = idx +1
            #         y_labels.append(ev)
            #         ev_array[idx] = EVs[model][ev]
            
            # ev_array_all = np.zeros((int(len(EVs[model])), len(models[model])))
            # y_labels_all = []
            # for idx, ev in enumerate(EVs[model]):
            #     y_labels_all.append(ev)
            #     ev_array_all[idx] = EVs[model][ev]
    
            mc.plotting.results.plot_model_rdm_half(
                ev_array,
                labels=y_labels,
                method="crosscorr",
                label_half="first",
                group_size=4,
                title=model,
            )

    #import pdb; pdb.set_trace()          
    if save_RDMs: 
        # then save these matrices.
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        
        with open(f"{RDM_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'wb') as file:
            pickle.dump(EVs, file)
            
        print(f"saved EV dictionary as {RDM_dir}/{sub}_modelled_EVs_{EV_string}.pkl")
