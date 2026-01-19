#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 13:31:21 2025

It saves EV files for FEAT, as well as an .fsf file that can be used as an input for the EVs,
making sure to order the EVs correctly.

based on 
1. clean_fmri_behaviour.py
2. EV_config file

@author: Svenja Küchenhoff
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import mc
import pickle
import sys
import json
import shutil

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    data_dir_deriv = f"{source_dir}/data/derivatives"
    data_dir = f"{source_dir}/data/pilot"
    analysis_dir = f"{source_dir}/multiple_clocks_repo/mc/fmri_analysis"  
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    data_dir_deriv = f"{source_dir}/data/derivatives"
    data_dir = f"{source_dir}/data/pilot"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    analysis_dir = f"{source_dir}/analysis"  
    print(f"Running on Cluster, setting {source_dir} as data directory")

         
# import pdb; pdb.set_trace()      
# --- Load configuration ---
config_file = sys.argv[2] if len(sys.argv) > 2 else "EV_config_all_paths_stickrews_split-buttons.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
version = config.get("name")
split_buttons = config.get("split_buttons", False)
regress_rewards = config.get("regress_rewards", True)
rewards_as_stick_function = config.get("rewards_as_stick_function", False)
regress_subpaths = config.get("regress_subpaths", False)
precise_paths = config.get("precise_paths", False)
tasks_included = config.get("tasks_included", ['A', 'B', 'C', 'D', 'E'])
repeats_included = config.get("repeats_included", [1,2,3,4,5])

fut_step_x_loc_regs = config.get("fut_step_x_loc_regs", False)

states_included = config.get("states_included", ['A', 'B', 'C', 'D'])
duration_state = config.get("duration_state", False)
state_regs = config.get("state_regs", False)



# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'  
subjects = [f"sub-{subj_no}"]

for sub in subjects:
    # load the cleaned behavioural table.
    beh_df = pd.read_csv(f"{data_dir_deriv}/{sub}/beh/{sub}_beh_fmri_clean.csv")
    
    # define and make paths
    for th in [1,2]:
        print(f"Now creating EVs for fmri file {th} and {sub}")
        EV_folder = f'{data_dir_deriv}/{sub}/func/EVs_{version}_pt0{th}/'
        if os.path.exists(EV_folder):
            print("careful, the EV folder does exist- there might be other EVs and thus not all files will be output correctly! Deleting dir.")
            shutil.rmtree(EV_folder)
            os.makedirs(EV_folder)
        if not os.path.exists(EV_folder):
            os.makedirs(EV_folder)
    
        file_all = f"{sub}_fmri_pt{th}_all.csv"
        
        # load behavioural file
        df_all = pd.read_csv(f"{data_dir}/{sub}/beh/"+file_all)
        first_TR_at = df_all['TR_received_no0'].dropna().unique().tolist()[0]
        beh_th = beh_df[beh_df['task_half'] == th].copy()


        # Button press EV -> will be a nuisance regressor.
        # for button press EVs I need to add the entries in nav_key_task.rt to 
        end_task = beh_th[(~beh_th['button_rts'].isna())]
        end_task_idx = end_task.index.to_list()
        start_task_idx = [beh_th.index.to_list()[0]] + [e + 1 for e in end_task_idx]
        end_task = end_task.reset_index(drop = True)
        rt_press, key_press = [], []
        for i, row in end_task.iterrows():
            onset_curr_task = beh_th.at[start_task_idx[i], 't_curr_loc']
            # extract button presses from the rt item with all presses
            presses_curr_task = row['button_rts'].strip('[""]').split(', ') # Split the string into a list using a comma as the separator
            buttons_curr_task = row['button_keys'].strip('[""]').split(', ')
            # Convert the elements to floats and add to the point in time where they actually started
            presses_curr_task = [(float(time)+onset_curr_task) for time in presses_curr_task]
            buttons_curr_task = [button.strip("''") for button in buttons_curr_task]
            rt_press=rt_press+presses_curr_task
            key_press=key_press+buttons_curr_task

        if split_buttons == True:
            mapping = {'1':'left', '2':'up', '3':'down', '4':'right'}
            # import pdb; pdb.set_trace()
            for button_val, button_name in mapping.items():
                # pick times where the key matches this button
                on_press = [t for k, t in zip(key_press, rt_press) if str(k) == button_val]
                dur_press = np.full(len(on_press), 0.02)
                mag_press = np.ones(len(on_press))
                button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(on_press, dur_press, mag_press, button_name, EV_folder, first_TR_at) 
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(button_press_EV)
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cutted EV {button_name} file. Happened for subject {sub} in task half {th}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"{button_name}" + '.txt', array, delimiter="    ", fmt='%f')
        else:
            dur_press = np.ones(len(on_press)) * 0.02
            mag_press = np.ones(len(on_press))
            
            button_press_EV = mc.analyse.analyse_MRI_behav.create_EV(on_press, dur_press, mag_press, 'press_EV', EV_folder, first_TR_at)
            
            deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(button_press_EV)
            if deleted_x_rows > 0:
                print(f"careful! I am saving a cutted EV buttonpress file. Happened for subject {sub} in task half {th}")
                np.savetxt(str(EV_folder) + 'ev_' + "button" + '.txt', array, delimiter="    ", fmt='%f')
        

        if fut_step_x_loc_regs == True:
            # defines each regressor as one hot encoding
            # timings are in 't_curr_loc' column.
            fut_step_regs = mc.analyse.extract_and_clean.define_futsteps_x_locs_regressors(beh_th)
            future_step_EV_names = [c for c in fut_step_regs.columns if c.startswith('loc')]
            for fut_step_EV_name in future_step_EV_names:
                fut_step_regs_curr_EV = fut_step_regs[(fut_step_regs[fut_step_EV_name]==1) & (~fut_step_regs['t_curr_rew'].isna())].copy()
                on_fut_step_EV = fut_step_regs_curr_EV['t_curr_rew'].to_list()
                if len(on_fut_step_EV) == 0:
                    print(f"not creating {fut_step_EV_name} as there are no such steps.")
                    continue
                if rewards_as_stick_function == True:
                    dur_fut_step_EV = np.ones(len(on_fut_step_EV))
                else:
                    dur_fut_step_EV = fut_step_regs_curr_EV['reward_delay'].to_list()
                mag_fut_step_EV = np.ones(len(on_fut_step_EV))
                fut_step_EV = mc.analyse.analyse_MRI_behav.create_EV(on_fut_step_EV, dur_fut_step_EV, mag_fut_step_EV, fut_step_EV_name, EV_folder, first_TR_at)
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(fut_step_EV)
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cutted future step EV {fut_step_EV_name} file. Happened for subject {sub} in task half {th}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"{fut_step_EV_name}" + '.txt', array, delimiter="    ", fmt='%f')

        if state_regs == True:
            # first filter for those tasks that shall be included.
            mask = beh_th['task_config_seq'].str.startswith(tuple(tasks_included))
            beh_th = beh_th[mask]
            
            # 1) Runs: increment when the state changes
            run_id = beh_th['state'].ne(beh_th['state'].shift()).cumsum()
        
            # 2) For each run: state, first time (onset), last time (offset), duration
            runs = (beh_th.groupby(run_id, as_index=False)   # <- as_index=False
                      .agg(state=('state','first'),
                           onset=('t_curr_loc','first'),
                           offset=('t_curr_loc','last')))
            #
            #
            runs = (beh_th.groupby(run_id, as_index=False)   # <- as_index=False
                      .agg(state=('state','first'),
                           onset=('t_curr_loc','first'),
                           rew_onset=('t_curr_loc','last'),
                           rew_delay=('reward_delay','last')))
            runs['duration'] = runs['rew_onset'] +  runs['rew_delay']- runs['onset']

            # a state is defined as starting from setting foot on the first location of a new subpath to not seing the coin anymore at the rewarded location.
            # note that there might be a time lag until they move again!

            # 3) Per-state lists in state dicts
            onsets_by_state    = runs.groupby('state', sort=False)['onset'].apply(list).to_dict()
            
            if duration_state == 'full_path':
                durations_by_state = runs.groupby('state', sort=False)['duration'].apply(list).to_dict()

            # add other options!
            
            # 4) Create one EV file per state in states_included
            for curr_state in states_included:
                onsets = onsets_by_state[curr_state]
                durs   = durations_by_state[curr_state]
                mags   = np.ones(len(onsets))
                state_EV = mc.analyse.analyse_MRI_behav.create_EV(onsets, durs, mags, f"state_{curr_state}", EV_folder, first_TR_at)
                deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(state_EV)
                if deleted_x_rows > 0:
                    print(f"careful! I am saving a cut state EV {curr_state} file. Happened for subject {sub} in task half {th}")
                    np.savetxt(str(EV_folder) + 'ev_' + f"{curr_state}" + '.txt', array, delimiter="    ", fmt='%f')
                
         
        if regress_rewards == True:
            all_tasks = beh_th['task_config_ex'].unique()
            for task in all_tasks:
                if task.startswith(tuple(tasks_included)):
                    for rew in states_included:
                        onset_curr_rew = beh_th[(beh_th['task_config_ex']==task) & (beh_th['state']==rew) & (~beh_th['t_curr_rew'].isna())]['t_curr_rew'].to_list()
                        dur_curr_rew = beh_th[(beh_th['task_config_ex']==task) & (beh_th['state']==rew) & (~beh_th['reward_delay'].isna())]['reward_delay'].to_list()
                        if rewards_as_stick_function== True:
                            dur_curr_rew = np.ones(len(onset_curr_rew))
                        mag_curr_rew = np.ones(len(onset_curr_rew))
                        rew_EV = mc.analyse.analyse_MRI_behav.create_EV(onset_curr_rew, dur_curr_rew, mag_curr_rew, f"{task}_{rew}_reward", EV_folder, first_TR_at)
                        deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(rew_EV)
                        if deleted_x_rows > 0:
                            print(f"careful! I am saving a cut state EV {task}_{rew}_reward file. Happened for subject {sub} in task half {th}")
                            np.savetxt(str(EV_folder) + 'ev_' + f"{task}_{rew}_reward" + '.txt', array, delimiter="    ", fmt='%f')
        
                       
        if regress_subpaths == True:
            all_tasks = beh_th['task_config_ex'].unique()
            for task in all_tasks:
                if task.startswith(tuple(tasks_included)):
                    for rew in states_included:
                        # the subpath is defined as 'time_bin_type' == 'path'
                        # ranging from t_curr_loc to t_curr_loc reward.
                        step_curr_path = beh_th[(beh_th['task_config_ex']==task) & (beh_th['state']==rew) & (beh_th['time_bin_type']=='path')]['t_curr_loc'].to_numpy()
                        onset_curr_rew = beh_th[(beh_th['task_config_ex']==task) & (beh_th['state']==rew) & (~beh_th['t_curr_rew'].isna())]['t_curr_rew'].to_numpy()

                        prev_rewards = np.r_[-np.inf, onset_curr_rew[:-1]]                # previous reward for each reward
                        idx = np.searchsorted(step_curr_path, prev_rewards, side='right')   # first step > prev reward
                        
                        valid = (idx < len(step_curr_path)) & (step_curr_path[idx] < onset_curr_rew)        # step exists and precedes reward
                        onset_curr_path    = np.where(valid, step_curr_path[idx], np.nan)
                        duration_curr_path = np.where(valid, onset_curr_rew - onset_curr_path, np.nan)
                        mag_curr_path = np.ones(len(onset_curr_path))
                        
                        path_EV = mc.analyse.analyse_MRI_behav.create_EV(onset_curr_path, duration_curr_path, mag_curr_path, f"{task}_{rew}_path", EV_folder, first_TR_at)
                        deleted_x_rows, array = mc.analyse.analyse_MRI_behav.check_for_nan(rew_EV)
                        if deleted_x_rows > 0:
                            print(f"careful! I am saving a cut state EV {task}_{rew}_path file. Happened for subject {sub} in task half {th}")
                            np.savetxt(str(EV_folder) + 'ev_' + f"{task}_{rew}_path" + '.txt', array, delimiter="    ", fmt='%f')


        # then, lastly, adjust the .fsf file I will use for the regression.
        print(f're-writing the .fsf file for {sub} for fmri file {th} now!')
        
        # collect all filepaths I just created.
        files_in_EV_folder = os.listdir(EV_folder) 
        EV_paths = []
        for EV in files_in_EV_folder:
            if EV.startswith("ev_") and EV.endswith(".txt"):
                EV_paths.append(os.path.join(EV_folder, EV)) 
        print(f"I collected {len(EV_paths)} EVs to put into the fsf file.")
        sorted_EVs = sorted(EV_paths)
        
        text_to_write = []
        with open(f"{EV_folder}task-to-EV.txt", 'w') as file:
            for i, EV_path in enumerate(sorted_EVs): 
                EV_file_name = EV_path.split('/')[-1].replace('.txt', '')
                file.write(f'{i} {EV_file_name}\n')
                
        if sub in ['sub-04', 'sub-06', 'sub-30', 'sub-31', 'sub-34']:
            template_name = 'new_fsf_file_84.fsf'
            #template_name = 'my_RDM_GLM_v2.fsf'
        elif sub in ['sub-05', 'sub-35'] and th == 1:
            #template_name = 'my_RDM_GLM_v2.fsf'
            template_name = 'new_fsf_file_84.fsf'
        else:
            #template_name = 'my_RDM_GLM_pnm.fsf'
            template_name = 'new_fsf_file_pnm_84.fsf'

        #template_name = 'new_fsf_file_84.fsf'
        # else:
        with open(f"{analysis_dir}/templates/{template_name}", "r") as fin:                    
            for line in fin:
                for i, EV_path in enumerate(sorted_EVs): 
                    # the count in the EV file starts from 1, not 0 -> so do +1
                    if line.startswith(f"set fmri(custom{i+1})"):
                        # print(f"my old line was: {line}")
                        line = f'set fmri(custom{i+1}) "{EV_path}"\n'
                    if line.startswith(f"set fmri(evtitle{i+1})"):
                        EV_name_ext = os.path.basename(EV_path)
                        EV_name = EV_name_ext.rsplit('.',1)[0]
                        # print(f"changing evtitle{i+1} to {EV_name}")
                        line = f'set fmri(evtitle{i+1}) "{EV_name}"\n'
                    if line.startswith("set fmri(evs_orig)"):
                        line = f"set fmri(evs_orig) {len(EV_paths)}\n"
                    if line.startswith("set fmri(evs_real)"):
                        line = f"set fmri(evs_real) {len(EV_paths)+1}\n"   
                        # import pdb; pdb.set_trace();
                text_to_write.append(line)
        


        n_EVs = len(sorted_EVs)
        
        # this looks pretty well, but sadly, feat doesnt seem to run with it.
        #max_EVs_og_fsf = 81
        #if n_EVs > max_EVs_og_fsf:
        #    print(f"n EVs is {n_EVs} but fsf file only covers {max_EVs_og_fsf}. starting the helper function.")
        #    text_to_write_cleaned = mc.analyse.analyse_MRI_behav.extend_for_more_evs(text_to_write.copy(), sorted_EVs, n_EVs, max_EVs_og_fsf)
        
        #else:
            
        # then, in the next round, delete all the EVs that I don't actually include.
        # first, do this for the orthogonalisation of the EVs + contrasts you want with the ones you don't.
        skip = 0
        text_to_write_half_cleaned = []
        for line in text_to_write:
            if skip > 0:
                # if the counter is increased, skip next line and decrease counter
                skip -= 1
                continue
            if (line.startswith("# Orthogonalise EV") and int(line[-3:-1]) > n_EVs) or (line.startswith("# Real contrast_orig") and int(line[-3:-1]) > n_EVs) or (line.startswith("# Real contrast_real vector") and int(line[-3:-1]) > n_EVs):
                #print(f"end of line is {line[-3:-1]}, so skip these next 3")
                skip = 2
            else:
                #import pdb; pdb.set_trace();
                text_to_write_half_cleaned.append(line)
                
        # then, delete all the configurations of the actual EVs I don't want.
        skip_until_marker = False
        marker_line = "# Contrast & F-tests mode"
        text_to_write_cleaned = []
        for line in text_to_write_half_cleaned:
            if skip_until_marker:
                if line.strip() == marker_line:
                    # add marker line to text and stop skipping
                    text_to_write_cleaned.append(line)
                    skip_until_marker = False
                continue
            if line.startswith("# EV") and int(line[5:7]) > n_EVs:
                skip_until_marker = True
            else:
                text_to_write_cleaned.append(line)
                    
        with open(f"{data_dir_deriv}/{sub}/func/{sub}_draft_GLM_0{th}_{version}.fsf", "w") as fout:
            for line in text_to_write_cleaned:
                fout.write(line)
            print(f"stored updated fsf file here: {data_dir_deriv}/{sub}/func/{sub}_draft_GLM_0{th}_{version}.fsf")
           
        # --- SETTINGS SUMMARY (per subject) ---
        summary = {
            "subject": sub,
            "name": version,
            "split_buttons": split_buttons,
            "repeats_included": repeats_included,
            "regress_rewards": regress_rewards,
            "rewards_as_stick_function": rewards_as_stick_function,
            "regress_subpaths": regress_subpaths,
            "tasks_included": tasks_included,
            "states_included": states_included,
            "fut_step_x_loc_regs": fut_step_x_loc_regs,
            "state_regs": state_regs,
            "OG_template_used": f"{analysis_dir}/templates/{template_name}",
            "fsf_stored_as": f"{data_dir_deriv}/{sub}/func/{sub}_draft_GLM_0{th}_{version}.fsf",
            "EVs_stored_in": EV_folder,
        }
        
        print("\n=== SETTINGS SUMMARY ===")
        for k, v in summary.items():
            print(f"{k:>20}: {v}")
        
        # Save a copy alongside results for provenance
        with open(os.path.join(EV_folder, f"{sub}_th-{th}_settings_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"(Saved summary → {os.path.join(EV_folder, f'{sub}_th-{th}_settings_summary.json')})\n")
  
