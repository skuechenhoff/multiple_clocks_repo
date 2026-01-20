#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:21:08 2025

@author: Svenja Küchenhoff

This script takes the raw .csv file from PsychoPy and cleans it such that I can
easily read out the relevant behavioural details (combine pt 1 and 2!):
    1) task_config [e.g. E1]
    3) forwards/backwards
    4) config type [e.g. E1_forw]
    5) curr rew_loc [location 1-9]
    6) time start reward
    7) curr_loc [location 1-9]
    8) time start at curr_loc
    9) button press for respective loc [1/4 keys]
    10) state
    11) time_bin_type (state+reward/path) [e.g. E1_forw_A_path]
    12) repeat [1-5]
    13) session [1/2]
    

    
"""

import pandas as pd
import os
import sys
import mc
import numpy as np

from glob import glob

# for mapping locations
# mapping dictionary: (x,y) -> grid number
coord_to_loc = {
    (-0.21,  0.29): 1, (0.0,  0.29): 2, (0.21,  0.29): 3,
    (-0.21,  0.0 ): 4, (0.0,  0.0 ): 5, (0.21,  0.0 ): 6,
    (-0.21, -0.29): 7, (0.0, -0.29): 8, (0.21, -0.29): 9,
}

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'

alternative_regs = False

# Find the source dir first, outside of the loop
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data"
if os.path.isdir(source_dir):
    print("Running on laptop")
else:
    source_dir = "/home/fs0/xpsy1114/scratch/data"
    print(f"Running on Cluster, setting {source_dir} as data source")


data_dir_beh = f"{source_dir}/pilot"

# import pdb; pdb.set_trace()
all_sub_paths = glob(f"{data_dir_beh}/sub-*")
subjects = [
    os.path.basename(p)
    for p in all_sub_paths
    if os.path.isdir(p)
]

subjects.remove('sub-21')
subjects.remove('sub-29')

for sub in subjects:
    out_dir = f"{source_dir}/derivatives/{sub}/beh"
    

    print(f"Now subject {sub}")
    
    both_halves = []   # collect cleaned tables for both halves

    # Then here inside the loop, we know which subject we are looking at so we can define the correct folders

    for task_half in [1,2]:
        file = data_dir_beh + f"/{sub}/beh/{sub}_fmri_pt{task_half}.csv"
        if not os.path.exists(file):
            print(f"This file doesn't exist: {file}")
            continue  # skip to next loop iteration
        df = pd.read_csv(file)
        
        # create a new df 
        beh_clean = pd.DataFrame()

        # to start with, a particularity of the df has to be accounted for:
        # every ABCD-loop starts with 'being' at the location where reward 'D' has been found.
        # this is, however, without having taken a new step.
        # thus: 1. remove the first 'step' of subpath A
        #       2. take the t_step_press_curr_run and the t_step_press_global values from AFTER the 'start_ABCD_screen' and fill the empties.
        df_test = df.copy()

        # before cleaning rows, make sure all crucial information is everywhere.
        # forward fill for anything below
        df['task_config'] = df['task_config'].ffill()
        df['repeat'] = df['repeat'].ffill()
        
        # backward fill these as this is a 'pre-step' that shall count for the reward before [except if this is the last repeat.]
        df['t_step_press_global'] = df['t_step_press_global'].bfill()
        # but not in the last repeat of a task, as then a different screen followed and no button was pressed.
        df.loc[df['task_config'].ne(df['task_config'].shift(-1)), 't_step_press_global'] = np.nan



        # remove these rows as it's not actually a step.
        beh_raw = df[df['start_ABCD_screen'].isna()].copy()
        
        # every valid row has a 'type' entry. filter for none-valid rows.
        beh_raw = beh_raw[beh_raw['type'].notna()].copy()
        
        # 1) repeat [1-5]
        beh_clean['repeat'] = beh_raw['repeat']
        
        # 1) task_config [e.g. E1]
        beh_clean['task_config_seq'] = beh_raw['task_config'].ffill()
        
        # 2) forwards/backwards
        beh_clean['instruction'] = beh_raw['type']
        
        # 3) config type [e.g. E1_forw]
        beh_raw['task_config_ex']   = beh_clean['task_config_seq'] + '_' + beh_clean['instruction']
        
        beh_clean['task_config_ex'] = beh_clean['task_config_seq'] + '_' + beh_clean['instruction']
        
        # 4) curr rew_loc [location 1-9]
        # Sometimes a trial ended early and coordinates are not defined so we put NA in the df
        beh_clean['curr_rew'] = beh_raw.apply(
            lambda row: coord_to_loc.get((row['curr_rew_x'], row['curr_rew_y']), pd.NA),
            axis=1
        )
        

        # 5) time start reward and length
        # for rewards:
        # -> arrival at location = t_step_end_global
        # -> appearance reward = arrival at location = t_step_end_global = t_reward_start
        # -> appearance reward end = t_reward_afterwait
        # -> dwell time at reward = t_step_press_global- t_step_end_global
        beh_clean['t_curr_rew']   = beh_raw['t_reward_start']
        beh_clean['reward_delay'] = beh_raw['reward_delay']
        
        
        # 6) curr_loc [location 1-9]
        beh_clean['curr_loc'] = beh_raw.apply(
            lambda row: coord_to_loc.get((row['curr_loc_x'], row['curr_loc_y']), pd.NA),
            axis=1
        )
        
        # 7) time start at curr_loc -> arrival at location = t_step_end_global
        beh_clean['t_curr_loc'] = beh_raw['t_step_end_global']

        # press to leave the location 
        beh_clean['t_press_leave_curr_loc'] = beh_raw['t_step_press_global'] 
        
        # -> dwell time at location = t_step_press_global- t_step_end_global
        beh_clean['t_dwell_curr_loc'] =  beh_clean['t_press_leave_curr_loc'] - beh_clean['t_curr_loc']
        # -> walking time to next location = length_step
        beh_clean['t_move_to_next_loc'] = beh_raw['length_step']
        
        # if t_press_leave_curr_loc and t_dwell_curr_loc are empty, fill as follows:
        # t_press_leave_curr_loc, t_dwell_curr_loc = t_curr_rew + reward_delay
        mask = (beh_clean['t_press_leave_curr_loc'].isna() & beh_clean['t_dwell_curr_loc'].isna())
        beh_clean.loc[mask, 't_press_leave_curr_loc'] = beh_clean.loc[mask, 't_curr_rew'] + beh_clean.loc[mask, 'reward_delay']
        beh_clean.loc[mask, 't_dwell_curr_loc'] = beh_clean.loc[mask, 'reward_delay']

        # -> overall time related to this location = anything from arrival to next location.
        beh_clean['t_spent_at_curr_loc'] = beh_clean['t_move_to_next_loc'] + beh_clean['t_dwell_curr_loc']
        
        
        # 8) button press for respective loc [1/4 keys]
        # leave this for now!
        # a bit more complicated given there might have been more buttons stored
        # than only the ones executed.
        # x = beh_raw[beh_clean['repeat'] == 1][beh_clean['task_config_ex']=='B1_backw']
        #timings = beh_clean[beh_clean['repeat'] == 1][beh_clean['task_config_ex']=='B1_backw']['t_curr_loc'].to_numpy()
        #import ast; press_timings_local = ast.literal_eval(x['nav_key_task.rt'][x['nav_key_task.rt'].notna()].iloc[0])
        
        # for now just keep the relevant rows.
        beh_clean['button_rts']  = beh_raw['nav_key_task.rt']
        beh_clean['button_keys'] = beh_raw['nav_key_task.keys']
        

        # 9) state
        # Sometimes at the start state is empty but since we know things start at A we just fill it in
        beh_clean['state'] = beh_raw['state'].fillna('A')
        
        # 10) time_bin_type (state+reward/path) [e.g. E1_forw_A_path]
        rewards_mask = beh_clean['t_curr_rew'].notna()
        beh_clean['time_bin_type'] = 'path'
        beh_clean.loc[rewards_mask, 'time_bin_type'] = 'reward'
        beh_clean['unique_time_bin_type'] = beh_clean['task_config_ex'] + '_' + beh_clean['state'] + '_path'
        beh_clean.loc[rewards_mask, 'unique_time_bin_type'] = beh_clean.loc[rewards_mask, 'task_config_ex'] + '_' + beh_clean.loc[rewards_mask, 'state']+ '_reward'


        # 12) session [1/2]
        beh_clean['task_half'] = task_half
        
        # if wanted, you can analyse Brooke's alternative regressors here.
        if alternative_regs == True:
            # beh_clean['task_half'] = mc.analyse.extract_and_clean.define_futsteps_x_locs_regressors(beh_clean)
            df = mc.analyse.extract_and_clean.define_futsteps_x_locs_regressors(beh_clean)
            for step in ['curr','one_fut','two_fut','three_fut']:
                for loc in range(1, 10):
                    row_name = f'loc_{loc}_{step}'
                    print(f"{step} {loc} has {np.sum(df[row_name])} occurances")

        both_halves.append(beh_clean)
    
    # concatenate both halves and save
    beh_both = pd.concat(both_halves, ignore_index=True)
    
    # import pdb; pdb.set_trace()
    # store where same reward-states appear at the same locations for later masking
    # mc.analyse.extract_and_clean.store_same_locs_in_same_state(beh_both, out_dir)
    
    out_file = f"{out_dir}/{sub}_beh_fmri_clean.csv"
    beh_both.to_csv(out_file, index=False)
    print(f"Saved {out_file}")
