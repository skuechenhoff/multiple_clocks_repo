#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:41:06 2025

@author: Svenja Küchenhoff
helpers functions to analyse human cells


"""
import pickle
import numpy as np
import os
import re
import scipy
import math
import mc
from matplotlib import pyplot as plt
import glob
import pandas as pd
import copy
# import pdb; pdb.set_trace()

def read_files_to_list_ordered(folder, pattern_string):
    # import pdb; pdb.set_trace()
    # This pattern matches "timings_rewards_grid<number>_sub50.csv" where <number> can be any integer
    pattern = re.compile(pattern_string)
    # List all files in the directory and filter them based on the pattern
    files = [f for f in os.listdir(folder) if pattern.match(f)]
    # Sort files by the numerical value following "grid"
    files.sort(key=lambda x: int(pattern.search(x).group(1)))
    # Read each file into a pandas DataFrame and store them in a list
    data_list = [np.genfromtxt(f"{folder}/{file}", delimiter=',') for file in files] 
    return data_list

def read_string_files_to_list_ordered(folder, pattern_string):
    # Create a list to store the file contents
    file_contents = []
    pattern = re.compile(pattern_string)
    files = [f for f in os.listdir(folder) if pattern.match(f)]
    # Sort files by the numerical value following "grid"
    files.sort(key=lambda x: int(pattern.search(x).group(1)))
    
    
    # # Construct the full pattern to search for files
    # full_pattern = f"{source_folder}/{pattern}"
    # # Use glob to find all files that match the pattern
    # file_list = sorted(glob.glob(full_pattern))
    # # Read each file and append the contents to the list
    for file in files:
        df = pd.read_csv(f"{folder}/{file}", header=None)
        file_contents.append(df.iloc[0].tolist())
    return file_contents



def buttons_to_ints(button_list):
    buttons_int = []
    for elem in button_list:
        if elem == 'LeftArrow':
            buttons_int.append(0)
        if elem == 'UpArrow':
            buttons_int.append(1)
        if elem == 'RightArrow':
            buttons_int.append(2)
        if elem == 'DownArrow':
            buttons_int.append(3)
        if elem == 'Return':
            buttons_int.append(99)
    return buttons_int
    


def button_change_indices(buttons):
    # Convert list to NumPy array
    arr = np.array(buttons)
    # Find the differences between consecutive elements
    changes = np.where(arr[:-1] != arr[1:])[0] + 1
    return changes



def filter_data(data, session, rep_filter):
    # filter can be 'all', 'all_correct', 'early', 'late', 'all_minus_explore'
    filtered_data = copy.deepcopy(data)
    if rep_filter =='all_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['correct']==1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['correct']==1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['correct']==1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['correct']==1].reset_index(drop = True)    
    
    elif rep_filter == 'early_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4,5]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4,5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4,5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1,2,3,4,5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'late_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6,7,8,9,10])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6,7,8,9,10])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6,7,8,9,10])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6,7,8,9,10])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    

    elif rep_filter == 'all_minus_explore' or rep_filter == 'residualised':
        # import pdb; pdb.set_trace()
        # exclude aanything where both 'rep_correct' == 0 and 'correct' == 0
        keep_mask = data[f"sub-{session:02}"]['beh'][['correct','rep_correct']].ne(0).any(axis=1)
        
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][keep_mask].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][keep_mask].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][keep_mask].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][keep_mask].reset_index(drop = True)

    elif rep_filter == 'late':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'] > 4].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'] > 4].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'] > 4].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'] > 4].reset_index(drop = True)
     
    elif rep_filter == 'early':
        keep_mask = ((data[f"sub-{session:02}"]['beh']['rep_correct'] > 0) & (data[f"sub-{session:02}"]['beh']['rep_correct'] < 5))
        filtered_data[f"sub-{session:02}"]['beh'] = (data[f"sub-{session:02}"]['beh'].loc[keep_mask].reset_index(drop=True))
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'].loc[keep_mask].reset_index(drop=True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'].loc[keep_mask].reset_index(drop=True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron].loc[keep_mask].reset_index(drop=True)
            
    elif rep_filter == 'first_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([0]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([0])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([0])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([0])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
        
    elif rep_filter == 'one_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([1])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'two_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([2]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([2])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([2])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([2])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'three_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([3]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([3])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([3])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([3])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'four_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([4]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([4])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([4])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([4])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'five_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([5]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([5])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'six_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([6])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'seven_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([7]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([7])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([7])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([7])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'eight_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([8]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([8])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([8])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([8])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    
    
    elif rep_filter == 'nine_correct':
        filtered_data[f"sub-{session:02}"]['beh'] = data[f"sub-{session:02}"]['beh'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([9]) & data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['timings'] = data[f"sub-{session:02}"]['timings'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([9])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        filtered_data[f"sub-{session:02}"]['locations'] = data[f"sub-{session:02}"]['locations'][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([9])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)
        for neuron in data[f"sub-{session:02}"]['normalised_neurons']:
            filtered_data[f"sub-{session:02}"]['normalised_neurons'][neuron] = data[f"sub-{session:02}"]['normalised_neurons'][neuron][data[f"sub-{session:02}"]['beh']['rep_correct'].isin([9])& data[f"sub-{session:02}"]['beh']['correct']== 1].reset_index(drop = True)    

    # import pdb; pdb.set_trace()
    return filtered_data


def rename_rois(df, collapse_pfc = False, plot_by_cingulate_and_MTL = False):
    roi_label = []
    if collapse_pfc == True:
        for _, row in df.iterrows():
            cell_label = row['neuron_id']
            if 'ACC' in cell_label or 'vCC' in cell_label or 'AMC' in cell_label or 'vmPFC' in cell_label or 'OFC' in cell_label or 'PCC' in cell_label:
                roi = 'PFC'
            elif 'MCC' in cell_label or 'HC' in cell_label:
                roi = 'hippocampal'
            elif 'EC' in cell_label:
                roi = 'entorhinal'
            elif 'AMYG' in cell_label:
                roi = 'amygdala'
            else:
                roi = 'mixed'
            roi_label.append(roi)
    elif plot_by_cingulate_and_MTL == True:
        for _, row in df.iterrows():
            cell_label = row['neuron_id']
            if 'ACC' in cell_label or 'vCC' in cell_label or 'AMC' in cell_label or 'vmPFC' in cell_label or 'PCC' in cell_label:
                roi = 'Cingulate'
            elif 'MCC' in cell_label or 'HC' in cell_label or 'EC' in cell_label or 'AMYG' in cell_label:
                roi = 'MTL'
            elif 'OFC' in cell_label:
                roi = 'OFC'
            else:
                roi = 'mixed'
            roi_label.append(roi)
    else:
        for _, row in df.iterrows():
            cell_label = row['neuron_id']
            if 'ACC' in cell_label or 'vCC' in cell_label or 'AMC' in cell_label or 'vmPFC' in cell_label:
                roi = 'ACC'
            elif 'PCC' in cell_label:
                roi = 'PCC'
            elif 'OFC' in cell_label:
                roi = 'OFC'
            elif 'MCC' in cell_label or 'HC' in cell_label:
                roi = 'hippocampal'
            elif 'EC' in cell_label:
                roi = 'entorhinal'
            elif 'AMYG' in cell_label:
                roi = 'amygdala'
            else:
                roi = 'mixed'
            roi_label.append(roi)
    return roi_label


def extract_consistent_grids(neuron, cell_name, beh):
    # DIFFERENCE BETWEEN grid-blocks AND unique grids
    # goal: kick out grid-blocks that are unreliable.
    
    # per grid-block, identify firing rate
    # exclude grid if firing rate lower than 20% of mean firing.
    # also make sure to leave at least 3 unique grids.

    # beh[f'mean_FR_{cell_name}'] = np.nanmean(neuron)
    # identify firing rate per grid-block.
    grid_nos = np.unique(beh['grid_no'].to_numpy())
    # FR per grid_no
    grid_fr = {}
    for g in grid_nos:
        mask_g = (beh['grid_no'] == g)
        grid_fr[g] = np.nanmean(neuron[mask_g])
        
    # attach column (row-wise) for convenience/inspection
    beh[f'grid_FR_{cell_name}'] = beh['grid_no'].map(grid_fr)
    

    # --- tentative exclusion: FR < 20% of overall mean (treat NaN FR as low) ---
    excluded_grid_nos = []
    thresh = 0.2 * np.nanmean(neuron) if not np.isnan(np.nanmean(neuron)) else np.nan
    for g in grid_nos:
        fr = grid_fr[g]
        if np.isnan(fr) or (not np.isnan(thresh) and fr < thresh):
            excluded_grid_nos.append(g)
    
    # 3) Tentative keep-mask with low-rate grids removed
    tentative_keep_mask = ~beh['grid_no'].isin(excluded_grid_nos)
    # tentative_keep_mask = ~beh['new_grid_idx'].isin(excluded_grid_nos)

    
    # next test based on this new selection, how many UNIQUE GRIDS are left?
    # at least 3 so cross-validation is possible.
    kept_identities = beh['idx_same_grids'][tentative_keep_mask].to_numpy()
    no_unique_good_grids = len(np.unique(kept_identities))
    target_unique_min = 3
    add_back_grids = []
    if no_unique_good_grids < target_unique_min:
        # stepwise add best bad grid in
        # sort excluded grids by FR descending (NaN treated as -inf)
        def fr_key(g):
            fr = grid_fr[g]
            return -np.inf if np.isnan(fr) else fr
        excluded_sorted = sorted(excluded_grid_nos, key=fr_key, reverse=True)

        # prefer adding grids that increase identity diversity
        # ADD GRIDS THAT FIRE MOST BACK IN
        for g in excluded_sorted:
            if len(kept_identities) >= target_unique_min:
                break
            unique_id = np.unique(beh['idx_same_grids'][beh['grid_no'] == g].to_numpy())
            if unique_id not in kept_identities:
                kept_identities.add(unique_id)
                add_back_grids.append(g)
        # if still short (e.g., identity overlap), add best remaining anyway
        if len(kept_identities) < target_unique_min:
            for g in excluded_sorted:
                if g not in add_back_grids:
                    add_back_grids.append(g)


    # final per-row keep decision (BUT DO NOT FILTER beh)
    final_keep_mask = tentative_keep_mask | beh['grid_no'].isin(add_back_grids)
    
    beh[f'consistent_FR_{cell_name}'] = final_keep_mask
   
    return beh


def load_norm_data(source_folder, subject_list, res_data = False):
    # load all files I prepared with Matlab into a subject dictionary
    data_dict = {}
    for sub in subject_list:
        print(f"loading files for subject {sub}")
        sub_folder = f"{source_folder}/s{sub}/cells_and_beh"
        if not os.path.exists(sub_folder):
            print(f"files for session {sub} not found!")
            continue
        
        data_dict[f"sub-{sub}"] = {}
        #data_dict[f"sub-{sub}"]["reward_configs"] = np.genfromtxt(f"{sub_folder}/all_configs_sub{sub}.csv", delimiter=',')
        with open(f"{sub_folder}/all_cells_region_labels_sub{sub}.txt", 'r') as file:
            data_dict[f"sub-{sub}"]["cell_labels"] = [line.strip() for line in file]
                                                      
        with open(f"{sub_folder}/all_electrode_labels_sub{sub}.txt", 'r') as file:
            data_dict[f"sub-{sub}"]["electrode_labels"] = [line.strip() for line in file]
        

        data_dict[f"sub-{sub}"]["locations"] = pd.read_csv(f"{sub_folder}/locations.csv", header=None)
        data_dict[f"sub-{sub}"]["timings"] = pd.read_csv(f"{sub_folder}/state_boundaries.csv", header=None)
        data_dict[f"sub-{sub}"]["normalised_neurons"] = {}
        data_dict[f"sub-{sub}"]['beh'] = pd.read_csv(f'{sub_folder}/all_trial_times_{sub}.csv', header = None)
        column_names = ['rep_correct', 't_A', 't_B', 't_C', 't_D', 'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall', 'new_grid_onset', 'session_no', 'grid_no', 'correct']
        data_dict[f"sub-{sub}"]['beh'].columns = column_names
        if res_data == True:
            file_pattern = os.path.join(sub_folder,'cleaned_from_reps', "cell*360_bins_residualised.csv")
        else:
            file_pattern = os.path.join(sub_folder, "cell*360_bins_passed.csv")
            
        # # Use glob to find all matching CSV files
        # # file_pattern = os.path.join(sub_folder, "cell-*-360_bins.csv")
        # file_pattern = os.path.join(sub_folder,'cleaned_from_reps', "*360_bins_residualised.csv")
        # #file_pattern = os.path.join(sub_folder, "cell*360_bins_passed.csv")
        cell_files = glob.glob(file_pattern)
        for cell_file_path in cell_files:
            file_name = os.path.basename(cell_file_path)
            # cell_name = file_name[len("cell_"):-len("-360_bins.csv")]
            if res_data == True:
                cell_name = file_name[len("cell-"):-len("-360_bins_residualised.csv")]
            else:
                cell_name = file_name[len("cell-"):-len("-360_bins_passed.csv")]
            cell_name = f"{sub}_{cell_name}"
            data_dict[f"sub-{sub}"]["normalised_neurons"][cell_name] = pd.read_csv(cell_file_path, header=None)
    # import pdb; pdb.set_trace()
    return data_dict


    
    
    
def load_cell_data(source_folder, subject_list):
    # load all files I prepared with Matlab into a subject dictionary
    data_dir = {}
    for sub in subject_list:
        print(f"loading files for subject {sub}")
        data_dir[f"sub-{sub}"] = {}
        data_dir[f"sub-{sub}"]["reward_configs"] = np.genfromtxt(f"{source_folder}/s{sub}/cells_and_beh/all_configs_sub{sub}.csv", delimiter=',')
        with open(f"{source_folder}/s{sub}/cells_and_beh/all_cells_region_labels_sub{sub}.txt", 'r') as file:
            data_dir[f"sub-{sub}"]["cell_labels"] = [line.strip() for line in file]
        with open(f"{source_folder}/s{sub}/cells_and_beh/all_electrode_labels_sub{sub}.txt", 'r') as file:
            data_dir[f"sub-{sub}"]["electrode_labels"] = [line.strip() for line in file]
        # import pdb; pdb.set_trace() 
        timinges_pattern = r'timings_rewards_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        location_pattern = r'locations_per_25ms_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        cells_pattern = r'all_cells_firing_rate_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        button_pattern = r'buttons_per_25ms_grid(\d+)_sub{sub}\.csv'.replace('{sub}', str(sub))
        data_dir[f"sub-{sub}"]["timings"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/cells_and_beh", timinges_pattern)
        data_dir[f"sub-{sub}"]["locations"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/cells_and_beh", location_pattern)
        data_dir[f"sub-{sub}"]["neurons"] = read_files_to_list_ordered(f"{source_folder}/s{sub}/cells_and_beh", cells_pattern)
        
        data_dir[f"sub-{sub}"]["buttons"] = read_string_files_to_list_ordered(f"{source_folder}/s{sub}/cells_and_beh", button_pattern)
        # import pdb; pdb.set_trace() 
    return data_dir


def neurons_concat_per_ROI_acrosstasks(neuron_dict, order, unique_tasks = False, dont_average_tasks = False):
    # import pdb; pdb.set_trace()
    
    if unique_tasks == True:
        tasks_to_include = order[0:8]
    elif unique_tasks == False:
        tasks_to_include = order[0:6]
    if dont_average_tasks == True:
        tasks_to_include = ['task_A_0','task_A_1','task_A_2', 
                        'task_B_0','task_B_1','task_B_2',
                        'task_C_0', 'task_C_1','task_C_2',
                        'task_D_0', 'task_D_1', 'task_D_2',
                        'task_E_0', 'task_E_1', 'task_E_2',
                        'task_F_0', 'task_F_1', 'task_F_2']
        
    # next concatenate the neurons per ROI according to the order_of_task list
    neuron_temp_dict = {}
    for ROI in neuron_dict:
        if ROI not in neuron_temp_dict:
            neuron_temp_dict[ROI] = {}
        for task in tasks_to_include:
            task_data = {}
            for label in sorted(neuron_dict[ROI]):
                if label.startswith(task):
                    # Extract the cell label which follows the task label format "task_cellLabel"
                    _, cell_label = label.split(f"{task}_", 1)
            
                    # Initialize the cell label key if not present
                    if cell_label not in task_data:
                        task_data[cell_label] = neuron_dict[ROI][label]
                        
            # After processing all labels for the current task, add them to the main dictionary
            neuron_temp_dict[ROI][task] = task_data

    #import pdb; pdb.set_trace()
    neuron_concat_dict = {}
    for ROI in sorted(neuron_temp_dict):
        task_matrices = []
        for task in sorted(neuron_temp_dict[ROI]):
            cell_concat = []
            for cell in sorted(neuron_temp_dict[ROI][task]):
                #import pdb; pdb.set_trace()
                if len(neuron_temp_dict[ROI][task][cell]) < 2:
                    empty_cell = np.zeros((1,4))
                    cell_concat.append(empty_cell)
                else:
                    cell_concat.append(np.transpose(np.expand_dims(neuron_temp_dict[ROI][task][cell], axis = 1)))
            if cell_concat:
                #import pdb; pdb.set_trace()   
                task_concat = np.concatenate(cell_concat, axis=0)
                task_matrices.append(task_concat)

        # Concatenate all task matrices along columns (axis 1)
        if task_matrices:
            neuron_concat_dict[f"{ROI}_concat"] = np.concatenate(task_matrices, axis=1)
    # clean the 0 rows if there are any!!
    # i think some of the tasks dont have 3 repeats. I'm wondering how to solve this
    # including the same data twice would make some tasks a lot more similar to each other later
    # maybe just exclude those subjects???
    
    
    for ROI in neuron_concat_dict:
        # import pdb; pdb.set_trace()
        zero_rows = np.all(neuron_concat_dict[ROI] == 0, axis=1)
        if np.sum(zero_rows):
            print(f"careful - excluding cells for ROI {ROI}")
        not_all_zero_rows = ~zero_rows
        neuron_concat_dict[ROI] = neuron_concat_dict[ROI][not_all_zero_rows]
        
    
    return neuron_concat_dict


def count_cells_per_roi(subject_dict):
    # import pdb; pdb.set_trace()
    all_cells = []
    no_ACC, no_PCC, no_OFC, no_HC, no_EC, no_AMYG, no_mixed = 0,0,0,0,0,0,0
    for sub in subject_dict:
        for cell_idx, cell in enumerate(subject_dict[sub]['neurons'][0]):
            # create one entry in result_dict per cell 
            cell_label = f"{subject_dict[sub]['cell_labels'][cell_idx]}_{cell_idx}_{sub}"
            all_cells.append(cell_label)
            if 'ACC' in cell_label:
                no_ACC = no_ACC+1
            elif 'vCC' in cell_label:
                no_ACC = no_ACC+1
            elif 'PCC' in cell_label:
                no_PCC = no_PCC+1
            elif 'OFC' in cell_label:
                no_OFC = no_OFC+1
            elif 'HC' in cell_label:
                no_HC = no_HC+1
            elif 'MCC' in cell_label:
                no_HC = no_HC+1
            elif 'EC' in cell_label:
                no_EC = no_EC+1
            elif 'AMYG' in cell_label:
                no_AMYG= no_AMYG+1
            else:
                no_mixed=no_mixed+1
    sum_all_cells = no_ACC + no_PCC + no_OFC+ no_HC+ no_EC+ no_AMYG + no_mixed
    sum_PFC_cells = no_ACC + no_PCC + no_OFC
    sum_MTL_cells = no_HC+ no_EC+ no_AMYG
    
    print(f"We overall have {sum_all_cells} cells.")
    print(f"out of this, {no_ACC} are in ACC,{no_PCC} in PCC, and {no_OFC} in OFC. \n this makes it {sum_PFC_cells} prefrontal cells.")
    print(f"out of this, {no_HC} are in HC ,{no_EC} in EC, and {no_AMYG} in Amygdala. \n this makes it {sum_MTL_cells} MTL cells.")

    
                    
    
                    
def run_RSA(data, only_specific_model = False, per_ROI = True, plotting = True, simple_models = False, dont_avg_rep_tasks = False):
    # import pdb; pdb.set_trace()
    # normal version is going to be collapse all subjects, but split cells by ROIs
    # start by labelling the grids.
    # CONTINUE HERE!!
    if only_specific_model:
        # DO SOMETHING DIFFERENT THAN SORT THE GRIDS!!!
        prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data, unique = False)
        neurons, grid_labels = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data, specific_model = True)
        # now next step is concatenate all grids in the correct order
        # for all models.
        simulated_data_concat, order_of_tasks =  mc.analyse.helpers_human_cells.models_concat_and_avg_across_subj(prepared_data, specific_model = True)
    else:
        if dont_avg_rep_tasks == True:
            # CHANGE THIS HERE SUCH THAT ONE CAN ALSO CHOOSE TO NOT AVERAGE A TASK!
            # THIS WOULD MAKE THE RDMS BIGGER
            # WIHTOUT AVERAGING; JUST SORT BY TASK_A ETC
            prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data, unique = True, dont_average_tasks= True)
            neurons, grid_labels = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data, specific_model = False, collapse_PFC = True, dont_avg_grids=True)
            simulated_data_concat, order_of_tasks =  mc.analyse.helpers_human_cells.models_concat_and_avg_across_subj(prepared_data, specific_model = False, dont_average= True)

        else:
            # now next step is concatenate all grids in the correct order for all models.
            prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data, unique = True)
            neurons, grid_labels = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data, specific_model = False, collapse_PFC = True)
            simulated_data_concat, order_of_tasks =  mc.analyse.helpers_human_cells.models_concat_and_avg_across_subj(prepared_data, specific_model = False, dont_average= False)
      
    #import pdb; pdb.set_trace()
    
    if dont_avg_rep_tasks == False:
        # instead of pooling by subject, pool by ROI.
        # next concatenate the neurons per ROI according to the order_of_task list
        neurons_concat = mc.analyse.helpers_human_cells.neurons_concat_per_ROI_acrosstasks(neurons, order_of_tasks, unique_tasks = True)
        task_configs = order_of_tasks[0:8]
    
    elif dont_avg_rep_tasks == True:
        # instead of pooling by subject, pool by ROI.
        # next concatenate the neurons per ROI according to the order_of_task list
        neurons_concat = mc.analyse.helpers_human_cells.neurons_concat_per_ROI_acrosstasks(neurons, order_of_tasks, unique_tasks = True, dont_average_tasks=True)
        task_configs = order_of_tasks[0:8*2]
    
    RDM_dict = {}
    # plot the averaged simulated and cleaned data
    all_models_string = []
    for model in simulated_data_concat:
        all_models_string.append(model)
        # mc.simulation.predictions.plot_without_legends(simulated_data_concat[model], titlestring= f"{model} model, averaged across runs", intervalline= 4, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
        RDM_dict[model] = mc.simulation.RDMs.within_task_RDM(simulated_data_concat[model], plotting = True, titlestring = f"Between tasks {model} RSM, 4*8, averaged over runs", intervalline= 4)
    
    if only_specific_model:
        RDM_dict = {}
        all_models_string = only_specific_model.copy()
        for model in all_models_string:
            RDM_dict[model] = mc.simulation.RDMs.within_task_RDM(simulated_data_concat[model], plotting = True, titlestring = f"Between tasks {model} RSM, 4*8, averaged over runs", intervalline= 4)
    
    mc.simulation.RDMs.plot_RDMs(RDM_dict, len(task_configs))       

    if simple_models == True:
        regressors_to_include = ['location', 'curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew', 'state']
    else:
        # midn_model, phas_model, loc_model, stat_model, phas_stat, clo_model, curr_neurons
        regressors_to_include = ['clo_model', 'phas_model', 'loc_model', 'stat_model'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
    
    if only_specific_model: 
        regressors_to_include = all_models_string
        
    results= {}
    neuron_RDMs = {}
    all_rois_string = []
    for ROI in sorted(neurons_concat): 
        all_rois_string.append(ROI)
        plt.figure();
        plt.imshow(neurons_concat[ROI], aspect = 'auto')
        plt.title(ROI)
        neuron_RDMs[ROI] = mc.simulation.RDMs.within_task_RDM(neurons_concat[ROI], plotting = True, titlestring = f"Between tasks {ROI} RSM, 4*8, averaged over runs", intervalline= 4)
        regressors = {}
        for i, model in enumerate(sorted(regressors_to_include)):
            regressors[model] = RDM_dict[model].copy()
        results[ROI] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)

    mc.simulation.RDMs.plot_RDMs(neuron_RDMs, len(task_configs), flexyscale=False)       
    
    if only_specific_model == True:         
        mc.plotting.results.overview_regression(results, all_rois_string, regressors_to_include, combo = True)  
    
    if only_specific_model == False:  
        results_single_model = {}
        for ROI in neurons_concat:
            results_single_model[ROI] = {}
            for model in all_models_string:
                regressors = {}
                regressors[model] = RDM_dict[model].copy()
                results_single_model[ROI][model] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)
       
        mc.plotting.results.overview_regression(results_single_model, all_rois_string, all_models_string)
        
        if simple_models == True:
            regressors_to_include = ['curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew']
        else:
            # ['two_fut_rings_split_clock', 'phas_stat', 'phas_model', 'three_fut_rings_split_clock', 'loc_model', 'curr_rings_split_clock', 'stat_model', 'clo_model', 'midn_model', 'one_fut_rings_split_clock']
            regressors_to_include = ['clo_model', 'midn_model'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
        regressors = {}
        for i, model in enumerate(sorted(regressors_to_include)):
            regressors[model] = RDM_dict[model].copy()
        results= {}
        for ROI in neurons_concat: 
            results[ROI] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)
        mc.plotting.results.overview_regression(results, all_rois_string, regressors_to_include, combo = True)  
        
            
        if simple_models == True:
            regressors_to_include = ['curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew', 'state']
        else:
            # ['two_fut_rings_split_clock', 'phas_stat', 'phas_model', 'three_fut_rings_split_clock', 'loc_model', 'curr_rings_split_clock', 'stat_model', 'clo_model', 'midn_model', 'one_fut_rings_split_clock']
            regressors_to_include = ['clo_model', 'phas_model'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
        regressors = {}
        for i, model in enumerate(sorted(regressors_to_include)):
            regressors[model] = RDM_dict[model].copy()
        results= {}
        for ROI in neurons_concat: 
            results[ROI] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)
        mc.plotting.results.overview_regression(results, all_rois_string, regressors_to_include, combo = True)  
        
        if simple_models == False:
            # ['two_fut_rings_split_clock', 'phas_stat', 'phas_model', 'three_fut_rings_split_clock', 'loc_model', 'curr_rings_split_clock', 'stat_model', 'clo_model', 'midn_model', 'one_fut_rings_split_clock']
            regressors_to_include = ['curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
            regressors = {}
            for i, model in enumerate(sorted(regressors_to_include)):
                regressors[model] = RDM_dict[model].copy()
            results= {}
            for ROI in neurons_concat: 
                results[ROI] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], regressors, mask_within = True, no_tasks = len(task_configs), plotting= False)
            mc.plotting.results.overview_regression(results, all_rois_string, regressors_to_include, combo = True)  
            

    import pdb; pdb.set_trace()
    return results



def run_state_RSA(data,  per_ROI = True, plotting = True, only_reward_times = None, no_bins_per_state= None, sim_fake_data =False):
    
    # normal version is going to be collapse all subjects, but split cells by ROIs
    # start by labelling the grids.

    prepared_data = mc.analyse.helpers_human_cells.label_unique_grids(data, unique = False)
    neurons, grids = mc.analyse.helpers_human_cells.pool_by_ROI_and_grid(prepared_data, specific_model = True)

        
    # now next step is concatenate all grids in the correct order
    # for all models.

    # instead, create state model in the correct order.
    states = ['A','B','C','D']
    for ROI in neurons:
        for task in neurons[ROI]:
            length_averaged_task = len(neurons[ROI][task])
            
    state_model = np.zeros((int(length_averaged_task),len(states)))
    length_state = int(len(neurons[ROI][task])/len(states))

    for s_i, state in enumerate(states):
        if s_i == 0:
            state_model[0:length_state,s_i] = 1
        else:
            state_model[length_state*s_i:length_state*(s_i+1),s_i] = 1
    state_model = np.tile(np.transpose(state_model), (1,len(grids)))
    # pool by ROI and concatenate the neurons per ROI according to the order_of_task list
    neurons_concat = mc.analyse.helpers_human_cells.neurons_concat_per_ROI_acrosstasks(neurons, grids)
    
    RDM_dict = {}
    RDM_dict['state'] = mc.simulation.RDMs.within_task_RDM(state_model, plotting = True, titlestring = f"Between tasks state RSM, {4*len(grids)}, averaged over runs", intervalline= length_averaged_task)
    
    
    if sim_fake_data:
        fake_neurons = {}
        fake_neurons['fake_ROI'], grid_labels = mc.analyse.helpers_human_cells.pool_by_grid(prepared_data)
        fake_neurons_concat = mc.analyse.helpers_human_cells.neurons_concat_per_ROI_acrosstasks(fake_neurons, grids)
        #RDM_dict['fake_neurons_RDM'] = mc.simulation.RDMs.within_task_RDM(fake_neurons_concat['fake_ROI_concat'], plotting = True, titlestring = f"Between tasks simualted neurons RSM, {4*len(grids)}, averaged over runs", intervalline= length_averaged_task)
        neurons_concat['fake_neurons'] = fake_neurons_concat['fake_ROI_concat'].copy()

    mc.simulation.RDMs.plot_RDMs(RDM_dict, len(grids))       
    results= {}
    neuron_RDMs = {}
    all_rois_string = []
    for ROI in sorted(neurons_concat): 
        results[ROI] = {}
           
    for ROI in sorted(neurons_concat): 
        all_rois_string.append(ROI)
        plt.figure();
        plt.imshow(neurons_concat[ROI], aspect = 'auto')
        plt.title(ROI)
        neuron_RDMs[ROI] = mc.simulation.RDMs.within_task_RDM(neurons_concat[ROI], plotting = True, titlestring = f"Between tasks {ROI} RSM, {4*len(grids)}, averaged over runs", intervalline= length_averaged_task)
        results[ROI]['state'] = mc.simulation.RDMs.GLM_RDMs(neuron_RDMs[ROI], RDM_dict, mask_within = True, no_tasks = len(grids), t_val = True, plotting= False)
    mc.simulation.RDMs.plot_RDMs(neuron_RDMs, len(grids), flexyscale=False)       
    mc.plotting.results.overview_regression(results, all_rois_string, ['state'], combo = False, only_reward_times =only_reward_times, no_bins_per_state =no_bins_per_state)  

    import pdb; pdb.set_trace()
    return results    







def models_concat_and_avg_across_subj(data, specific_model = False, dont_average = False):
    # concatenate all grids in the correct order
    # first loop: collect the same model for the same task for all subjects with np.stack
    # second loop: average the same model same task across all subjects
    # and in the end average across task repeats
    # first figure out which grids to consider.
    temp_stacked = {}
    all_models_set = set()
    
    if specific_model == True:
        grid_labels = ['average_task_A', 'average_task_B', 'average_task_C', 
                       'average_task_D', 'average_task_E', 'average_task_F']
        grid_distribution = mc.analyse.helpers_human_cells.redistribute_grids(data, grid_labels)

    if dont_average == True:
        grid_labels = ['task_A', 'task_B', 'task_C', 
                           'task_D', 'task_E', 'task_F']
        

        for sub in data:
            for task in grid_labels:  
                count_to_three = 0
                for grid in data[sub]:
                    if count_to_three > 1:
                        continue
                    elif grid.startswith(task):
                        temp_stacked[f"{task}_{count_to_three}_{sub}"] = {}
                        count_to_three = count_to_three+1
        
        # if no average across tasks, only average across subjects.
        for sub in data:
            for task in grid_labels:  
                count_to_three = 0
                for grid in data[sub]:
                    if count_to_three > 1:
                        continue
                    elif grid.startswith(task):
                        curr_grid = f"{task}_{count_to_three}_{sub}"
                        count_to_three = count_to_three+1
                        
                        if grid == 'task_A_3_sub_sub-43':
                            grid = 'task_A_14_sub_sub-43'
                        if grid == 'task_A_19_sub_sub-51':
                            grid = 'task_A_11_sub_sub-51'
                        for model in data[sub][grid]:
                            if model not in ['neurons']:
                                all_models_set.add(model)
                                # here always append 3x 
                                if model not in temp_stacked[curr_grid]:
                                    temp_stacked[curr_grid][model] = [data[sub][grid][model]]
                                else:
                                    temp_stacked[curr_grid][model].append(data[sub][grid][model])
            
        temp_across_subs = {}
        for task in temp_stacked:
            task_across_subs = task.split("_sub")[0]
            temp_across_subs[task_across_subs] = {}
            for model in temp_stacked[task]:
                temp_across_subs[task_across_subs][model] = np.stack(temp_stacked[task][model], axis = 0)
        
        for task in temp_across_subs:
            for model in temp_across_subs[task]:
                temp_across_subs[task][model] = np.mean(temp_across_subs[task][model], axis = 0)
    
        # then, last step, concatenate all tasks. store which order you concatenated.
        models_task_concat = {}
        order_of_tasks = []
        for model in all_models_set:
            for task in sorted(temp_across_subs):
                order_of_tasks.append(task)
                if model not in models_task_concat:
                    models_task_concat[model] = temp_across_subs[task][model]
                else:
                    models_task_concat[model] = np.concatenate((models_task_concat[model], temp_across_subs[task][model]), axis = 1)
            
    
    

    elif dont_average == False:                  
        for sub in data:
            count_to_three = 0
            for grid in data[sub]: 
                if grid.startswith('average'):
                    if specific_model == True:
                        # first redistribute the tasks.
                        # Determine which predefined grid label this grid should be associated with
                        for label, grids in grid_distribution.items():
                            if grid in grids or grid == label:
                                current_grid = label  # Use the predefined grid label for processing
                                break
                    else: 
                        current_grid = grid 
                    temp_stacked[current_grid] = {}

        # reduce data for models and grids by subject dimension, make a list instead.
        for sub in data:
            for grid in data[sub]:
                if grid.startswith('average'):
                    if specific_model == True:
                        # first redistribute the tasks.
                        # Determine which predefined grid label this grid should be associated with
                        for label, grids in grid_distribution.items():
                            if grid in grids or grid == label:
                                current_grid = label  # Use the predefined grid label for processing
                                break
                    else: 
                        current_grid = grid 
                    for model in data[sub][current_grid]:
                        if model not in ['neurons']:
                            all_models_set.add(model)
                            if model not in temp_stacked[current_grid]:
                                temp_stacked[current_grid][model] = [data[sub][current_grid][model]]
                            else:
                                temp_stacked[current_grid][model].append(data[sub][current_grid][model])

        # stack all lists and average across subjects
        for task in temp_stacked:
            # probably do something like split after sub and then average across subs.
            for model in temp_stacked[task]:
                temp_stacked[task][model] = np.stack(temp_stacked[task][model], axis = 0)
            for model in temp_stacked[task]:
                temp_stacked[task][model] = np.mean(temp_stacked[task][model], axis = 0)
                      
        # then, last step, concatenate all tasks. store which order you concatenated.
        models_task_concat = {}
        order_of_tasks = []
        for model in all_models_set:
            for task in sorted(temp_stacked):
                order_of_tasks.append(task)
                if model not in models_task_concat:
                    models_task_concat[model] = temp_stacked[task][model]
                else:
                    models_task_concat[model] = np.concatenate((models_task_concat[model], temp_stacked[task][model]), axis = 1)
        
    return models_task_concat, order_of_tasks


def threshold_round(v, threshold=0.5):
    return np.round(v).astype(int) if threshold == 0.5 else np.floor(v + (1 - threshold)).astype(int)


def neurons_to_bins_multimodel(time_course, model_dict):
    for i, m in enumerate(model_dict):
        model_idx = np.transpose(copy.deepcopy(model_dict[m]))
        for idx, neuron in enumerate(model_idx):
            model_idx[idx] = neuron*(idx+1)
        if i == 0:
            model_bins = np.sum(model_idx, axis = 0)
        else:
            # import pdb; pdb.set_trace()
            model_bins = model_bins + np.sum(model_idx, axis = 0)
        
    # Identify change points (boundaries) in the index array
    # Apply custom rounding
    rounded_vec = threshold_round(model_bins, threshold=0.2)
    
    change_points = np.where(rounded_vec[1:] != rounded_vec[:-1])[0] + 1
    
    bins = np.split(time_course, change_points)  # Split the neural array at change points
        
    # Compute the average for each segment
    timecourse_binned = np.array([segment.mean(axis = 0) for segment in bins])
    return timecourse_binned

    


    

def neurons_to_state_bins(time_course, state_model):
    state_model_idx = np.transpose(copy.deepcopy(state_model))
        
    for idx, state in enumerate(state_model_idx):
        state_model_idx[idx] = state*(idx+1)
    
    state_model_bins = np.sum(state_model_idx, axis = 0)
    # Define threshold (e.g., values within 0.5 of an integer are rounded to that integer)

    # Apply custom rounding
    rounded_vec = threshold_round(state_model_bins, threshold=0.2)
    # Identify change points
    state_change_points = np.where(rounded_vec[1:] != rounded_vec[:-1])[0] + 1

    state_bins = np.split(time_course, state_change_points)  # Split the neural array at change points
    
    # Compute the average for each segment
    timecourse_binned = np.array([segment.mean(axis = 0) for segment in state_bins])
     
    # import pdb; pdb.set_trace()
    # ok i dont know what the fuck is happening
    # there is some weird shit going on about float to integer conversion
    # fix this!!
    # i need these to be exactly number/3 state change points!!
    # also make sure to include the last one...
    # test_idx = np.transpose(state_model)
    # for idx, state in enumerate(state_model_idx):
    #     test_idx[idx] = np.ceil(state*(idx+1))
            
    # state_model_bins = [int(e) for e in state_model_bins]
    # # Identify change points (boundaries) in the index array
    # state_change_points = np.where(np.diff(state_model_bins) != 0)[0] + 1  # Find where index changes
    # state_change_points = np.where(state_model_bins[:-1] != vec[1:])[0] + 1
    return timecourse_binned
    


def redistribute_grids(data_dict, grid_string):
    # Step 1: Find all unique grids
    all_grids = set()
    for sub in data_dict:
        for grid in data_dict[sub]:
            if grid.startswith('average'):
                all_grids.add(grid)
    
    # Step 2: Separate additional grids
    additional_grids = [grid for grid in all_grids if grid not in grid_string]
    
    # Initialize a counter for distributing grids
    from collections import defaultdict
    grid_distribution = defaultdict(list)
    
    # Step 3: Distribute additional grids equally among predefined grid_labels
    for i, grid in enumerate(additional_grids):
        assigned_label = grid_string[i % len(grid_string)]
        grid_distribution[assigned_label].append(grid)
        
    return grid_distribution
      

def pool_by_grid(data, specific_model = False):
    neurons = {}
    if specific_model == True:
        grid_labels = ['average_task_A', 'average_task_B', 'average_task_C', 
                       'average_task_D', 'average_task_E', 'average_task_F']
        grid_distribution = mc.analyse.helpers_human_cells.redistribute_grids(data, grid_labels)
    else:
        grid_labels = []
        
    for sub in data:
        for grid in data[sub]:
            if grid.startswith('average'):
                if specific_model == True:
                    # first redistribute the tasks.
                    # Determine which predefined grid label this grid should be associated with
                    for label, grids in grid_distribution.items():
                        if grid in grids or grid == label:
                            current_grid = label  # Use the predefined grid label for processing
                            break
                else:
                    current_grid = grid
                # Loop through cells in the subject's data    
                for i_c, cell in enumerate(data[sub][grid]['fake_neurons']):    
                    neurons[f"{current_grid}_{sub}_fake_ROI_{i_c}"] = cell
                                
    return neurons, grid_labels

    

                          
def pool_by_ROI_and_grid(data, specific_model = False, collapse_PFC = False, dont_avg_grids = False):
    # first create the ROI sets
    # import pdb; pdb.set_trace()
    ROIs = {}
        
    if collapse_PFC == True:
        ROI_labels = ['hippocampal', "PFC", "entorhinal", "amygdala", "mixed"]
    else:
        ROI_labels = ['hippocampal', "PCC", "ACC", "OFC", "entorhinal", "amygdala", "mixed"]
    for ROI in ROI_labels:
        ROIs[ROI] = set()
    for sub in data:
        for region in data[sub]['cell_labels']:
            if collapse_PFC == True:
                if 'ACC' in region or 'PCC' in region or 'OFC' in region:
                    ROIs['PFC'].add(region)
                elif 'HC' in region:
                    ROIs['hippocampal'].add(region)
                elif 'EC' in region:
                    ROIs['entorhinal'].add(region)
                elif 'AMYG' in region:
                    ROIs['amygdala'].add(region)
                    
                else:
                    ROIs['mixed'].add(region)
            else:
                if 'ACC' in region:
                    ROIs['ACC'].add(region)
                elif 'PCC' in region:
                    ROIs['PCC'].add(region)
                elif 'OFC' in region:
                    ROIs['OFC'].add(region)   
                elif 'HC' in region:
                    ROIs['hippocampal'].add(region)
                elif 'EC' in region:
                    ROIs['entorhinal'].add(region)
                elif 'AMYG' in region:
                    ROIs['amygdala'].add(region)
                    
                else:
                    ROIs['mixed'].add(region)

                    
                    
    print(f"mixed ROI includes: {ROIs['mixed']}")
    ROIs['entorhinal'] = {'REC', 'LEC'}
    # then create a neuron dictionary, split by ROIs, where I save how each cell 
    # of that ROI fires for each grid.
    
    if specific_model == True:
        grid_labels = ['average_task_A', 'average_task_B', 'average_task_C', 
                       'average_task_D', 'average_task_E', 'average_task_F']
        grid_distribution = mc.analyse.helpers_human_cells.redistribute_grids(data, grid_labels)
    else:
        grid_labels = []
    
        
    if dont_avg_grids == True:
        grid_labels = ['task_A', 'task_B', 'task_C', 
                       'task_D', 'task_E', 'task_F']
    neurons = {}
    for ROI in ROIs:
        neurons[ROI] = {}
    # import pdb; pdb.set_trace()  
    if dont_avg_grids == True:
        for sub in data:
            for task in grid_labels:  
                count_to_three = 0
                for grid in sorted(data[sub]):
                    if count_to_three > 1:
                        continue
                    elif grid.startswith(task):
                        if grid == 'task_A_3_sub_sub-43':
                            grid = 'task_A_14_sub_sub-43'
                        if grid == 'task_A_19_sub_sub-51':
                            grid = 'task_A_11_sub_sub-51'
                        
                        curr_grid = f"{task}_{count_to_three}_{sub}"
                        count_to_three = count_to_three+1
                        
                        # Loop through cells in the subject's data    
                        for i_c, cell in enumerate(data[sub]['cell_labels']):
                            # inner: loop through new dict
                            for ROI in ROIs:
                                if cell in ROIs[ROI]:
                                    if neurons[ROI].get(f"{curr_grid}_{i_c}_{cell}") is None:
                                        # if grid in ['task_A_3_sub_sub-43', 'task_A_19_sub_sub-51']:
                                        #     continue
                                        # else:
                                        neurons[ROI][f"{curr_grid}_{i_c}_{cell}"] = data[sub][grid]['neurons'][i_c]

    else:
        # outer: loop through old data dict
        for sub in data:
            for grid in data[sub]:
                if grid.startswith('average'):
                    if specific_model == True:
                        # first redistribute the tasks.
                        # Determine which predefined grid label this grid should be associated with
                        for label, grids in grid_distribution.items():
                            if grid in grids or grid == label:
                                current_grid = label  # Use the predefined grid label for processing
                                break
                    else:
                        current_grid = grid
                    # Loop through cells in the subject's data    
                    for i_c, cell in enumerate(data[sub]['cell_labels']):
                        # inner: loop through new dict
                        for ROI in ROIs:
                            if cell in ROIs[ROI]:
                                if neurons[ROI].get(f"{current_grid}_{sub}_{i_c}_{cell}") is None:
                                    neurons[ROI][f"{current_grid}_{sub}_{i_c}_{cell}"] = data[sub][grid]['neurons'][i_c]
                                    
                                
                                
    return neurons, grid_labels

        
        
    
  
def label_unique_grids(data_dict, unique = True, dont_average_tasks = False):
    # import pdb; pdb.set_trace()
    if unique == True:
        task_labels = {}
        for sub in data_dict:
            task_labels[sub] = {'task_A': ([1., 9., 5., 8.]), 'task_B': ([2., 5., 7., 6.]), 'task_C': ([3., 7., 9., 5.]), 'task_D': ([4., 8., 1., 3.]), 'task_E': ([6., 4., 2., 9.]), 'task_F': ([7., 3., 4., 2.]), 'task_G': ([8., 2., 6., 7.]), 'task_H': ([9., 1., 3., 4.])}
        #task_labels = {'task_A': ([1., 9., 5., 8.]), 'task_B': ([2., 5., 7., 6.]), 'task_C': ([3., 7., 9., 5.]), 'task_D': ([4., 8., 1., 3.]), 'task_E': ([6., 4., 2., 9.]), 'task_F': ([7., 3., 4., 2.]), 'task_G': ([8., 2., 6., 7.]), 'task_H': ([9., 1., 3., 4.])}
    else:
        # import pdb; pdb.set_trace()
        task_labels = {}
        for sub in data_dict:
            task_labels[sub] = {}
            unique_quadruplets = {tuple(row) for row in data_dict[sub]['reward_configs']}
            for i, quad in enumerate(unique_quadruplets):
                task_labels[sub][f'task_{chr(65 + i)}'] = [int(num) for num in quad]  # chr(65) is 'A'
                
    
        for sub in data_dict:
            keys_to_modify = []
            for unique_task in task_labels[sub]:
                for grid_idx, grid in enumerate(data_dict[sub]['reward_configs']):
                    if list(grid) == list(task_labels[sub][unique_task]):
                        if sub == 'sub-43' and grid_idx == 3:
                            continue
                        if sub == 'sub-15' and grid_idx == 23:
                            continue
                        new_key = f"{unique_task}_grid_{grid_idx}_sub_{sub}"
                        old_key = f"grid_{grid_idx}"
                        keys_to_modify.append((old_key, new_key))
            for old_grid_key, new_grid_key in keys_to_modify:
                # print(old_grid_key, new_grid_key)
                data_dict[sub][new_grid_key] = data_dict[sub].pop(old_grid_key)
    
    if dont_average_tasks == False:                      
        for sub in data_dict:
            # last, but not least, create grid averages.
            # if there aren't 3 averages, just make 2 averages instead.
            # import pdb; pdb.set_trace()
            for unique_task in task_labels[sub]:
                data_dict[sub][f"average_{unique_task}"] = {}
                for grid in data_dict[sub]:
                    if grid.startswith(unique_task):
                        for model in data_dict[sub][grid]:
                            if model not in data_dict[sub][f"average_{unique_task}"]:
                                data_dict[sub][f"average_{unique_task}"][model] = [data_dict[sub][grid][model]]
                            else:
                               data_dict[sub][f"average_{unique_task}"][model].append(data_dict[sub][grid][model]) 
        # and in the end average across task repeats
        for sub in data_dict:
            for grid in data_dict[sub]:
                temp_stacked = {}
                if grid.startswith('average'):
                    for model in data_dict[sub][grid]:
                        # first collect all grids that are the same
                        # Stack the arrays along a new axis (axis 0)
                        temp_stacked[model] = np.stack(data_dict[sub][grid][model], axis=0)
                        # Compute the mean across the stacked dimension
                    # then average
                    for model in temp_stacked:
                        data_dict[sub][grid][model] = np.mean(temp_stacked[model], axis=0)
                        
    # i think basically don't do anything here if no average is needed.
    
    # elif dont_average_tasks == True:
    #     import pdb; pdb.set_trace()
    #     for sub in data_dict:
    #         # dont create averages across tasks.
    #         for unique_task in task_labels[sub]:
    #             data_dict[sub][f"average_{unique_task}"] = {}
    #             for grid in data_dict[sub]:
    #                 if grid.startswith(unique_task):
    #                     for model in data_dict[sub][grid]:
    #                         if model not in data_dict[sub][f"average_{unique_task}"]:
    #                             data_dict[sub][f"average_{unique_task}"][model] = [data_dict[sub][grid][model]]
    #                         else:
    #                            data_dict[sub][f"average_{unique_task}"][model].append(data_dict[sub][grid][model]) 
        
        
    #     for sub in data_dict:
    #         for grid in data_dict[sub]:
    #             temp_stacked = {}
    #             for task in task_labels[sub]:
    #                 if grid.startswith(task):
    #                     for model in data_dict[sub][grid]:
    #                         # first collect all grids that are the same
    #                         # Stack the arrays along a new axis (axis 0)
    #                         temp_stacked[model] = np.stack(data_dict[sub][grid][model], axis=0)
    #                         # Compute the mean across the stacked dimension
    #                     # then average
    #                     for model in temp_stacked:
    #                         data_dict[sub][grid][model] = np.mean(temp_stacked[model], axis=0)
    return data_dict


def clean_data(data, s):
    # hard-code problemgrids for each s
    # import pdb; pdb.set_trace()
    if s == 'sub-27':
        problem_grid_idx = [4,7]
    if s == 'sub-59':
        problem_grid_idx = [21]
    if s == 'sub-15':
        problem_grid_idx = [23]
    if s == 'sub-43':
        problem_grid_idx = [3]
    if s == 'sub-44':
        problem_grid_idx = [3]
        
    data_prep_tmp = copy.deepcopy(data)
    indices_to_keep = [i for i in range(len(data_prep_tmp[s]['buttons'])) if i not in problem_grid_idx]
    for entry in ['buttons', 'locations', 'neurons', 'timings']:
        data[s][entry] = [data_prep_tmp[s][entry][i] for i in indices_to_keep]
    
    data[s]['reward_configs'] = np.delete(data_prep_tmp[s]['reward_configs'], problem_grid_idx, axis=0)

        # data_prep_tmp = copy.deepcopy(data)
        # for entry in ['buttons', 'locations', 'neurons', 'timings']:
        #     data[s][entry] = [x for i, x in enumerate(data_prep_tmp[s][entry]) if i != problem_grid_idx]
        # data[s]['reward_configs'] =  np.delete(data_prep_tmp[s]['reward_configs'], problem_grid_idx, axis = 0)
        
        
        
        # for problem_grid_idx in problems:
        #     for entry in ['buttons', 'locations', 'neurons', 'timings']:
        #         data[s][entry] = [x for i, x in enumerate(data_prep_tmp[s][entry]) if i != problem_grid_idx]
        #     data[s]['reward_configs'] =  np.delete(data_prep_tmp[s]['reward_configs'], problem_grid_idx, axis = 0)
            
    
    return data[s]


    
def generate_circular_timepoint_permutations_neurons(neurons, n_perms = 10):
    """
    Perform circular timepoint permutations on multi-task neural data.

    Parameters:
    - neurons: list of np.arrays with shape (n_neurons, n_timepoints), one per task
    - n_perms: number of circular permutations to generate

    Returns:
    - permuted_neurons: list of length n_perms
        Each element is a list of 19 arrays (one per task), like the input
    """
    # import pdb; pdb.set_trace()
    # Step 1: Concatenate all neurons along the time axis
    original_shapes = [arr.shape[1] for arr in neurons]  # timepoints per task
    concatenated = np.concatenate(neurons, axis=1)       # shape: (n_neurons, total_time)

    total_time = concatenated.shape[1]
    shift_step = total_time // n_perms                   # how far to shift each time

    permuted_neurons = []

    for i in range(n_perms):
        shift = shift_step * (i + 1)                     # avoid shift=0 for the first perm
        rotated = np.roll(concatenated, shift=shift, axis=1)

        # Split back into original segments
        split_data = []
        idx = 0
        for length in original_shapes:
            split_data.append(rotated[:, idx:idx + length])
            idx += length

        permuted_neurons.append(split_data)   

    return permuted_neurons


def pair_grids_to_increase_spatial_coverage(locs, beh, cell_name, min_coverage=100, min_groups=3,max_groups=5):
    """
    Merge original grids into groups to maximize spatial coverage.
    Rules:
      - <3 unique grids  -> discard neuron (output False).
      - 3 -> keep as 3 singletons.
      - 4 -> pair the two worst; others singleton (3 groups).
      - 5 -> pair worst 4 (2 pairs), keep best alone (3 groups).
      - 6 -> 3 pairs (maximize coverage).
      - 7 -> 3 pairs + best alone (4 groups).
      - 8 -> 4 pairs (maximize coverage).
      - 9 -> 4 pairs + best alone (5 groups).
      - 10 -> 5 pairs.
      - >10 -> make 5 groups: start with 5 best pairs, then add the rest to
               existing groups (triplets) maximizing coverage gain.
    Writes integer labels to beh[f'paired_grid_idx_{cell_name}'].
    """
    # --- get same_grids (optionally filter by consistent_FR) ---
    if f"consistent_FR_{cell_name}" in beh:
        # first filter locations and same_grids for grids that are reliable.
        reliable_FR_mask = beh[f"consistent_FR_{cell_name}"].to_numpy()
        locs_used = locs[reliable_FR_mask]
        same_grids = beh['idx_same_grids'][reliable_FR_mask].to_numpy()
    else:
        reliable_FR_mask = None
        locs_used = locs
        same_grids = beh['idx_same_grids'].to_numpy()
    
    # if cell_name == '05-05-mRF3cVPF04-RPvmPFC':
    #     import pdb; pdb.set_trace()
    #     # WHY DOES THIS CLUMP MORE THAN 3 GRIDS TOEGTEHR????
    
    unique_grids = np.unique(same_grids)
    

    # discard if fewer than 3 unique grids (column = False)
    if len(unique_grids) < 3:
        col = f'paired_grid_idx_{cell_name}'
        if reliable_FR_mask is not None:
            out = np.full(reliable_FR_mask.shape, False, dtype=object)
            beh[col] = out
        else:
            beh[col] = np.full(same_grids.shape, False, dtype=object)
        return beh
    

    # --- 1) build coverage dict for each original grid ---
    grid_cvg = {}
    
    for grid_idx in unique_grids:
        grid_cvg_vec = np.zeros(9, dtype=int)
        all_locs_curr_grid = locs_used[same_grids == grid_idx]
        for loc in range(1,10):
            grid_cvg_vec[loc-1] = np.count_nonzero(all_locs_curr_grid == loc)
        grid_cvg[grid_idx] = grid_cvg_vec
        
    
    # --- helpers for scoring coverage and choosing pairs ---
    def group_score(grids):
        """
        Score to maximize:
          1) # of locations with coverage >= min_coverage (higher is better)
          2) minimum coverage across locations (higher is better)
        """
        tot = sum((grid_cvg[g] for g in grids), np.zeros(9, dtype=int))
        n_cov_good = int(np.sum(tot >= min_coverage)) 
        min_cvg = int(np.min(tot)) if tot.size else 0
        return (n_cov_good, min_cvg)
    
    def worst_sort_key(g):
        """
        Higher = worse:
          1) more weak locations (< min_coverage)  -> worse
          2) lower minimum coverage                -> worse
        """
        v = grid_cvg[g]
        weak = int(np.sum(v < min_coverage))
        minv = int(np.min(v))
        return (weak, -minv)   # more weak first; for ties, lower min (i.e., -minv higher)

    def best_sort_key(g):
        """
        Higher = better:
          1) fewer weak locations (< min_coverage) -> better
          2) higher minimum coverage               -> better
        """
        v = grid_cvg[g]
        weak = -int(np.sum(v < min_coverage))  # fewer weak -> larger value
        minv = int(np.min(v))                  # higher min -> larger value
        return (weak, minv)

    def best_pair(rem):
        """Pick (a,b) maximizing group_score({a,b}) with the above priority."""
        rem = list(rem)
        best = None
        for i in range(len(rem)):
            for j in range(i+1, len(rem)):
                a, b = rem[i], rem[j]
                # Comparisons are lexicographic on (n_good, min_cvg), 
                # so “passing the threshold” dominates, and “raising the floor” is second.
                sc = group_score([a, b]) # (n_good, min_cvg)
                if best is None or sc > best[0]:
                    best = (sc, (a, b))
        # best[1] = pair of best fitting grids.
        return best[1] if best else None


     # --- 2) build groups according to n and your rules ---
    groups = []  # list of lists of grid ids
    remaining = set(unique_grids.tolist())


    # choose a single "best" to leave alone in cases 5,7,9
    def pick_best_single(rem):
        # builds scores out of neg 'passes coverage' and minimum coverage
        # takes max(weak, minv) per grid
        return max(rem, key=best_sort_key)
    
    # choose k pairs greedily from remaining
    def add_k_pairs(rem, k):
        for _ in range(k):
            if len(rem) < 2: break
            a_b = best_pair(rem) # a_b = pair of best fitting grids.
            if a_b is None: break
            a, b = a_b
            groups.append([a, b])
            # then remove the pairs that have just been added
            rem.remove(a); rem.remove(b)
    
    if len(unique_grids) == 3:
        # 3 singles
        groups = [[g] for g in unique_grids]
        remaining.clear()
    
    elif len(unique_grids) == 4:
        # pair worst 2; others singleton
        worst2 = sorted(remaining, key=worst_sort_key, reverse=True)[:2]
        groups.append(list(worst2))
        for g in worst2: remaining.remove(g)
        for g in sorted(remaining): groups.append([g])
        remaining.clear()

        
    elif len(unique_grids) in [5,7,9]:
        # best alone; pair worst 4 (2 pairs)
        best_single = pick_best_single(remaining)
        remaining.remove(best_single)
        # after removing the best grid, pair the remaining ones up as usual
        add_k_pairs(remaining, int((len(unique_grids)-1)/2))
        # in the end, add the single grid to the groups and delete all used grids
        groups.append([best_single])
        remaining.clear()
    
    elif len(unique_grids) in [6,8,10]:
        add_k_pairs(remaining, int(len(unique_grids)/2))
    
    else:
        # n > 10  →  make exactly 5 groups max
        add_k_pairs(remaining, max_groups)  # start with 5 best pairs
        # add leftovers to existing groups to maximize coverage gain
        while remaining:
            g = remaining.pop()
            best_gain, best_idx = None, None
            for idx, grp in enumerate(groups):
                # find out which group improves most if left-over is added
                base = group_score(grp)
                new  = group_score(grp + [g])
                gain = tuple(np.array(new) - np.array(base))
                if best_gain is None or gain > best_gain:
                    best_gain, best_idx = gain, idx
            groups[best_idx].append(g)

    # --- 3) map old -> new labels and scatter back to DataFrame ---
    label = {}
    for new_id, grp in enumerate(groups):
        for g in grp:
            label[int(g)] = int(new_id)

    same_grid_idx_new = np.array([label[int(g)] for g in same_grids], dtype=int)

    if reliable_FR_mask is not None:
        # put back into shape with False for grids that have insufficient firing
        out = np.full(reliable_FR_mask.shape, False, dtype=object)  
        out[reliable_FR_mask] = same_grid_idx_new
        beh[f'paired_grid_idx_{cell_name}'] = out
    else:
        beh[f'paired_grid_idx_{cell_name}'] = same_grid_idx_new
        
    return beh

#
#
#
# new 06.09.2025
# adjusted such that it matches spatial consistency analysis

def prep_regressors_for_neurons_360(data_dict):
    no_state = 4
    state_width = 90
    no_locations = 9
    no_phase = 3
    phase_width = 30
        
    models = ['state_model', 'phase_model', 'phase_state_model', 'location_model', 'reward_loc_model', 'future_location_x_phase_model', 'future_reward_x_state_model']   
    # clean subjects that messed up certain grids 
    for sub in data_dict:
        print(f"now starting to process data from subject {sub}")

    data_prep = copy.deepcopy(data_dict)

    for sub in data_dict:
        print(f"now starting to process data from subject {sub}")
        for m in models:
            if m == 'state_model':
                data_prep[sub][m]= np.zeros((len(data_dict[sub]['beh']), no_state, 360))
            elif m == 'phase_model':
                data_prep[sub][m]= np.zeros((len(data_dict[sub]['beh']), no_phase, 360))
            elif m == 'phase_state_model':
                data_prep[sub][m]= np.zeros((len(data_dict[sub]['beh']), no_phase*no_state, 360))
            elif m == 'location_model':
                data_prep[sub][m]= np.zeros((len(data_dict[sub]['beh']), no_locations, 360))
            elif m == 'reward_loc_model':
                data_prep[sub][m]= np.zeros((len(data_dict[sub]['beh']), no_locations, 360))
            elif m == 'future_location_x_phase_model':
                data_prep[sub][m]= np.zeros((len(data_dict[sub]['beh']), no_locations*no_phase*no_state, 360))
            elif m == 'future_reward_x_state_model':
                data_prep[sub][m]= np.zeros((len(data_dict[sub]['beh']), no_locations*no_state, 360))
        
        # reward_configurations = data_dict[sub]['reward_configs'].copy()
        # unique_grids = np.unique(data_dict[sub]['beh_clean']['grid_no'].to_numpy())
        
        # ok new attempt 07.09.2025.
        # just model whatever is in each row.
        # the indices should align with the location data.
        # state + phase is easy to define.
        # locations are just a read out.
        # model neurons for each row!
        for idx, row in data_dict[sub]['beh'].iterrows():
            for m in models:
                if m == 'state_model':
                    for s in range(no_state):
                        data_prep[sub][m][idx, s, state_width*s:state_width*(s+1)] = 1

                elif m == 'phase_model':
                    mask = ((np.arange(360)//phase_width) % 3)[None, :] == np.arange(3)[:, None]
                    # assign to the first three rows for this idx
                    data_prep[sub][m][idx, :3, :] = mask.astype(data_prep[sub][m].dtype)
                    
                elif m == 'phase_state_model':
                    for s in range(no_state):       # state 0..3
                       for p in range(no_phase):   # phase 0..2
                           r = s*3 + p
                           start = s*state_width + p*phase_width
                           data_prep[sub][m][idx, r, start:start+phase_width] = 1
                           
                elif m == 'reward_loc_model':
                    data_prep[sub][m][idx, int(row['loc_A']-1), 0:90] = 1
                    data_prep[sub][m][idx, int(row['loc_B']-1), 90:180] = 1
                    data_prep[sub][m][idx, int(row['loc_C']-1), 180:270] = 1
                    data_prep[sub][m][idx, int(row['loc_D']-1), 270:] = 1
                    
                elif m == 'future_reward_x_state_model':
                    # First write the base pattern into the FIRST 9 ROWS (NOW)
                    data_prep[sub][m][idx, int(row['loc_A']-1), 0:90] = 1
                    data_prep[sub][m][idx, int(row['loc_B']-1), 90:180] = 1
                    data_prep[sub][m][idx, int(row['loc_C']-1), 180:270] = 1
                    data_prep[sub][m][idx, int(row['loc_D']-1), 270:] = 1
                    
                    # Use the first 9 rows as the source
                    src = data_prep[sub][m][idx, :no_locations, :]
                    
                    # Fill the remaining blocks by circularly shifting columns by 90° per state
                    # (future rewards)
                    for s in range(1, no_state):  # skip s=0; it's already written
                        r0 = s * no_locations
                        r1 = r0 + no_locations
                        data_prep[sub][m][idx, r0:r1, :] = np.roll(src, s*state_width, axis=1)
                
                elif m == 'location_model':
                    curr_locs = data_dict[sub]['locations'].iloc[idx].to_list()
                    for l_idx, l in enumerate(curr_locs):
                        data_prep[sub][m][idx, int(l-1), l_idx] = 1
                        
                elif m == 'future_location_x_phase_model':
                    # --- current block (state 0, phase 0) in the FIRST 9 ROWS ---
                    curr_locs = data_dict[sub]['locations'].iloc[idx].tolist()
                    for l_idx, l in enumerate(curr_locs):
                        data_prep[sub][m][idx, int(l) - 1, l_idx] = 1
                
                    # Use the first 9 rows as the source (copy to avoid accidental in-place coupling)
                    src = data_prep[sub][m][idx, :no_locations, :].copy()
                
                    # Fill all state×phase blocks by circularly shifting columns
                    for s in range(no_state):          # states 0..3
                        for p in range(no_phase):      # phases 0..2
                            shift = s*state_width + p*phase_width
                            r0 = (s*no_phase + p) * no_locations
                            r1 = r0 + no_locations
                            block = np.roll(src, shift, axis=1)
                            data_prep[sub][m][idx, r0:r1, :] = block
                            
    # import pdb; pdb.set_trace()                           
    return data_prep


#
#
#

def prep_regressors_for_neurons(data_dict, models_I_want = None, only_repeats_included = None, randomised_reward_locations = False, avg_across_runs = False, comp_circular_perms = None):
    no_state = 4
    no_locations = 9
    no_buttons = 4
    if avg_across_runs==True:
        bins_per_state = 5
    all_models = ['state_reg','complete_musicbox_reg','reward_musicbox_reg','buttonbox_reg','location_reg','current_reward_reg', 'next_reward_reg', 'second_next_reward_reg', 'previous_reward_reg']
    models_same_length = ['location_reg','current_reward_reg', 'next_reward_reg', 'second_next_reward_reg', 'previous_reward_reg']
    
    # clean subjects that messed up certain grids 
    for sub in data_dict:
        if sub in ['sub-15', 'sub-27','sub-43','sub-44', 'sub-59']:
            data_dict[sub] = clean_data(data_dict, sub)

    data_prep = copy.deepcopy(data_dict)

    if comp_circular_perms:
        data_prep[sub]['perm_neurons'] = mc.analyse.helpers_human_cells.generate_circular_timepoint_permutations_neurons(data_dict[sub]['neurons'], n_perms = comp_circular_perms)
        
        
    for sub in data_dict:
        print(f"now starting to process data from subject {sub}")
        for m in all_models:
            data_prep[sub][m]=[]
        import pdb; pdb.set_trace()
        if models_I_want:
            if models_I_want[0] == 'only':
                # data_prep[sub] = {k: v for k, v in data_prep[sub].items() if not (k in ['state_reg', 'complete_musicbox_reg', 'reward_musicbox_reg', 'buttonbox_reg', 'location_reg'] and v == [])}
                for model in models_I_want[1:]:
                    data_prep[sub][f"musicbox_{model}_rew_reg"] = []
                    data_prep[sub][f"musicbox_{model}_complete_reg"] = []
            else:
                for model in models_I_want:
                    data_prep[sub][f"musicbox_{model}_rew_reg"] = []
                    data_prep[sub][f"musicbox_{model}_complete_reg"] = []
        
        if randomised_reward_locations == False:
            reward_configurations = data_dict[sub]['reward_configs'].copy()
            
        elif randomised_reward_locations == True:
            # randomize reward configurations 
            shuffled_tasks = np.zeros((data_dict[sub]['reward_configs'].shape))
            indices = np.random.permutation(len(data_dict[sub]['reward_configs']))
            for j, i in enumerate(indices):
                if i != j:
                    shuffled_tasks[j] = data_dict[sub]['reward_configs'][i]
                elif i == j:
                    if i+1 == len(indices):
                        shuffled_tasks[j] = data_dict[sub]['reward_configs'][i-1]
                    else:
                        shuffled_tasks[j] = data_dict[sub]['reward_configs'][i+1]
       
            reward_configurations = shuffled_tasks.copy()
            data_prep[sub]['shuffled_tasks'] = shuffled_tasks.copy()
            
            
        for grid_idx, grid_config in enumerate(reward_configurations):
            # import pdb; pdb.set_trace()
            if only_repeats_included:
                if 'max' in only_repeats_included:
                    timings_task = data_dict[sub]['timings'][grid_idx][only_repeats_included[0]:]
                    if (sub == 'sub-25' and grid_idx == 9) or (sub == 'sub-52' and grid_idx == 6) or (sub == 'sub-44' and grid_idx == 3) or (sub == 'sub-28' and grid_idx == 16) or (sub == 'sub-02' and grid_idx == 18):
                        # cut the last row of the timings as they didn't finish it in that repeat
                        timings_task = timings_task[:-1, :]
                else:
                    start_from_repeat = np.min(only_repeats_included)
                    end_at_repeat = np.max(only_repeats_included)
                    timings_task = data_dict[sub]['timings'][grid_idx][start_from_repeat:end_at_repeat+1]
                    if end_at_repeat > 3 and (sub == 'sub-02' and grid_idx == 18):
                        timings_task = timings_task[:-1, :]
                    
                # timings_task = data_dict[sub]['timings'][grid_idx][start_from_repeat:end_at_repeat]
                # if 'max' in exclude_x_repeats:
                #     start_from_repeat = np.min(exclude_x_repeats)
                #     start_from_repeat = np.max(exclude_x_repeats)
                #     timings_task = data_dict[sub]['timings'][grid_idx][start_from_repeat:]
            else:  
                timings_task = data_dict[sub]['timings'][grid_idx]

            
            if randomised_reward_locations == False:
                # check the match between timings and reward configs
                mc.simulation.predictions.test_timings_rew(sub, data_dict[sub]['locations'][grid_idx],timings_task, grid_config, grid_idx)
            
            # first run the old way of modelling for the complete models (plus controls. Will end with 'model')
            neurons_for_task = data_dict[sub]['neurons'][grid_idx].copy()
            # fields need to be between 0 and 8, and keep them as integers
            locations_curr_grid = [int((field_no-1)) for field_no in data_dict[sub]['locations'][grid_idx]]
            task_config = [int((field_no-1)) for field_no in data_dict[sub]['reward_configs'][grid_idx]]
            models_per_rep_dict, per_rep_prep = {}, {}
            regression_across_repeats = []
            for i, reps in enumerate(timings_task):
                timings_repeat = [int(r) for r in reps]
                # and for next repeat
                if i+1 == len(timings_task):
                    # if this is the last repeat, cut the file differently
                    timings_next_rep = None
                else:
                    timings_next_rep = [int(elem) for elem in timings_task[i+1]]
                    
                per_rep_prep[f"rep_{i}"] = mc.analyse.helpers_human_cells.prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_rep, task_config, neurons_for_task)
                
                if avg_across_runs == True:
                    # prepare binning regressors if wanted
                    regression_across_repeats.append(mc.simulation.predictions.create_x_regressors_per_state(per_rep_prep[f"rep_{i}"], no_regs_per_state=bins_per_state, only_for_rewards=False))
                
                if models_I_want[0] != 'only':
                    models_per_rep = mc.simulation.predictions.set_continous_models_ephys(per_rep_prep[f"rep_{i}"],  grid_size = 3, no_phase_neurons=3, fire_radius = 0.25, wrap_around = 1, plot = False, split_clock = True, use_orig_timings= True)
                    # then prepare concatenating all of them
                    for model in models_per_rep:
                        if model not in models_per_rep_dict:
                            models_per_rep_dict[model] = []
                        models_per_rep_dict[model].append(models_per_rep[model])
                        if model not in data_prep[sub]:
                            data_prep[sub][model] = []
   
            for m in models_per_rep_dict:
                if m in data_prep[sub]:
                    data_prep[sub][m].append(np.concatenate(models_per_rep_dict[m], axis = 1))
                    
            if avg_across_runs == True:
                # prepare binning regressors if wanted
                regression_across_repeats_concat = np.concatenate(regression_across_repeats, axis = 1)

            # this version cuts everything between starting the trial, and finding the very last reward.
            data_prep[sub]['neurons'][grid_idx]= data_prep[sub]['neurons'][grid_idx][:, int(timings_task[0,0]):int(timings_task[-1,-1])]
            data_prep[sub]['locations'][grid_idx] = data_prep[sub]['locations'][grid_idx][int(timings_task[0,0]):int(timings_task[-1,-1]+1)]
            data_prep[sub]['buttons'][grid_idx] = data_prep[sub]['buttons'][grid_idx][int(timings_task[0,0]):int(timings_task[-1,-1]+1)]
            
            if comp_circular_perms:
                for perm in data_prep[sub]['perm_neurons']:
                    perm[grid_idx]= perm[grid_idx][:, int(timings_task[0,0]):int(timings_task[-1,-1])]

            # length_curr_grid = len(data_prep[sub]['locations'][grid_idx])

            # this is still linked to the differences in slicing 9th of may 2025
            # Ideally, i'd cut all timings for regressors, models and regs at the SAME PLACE
            # and then use the same slices for each.
            # this is a fix for now, but it's not ideal.
            length_curr_grid = int(timings_task[-1,-1]) - int(timings_task[0,0])



            # here, go and test if the dimensions are right. if not, dublicate last column.
            # this should NOT be needed anymore!!! 9.05.2025 
            # that's why there is a breakpoint. still good to check...
            # it dublicates the last column in case I cut one too many- but this
            # has the power to substantially change the regression later which i want to avoid
            for m in data_prep[sub]:
                if m.endswith('model'):
                    if data_prep[sub][m][grid_idx].shape[1] < length_curr_grid :
                        import pdb; pdb.set_trace()
                        x = data_prep[sub][m][grid_idx]
                        while x.shape[1] < length_curr_grid :
                            last_column = data_prep[sub][m][grid_idx][:, -1].reshape(-1, 1)  # Extract and reshape the last column
                            x = np.hstack((x, last_column))                            
                        data_prep[sub][m][grid_idx] = x 
            

            #data_prep[sub]['state_reg'].append(np.zeros((no_state-1, length_curr_grid)))
            data_prep[sub]['state_reg'].append(np.zeros((no_state, length_curr_grid)))
            data_prep[sub]['complete_musicbox_reg'].append(np.zeros((no_state*no_locations, length_curr_grid)))
            # data_prep[sub]['complete_musicbox_reg'].append(np.zeros((3*no_state*no_locations, length_curr_grid)))
            data_prep[sub]['reward_musicbox_reg'].append(np.zeros((no_state*no_locations, length_curr_grid)))
            # data_prep[sub]['location_reg'].append(np.zeros((no_locations, length_curr_grid)))
            data_prep[sub]['buttonbox_reg'].append(np.zeros((no_state*no_buttons, length_curr_grid)))
            # split simple musicbox
            for model in models_same_length:
                data_prep[sub][model].append(np.zeros((no_locations, length_curr_grid)))
            
            if models_I_want:
                if models_I_want[0] == 'only':
                    for model in models_I_want[1:]:
                        if model == 'withoutnow':
                            less_rows = 1
                        if model in ['only2and3future', 'onlynowandnext', 'onlynowand3future', 'onlynextand2future']:
                            less_rows = 2
                        
                        data_prep[sub][f"musicbox_{model}_rew_reg"].append(np.zeros(((no_state-less_rows)*no_locations, length_curr_grid)))
                        data_prep[sub][f"musicbox_{model}_complete_reg"].append(np.zeros(((no_state-less_rows)*no_locations, length_curr_grid)))
                else:
                    for model in models_I_want:
                        if model == 'withoutnow':
                            less_rows = 1
                        if model in ['only2and3future', 'onlynowandnext', 'onlynowand3future', 'onlynextand2future']:
                            less_rows = 2
                        
                        data_prep[sub][f"musicbox_{model}_rew_reg"].append(np.zeros(((no_state-less_rows)*no_locations, length_curr_grid)))
                        data_prep[sub][f"musicbox_{model}_complete_reg"].append(np.zeros(((no_state-less_rows)*no_locations, length_curr_grid)))
                
            
            # the following ones should take the timings as input, and fill in the 
            # empty arrays of the following regressors, according to the timings
            # and locations/buttonpresses:
            if randomised_reward_locations == True:
                # if shuffled_tasks[grid_idx][0] != data_dict[sub]['reward_configs'][grid_idx][0]:
                #     grid_config = shuffled_tasks[grid_idx]
                # elif grid_idx == 0:
                #     grid_config = shuffled_tasks[grid_idx+1]
                # elif grid_idx == len(data_dict[sub]['reward_configs']):
                #     grid_config = shuffled_tasks[grid_idx-1]
                print(f"shuffled reward config is {grid_config} and actual was {data_dict[sub]['reward_configs'][grid_idx]}")
            

            data_prep[sub]['state_reg'][grid_idx] = mc.simulation.predictions.state_cells(data_prep[sub]['state_reg'][grid_idx],timings_task, grid_config)
            data_prep[sub]['location_reg'][grid_idx] = mc.simulation.predictions.locations_cells(data_prep[sub]['locations'][grid_idx], data_prep[sub]['location_reg'][grid_idx])
            data_prep[sub]['reward_musicbox_reg'][grid_idx] = mc.simulation.predictions.music_box_simple_cells(data_prep[sub]['locations'][grid_idx], data_prep[sub]['reward_musicbox_reg'][grid_idx], timings_task, grid_config)
            data_prep[sub]['complete_musicbox_reg'][grid_idx] = mc.simulation.predictions.musicbox_cells_complete(data_prep[sub]['locations'][grid_idx], data_prep[sub]['complete_musicbox_reg'][grid_idx], timings_task, grid_config)
            
            # split musicbox
            data_prep[sub]['current_reward_reg'][grid_idx] = mc.simulation.predictions.single_reward(data_prep[sub]['locations'][grid_idx], data_prep[sub]['current_reward_reg'][grid_idx], timings_task, grid_config, setting='current')
            data_prep[sub]['next_reward_reg'][grid_idx] = mc.simulation.predictions.single_reward(data_prep[sub]['locations'][grid_idx], data_prep[sub]['next_reward_reg'][grid_idx], timings_task, grid_config, setting='next')
            data_prep[sub]['second_next_reward_reg'][grid_idx] = mc.simulation.predictions.single_reward(data_prep[sub]['locations'][grid_idx], data_prep[sub]['second_next_reward_reg'][grid_idx], timings_task, grid_config, setting='second_next')
            data_prep[sub]['previous_reward_reg'][grid_idx] = mc.simulation.predictions.single_reward(data_prep[sub]['locations'][grid_idx], data_prep[sub]['previous_reward_reg'][grid_idx], timings_task, grid_config, setting='previous')
            
            data_prep[sub]['buttonbox_reg'][grid_idx] = mc.simulation.predictions.button_box_simple_cells(data_prep[sub]['buttons'][grid_idx], data_prep[sub]['buttonbox_reg'][grid_idx], timings_task)
            if models_I_want:
                for model in models_I_want:
                    if models_I_want[0] == 'only':
                        for model in models_I_want[1:]:
                            data_prep[sub][f"musicbox_{model}_rew_reg"][grid_idx] = mc.simulation.predictions.music_box_simple_cells(data_prep[sub]['locations'][grid_idx], data_prep[sub][f"musicbox_{model}_rew_reg"][grid_idx], timings_task, grid_config, setting = model)
                            data_prep[sub][f"musicbox_{model}_complete_reg"][grid_idx] = mc.simulation.predictions.musicbox_cells_complete(data_prep[sub]['locations'][grid_idx], data_prep[sub]['complete_musicbox_reg'][grid_idx], timings_task, grid_config, setting = model)
                    else:
                        for model in models_I_want:
                            data_prep[sub][f"musicbox_{model}_rew_reg"][grid_idx] = mc.simulation.predictions.music_box_simple_cells(data_prep[sub]['locations'][grid_idx], data_prep[sub][f"musicbox_{model}_rew_reg"][grid_idx], timings_task, grid_config, setting = model)
                            data_prep[sub][f"musicbox_{model}_complete_reg"][grid_idx] = mc.simulation.predictions.musicbox_cells_complete(data_prep[sub]['locations'][grid_idx], data_prep[sub]['complete_musicbox_reg'][grid_idx], timings_task, grid_config, setting = model)
                       
 
            # on this level, there is the option to average across repeats
            if avg_across_runs == True:
                data_prep_tmp = copy.deepcopy(data_prep)
                for m in data_prep[sub]:
                    if m.endswith('reg') or m.endswith('model') or m == 'neurons':
                        # this needs to be concatenated and all
                        data_prep[sub][m][grid_idx] = mc.simulation.predictions.transform_data_to_betas(data_prep_tmp[sub][m][grid_idx], regression_across_repeats_concat)
                if comp_circular_perms:
                    for perm in data_prep[sub]['perm_neurons']:
                        perm[grid_idx]= mc.simulation.predictions.transform_data_to_betas(perm[grid_idx], regression_across_repeats_concat)
            
            #import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()            
    if avg_across_runs == True:
        # finally, also average across tasks: each tasks in the end only exists once.
        # import pdb; pdb.set_trace()
        data_prep_tmp = copy.deepcopy(data_prep)
        data_prep = {}
        data_prep[sub] = {}
        # copy cell_labels, excluded_cells, og_timings, og_buttons
        to_copy = ['cell_labels', 'excluded_cells', 'buttons', 'timings', 'electrode_labels']
        for c in to_copy:
            data_prep[sub][c] = data_prep_tmp[sub][c]
        
        # store the new order of reward_configs here
        data_prep[sub]['reward_configs'], idx_unique, idx_inverse, counts = np.unique(
            data_prep_tmp[sub]['reward_configs'], axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        
        if comp_circular_perms:
            data_prep[sub]['perm_neurons'] = []
            for perm in data_prep_tmp[sub]['perm_neurons']:
                data_prep[sub]['perm_neurons'].append([])
                
        for unique_grid_idx in range(len(idx_unique)):
            # collect all indices that are the same grid
            same_grids_at = np.where(idx_inverse == unique_grid_idx)[0]
            
            # first test if these are actually really the same grids!
            same_grids = []
            for same_idx in same_grids_at:
                same_grids.append(data_prep_tmp[sub]['reward_configs'][same_idx])
                _, test_count = np.unique(same_grids, return_counts=True)
                # if these grid configs aren't the same, the count will be lower
            if test_count[0] != len(same_grids_at):
                import pdb; pdb.set_trace()
  
            # then average these grids
            for m in data_prep_tmp[sub]:
                if m.endswith('reg') or m.endswith('model') or m == 'neurons':
                    if m not in data_prep[sub]:
                        data_prep[sub][m] = []
                    data_prep[sub][m].append(np.nanmean([data_prep_tmp[sub][m][i] for i in same_grids_at], axis=0))
            
            if comp_circular_perms:
                for p, perm in enumerate(data_prep_tmp[sub]['perm_neurons']):
                    data_prep[sub]['perm_neurons'][p].append(np.nanmean([perm[i] for i in same_grids_at], axis = 0))
    return data_prep



def remove_top_x_percent_of_x_model(result_df, delete_from_model, x_percent):
    # import pdb; pdb.set_trace()
    overview = {}
    overview['all_cells'] = result_df['cell'].unique().tolist()
    
    df_filtered = result_df[result_df['model'] == delete_from_model].sort_values(by=['average_corr'], ascending=True).reset_index(drop=True)
    n_cells_in_model = len(df_filtered)
    n_to_remove = int(n_cells_in_model*x_percent/100)
    
    
    # import pdb; pdb.set_trace()
    # z_scores = (real_corrs - np.mean(null_corrs)) / np.std(null_corrs)
    # cutoff = np.percentile(null_corrs, 95)



    n_to_keep = n_cells_in_model - n_to_remove
    cells_without_best_model_cells = df_filtered['cell'][0:n_to_keep].unique().tolist()
    # Remove these cells from the original dataframe
    results_clean = result_df[result_df['cell'].isin(cells_without_best_model_cells)].reset_index(drop=True)
    
    overview['cells_to_keep'] = cells_without_best_model_cells
    
    
    return results_clean, overview




def remove_certain_cells_for_x_model(result_df, path_to_csv):
    
    # Load the list of cells to delete
    delete_cells_df = pd.read_csv(path_to_csv)  # path_to_csv = 'path/to/delete_cells.csv'
    cells_to_delete = delete_cells_df['cell'].tolist()
    
    # Remove these cells from the original DataFrame
    results_clean = result_df[~result_df['cell'].isin(cells_to_delete)].reset_index(drop=True)
    
    # Overview
    overview = {}
    overview['all_cells'] = result_df['cell'].unique().tolist()
    overview['cells_to_keep'] = results_clean['cell'].unique().tolist()
    
    return results_clean, overview




def identify_max_cells_for_model(result_dir, x_percentage_of_cells = None):
    # import pdb; pdb.set_trace()
    df = pd.DataFrame()
    all_cells = {}
    i = 0
    for sub in result_dir:
        for model in result_dir[sub]:
            for cell in result_dir[sub][model]:
                df.at[i, 'cell'] = cell
                df.at[i, 'average_corr'] = np.mean(result_dir[sub][model][cell])
                df.at[i, 'model'] = model
                if len(sub) == 5:
                    prefix, num_str = sub.split('-')
                    # Convert the number to an integer, then format it as a two-digit string.
                    formatted_num = f"{int(num_str):02}"
                    df.at[i, 'subject'] = f"{prefix}-{formatted_num}"
                else:
                    df.at[i, 'subject'] = sub
                i = i + 1
        
    for model in result_dir[sub]:
        all_cells[model] = df[df['model'] == model]
    
    
    top_ten = {}
    if x_percentage_of_cells:
        for model in all_cells:
            # find out how many cells are x percent of all cells in this model
            n_cells_in_model = len(all_cells[model])
            percent = int(n_cells_in_model*x_percentage_of_cells/100)
            top_ten[model] = all_cells[model].sort_values(by=['average_corr'], ascending=False)[0:percent]
    else:
        for model in all_cells:
            top_ten[model] = all_cells[model].sort_values(by=['average_corr'], ascending=False)[0:10]
    # import pdb; pdb.set_trace()
    return top_ten, all_cells
      

    

def store_best_cells(best_cells, all_data, name_extension_string = None):
    result_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/best_cells"
    # import pdb; pdb.set_trace()
    for model in best_cells:
        # import pdb; pdb.set_trace()
        for index, row in best_cells[model].iterrows():
            subject = row['subject']
            cell_label = row['cell']
            cell_idx_str = cell_label.split('_', 2)[1]
            cell_idx = int(cell_idx_str)
            cells_to_store = []
            for task in all_data[subject]['neurons']:
                cells_to_store.append(task[cell_idx])
            subset_dict = {}
            subset_dict[cell_label] = cells_to_store.copy()
            subset_dict['reward_configs'] = all_data[subject]['reward_configs'].copy()
            subset_dict['locations'] = all_data[subject]['locations'].copy()
            subset_dict['timings'] = all_data[subject]['timings'].copy()
            file_name = f"{cell_label}_best_for_{model}"
            if name_extension_string:
                file_name = f"{cell_label}_best_for_{model}_{name_extension_string}"
            with open(os.path.join(result_folder,file_name), 'wb') as f:
                pickle.dump(subset_dict, f)
    
    
            

           
def prep_neurons_and_state(data_dict, repeats, only_reward_times, no_bins_per_state, sim_fake_data = False):
    # import pdb; pdb.set_trace()
    # first, do some modifications (integers, start from 0, make timings to bins)
    # then, call the simulate functions that model the different neurons
    # do this for every repeat < for ever grid < for every subject.
    data_prep = {}
    
    # assume that every task has at least 10 repeats.
    for sub in data_dict:
        print(f"now starting to process data from subject {sub}")
        data_prep[sub] = {}
        data_prep[sub]['cell_labels'] = data_dict[sub]['cell_labels'].copy()
        data_prep[sub]['reward_configs'] = data_dict[sub]['reward_configs'].copy()
        #import pdb; pdb.set_trace()
        for grid_idx, grid_config in enumerate(data_dict[sub]['reward_configs']):
            if (sub == 'sub-43' and grid_idx == 3) or (sub == 'sub-15' and grid_idx == 23) or (sub == 'sub-02' and grid_idx == 18):
                # probably aborted the experiment here
                data_prep[sub][f"grid_{grid_idx}"] = {}
                continue
            data_prep[sub][f"grid_{grid_idx}"] = {}
            # timings task is start A - find A - find B - find C - find D
            timings_task = data_dict[sub]['timings'][grid_idx].copy()
            # fields need to be between 0 and 8, and keep them as integers
            locations_curr_grid = [int((field_no-1)) for field_no in data_dict[sub]['locations'][grid_idx]]
            # current neural recordings 
            neurons_for_task = data_dict[sub]['neurons'][grid_idx].copy()
            task_config = [int((field_no-1)) for field_no in data_dict[sub]['reward_configs'][grid_idx]]
            regression_across_repeats, simulated_state_data, prep_repeat_dict = {}, {}, {}
            repeats_list = []
            for repeat in range(repeats[0], repeats[1]):
                repeats_list.append(repeat)
                if repeat == 9 and sub == 'sub-52':
                    continue
                # then figure out the precise timings.
                if math.isnan(timings_task[repeat][-1]) == True:
                    # test if there are more nans than only the last, and if 
                    # this anyways the last repeat, skip this one
                    if math.isnan(timings_task[repeat][-2]) == True and repeat+1 == len(timings_task):
                        continue
                    if repeat+1 == len(timings_task):
                        # note this is a dumb fix. 
                        # cleaner if ignoring this trial as patient probably didn't finish
                        timings_task[repeat][-1] = timings_task[repeat][-2]+2
                timings_repeat = [int(elem) for elem in timings_task[repeat]]
                
                # I also need the next repeat
                if repeat+1 < len(timings_task):
                    if math.isnan(timings_task[repeat+1][-1]) == True:
                        # if this is the penultimum repeat and there are more nans, skip this one
                        if math.isnan(timings_task[repeat+1][-2]) == True and repeat+2 == len(timings_task):
                            continue
                        # note this is a dumb fix. 
                        # cleaner if ignoring this trial as patient probably didn't finish
                        timings_task[repeat+1][-1] = timings_task[repeat+1][-2]+2
                    
                    timings_next_repeat = [int(elem) for elem in timings_task[repeat+1]]
                else:
                    timings_next_repeat = None
                
                # then create the regressors 
                prep_repeat_dict[f"rep_{repeat}"] = mc.analyse.helpers_human_cells.prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_repeat, task_config, neurons_for_task)
                regression_across_repeats[f"rep_{repeat}"] = mc.simulation.predictions.create_x_regressors_per_state(prep_repeat_dict[f"rep_{repeat}"], no_regs_per_state=no_bins_per_state, only_for_rewards=only_reward_times)
                if sim_fake_data == True:
                    simulated_state_data[f"rep_{repeat}"] = mc.simulation.predictions.simulate_fake_data(prep_repeat_dict[f"rep_{repeat}"], model_to_simulate = 'state', repeat_idx = repeat)

            # lastly, run the same regression on the cells
            # the regressor sometimes is a few timebins shorter. make equal in size.
            # before doing so, normalise the neurons.
            # subtract mean and divide by standard deviation.
            # first, concatenate all repeats.
            for i, repeat in enumerate(sorted(regression_across_repeats)):
                if i == 0:
                    prep_repeat_dict["concat_neurons"] = prep_repeat_dict[repeat]['neuron_rep'].copy()
                    regression_across_repeats["concat"] = regression_across_repeats[repeat].copy()
                    if sim_fake_data == True:
                        simulated_state_data["concat"]  = simulated_state_data[repeat].copy()
                else:
                    prep_repeat_dict["concat_neurons"] = np.concatenate((prep_repeat_dict["concat_neurons"], prep_repeat_dict[repeat]['neuron_rep']), axis = 1)
                    regression_across_repeats["concat"] = np.concatenate((regression_across_repeats["concat"], regression_across_repeats[repeat]), axis = 1)
                    if sim_fake_data == True:
                        simulated_state_data["concat"]  = np.concatenate((simulated_state_data["concat"], simulated_state_data[repeat]), axis = 1)

            prep_repeat_dict["concat_neurons_normal"] = mc.simulation.predictions.normalise_neurons(prep_repeat_dict["concat_neurons"])
            data_prep[sub][f"grid_{grid_idx}"]['neurons'] = mc.simulation.predictions.transform_data_to_betas(prep_repeat_dict["concat_neurons_normal"], regression_across_repeats["concat"])     
            if sim_fake_data == True:
                data_prep[sub][f"grid_{grid_idx}"]['fake_neurons'] = mc.simulation.predictions.transform_data_to_betas(simulated_state_data["concat"], regression_across_repeats["concat"])  
                
    return data_prep           
 

def get_rid_of_low_firing_cells(data_one_sub, hz_exclusion_threshold = 0.1, sd_exclusion_threshold = 1.5):
     
    neuron_dict = {}
    for task in data_one_sub['neurons']:
        for i, neuron in enumerate(task):
            curr_cell = f"{data_one_sub['cell_labels'][i]}_{i}"
            if curr_cell not in neuron_dict:
                neuron_dict[curr_cell] = []            
            avg_rate_hz = np.sum(neuron) / (len(neuron) * 0.025)
            neuron_dict[curr_cell].append(avg_rate_hz)
       
    excl_dict = {}
    neurons_to_exclude = []
    neurons_to_exclude_str = []
    for neuron in neuron_dict:
        excl_dict[f"avg_{neuron}"] = np.mean(neuron_dict[neuron])
        excl_dict[f"dev_{neuron}"] = np.std(neuron_dict[neuron])/np.mean(neuron_dict[neuron])
        neuron_index = neuron.split('_')[-1]
        if excl_dict[f"avg_{neuron}"] < hz_exclusion_threshold:
            neurons_to_exclude.append(int(neuron_index))
            neurons_to_exclude_str.append(neuron)
        elif excl_dict[f"dev_{neuron}"] > sd_exclusion_threshold:
            neurons_to_exclude.append(int(neuron_index))
            neurons_to_exclude_str.append(neuron)

    remaining_neurons = {}
    stuff_to_copy = ['reward_configs', 'timings', 'locations', 'buttons']
    to_clean = ['cell_labels', 'neurons']
    
    
    for m in data_one_sub:
        if m in stuff_to_copy:
            remaining_neurons[m] = data_one_sub[m]
        elif m in to_clean:
            # # import pdb; pdb.set_trace() 
            # if m not in remaining_neurons[sub]:
            #     remaining_neurons[sub][m] = []
            if m == 'neurons':
                if m not in remaining_neurons:
                    remaining_neurons[m] = []
                for el in data_one_sub[m]:
                    cleaned = np.delete(el, neurons_to_exclude, axis=0)
                    remaining_neurons[m].append(cleaned)
            elif m == 'cell_labels':
                remaining_neurons[m] = [x for i, x in enumerate(data_one_sub[m]) if i not in neurons_to_exclude]

    
    print(f"exlcuding neurons {neurons_to_exclude_str} because average spike rate lower than {hz_exclusion_threshold} or variability bigger than std/mean = {sd_exclusion_threshold}")
    # import pdb; pdb.set_trace() 
    return remaining_neurons

   
def prep_and_model_human_cells(data_dict, repeats, model_simple = True):
    
    task_labels = {'task_A': ([1., 9., 5., 8.]), 'task_B': ([2., 5., 7., 6.]), 'task_C': ([3., 7., 9., 5.]), 'task_D': ([4., 8., 1., 3.]), 'task_E': ([6., 4., 2., 9.]), 'task_F': ([7., 3., 4., 2.]), 'task_G': ([8., 2., 6., 7.]), 'task_H': ([9., 1., 3., 4.])}
    
    # import pdb; pdb.set_trace()
    # first, do some modifications (integers, start from 0, make timings to bins)
    # then, call the simulate functions that model the different neurons
    # do this for every repeat < for ever grid < for every subject.
    
    modelled_data = {}
    for sub in data_dict:
        print(f"now starting to process data from subject {sub}")
        
        # clean up the data per subject
        # only consider those neurons that are firing enough
        modelled_data[sub] = mc.analyse.helpers_human_cells.get_rid_of_low_firing_cells(data_dict[sub])
        
        for grid_idx, grid_config in enumerate(data_dict[sub]['reward_configs']):
            # first identify which task you are currently modelling.
            for task_config_pre_defined in task_labels:
                if task_labels[task_config_pre_defined] == list(grid_config):
                    grid_label = f"{task_config_pre_defined}_{grid_idx}_sub_{sub}"
                    
            modelled_data[sub][grid_label] = {}
            # wrong timings for some reason in subject 51 in grid 17, 18 and probably more
            if (sub == 'sub-43' and grid_idx == 3) or (sub == 'sub-15' and grid_idx == 23) or (sub == 'sub-02' and grid_idx == 18) or (sub == 'sub-51' and grid_idx > 17):
                # probably aborted the experiment here
                continue
            # timings task is start A - find A - find B - find C - find D
            timings_task = data_dict[sub]['timings'][grid_idx]
            neurons_for_task = modelled_data[sub]['neurons'][grid_idx].copy()

            # fields need to be between 0 and 8, and keep them as integers
            locations_curr_grid = [int((field_no-1)) for field_no in data_dict[sub]['locations'][grid_idx]]
            task_config = [int((field_no-1)) for field_no in data_dict[sub]['reward_configs'][grid_idx]]
                
            regression_across_repeats, models_per_repeat, prep_repeat_dict  = {}, {}, {}
            # for repeat in range(1, len(timings_task)):  
            for repeat in range(repeats[0], repeats[1]):
                # first check if this repeat wasn't completed
                # for current repeat
                if math.isnan(timings_task[repeat][-1]) == True:
                    # test if there are more nans than only the last, and if 
                    # this anyways the last repeat, skip this one
                    if math.isnan(timings_task[repeat][-2]) == True and repeat+1 == len(timings_task):
                        continue
                    if repeat+1 == len(timings_task):
                        timings_task[repeat][-1] = timings_task[repeat][-2]+2
                timings_repeat = [int(elem) for elem in timings_task[repeat]]
                
                # and for next repeat
                if repeat+1 < len(timings_task):
                    if math.isnan(timings_task[repeat+1][-1]) == True:
                        # if this is the penultimum repeat and there are more nans, skip this one
                        if math.isnan(timings_task[repeat+1][-2]) == True and repeat+2 == len(timings_task):
                            continue
                        timings_task[repeat+1][-1] = timings_task[repeat+1][-2]+2
                    timings_next_repeat = [int(elem) for elem in timings_task[repeat+1]]
                else:
                    timings_next_repeat = None
                # this includes the trajectory, and the according neural recording
                prep_repeat_dict[f"rep_{repeat}"] = mc.analyse.helpers_human_cells.prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_repeat, task_config, neurons_for_task)
                
                # then model per repeat/grid/subject
                if model_simple == True:
                    models_per_repeat[f"rep_{repeat}"] = mc.simulation.predictions.set_simple_models_cells(prep_repeat_dict[f"rep_{repeat}"])
                else:
                    models_per_repeat[f"rep_{repeat}"] = mc.simulation.predictions.set_continous_models_ephys(prep_repeat_dict[f"rep_{repeat}"], split_clock = True)
                # then prepare regressors per repeat; will be same length as the walked path/trajectory/modelled data.
                regression_across_repeats[f"rep_{repeat}"] = mc.simulation.predictions.create_x_regressors_per_state(prep_repeat_dict[f"rep_{repeat}"], only_for_rewards=True)
            
            model_list = []
            for key in models_per_repeat[f"rep_{repeats[0]}"]:
                model_list.append(key)
            
            # then, per grid, average the runs to one. 
            # NOTE: later, HERE IS WHERE YOU COULD DO EARLY VS LATE !!
            # first, concatenate all repeats.
            # import pdb; pdb.set_trace()
            for i, rep in enumerate(sorted(models_per_repeat)):
                if i == 0:
                    prep_repeat_dict["concat_neurons"] = prep_repeat_dict[rep]['neuron_rep'].copy()
                    regression_across_repeats["concat"] = regression_across_repeats[rep].copy()
                    for model in model_list:
                        models_per_repeat[f"{model}_concat"] = models_per_repeat[rep][model].copy()
                else:
                    regression_across_repeats["concat"] = np.concatenate((regression_across_repeats["concat"], regression_across_repeats[rep]), axis = 1)
                    prep_repeat_dict["concat_neurons"] = np.concatenate((prep_repeat_dict["concat_neurons"], prep_repeat_dict[rep]['neuron_rep']), axis = 1)
                    for model in model_list:
                        models_per_repeat[f"{model}_concat"] = np.concatenate((models_per_repeat[f"{model}_concat"], models_per_repeat[rep][model]), axis = 1)
            # then average across all trials by running the regression and save each grid and model per subject.
            for model in model_list:
                modelled_data[sub][grid_label][model] = mc.simulation.predictions.transform_data_to_betas(models_per_repeat[f"{model}_concat"], regression_across_repeats["concat"])
            # lastly, run the same regression on the cells
            prep_repeat_dict["concat_neurons_normal"] = mc.simulation.predictions.normalise_neurons(prep_repeat_dict["concat_neurons"])
            modelled_data[sub][grid_label]['neurons'] = mc.simulation.predictions.transform_data_to_betas(prep_repeat_dict["concat_neurons_normal"], regression_across_repeats["concat"])     
            # import pdb; pdb.set_trace()
            # # the regressor sometimes is a few timebins shorter. make equal in size.
           
            # # before doing so, normalise the neurons.
            # # subtract mean and divide by standard deviation.
            # data_dict[sub]['neurons'][grid_idx] = mc.simulation.predictions.normalise_neurons(data_dict[sub]['neurons'][grid_idx])
            # # import pdb; pdb.set_trace()
            # # actually, 0 needs to be the first repeat index. How do I find that one??
            # start_modelling_from = int(timings_task[repeats[0]][0])
            # import pdb; pdb.set_trace()
            # # THIS ISNT RIGHT!!!!
            
            # modelled_data[sub][grid_label]['neurons'] = mc.simulation.predictions.transform_data_to_betas(prep_repeat_dict["concat_neurons"], regression_across_repeats["concat"])     
            # modelled_data[sub][grid_label]['neurons'] = mc.simulation.predictions.transform_data_to_betas(data_dict[sub]['neurons'][grid_idx][:, start_modelling_from:start_modelling_from + regression_across_repeats["concat"].shape[1]], regression_across_repeats["concat"])     
            # import pdb; pdb.set_trace()
    #fake_data = mc.analyse.helpers_human_cells(data_dict, modelled_data)
    
    print(f"the following models have been simulated and averaged for all repeats and all grids: {model_list}")
    return modelled_data
    

# THIS IS A LITTLE HELPER.
#def create_fake_data(og_data, model):
    
    # then store the 





def prep_behaviour_one_repeat(timings_repeat, locations_curr_grid, timings_next_repeat, reward_locs, neurons):
    # some pre-processing to create my models.
    # cut the neurons file such to include only the current repeat
    prep_dict = {}
    prep_dict['reward_locs'] = reward_locs
    prep_dict['timings_repeat'] = timings_repeat
    # because in the following, I will only regard the single repeats for themselves,
    # cut the timings such that it starts with the first bin of the repeat.
    prep_dict['timings_repeat'] = [elem - timings_repeat[0] for elem in timings_repeat]

    # locations for current repeat
    
    # 22/04/2025 update: don't do this, as it may use parts of the data twice and fucks up timings
    # 26.02.25 
    # curr trajectory should actually end only once you leave reward position. 
    # so instead of using timings_repeat[-1], find the relevant index.
    # this simply cuts longer chunks so that I can later select those longer chunks
    # in my regression and simulation.
    # start_at_rew, end_at_rew = mc.simulation.predictions.find_start_end_indices(locations_curr_grid, timings_repeat[-1])
    # prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:end_at_rew]
    # prep_dict['neuron_rep'] = neurons[:, timings_repeat[0]:end_at_rew] 

    # import pdb; pdb.set_trace()
    if timings_next_repeat:
        # note that timings_next_repeat[0] == timings_repeat[-1]
        # so collecting timings_next_repeat is not technically necessary
        prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:timings_next_repeat[0]]
        prep_dict['neuron_rep'] = neurons[:, timings_repeat[0]:timings_next_repeat[0]] 
        
        # this is if I want the actual time where they leave the rew location.
        # subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
        #                 locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
        #                 locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
        #                 locations_curr_grid[timings_repeat[3]:end_at_rew]]
        
        subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
                        locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
                        locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
                        locations_curr_grid[timings_repeat[3]:timings_next_repeat[0]]]
        
    elif not timings_next_repeat:
        # this means it's the last repeat and you have to include the last bin.
        # note: if you want to include the entire file after finding the reward,
        # do
        # prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:]
        # prep_dict['neuron_rep'] = neurons[:, timings_repeat[0]:] 
        # subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
        #                 locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
        #                 locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
        #                 locations_curr_grid[timings_repeat[3]:]]
        
        
        prep_dict['trajectory'] = locations_curr_grid[timings_repeat[0]:timings_repeat[-1]+1]
        prep_dict['neuron_rep'] = neurons[:, timings_repeat[0]:timings_repeat[-1]+1] 
        subpath_locs = [locations_curr_grid[timings_repeat[0]:timings_repeat[1]], 
                        locations_curr_grid[timings_repeat[1]:timings_repeat[2]], 
                        locations_curr_grid[timings_repeat[2]:timings_repeat[3]], 
                        locations_curr_grid[timings_repeat[3]:timings_repeat[4]+1]]
    
    
    # actually, the fourth subpath has to last until the next one starts.
    # to find out the step number per subpath
    prep_dict['step_number'] = [0,0,0,0] 
    for path_no, subpath in enumerate(subpath_locs):
        for i, field in enumerate(subpath):
            if i == 0:
                count = 0
            elif field != subpath[i-1]:
                count+=1
        prep_dict['step_number'][path_no] = count     
    # mark where steps are made (vs. staying on same location)
    for field_no, field in enumerate(prep_dict['trajectory']):
        if field_no == 0:
            prep_dict['index_make_step'] = [0]
        elif field != prep_dict['trajectory'][field_no-1]:
            prep_dict['index_make_step'].append(field_no)
    return prep_dict


def prep_repeat_neurons(timings_repeat, t_first_bin, neurons):
    prep_dict = {}
    curr_neurons = neurons[:, t_first_bin:timings_repeat[-1]]
    # z-score neurons
    prep_dict['curr_neurons'] = scipy.stats.zscore(curr_neurons, axis=1)
    return prep_dict
