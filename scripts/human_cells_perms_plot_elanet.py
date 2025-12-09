#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:41:15 2025

@author: Svenja Kuchenhoff

This script loads the original correlation values per cell and the permutation
and tests where in the distribution they lie.


"""


import mc
import fire
import os
import pickle
import matplotlib.pyplot as plt


def load_result_dirs(file_name, subs, perms_locs=None, perms_time=None):
    result_dir = {"binned": {}, "raw":{}}
    if perms_locs:
        result_dir = {"binned": {}, "raw":{}, "perm_locs":{}}
    if perms_time:
        result_dir = {"binned": {}, "raw":{}, "perm_time":{}}
    if perms_locs and perms_time:
        result_dir = {"binned": {}, "raw":{}, "perm_time":{}, "perm_locs":{}}
            
    results_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/corrs"
    
    # subjects = [f"sub-{i}" for i in range(1, 65)]
    subjects = [f"sub-{i}" for i in subs]
    #subjects = ['sub-1', 'sub-2']
    actual_subjects = []
    # check if on server or local
    if not os.path.isdir(results_folder):
        print("running on ceph")
        results_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group/elastic_net_reg/corrs"

    # loop through all subjects
    # first find how many subjects there are
    # import pdb; pdb.set_trace()
    for sub in subjects:
        path_to_subfile = f"{results_folder}/{sub}_corrs_{file_name}"
        # build in some thing that doesnt make it crash all the time if the file doesnt exist
        if os.path.isfile(path_to_subfile):
            actual_subjects.append(sub)

            with open(path_to_subfile, 'rb') as f:
                sub_dir = pickle.load(f)
                result_dir['raw'][sub] = {}
                for neuron, model in sub_dir.items():
                    for model, inner in model.items():
                        if model not in result_dir['raw'][sub]:
                            result_dir['raw'][sub][model] = {}
                        for inner_neuron, values in inner.items():
                            result_dir['raw'][sub][model][inner_neuron] = values
            if os.path.isfile(f"{path_to_subfile}_fit_binned_by_state"):
                with open(f"{path_to_subfile}_fit_binned_by_state", 'rb') as f:
                    sub_dir = pickle.load(f)
                    result_dir['binned'][sub] = {}
                    for neuron, model in sub_dir.items():
                        for model, inner in model.items():
                            if model not in result_dir['binned'][sub]:
                                result_dir['binned'][sub][model] = {}
                            for inner_neuron, values in inner.items():
                                result_dir['binned'][sub][model][inner_neuron] = values
                    
            if perms_time:
                if os.path.isfile(f"{results_folder}/{perms_time}_{sub}_corrs_{file_name}"):         
                    with open(f"{results_folder}/{perms_time}_{sub}_corrs_{file_name}", 'rb') as f:
                        sub_dir = pickle.load(f)
                        result_dir['perm_time'][sub] = {}
                        for neuron, model in sub_dir.items():
                            for model, inner in model.items():
                                if model not in result_dir['perm_time'][sub]:
                                    result_dir['perm_time'][sub][model] = {}
                                for inner_neuron, values in inner.items():
                                    result_dir['perm_time'][sub][model][inner_neuron] = values
            if perms_locs:
                if os.path.isfile(f"{results_folder}/{perms_locs}_{sub}_corrs_{file_name}"):
                    with open(f"{results_folder}/{perms_locs}_{sub}_corrs_{file_name}", 'rb') as f:
                        sub_dir = pickle.load(f)
                        result_dir['perm_locs'][sub] = {}
                        for neuron, model in sub_dir.items():
                            for model, inner in model.items():
                                if model not in result_dir['perm_locs'][sub]:
                                    result_dir['perm_locs'][sub][model] = {}
                                for inner_neuron, values in inner.items():
                                    result_dir['perm_locs'][sub][model][inner_neuron] = values

    return result_dir, actual_subjects




def load_data(subs):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    # check if on server or local
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    group_dir = f"{data_folder}/group/elastic_net_reg"
    
    if not os.path.isdir(group_dir):
        os.mkdir(group_dir)
    
    # first bring the subject list in the right format
    formatted_subjects = [f"{int(s.split('-')[1]):02}" for s in subs]
    data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, formatted_subjects)

    return data





def plot_all(model_name_string, sub_list, perms_locs=None, perms_time=None, plot_cells_corr_higher_than=0.05, save=False):
    results, subjects = load_result_dirs(model_name_string, subs=sub_list, perms_locs=perms_locs, perms_time=perms_time)
    # import pdb; pdb.set_trace()
    if perms_locs:
        task_perm_results=results['perm_locs']
    else:
        task_perm_results = perms_locs
    
    if perms_time:
        time_perm_results=results['perm_time']
    else:
        time_perm_results = perms_time
        
    results_corr_by_roi_df, no_perms = mc.analyse.plotting_cells.prep_result_df_perms_for_plotting_by_rois(results['raw'], time_perm_results=time_perm_results, task_perm_results=task_perm_results)
    
    mc.plotting.results.plot_perms_per_cell_and_roi(results_corr_by_roi_df, no_perms, corr_thresh=plot_cells_corr_higher_than, save=save, model_name_string=model_name_string)
    
    
    # import pdb; pdb.set_trace()
    
    
    
    
 
# if running from command line, use this one!   
# if __name__ == "__main__":
#     fire.Fire(plot_all)
#     # call this script like
#     # python wrapper_plot_elnetreg_results.py --model_name_string='w_partial_musicboxes_excl_rep1-2', --models_I_want='['withoutnow', 'only2and3future','onlynowandnext']' --exclude_x_repeats='[1,2]' --randomised_reward_locations=False --save_regs=True


    
if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    # no for timepoints: 27, 40, 44, 50
    # no for tasks: 7, 27, 40, 43, 44, 50, 
    plot_all(
        # for late, 2, 28, 52 doesnt work
        #model_name_string='w_partial_musicboxes_only_reps_6-10_avg_in_20_bins_across_runs',
        #sub_list = [i for i in range(60) if i not in [2, 28, 40, 52, 57, 58, 59]],
        # everything works for early
        #model_name_string='w_partial_musicboxes_only_reps_1-5_avg_in_20_bins_across_runs',
        #sub_list = [i for i in range(60)],
        # for all, 28,52 doesnt exist
        model_name_string=['360bins_early','360bins_late', '360bins_all_minus_explore'],
        sub_list = [i for i in range(64) if i not in [2]],
        #perms_time = '500perms_circular_shuffle',
        perms_time = None,
        plot_cells_corr_higher_than=0.05,
        save=True
        # sub-1_corrs_w_partial_musicboxes_excl_rep1-2_avg_in_20_bins_across_runs_fit_binned_by_state
    )
    
    
#sub-18_corrs_360bins


# if __name__ == "__main__":
#     # For debugging, bypass Fire and call compute_one_subject directly.
#     plot_all(
#         model_name_string='w_partial_musicboxes_excl_rep1-1_avg_in_20_bins_across_runs',
#         sub_list=[1,2,3,5,6,7,8,9,10],
#         perms_locs = '265perms_configs_shuffle',
#         perms_time = '300perms_timepoints_shuffle',
#         plot_cells_corr_higher_than=0.05,
#         save=True
#         # sub-1_corrs_w_partial_musicboxes_excl_rep1-2_avg_in_20_bins_across_runs_fit_binned_by_state
#     )
    
# w_partial_musicboxes_excl_rep1-3_excl_rep1-3_pre_corr_binned-None_only_pos

