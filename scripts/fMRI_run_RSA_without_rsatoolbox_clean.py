#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17th of October 2025.

Running the RSA [mainly] without RSA toolbox,
based on 
1. clean_fmri_behaviour
2. create_fMRI_model_RDMs_on_clean_beh


# this is the old GLM I used:
GLM ('regression') settings (creating the 'bins'):
    03-4 -24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)
# this is the new GLM I want to use:
    # glm_all-rews-split_buttons
    
    
@author: Svenja Küchenhoff, 2025
"""


from tqdm import tqdm
import numpy as np
import os
from nilearn.image import load_img
from rsatoolbox.util.searchlight import get_volume_searchlight
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc
import pickle
import sys
from datetime import date
import json
from fnmatch import fnmatch

def pair_correct_tasks(data_dict, keys_list):
    """
    data_dict: dict with keys like 'A1_forw_A_reward'
    keys_list: ordered list of keys you want to include and in what order
    Returns two matrices: one for the first element of each pair, one for its match.
    """
    # Define task pairing relationships
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    th_1, th_2, paired_list_control  = [], [], []
    # Loop through keys in the *specified order*
    for key in keys_list:
        assert key in data_dict, "Missmatch between model rdm keys and data RDM keys"
        task, direction, state, phase = key.split('_')  # e.g. ['A1', 'forw', 'A', 'reward']
        # Create the pairing suffix (e.g. from '1_forw' → '2_backw')
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        # Build the paired key (e.g. 'A2_backw_A_reward')
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        # Only add if both keys exist
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} with {pair_key}")

    # import pdb; pdb.set_trace()       
    th_1 = np.vstack(th_1)
    th_2 = np.vstack(th_2)
    # print(paired_list_control)
    return th_1, th_2, paired_list_control

#
#
# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
    
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
      
# --- Load configuration ---
# config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_simple.json"
config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_DSR_stepwise_combos.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
EV_string = config.get("load_EVs_from")
regression_version = config.get("regression_version")


today_str = date.today().strftime("%d-%m-%Y")
name_RSA = config.get("name_of_RSA")
RDM_version = f"{name_RSA}_{today_str}"


# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]

# Flags
smoothing = config.get("smoothing", True)
fwhm = config.get("fwhm", 5)
load_searchlights = config.get("load_searchlights", False)
masked_conditions = config.get("masked_conds", None)

# conditions selection
conditions = config.get("EV_condition_selection")
parts_to_use = conditions["parts"]

# model selection happens later in script


print(f"Now running RSA based on subj GLM {regression_version} for subj {subj_no}")


for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        # DONT FORGET TO COMMENT THIS OUT!!!!
        regression_version = '03-4'
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")
      
    modelled_conditions_dir = f"{data_dir}/beh/modelled_EVs"
    data_rdm_dir = f"{data_dir}/func/data_RDMs_{RDM_version}_glmbase_{regression_version}"

    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results" 
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}_smooth{fwhm}/results" 
    os.makedirs(results_dir, exist_ok=True)

    if masked_conditions:
        if masked_conditions[0] == 'load_same_loc_instate_mask':
            with open (f"{data_dir}/{masked_conditions[1]}", "r") as f:
                json_mask = json.load(f)
            masked_conditions = json_mask.get("masked_conditions")
        

    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    mask_file = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    mask = mask_file.get_fdata()  
    # save this file to save time
    if load_searchlights:
        with open(f"{data_dir}/func/searchlight_centers.pkl", 'rb') as file:
            centers = pickle.load(file)
        with open(f"{data_dir}/func/searchlight_neighbors.pkl", 'rb') as file:
            neighbors = pickle.load(file)
    else:
        # creating the searchlights
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5) # Found 175.483 searchlights
        # if I use the grey matter mask, then I find 144.905 searchlights
        # save this structure
        with open(f"{data_dir}/func/searchlight_centers.pkl", 'wb') as file:
            pickle.dump(centers, file)
            print("stored searchlight centres")
        with open(f"{data_dir}/func/searchlight_neighbors.pkl", 'wb') as file:
            pickle.dump(neighbors, file)   
            print("stored searchlight neighbors")

    #
    # Step 2: loading conditions for model and data RDMs
    #
    # loading the model EVs into dict
    with open(f"{modelled_conditions_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'rb') as file:
        model_EVs = pickle.load(file)
    selected_models = config.get("models", list(model_EVs.keys()))
    # loading the data EVs into dict
    #data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(data_dir, regression_version=regression_version, only_load_labels=True)
    data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(data_dir, regression_version=regression_version)
    # if you don't want all conditions created through FSL, exclude some here!
    # based on config file:
    # Ensure all four parts exist in config
    for _p in ("task", "direction", "state", "phase"):
        if _p not in parts_to_use:
            raise ValueError(f"Missing selection.parts['{_p}'] in config.")
            
    EV_keys = []        
    for ev in sorted(all_EV_keys):
        task, direction, state, phase = ev.split('_')
        # simple include/exclude logic
        for name, value in zip(["task", "direction", "state", "phase"], [task, direction, state, phase]):
            part = parts_to_use[name]
            includes = part.get("include", [])
            excludes = part.get("exclude", [])
            # Exclude first
            if any(fnmatch(value, pat) for pat in excludes):
                break  
            # If include list non-empty → must match at least one
            if includes and not any(fnmatch(value, pat) for pat in includes):
                break
        else:
            # only append if none of the 4 parts triggered 'break'
            EV_keys.append(ev)
    
    print(f"including the following EVs in the RDMs: {EV_keys}")
    data_th1, data_th2, paired_labels = pair_correct_tasks(data_EVs, EV_keys)
    data_concat = np.concatenate((data_th1, data_th2), axis = 0)
    # 
    # Step 3: compute the model and data RDMs.
    models_concat = {}
    model_RDM_dir = {}
    
    for model in model_EVs:
        model_th1, model_th2, model_paired_labels = pair_correct_tasks(model_EVs[model], EV_keys)
        # finally, concatenate th1 and th2 to do the cross-correlation after
        models_concat[model] = np.concatenate((model_th1, model_th2), axis = 0)
        
        if masked_conditions:
            # here, I want to now mask all within-task similarities.
            model_RDM_dir[model] = mc.analyse.my_RSA.compute_crosscorr_and_filter(models_concat[model], plotting = False, labels = model_paired_labels, mask_pairs= masked_conditions, full_mask=None, binarise = False)
            print(f"excluding n = {np.sum(np.isnan(model_RDM_dir[model]))} datapoints from {len(model_RDM_dir[model][0])}.")
            # import pdb; pdb.set_trace()
        else:  
            model_RDM_dir[model] = mc.analyse.my_RSA.compute_crosscorr(models_concat[model], plotting= True)
            if model == 'A-state':
                A_state_mask = ~np.isnan(model_RDM_dir['A-state'][0])
                # import pdb; pdb.set_trace()
                # first make sure to not do this for the 'correct nans'
                nan_mask = np.isnan(model_RDM_dir['state'][0])
                # then turn all nans into 1s
                nan_mask_other_states = np.isnan(model_RDM_dir[model][0])
                model_RDM_dir[model][0][nan_mask_other_states] = 1
                #plt.figure(); plt.imshow(model_RDM_dir[model][0])
    
    if 'A-state-ones' in selected_models:
        model_RDM_dir['state'][0][A_state_mask] = model_RDM_dir['state'][0][3]
        print(f"set state A in state regressor to {model_RDM_dir['state'][0][3]}")
    if 'A-state-mask' in selected_models:
        # A_state_mask = ~np.isnan(model_RDM_dir['A-state'][0])
        for model in model_RDM_dir:
            model_RDM_dir[model][0][A_state_mask] = np.nan
            print(f"excluding n = {np.sum(np.isnan(model_RDM_dir[model]))} datapoints from {len(model_RDM_dir[model][0])} because of state-A masking.")
            # if plotting == True:
            #     # plot to test how it looks like
            #     rdm_recon = np.zeros((40, 40))
            #     tri_u = np.triu_indices(40, k=0)
            #     # fill upper triangle
            #     rdm_recon[tri_u] = model_RDM_dir[model][0]
            #     plt.figure()
            #     plt.imshow(rdm_recon)
            #     plt.title(model)
    # import pdb; pdb.set_trace()

        
    
    if not os.path.exists(f"{data_rdm_dir}/data_RDM.npy"): 
         # and searchlight-wise for data RDMs
         if masked_conditions:
             # here, I want to now mask all within-task similarities.
             data_RDMs = mc.analyse.my_RSA.get_RDM_per_searchlight(data_concat, centers, neighbors, method = 'crosscorr_and_filter', labels = paired_labels, mask_pairs= masked_conditions)
         else:
             data_RDMs = mc.analyse.my_RSA.get_RDM_per_searchlight(data_concat, centers, neighbors, method = 'crosscorr')
         mc.analyse.handle_MRI_files.save_data_RDM_as_nifti(data_RDMs, data_rdm_dir, "data_RDM", ref_img, centers) 
    else:
         data_RDMs = np.load(f"{data_rdm_dir}/data_RDM.npy")

    
    if smoothing == True:
        if not os.path.exists(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.npy"):
            path_to_save_smooth = f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}"
            print(f"now smoothing the RDM and saving it here: {path_to_save_smooth}")
            data_RDMs = mc.analyse.handle_MRI_files.smooth_RDMs(data_RDMs, ref_img, fwhm,use_rsa_toolbox = False, path_to_save=path_to_save_smooth,centers=centers)
        else:
            data_RDMs = np.load(f"{data_rdm_dir}/data_RDM_smooth_fwhm{fwhm}.npy")
           
    # if visualise_RDMs == True:
    #     # note that data_RDM_file_2d this is now equivalent to half (first 40) of data_RDMs.
    #     # adjust!
    #     # ACC [54, 63, 41]
    #     mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [54, 63, 41], ref_img, condition_names)
    #     mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDMs, [54, 63, 41], ref_img, condition_names, centers = centers, no_rsa_toolbox=True)
        
    #     # visual cortex [72, 17, 9]
    #     mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [72, 17, 9], ref_img, condition_names)
    #     mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDMs, [72, 17, 9], ref_img, condition_names, centers = centers, no_rsa_toolbox=True)
        
    #     # hippocampus [43, 50, 17]
    #     mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [43, 50, 17], ref_img, condition_names)
    #     mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDMs, [43, 50, 17], ref_img, condition_names, centers = centers, no_rsa_toolbox=True)
        
       
    #
    # STEP 4: evaluate the model fit between model and data RDMs.
    #
    RSA_results = {}
    for model in selected_models:
        if model == 'A-state-mask' or model == 'A-state-ones':
            print("skipping computing A-state-mask or A-state-ones, already masked")
            continue
        print(model)
        # first, compute similarity esitmate for each model separately.
        # ACTUAL RSA - single regressors
        RSA_results[model] = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(model_RDM_dir[model][0], d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {model}"))
        mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=RSA_results[model], centers=centers, file_path = results_dir, file_name= f"{model}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)

    import pdb; pdb.set_trace() 
    run_combo_models = config.get("run_combo_models", bool(config.get("combo_models")))
    if run_combo_models:
        combo_list = config["combo_models"]
        for combo in combo_list:
            combo_model_name = combo["name"]
            print(f"running combo model {combo_model_name}")
            models_to_combine = combo["regressors"]
            # check if these models have been computed in model_EVs
            missing = [m for m in models_to_combine if m not in model_EVs]
            if missing:
                raise ValueError(f"Combo model {combo_model_name} not possible, as {missing} not computed")
            stacked_model_RDMs = np.stack([model_RDM_dir[m][0] for m in models_to_combine], axis=1)
            # check how correlated each model is whith each other.
            corr = np.corrcoef(stacked_model_RDMs, rowvar=False)
            for i in range(len(models_to_combine)):
                for j in range(i+1, len(models_to_combine)):
                    print(f"{models_to_combine[i]} vs {models_to_combine[j]}: r={corr[i,j]:.3f}")

            estimates_combined_model_rdms = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(stacked_model_RDMs, d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {combo_model_name}"))
            for i, model in enumerate(models_to_combine):
                mc.analyse.handle_MRI_files.save_my_RSA_results(result_file=estimates_combined_model_rdms, centers=centers, file_path = results_dir, file_name= f"{model.upper()}-{combo_model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
            
    # --- SETTINGS SUMMARY (per subject) ---
    summary = {
        "subject": sub,
        "EV_string": EV_string,
        "EV_labels_in_RDM": model_paired_labels,
        "regression_version": regression_version,
        "RDM_version": RDM_version,
        "smoothing": smoothing,
        "fwhm": fwhm,
        "load_searchlights": load_searchlights,
        "n_all_EVs": len(all_EV_keys),
        "n_selected_EVs": len(EV_keys),
        "models_evaluated": selected_models,
        "run_combo_models": run_combo_models,
        "combo_model_names": [c["name"] for c in config.get("combo_models", [])] if run_combo_models else [],
        "data_dir": data_dir,
        "results_dir": results_dir
    }
    
    print("\n=== SETTINGS SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    
    # Save a copy alongside results for provenance
    with open(os.path.join(results_dir, f"{sub}_settings_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"(Saved summary → {os.path.join(results_dir, f'{sub}_settings_summary.json')})\n")
            

