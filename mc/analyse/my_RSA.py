#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:38:27 2025

@author: Svenja Küchenhoff


All things RSA


"""
import mc
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import statsmodels.api as sm
import os
import nibabel as nib


def load_data_EVs(data_dir, regression_version, old=False, only_load_labels = False):
    EV_dict = {}
    # import pdb; pdb.set_trace()
    # names need to be 'A1_backw_A_path' etc.
    list_loaded = []
    for th in [1,2]:
        if regression_version.startswith('0'):
            # the old versions are numbers starting with 0. here, only take the first letters.
            pe_path = f"{data_dir}/func/glm_{regression_version[0:2]}_pt0{th}.feat/stats"
            EV_path = f"{data_dir}/func/EVs_{regression_version[0:2]}_pt0{th}/task-to-EV.txt"
        else:
            pe_path = f"{data_dir}/func/glm_{regression_version}_pt0{th}.feat/stats"
            EV_path = f"{data_dir}/func/EVs_{regression_version}_pt0{th}/task-to-EV.txt"
        # order from FSL processed EVs to names is stored here:
        
        with open(EV_path, 'r') as file:
        # pe_path = f"{data_dir}/func/glm_{regression_version[0:2]}_pt0{th}.feat/stats"
        # # order from FSL processed EVs to names is stored here:
        # with open(f"{data_dir}/func/EVs_{regression_version[0:2]}_pt0{th}/task-to-EV.txt", 'r') as file:
            for line in file:
                index, name_ev = line.strip().split(' ', 1)
                name = name_ev.replace('ev_', '')
                EV_path = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
                if only_load_labels == False:
                    EV_dict[name] = np.array(nib.load(EV_path).get_fdata()).flatten()
                else:
                    EV_dict[name] = np.zeros((1,1))
                    # reshape data so we have 1 x n_voxels
                    # import pdb; pdb.set_trace()
                if name not in ['press_EV', 'up', 'down', 'left', 'right']:
                    list_loaded.append(name)
    print(f"loaded the following data EVs in dict: {list_loaded}")
    return EV_dict, list_loaded
    
def load_data_EVs_th(data_dir, regression_version):
    EV_dict = {}
    # names need to be 'A1_backw_A_path' etc.
    list_loaded = []
    for th in [1,2]:
        # NOTE: if you still want to run the old ones, adjust which part of the regression version you take to load.
        pe_path = f"{data_dir}/func/glm_{regression_version}_pt0{th}.feat/stats"
        # order from FSL processed EVs to names is stored here:
        with open(f"{data_dir}/func/EVs_{regression_version}_pt0{th}/task-to-EV.txt", 'r') as file:
        # pe_path = f"{data_dir}/func/glm_{regression_version[0:2]}_pt0{th}.feat/stats"
        # # order from FSL processed EVs to names is stored here:
        # with open(f"{data_dir}/func/EVs_{regression_version[0:2]}_pt0{th}/task-to-EV.txt", 'r') as file:
            for line in file:
                index, name_ev = line.strip().split(' ', 1)
                name = name_ev.replace('ev_', '')
                # reshape data so we have 1 x n_voxels
                # import pdb; pdb.set_trace()
                if name not in ['press_EV', 'up', 'down', 'left', 'right']:
                    list_loaded.append(f"th_{th}_{name}")
                    EV_path = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
                    EV_dict[f"th_{th}_{name}"] = np.array(nib.load(EV_path).get_fdata()).flatten()

    print(f"loaded the following data EVs in dict: {list_loaded}")
    return EV_dict, list_loaded


def get_RDM_per_searchlight(fmri_data, centers, neighbors, method = 'crosscorr', labels = None, full_mask=None, mask_pairs=None):
    # import pdb; pdb.set_trace()
    centers = np.array(centers)
    #n_conds = fmri_data['1'].shape[0]
    n_conds = int(fmri_data.shape[0]/2)
    
    # first step: parallelise centers/neighbors.
    n_centers = centers.shape[0]
    # For memory reasons, we chunk the data if we have more than 1000 RDMs
    # loop over chunks
    if n_centers > 1000:
        # we can't run all centers at once, that will take too much memory
        # so lets to some chunking
        chunked_center = np.split(np.arange(n_centers),
                                  np.linspace(0, n_centers,
                                              101, dtype=int)[1:-1])
        # output will INCLUDE the diagonal. so triangle number is:
        sl_rdms = np.zeros((n_centers, n_conds * (n_conds + 1) // 2))
        # if excluding the diagonal
        # sl_rdms = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
        #for chunks in chunked_center:
        for chunks in tqdm(chunked_center, desc='Calculating RDMs...'):            
            center_data= []
            for c in chunks:
                # grab this centers of this chunk and its and neighbors
                center_neighbors = neighbors[c]
                center_data.append(fmri_data[:, center_neighbors])
            # then compute the RDM per searchlight
            if method == 'crosscorr':
                RDM_corr = mc.analyse.my_RSA.compute_crosscorr(center_data)
            elif method == 'crosscorr_and_filter':
                RDM_corr = mc.analyse.my_RSA.compute_crosscorr_and_filter(center_data, labels=labels, full_mask=full_mask, mask_pairs=mask_pairs)
            else:
                assert False, "invalid method"
            sl_rdms[chunks, :] = RDM_corr # then store per voxel and return.
       
    return sl_rdms
        
     
def parse_label_pair(label):
    """
    Example:
        'A1_backw_A_reward with A2_forw_A_reward'
    ->  state = 'A'
        task1 = 'A1_backw'
        task2 = 'A2_forw'
    """
    left, right = label.split(" with ")

    def parse_side(side):
        side = side.replace("_reward", "")
        task, state = side.rsplit("_", 1)
        return task, state

    task1, state1 = parse_side(left)
    task2, state2 = parse_side(right)

    assert state1 == state2, f"State mismatch inside label: {label}"
    return state1, task1, task2

def plot_rdm_with_labels(rdm, labels, group_size=4):
    n = rdm.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(rdm, aspect='auto', cmap='coolwarm', vmin=0, vmax=2)

    # Tick positions and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    # White lines after each task (every `group_size` rows/cols)
    for k in range(group_size, n, group_size):
        ax.axhline(k - 0.5, color='white', linewidth=1)
        ax.axvline(k - 0.5, color='white', linewidth=1)

    plt.tight_layout()
    plt.show()
       
def compute_crosscorr_and_filter(data_chunk, labels = None, full_mask=None, mask_pairs=None, plotting = False, binarise = False):  
    RDM = []
    # import pdb; pdb.set_trace()

    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    
    for data in data_chunk:
        # centers the data around zero by subtracting the mean of each row
        data_demeaned = data - data.mean(axis=1, keepdims=True)
        # normalising data
        data_demeaned /= np.sqrt(np.einsum('ij,ij->i', data_demeaned, data_demeaned))[:, None]    
        # cosine dissimilarity
        rdm_both_halves = 1 - np.einsum('ik,jk', data_demeaned, data_demeaned)  

        # cutting the lower left square of the matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
    
        # making the matrix symmetric
        rdm = (rdm_small + np.transpose(rdm_small))/2
        # import pdb; pdb.set_trace()
        # if you want to mask/filter, do that first
        if full_mask:
            # collect the indices I want to mask
            idx_to_mask = []
            for i, label in enumerate(labels):
                for m in full_mask:
                    if m in label:
                        idx_to_mask.append(i)
            
            rdm[:, idx_to_mask] = np.nan
            rdm[idx_to_mask] = np.nan
            
            if binarise == True:
                # THIS IS ONLY FOR MODEL RDMS!!
                rdm = np.where(np.isnan(rdm), np.nan, (rdm > 0.5).astype(float))
        
        elif mask_pairs:
            # Two cases:
            # 1) mask_pairs is your OLD list of substrings  -> keep old logic
            # 2) mask_pairs is the NEW big dict (state -> loc -> [tasks])
            if isinstance(mask_pairs, dict):
                # import pdb; pdb.set_trace()
                # --- NEW: mask "same state, same location" ---
                # Prepare splitting labels
                parsed = [parse_label_pair(lab) for lab in labels]
                n_cond = len(labels)

                for i in range(n_cond):
                    # loop through all conditions i
                    state_i, t1_i, t2_i = parsed[i] # split the labels
                    # for the respective state of the current condition, call the paired mask.
                    loc_dict_i = mask_pairs.get(state_i, {})
                    if not loc_dict_i:
                        continue
                    # next, check each paired condition j
                    for j in range(i, n_cond):
                        state_j, t1_j, t2_j = parsed[j]
                        if state_j != state_i:
                            continue

                        # Check if there exists a location where BOTH conditions live
                        for loc, tasks in loc_dict_i.items():
                            task_set = set(tasks)
                            if (t1_i in task_set and t2_i in task_set and
                                t1_j in task_set and t2_j in task_set):
                                # same state, same location -> mask
                                rdm[i, j] = np.nan
                                rdm[j, i] = np.nan
                                break  # stop looping over locs for this (i,j)

            else:
                # --- OLD behaviour: substring-based mask_pairs ---
                import pdb; pdb.set_trace()
                for m_l in mask_pairs:
                    idx = [i for i, lab in enumerate(labels) if m_l in lab]
                    idx = np.array(idx, dtype=int)
                    if idx.size > 0:
                        rdm[np.ix_(idx, idx)] = np.nan
   
        #import pdb; pdb.set_trace()
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the upper triangle, including the diagonal.
        n = rdm.shape[1]    
        RDM.append(rdm[np.triu_indices(n, k=0)]) 
        if plotting == True:
            plot_rdm_with_labels(rdm, labels, group_size=4)
            #plt.figure()
            #plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm')
            #plt.figure()
            #plt.imshow(rdm_both_halves, aspect = 'auto', cmap = 'coolwarm')

    return RDM



def compute_crosscorr(data_chunk, plotting = False):  
    RDM = []
    #import pdb; pdb.set_trace()
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    
    for data in data_chunk:
        # centers the data around zero by subtracting the mean of each row
        data_demeaned = data - data.mean(axis=1, keepdims=True)
        # normalising data
        data_demeaned /= np.sqrt(np.einsum('ij,ij->i', data_demeaned, data_demeaned))[:, None]    
        # cosine dissimilarity
        rdm_both_halves = 1 - np.einsum('ik,jk', data_demeaned, data_demeaned)  
        
        # cutting the lower left square of the matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
        
        # making the matrix symmetric
        rdm = (rdm_small + np.transpose(rdm_small))/2
        
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the upper triangle, including the diagonal.
        n = rdm.shape[1]
        RDM.append(rdm[np.triu_indices(n, k=0)]) 
        if plotting == True:
            plt.figure()
            plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm', vmax=2, vmin=0)
            plt.figure()
            plt.imshow(rdm_both_halves, aspect = 'auto', cmap = 'coolwarm')
            
    return RDM



def mask_RDM(lower_tri, n, labels, mask=None, binarise = False, plotting = False):
    # import pdb; pdb.set_trace()
    # this puts it to the upper triangle 
    masked_RDM = np.full((n, n), np.nan, dtype = float) 
    iu = np.triu_indices(n, 0) 
    masked_RDM[iu] = lower_tri 

    # collect the indicesI want to mask
    idx_to_mask = []
    for i, label in enumerate(labels):
        for m in mask:
            if m in label:
                idx_to_mask.append(i)
    
    masked_RDM[:, idx_to_mask] = np.nan
    masked_RDM[idx_to_mask] = np.nan
    masked_vector = masked_RDM[np.triu_indices(n, 0)]
    
    if binarise == True:
        # THIS IS ONLY FOR MODEL RDMS!!
        masked_vector = np.where(np.isnan(masked_vector), np.nan, (masked_vector > 0.5).astype(float))
    if plotting == True:
        plt.figure()
        plt.imshow(masked_RDM, aspect = 'auto', cmap = 'coolwarm')

    return masked_vector


def evaluate_model(model_rdm, data_rdm):
    # import pdb; pdb.set_trace()

    #X = sm.add_constant(model_rdm.transpose());
    X = sm.add_constant(model_rdm);
    # first, normalize the regressors (but not the intercept, bc std = 0 -> division by 0!)
    # X = model_rdm.transpose()
    for i in range(1, X.shape[1]):
        X[:,i] = (X[:,i] - np.nanmean(X[:,i]))/ np.nanstd(X[:,i])
    
    # to check if a GLM is ill-conditioned
    # To check that you can check the “condition number” of the design matrix - 
    # the ration between the maximum singular value (similar to eigenvalue) and the minimum singular value.. 
    # If that ratio is close to 1, you’re good. If it’s very large (e.g. >1000), it means the matrix is ill-conditioned - 
    # one of your regressors is close to being a linear combination of the other two.
    # mc.analyse.analyse_MRI_behav.check_GLM_regressors(X)
    # import pdb; pdb.set_trace()
    
    Y = data_rdm;
    Y = (Y - np.nanmean(Y))/ np.nanstd(Y)
    
    # to filter out potential nans in the model part
    nan_filter = np.isnan(X).any(axis=1)
    filtered_X = X[~nan_filter]
    filtered_Y = Y[~nan_filter]
    
    est = sm.OLS(filtered_Y, filtered_X).fit()
    # import pdb; pdb.set_trace()
    return est.tvalues[1:], est.params[1:], est.pvalues[1:]
