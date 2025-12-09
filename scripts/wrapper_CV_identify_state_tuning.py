#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 20:42:38 2025

@author: Svenja Küchenhoff

"""

import os
import mc
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st
from joblib import Parallel, delayed


def get_data(sub, trials):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    if trials == 'residualised':
        res_data = True
    else:
        res_data = False
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [f"{sub:02}"], res_data = res_data)
    return data_norm, data_folder

def comp_state_tuning(neurons, perms = None, random_data = False, only_BCD = False):
    # import pdb; pdb.set_trace() 
    if random_data == True:
        n_rows = neurons.shape[0]
        n_cols =neurons.shape[1]
        neurons = np.random.randint(1,100, size = (n_rows, n_cols))
    
    if only_BCD:
        mean_firing_rates_states = np.full((3), np.nan, dtype=float)
    else:
        mean_firing_rates_states = np.full((4), np.nan, dtype=float)
    states = np.repeat((0,1,2,3), 90)
    states = np.tile(states, len(neurons))
    
    #2025 6th of October: this is BAD!
    # Removes repeat main effects. Within each neuron×repeat you subtract that repeat’s mean (across states), 
    # so any genuine baseline drift or learning-related gain change over repeats is forced to 0. 
    # Interaction can look bigger because only relative differences between states remain.
    # Rescales each repeat differently. Dividing by that repeat’s SD means repeats with low 
    # variance get blown up and high-variance repeats get shrunk. Averages across repeats/tasks no longer live in a single, comparable unit.
    
    # # z-score each repeat across time (axis=1) to remove task-wide gain
    # for i_r, rep in enumerate(neurons):
    #     m = np.nanmean(rep)
    #     s = np.nanstd(rep, ddof=0)
    #     neurons[i_r] = (rep - m) / s if s and np.isfinite(s) else (rep - m)

            
    if perms:
        for i_r, rep in enumerate(neurons):
            # import pdb; pdb.set_trace()
            # circular shift of location time series by random bins along time_axis
            T = neurons.shape[1]
            k = np.random.randint(0, T)
            #k = int(perms.integers(1, T)) if T > 1 else 0
            neurons[i_r] = np.roll(rep, -k)
            
    fr_all_reps  = neurons.reshape(-1).astype(float)
    nan_mask = np.isfinite(fr_all_reps)
    fr_clean = fr_all_reps[nan_mask]
    state_clean = states[nan_mask]
    
    #import pdb; pdb.set_trace()
    if only_BCD:
        for state in range(1,4):
            sel = (state_clean == state)
            mean_firing_rates_states[state-1] = fr_clean[sel].mean()
        idx = max(range(3), key=mean_firing_rates_states.__getitem__)
        pref_state = "BCD"[idx]
    else:
        for state in range(0,4):
            sel = (state_clean == state)
            mean_firing_rates_states[state] = fr_clean[sel].mean()
        idx = max(range(4), key=mean_firing_rates_states.__getitem__)
        pref_state = "ABCD"[idx]
        
    return mean_firing_rates_states, pref_state
    

# --- Helpers for stats ---
def one_tailed_ttest_greater_than_zero(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan  # t, p, mean
    t_stat, p_two = st.ttest_1samp(x, 0.0, nan_policy='omit')
    p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
    return float(t_stat), float(p_one), float(np.mean(x))

def stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'
    


def plot_results_per_roi(df, title_string_add, plot_by_pfc=False, plot_by_cingulate_and_MTL=False):
    # --- inputs ---
    metric_col = 'state_cv_consistency'
    p_col = 'p_perm'   # optional column
    alpha_sig = 0.05
    bins = 20

    # ROI labels
    df = df.copy()
    df['roi'] = mc.analyse.helpers_human_cells.rename_rois(
        df,
        collapse_pfc=plot_by_pfc,
        plot_by_cingulate_and_MTL=plot_by_cingulate_and_MTL
    )

    rois = df['roi'].dropna().unique().tolist()
    n_cols = len(rois)

    plt.rcParams.update({'font.size': 11})

    # single row, n_cols columns
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(max(6.5, 2.2 * n_cols), 4.0),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.3}
    )
    if n_cols == 1:
        axes = np.array([axes])

    # Precompute common bin edges
    all_vals = df[metric_col].to_numpy(dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    ylim_max = 0
    per_roi_data = []
    for roi in rois:
        sub = df.loc[df['roi'] == roi]
        vals = sub[metric_col].to_numpy(dtype=float)
        mask_valid = np.isfinite(vals)

        # If p_perm exists, split into significant vs. not
        if p_col in sub.columns:
            pvals = sub[p_col].to_numpy(dtype=float)
            mask_valid &= np.isfinite(pvals)
            vals = vals[mask_valid]
            pvals = pvals[mask_valid]

            sig_mask = pvals < alpha_sig
            vals_sig = vals[sig_mask]
            vals_nonsig = vals[~sig_mask]

            c_sig, _ = np.histogram(vals_sig, bins=bin_edges)
            c_nonsig, _ = np.histogram(vals_nonsig, bins=bin_edges)
            counts = c_sig + c_nonsig
            per_roi_data.append((vals, vals_sig, vals_nonsig))
        else:
            vals = vals[mask_valid]
            counts, _ = np.histogram(vals, bins=bin_edges)
            per_roi_data.append((vals, None, None))

        ylim_max = max(ylim_max, counts.max() if counts.size else 0)

    # plotting
    for ax, roi, vals_tuple in zip(axes, rois, per_roi_data):
        vals_all, vals_sig, vals_nonsig = vals_tuple

        if vals_sig is not None:  # p_perm available
            ax.hist(vals_nonsig, bins=bin_edges,
                    color='lightgray', edgecolor='black', alpha=1.0, label='n.s.')
            ax.hist(vals_sig, bins=bin_edges,
                    color='salmon', edgecolor='black', alpha=0.95, label=f'p<{alpha_sig:.2f}')
        else:  # no p_perm
            ax.hist(vals_all, bins=bin_edges,
                    color='lightgray', edgecolor='black')

        # zero line
        ax.axvline(0, color='k', linestyle='dashed', linewidth=1.3)

        # stats box (based on all vals)
        t_stat, p_one, mval = one_tailed_ttest_greater_than_zero(vals_all)
        sig = stars(p_one)
        ax.text(
            0.98, 0.96,
            f"n={vals_all.size}\nmean={mval:.2f}\n{sig} (p={p_one:.1e})",
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
            fontsize=10
        )

        ax.set_title(str(roi), pad=4)
        ax.tick_params(axis='both', labelsize=10, width=1.0, length=4)

    # consistent y-axis
    axes[0].set_ylim(0, max(1, int(ylim_max * 1.04)))

    # shared labels + title
    fig.supxlabel(metric_col)
    fig.supylabel("Frequency")
    fig.suptitle(
        f"Cross-validated state consistency per cell, split by ROI\n{title_string_add}",
        fontsize=12, fontweight='bold', y=0.98
    )

    # add legend only if p_perm was present
    if p_col in df.columns:
        axes[-1].legend(frameon=False, fontsize=9, loc='upper left')

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.85, wspace=0.3)
    plt.show()



# def plot_results_per_roi(df, title_string_add, plot_by_pfc = False, plot_by_cingulate_and_MTL = False):

#     # --- inputs ---
#     metric_col = 'state_cv_consistency'
#     bins = 20
#     title_string_add = title_string_add if 'title_string_add' in locals() else ''
#     plot_by_pfc = locals().get('plot_by_pfc', False)
#     plot_by_cingulate_and_MTL = locals().get('plot_by_cingulate_and_MTL', False)
    
#     # ROI labels
#     df['roi'] = mc.analyse.helpers_human_cells.rename_rois(
#         df,
#         collapse_pfc=plot_by_pfc,
#         plot_by_cingulate_and_MTL=plot_by_cingulate_and_MTL
#     )
    
#     rois = df['roi'].dropna().unique().tolist()
#     n_cols = len(rois)
    
#     plt.rcParams.update({'font.size': 11})
    
#     # single row, n_cols columns
#     fig, axes = plt.subplots(
#         1, n_cols,
#         figsize=(max(6.5, 2.2*n_cols), 4.0),
#         sharex=True, sharey=True,
#         gridspec_kw={'wspace': 0.3}
#     )
    
#     # make axes iterable even when n_cols == 1
#     if n_cols == 1:
#         axes = np.array([axes])
    
#     # precompute max y across all ROIs for consistent scaling
#     ylim_max = 0
#     vals_per_roi = []
    
#     for roi in rois:
#         vals = df.loc[df['roi'] == roi, metric_col].to_numpy(dtype=float)
#         vals = vals[np.isfinite(vals)]
#         counts, _ = np.histogram(vals, bins=bins)
#         ylim_max = max(ylim_max, counts.max() if counts.size else 0)
#         vals_per_roi.append(vals)
    
#     # plotting
#     for ax, roi, vals in zip(axes, rois, vals_per_roi):
#         ax.hist(vals, bins=bins)
#         ax.axvline(0, linestyle='dashed', linewidth=1.3)
    
#         # stats box
#         t_stat, p_one, mval = one_tailed_ttest_greater_than_zero(vals)
#         sig = stars(p_one)
#         ax.text(
#             0.98, 0.96,
#             f"n={vals.size}\nmean={mval:.2f}\n{sig} (p={p_one:.1e})",
#             transform=ax.transAxes, ha='right', va='top',
#             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
#             fontsize=10
#         )
    
#         ax.set_title(str(roi), pad=4)
#         ax.tick_params(axis='both', labelsize=10, width=1.0, length=4)
    
#     # consistent y-axis across ROIs (+4% headroom)
#     axes[0].set_ylim(0, max(1, int(ylim_max * 1.04)))
    
#     # shared labels + title
#     fig.supxlabel("state_cv_consistency")
#     fig.supylabel("Frequency")
#     fig.suptitle(
#         f"Cross-validated state consistency per cell, split by ROI\n{title_string_add}",
#         fontsize=12, fontweight='bold', y=0.98
#     )
    
#     fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.85, wspace=0.3)
#     plt.show()



def store_p_vals_perms(true_df, perm_df, out_path, trials):
    # merge obs with all its nulls, compute p for each group
    obs = true_df[["session_id","neuron_id","state_cv_consistency"]].rename(columns={"state_cv_consistency":"obs"})
    nulls = perm_df[["session_id","neuron_id","state_cv_consistency"]].rename(columns={"state_cv_consistency":"perm"})
    merged = obs.merge(nulls, on=["session_id","neuron_id"], how="left")
    
    def perm_p_one_sided(g):
        M = g["perm"].notna().sum()
        if M == 0 or not np.isfinite(g["obs"].iat[0]):
            return pd.Series({"n_perms": M, "p_perm": np.nan})
        k = (g["perm"] >= g["obs"].iat[0]).sum()       # how many nulls >= obs
        p = (k + 1) / (M + 1)
        return pd.Series({"n_perms": M, "p_perm": p})
    
    stats = merged.groupby(["session_id","neuron_id"], sort=False).apply(perm_p_one_sided).reset_index()

    # attach p-values back to the true table
    out = true_df.merge(stats, on=["session_id","neuron_id"], how="left")
    # ---------- FDR (Benjamini–Hochberg) ----------
    def bh_reject(pvals, alpha=0.05):
        p = np.asarray(pvals, float)
        mask = np.isfinite(p); m = mask.sum()
        sig = np.zeros_like(p, bool)
        if m == 0: return sig
        order = np.argsort(p[mask]); p_sorted = p[mask][order]
        thresh = alpha * (np.arange(1, m+1) / m)
        passed = p_sorted <= thresh
        if passed.any():
            cutoff = p_sorted[np.nonzero(passed)[0].max()]
            sig[mask] = p[mask] <= cutoff
        return sig
        
    # 4) write back onto the original table (align by index; fill others with NA)
    out['sig_FDR_all'] = bh_reject(out['p_perm'].to_numpy(), alpha=0.05)
    # ---------- save ----------
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    return out







def run_state_corr_perm_wise(perm_idx, unique_grids, idx_same_grids, data, beh_df, sesh, curr_neuron, no_perms=None, random_data=False, only_BCD=False, sparsity_c=None):
    consistency_train_test = []
    pref_states = []
    for count_test_task, test_task_id in enumerate(unique_grids):
        mask_test_task = (idx_same_grids == test_task_id)
        neurons_test_task = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].loc[mask_test_task].to_numpy()

        mask_train_task = (idx_same_grids != test_task_id)
        # create subset of df and neurons.
        neurons_train_task = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].loc[mask_train_task].to_numpy()


        # z-score both test and train neurons with estimated mean and std of train tasks.
        # this is necessary because I'd artificially get rid of any task-effects, while this is exactly
        # what I want to test here!
        
        mean_train_tasks = np.mean(neurons_train_task)
        std_train_tasks = np.mean(neurons_train_task)
        
        z_scored_neuron_train_tasks = (neurons_train_task - mean_train_tasks)/ std_train_tasks
        z_scored_neuron_test_tasks = (neurons_test_task - mean_train_tasks)/ std_train_tasks
        
        # and compute the state-tuning in the train-tasks
        fr_state_train_tasks, _ = comp_state_tuning(z_scored_neuron_train_tasks, random_data=random_data, only_BCD=only_BCD)

        # validate: compute the correlation with state-rate-map of held-out task
        fr_state_test_task, pref_state_curr_train = comp_state_tuning(z_scored_neuron_test_tasks, perms = no_perms, random_data=random_data, only_BCD=only_BCD)
        
        consistency_train_test.append(np.corrcoef(fr_state_test_task, fr_state_train_tasks)[1][0])
        # 
        pref_states.append(pref_state_curr_train)
        
        
    mean_state_consistency = np.mean(consistency_train_test)
    # import pdb; pdb.set_trace()
    states, counts_state = np.unique(pref_states, return_counts=True)
    pref_state = states[np.argmax(counts_state)]
    
    if sparsity_c:
        results_one_perm = {
        "session_id": sesh,
        "neuron_id": curr_neuron,
        "state_cv_consistency": mean_state_consistency,
        "perm_idx": perm_idx,
        "mean_firing_rate": beh_df[f'mean_FR_{curr_neuron}'].to_numpy()[0],
        "sparse_repeats": sum(~beh_df[f'consistent_FR_{curr_neuron}']),
        "pref_states": pref_states,
        "pref_state": pref_state
        }
    else:
        results_one_perm = {
        "session_id": sesh,
        "neuron_id": curr_neuron,
        "state_cv_consistency": mean_state_consistency,
        "perm_idx": perm_idx,
        "mean_firing_rate": beh_df[f'mean_FR_{curr_neuron}'].to_numpy()[0],
        "pref_states": pref_states,
        "pref_state": pref_state
        }
    
    return results_one_perm

    

def compute_state_tunings(sessions, trials = 'all_minus_explore', no_perms = None, sparsity_c = None, save_all = False, random_data = False, only_BCD=False):
    # determine results table
    
    if sparsity_c:
        COLUMNS = [
        "session_id", "neuron_id",
        "state_cv_consistency",
        "perm_idx", 
        "mean_firing_rate",
        "sparse_repeats",
        "pref_states",
        "pref_state"
        ]
    else:
        COLUMNS = [
        "session_id", "neuron_id",
        "state_cv_consistency",
        "perm_idx", 
        "mean_firing_rate",
        "pref_states",
        "pref_state"
        ]

    
    results = []
    included_neurons = []
    
    for sesh in sessions:
        # load data
        data_raw, source_dir = get_data(sesh, trials=trials)
        group_dir_state = f"{source_dir}/group/state_tuning"
        # import pdb; pdb.set_trace()
        # if this session doesn't exist, skip
        if not data_raw:
            print(f"no raw data found for {sesh}, so skipping")
            continue
    
        # filter data for only those repeats that were 1) correct and 2) not the first one
        data = mc.analyse.helpers_human_cells.filter_data(data_raw, sesh, trials)
        beh_df = data[f"sub-{sesh:02}"]['beh'].copy()


        if no_perms:
            perms = no_perms
            #no_perms = np.random.default_rng(123)
            # perms = permute_locations(data[f"sub-{sesh:02}"]['locations'], data[f"sub-{sesh:02}"]['beh'], no_perms = no_perms)
            if sparsity_c:
                include_these_cells = Path("/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/state_tuning/included_cells_all_minus_explore_reps_gridwise_qc_pct.txt").read_text().splitlines()
                # include_these_cells = Path(f"{group_dir_state}/included_cells_{trials}_reps_{sparsity_c}_pct.txt").read_text().splitlines()

        else:
            perms = 1
     
        # for each cell, cross-validate the peak task-lag shift for spatial consistency.
        for neuron_idx, curr_neuron in enumerate(data[f"sub-{sesh:02}"]['normalised_neurons']):
            # resetting unique tasks for each neuron.
            # determine identical grids
            grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
            unique_grids, _, idx_same_grids, _ = np.unique(
                beh_df[grid_cols].to_numpy(),
                axis=0,
                return_index=True,
                return_inverse=True,
                return_counts=True
            )
        
            beh_df['idx_same_grids'] = idx_same_grids
            if sparsity_c:
                if perms > 1:
                    if curr_neuron not in include_these_cells:
                        continue
                else: 
                    included_neurons.append(curr_neuron)
            
            # clean the data such that I don't consider 'bad' blocks of repeats
            # with super low firing 
            # import pdb; pdb.set_trace()
            beh_df[f'mean_FR_{curr_neuron}'] = np.nanmean(data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy())
            
            if sparsity_c:
                # print("defining tasks where neuron fires too sparsly (less than 20% of mean firing)")
                beh_df = mc.analyse.helpers_human_cells.extract_consistent_grids(data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy(), curr_neuron, beh_df)
                consistent_grids_mask = beh_df[f'consistent_FR_{curr_neuron}'].to_numpy()
                # set grid indexes I want to ignore to -1
                idx_same_grids[~consistent_grids_mask] = -1
                
                # if after excluding inconsistent grids there aren't enough grids for CV left,
                # kick this neuron.
                unique_grids = np.unique(beh_df['idx_same_grids'][beh_df[f'consistent_FR_{curr_neuron}']])
            else:
                unique_grids = np.unique(beh_df['idx_same_grids'])
            
            if sparsity_c and len(unique_grids) < 3:
                print(f"excluding {curr_neuron} in sesh {sesh} because there were not enough grids with consistent FR!")
                continue
            
            # # debug loop
            # for perm_idx in range(0,perms):
            #     results_curr_neuron = run_state_corr_perm_wise(perm_idx, unique_grids, idx_same_grids, data, beh_df, sesh, curr_neuron, no_perms=no_perms, random_data=random_data, only_BCD=only_BCD, sparsity_c=sparsity_c)
            # import pdb; pdb.set_trace()
            
            # parallelise this
            results_curr_neuron = Parallel(n_jobs = -1)(delayed(run_state_corr_perm_wise)(perm_idx, unique_grids, idx_same_grids, data, beh_df, sesh, curr_neuron, no_perms=no_perms, random_data=random_data, only_BCD=only_BCD, sparsity_c=sparsity_c) for perm_idx in range(0,perms))
            results.extend(results_curr_neuron)
            if not no_perms:
                print(f"average state consistency for neuron {curr_neuron} is {results[-1]['mean_firing_rate']}")
            if no_perms:
                print(f"now done computing {no_perms} permutations for neuron {curr_neuron}...")
            
                
    
    
    results_df = pd.DataFrame(results, columns = COLUMNS)
    results_df['roi'] = mc.analyse.helpers_human_cells.rename_rois(
        results_df,
        collapse_pfc=False,
        plot_by_cingulate_and_MTL=False
    )
    # import pdb; pdb.set_trace()
    if save_all == True:
        if not os.path.isdir(group_dir_state):
            os.mkdir(group_dir_state)
        
        if perms > 1:

            result_string = f"state_consistency_{trials}_repeats.csv"
            if only_BCD == True:
                result_string = f"state_consistency_{trials}_repeats_only_BCD.csv"
                
            elif sparsity_c:
                result_string = f"state_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons.csv"
                
            empirical_result = pd.read_csv(f"{group_dir_state}/{result_string}")
            perm_string = f"pval_for_perms200_{result_string}"
            name_result_stats = f"{group_dir_state}/{perm_string}"
            # DELETE THIS
            # cells = results_df['neuron_id'].unique()
            # for cell in cells:
            #     perm_vals = results_df[results_df['neuron_id']==cell]['state_cv_consistency'].to_numpy()
            #     plt.figure()
            #     plt.hist(perm_vals, bins = 50)
            #     plt.title(f"{cell} mean = {np.mean(perm_vals)}")
                
            # import pdb; pdb.set_trace()
            perm_pval_result = store_p_vals_perms(true_df = empirical_result, perm_df = results_df, out_path=name_result_stats, trials=trials)
            #mc.plotting.results.plot_perm_spatial_consistency(results_df, empirical_result, name_result_stats, group_dir_fut_spat)
            title_string = f'Binomial test after single-cell permutations, {trials} repeats'
            # plt_binomial_per_roi(perm_pval_result, title_string)
            results_df.to_csv(f"{group_dir_state}/perms_{result_string}")
            

        else:
            name_result = f"{group_dir_state}/state_consistency_{trials}_repeats.csv"
            plot_results_per_roi(results_df, title_string_add = f'{trials}_repeats',plot_by_pfc=False,plot_by_cingulate_and_MTL=True)
            plot_results_per_roi(results_df, title_string_add = f'{trials}_repeats',plot_by_pfc=False,plot_by_cingulate_and_MTL=False)
            plot_results_per_roi(results_df, title_string_add = f'{trials}_repeats',plot_by_pfc=True,plot_by_cingulate_and_MTL=False)
            
 
            if only_BCD == True:
                name_result = f"{group_dir_state}/state_consistency_{trials}_repeats_only_BCD.csv"
                
            elif sparsity_c:
                name_result = f"{group_dir_state}/state_consistency_{trials}_repeats_excl_{sparsity_c}_pct_neurons.csv"
                print(f"included {len(included_neurons)} neurons.")
                Path(f"{group_dir_state}/included_cells_{trials}_reps_{sparsity_c}_pct.txt").write_text("\n".join(included_neurons))
            

            results_df.to_csv(name_result)
        
        # results_df = pd.read_csv('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/state_tuning/perms_state_consistency_all_minus_explore_repeats_excl_gridwise_qc_pct_neurons.csv')
        # pval_df = pd.read_csv('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/state_tuning/pval_for_perms200_state_consistency_late_repeats_excl_gridwise_qc_pct_neurons.csv')
        
        import pdb; pdb.set_trace()
        print(f"saved cross-validated state tuning values in {name_result}")  
        
  
if __name__ == "__main__":
    # trials can be 'all', 'all_correct', 'early', 'late', 'all_minus_explore', 'residualised'
    # they can also be: 'first_correct', 'one_correct', ... 'nine_correct'
    # NOTE: if you do per-repeat estimates, use every grid! grid mean will be super unreliable anyways
    compute_state_tunings(sessions=list(range(0,64)), trials = 'all_correct', no_perms = None, sparsity_c = 'gridwise_qc', save_all=True, only_BCD = False)
    
    
    # compute_state_tunings(sessions=list(range(0,64)), trials = 'all_minus_explore', no_perms = None, sparsity_c = 'gridwise_qc', save_all=True, only_BCD = False)
    #compute_state_tunings(sessions=[4], trials = 'residualised', no_perms = None, sparsity_c = 'gridwise_qc', save_all=False, random_data=False)

    
    