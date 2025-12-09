#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:06:31 2025

easy linear model for states.


# START AGAIN.
# get one p-value per regressor, and bonferroni correct for amount of neurons (in this area)
# also test the distribution of slopes per ROI against zero


"""

import os
import mc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm

#import pdb; pdb.set_trace()

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
    # import pdb; pdb.set_trace()
    return data_norm, data_folder 

def make_long_df(
    neuron: np.ndarray,
    beh_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a long DataFrame (trials x bins) with y, state, state_phase,
    and repeated behavioral columns. Optionally add one-hot columns
    for each state and state_phase category.

    Parameters
    ----------
    neuron : np.ndarray, shape (n_trials, 360)
    beh_df : pd.DataFrame, length n_trials, with columns:
             'rep_correct', 'grid_no', 'correct'
    add_one_hot : bool, default True
        If True, add one-hot columns for each category in state and state_phase.
    drop_first : bool, default False
        If True, drop the first category in each set (useful if you'll include
        an intercept in a linear model).

    Returns
    -------
    df_long : pd.DataFrame, shape (n_trials*360, ...)
    """

    n_trials, n_bins = neuron.shape
    assert n_bins == 360, "Expected 360 bins per trial."
    assert len(beh_df) == n_trials, "beh_df length must match neuron.shape[0]."

    # --- category definitions ---
    state_categories = ["A", "B", "C", "D"]
    state_bins = {
        "A": slice(0, 90),
        "B": slice(90, 180),
        "C": slice(180, 270),
        "D": slice(270, 360),
    }
    state_phase_categories = [
        "A_early", "A_mid", "A_rew",
        "B_early", "B_mid", "B_rew",
        "C_early", "C_mid", "C_rew",
        "D_early", "D_mid", "D_rew",
    ]
    phase_bins = {
        "A_early": slice(0, 30),   "A_mid": slice(30, 60),   "A_rew": slice(60, 90),
        "B_early": slice(90, 120), "B_mid": slice(120, 150), "B_rew": slice(150, 180),
        "C_early": slice(180, 210),"C_mid": slice(210, 240), "C_rew": slice(240, 270),
        "D_early": slice(270, 300),"D_mid": slice(300, 330), "D_rew": slice(330, 360),
    }
    
    # --- per-trial bin labels ---
    state_per_bin = np.empty(n_bins, dtype=object)
    for lbl, sl in state_bins.items():
        state_per_bin[sl] = lbl

    state_phase_per_bin = np.empty(n_bins, dtype=object)
    for lbl, sl in phase_bins.items():
        state_phase_per_bin[sl] = lbl

    # --- expand over all trials ---
    y_long = neuron.reshape(-1)
    state_long = np.tile(state_per_bin, n_trials)
    state_phase_long = np.tile(state_phase_per_bin, n_trials)

    rep_correct_long = np.repeat(beh_df["rep_correct"].to_numpy(), n_bins)
    grid_no_long     = np.repeat(beh_df["grid_no"].to_numpy(), n_bins)
    correct_long     = np.repeat(beh_df["correct"].to_numpy(), n_bins)

    # --- assemble df ---
    df_long = pd.DataFrame({
        "y": y_long,
        "state": state_long,
        "state_phase": state_phase_long,
        "rep_correct": rep_correct_long,
        "grid_no": grid_no_long,
        "correct": correct_long,
    })

    # keep ordered categoricals
    df_long["state"] = pd.Categorical(df_long["state"],
                                      categories=state_categories, ordered=True)
    df_long["state_phase"] = pd.Categorical(df_long["state_phase"],
                                            categories=state_phase_categories, ordered=True)
    
    # which categories to encode (optionally drop first)
    state_cats_encode = state_categories
    phase_cats_encode = state_phase_categories

    # one-hot (int8 for memory efficiency)
    svals = df_long["state"].to_numpy()
    for cat in state_cats_encode:
        df_long[cat] = (svals == cat).astype("int8")

    spvals = df_long["state_phase"].to_numpy()
    for cat in phase_cats_encode:
        df_long[cat] = (spvals == cat).astype("int8")


    df_long = df_long.replace([np.inf, -np.inf], np.nan)
    df_long = df_long.dropna(subset=["y","A","B","C","D","rep_correct","correct"])
    # keep valid one-hot rows
    df_long = df_long[df_long[["A","B","C","D"]].sum(axis=1) == 1]
    # center repeat count per neuron
    mu_rep = df_long["rep_correct"].mean()
    df_long["rep_c"] = df_long["rep_correct"] - mu_rep
    
    # NEW: state × repeat interactions (keep it simple)
    df_long["A_rep"] = df_long["A"].astype("float32") * df_long["rep_c"].astype("float32")
    df_long["B_rep"] = df_long["B"].astype("float32") * df_long["rep_c"].astype("float32")
    df_long["C_rep"] = df_long["C"].astype("float32") * df_long["rep_c"].astype("float32")
    df_long["D_rep"] = df_long["D"].astype("float32") * df_long["rep_c"].astype("float32")

    return df_long



def determine_roi(n):
    if 'ACC' in n or 'vCC' in n or 'AMC' in n or 'vmPFC' in n:
                roi = 'ACC'
    elif 'PCC' in n:
        roi = 'PCC'
    elif 'OFC' in n:
        roi = 'OFC'
    elif 'MCC' in n or 'HC' in n:
        roi = 'hippocampal'
    elif 'EC' in n:
        roi = 'entorhinal'
    elif 'AMYG' in n:
        roi = 'amygdala'
    else:
        roi = 'mixed'
    return roi



def run_linear_models(df, session, neuron_id, roi, permute = False, rng_object = None, no_perms = None):
    result_rows = [] 
    # now, fit the full model with interaction terms.
    # get one p-value per regressor -> for group stats
    # and the beta (slope) per regressor -> for group stats
    # import pdb; pdb.set_trace()
    y = df["y"].to_numpy(float)
    y = stats.zscore(y)

    if permute == True:
        # do the circular shift per repeat on the data.
        n_bins = 360
        n_repeats = y.size // n_bins       # e.g. 120
        y_2d = y.reshape(n_repeats, n_bins) # shape (n_repeats, 360)
        shifts = rng_object.integers(n_bins, size=n_repeats)
        shifted = np.empty_like(y_2d)
        for r in range(n_repeats):
            shifted[r] = np.roll(y_2d[r], shifts[r])
        y = shifted.reshape(-1)

    # Model 1 (main effects): y ~ A+B+C+D + rep_c + correct (no intercept)
    # cols_m1 = ["A","B", "C", "D","rep_c","correct"]
    # don't include "correct" as some subjects are just 100% correct after the explore trial.
    cols_m1 = ["A","B", "C", "D","rep_c"]
    X_m1 = df[cols_m1].astype(float)
    # DONT demean abcd, but demean the data!!!
    # this is because if I 0-centre ABCD, the model is rank-defficient

    res1 = sm.OLS(y, X_m1).fit(cov_type="HC3")
    
    # beta = np.linalg.pinv(X_m1) @ y
    # is the same as res1.params
    
    # State main effect.
    # This corresponds to creating four separate T-contrasts 
    # [1 0 0 0 0 0], [0 1 0 0 0 0], [0 0 1 0 0 0], [0 0 0 1 0 0]
    f_res = res1.f_test('A = B = C = D = 0')
    # import pdb; pdb.set_trace()
    # per-regressor rows (M1)
    for term in cols_m1:
        result_rows.append({
            "session": session, "neuron": neuron_id, "roi": roi,
            "model": "M1_main", "term": term,
            "beta": float(res1.params.get(term, np.nan)),
            "t":    float(res1.tvalues.get(term, np.nan)),
            "p":    float(res1.pvalues.get(term, np.nan)),
            "F":    f_res.fvalue,
            "p_F": f_res.pvalue,
            "permuted": permute
        })
    
    # Model 2 (with interaction): y ~ A+B+C+D + correct + (A_rep+B_rep+C_rep+D_rep)
    # exclude rep_c here because the interactions would be a linear combination of it.
    # cols_m2 = ["A","B","C","D","correct","A_rep","B_rep","C_rep","D_rep"]
    cols_m2 = ["A","B","C","D","A_rep","B_rep","C_rep","D_rep"]
    X_m2 = df[cols_m2].astype(float)      
    res2 = sm.OLS(y, X_m2).fit(cov_type="HC3")

    # Main effect State Repeats
    f_res_int = res2.f_test('A_rep = B_rep = C_rep = D_rep = 0')

    for term in cols_m2:
        result_rows.append({
            "session": session, "neuron": neuron_id, "roi": roi,
            "model": "M2_interact", "term": term,
            "beta": float(res2.params.get(term, np.nan)),
            "t":    float(res2.tvalues.get(term, np.nan)),
            "p":    float(res2.pvalues.get(term, np.nan)),
            "F":    f_res_int.fvalue,
            "p_F": f_res_int.pvalue,
            "permuted": permute
        })

    return result_rows




def run_single_neuron(neuron_name, sesh, neuron_data, beh_df, perms = False, no_perms = None):
    df = make_long_df(neuron_data, beh_df)
    roi = determine_roi(neuron_name)
    neuron_results_rows = []
    if perms == True:
        seed = (hash(neuron_name) % 2**32)
        rng = np.random.default_rng(seed = seed)
        for i in range(no_perms):
            results = run_linear_models(df, sesh, neuron_name, roi, permute=perms, rng_object = rng, no_perms = no_perms)
            neuron_results_rows.extend(results)
            if (i + 1) % 100 == 0:
                print(f"[{neuron_name}] finished permutation {i + 1}/500", flush=True)
    else:      
        results = run_linear_models(df, sesh, neuron_name, roi, permute=perms)
        neuron_results_rows.extend(results)
    return neuron_results_rows
    
    
def compute_state_lin_mod_all(sessions, trials, perms = False, no_perms= None, save_all=False):
    all_result_rows = []
    for sesh in sessions:
        # # I THINK that session 50, 05 elect36 left insular, is an A state-neuron. look at that one first.
        # # '50_05-05-elec36-LINS'
        # sesh = 50
        # trials = 'all_minus_explore'

        # first step: load data of a single neuron.
        data_raw, source_dir = get_data(sesh, trials=trials)
        group_dir_state = f"{source_dir}/group/state_tuning"
        # if this session doesn't exist, skip
        if not data_raw:
            print(f"no raw data found for {sesh}, so skipping")
            continue
    
        # filter data for only those repeats that I define in 'trials'
        data = mc.analyse.helpers_human_cells.filter_data(data_raw, sesh, trials)
        behaviour = data[f"sub-{sesh:02}"]['beh'].copy()
        neurons = data[f"sub-{sesh:02}"]['normalised_neurons'].copy()
        
        neuron_results_rows = Parallel(n_jobs =-1)(delayed(run_single_neuron)(neuron_name = n, sesh = sesh, neuron_data = neurons[n].to_numpy(),beh_df =  behaviour, perms = perms, no_perms=no_perms) for n in neurons)
        for r_n in neuron_results_rows:
            all_result_rows.extend(r_n)
            
        # # import pdb; pdb.set_trace()
        # # 
        # for n in neurons:
        #     neuron_results_rows = run_single_neuron(neuron_name = n, sesh = sesh, neuron_data = neurons[n].to_numpy(),beh_df =  behaviour, perms = perms, no_perms=no_perms)
        #     all_result_rows.extend(neuron_results_rows)
        #     df = make_long_df(neurons[n].to_numpy(), behaviour)
        #     roi = determine_roi(n)
            
        #     # think about how to run the permutations and how to store this!!!
        #     if perms == True:
        #         # no_perms = np.array(range(0,200))
        #         results = Parallel(n_jobs=3)(delayed(run_linear_models)(df, sesh, n, roi, permute=perms) for p in tqdm(range(no_perms)))
        #         for r in results:
        #             all_result_rows.extend(r)
        #         # not sure if this is crrect.
        #     else:
        #         # import pdb; pdb.set_trace()        
        #         results = run_linear_models(df, sesh, n, roi, permute=perms)
        #         all_result_rows.extend(results)

    # 
    results_df = pd.DataFrame(all_result_rows, columns=["session","neuron","roi","model","term","beta","t","p","F","p_F", "permuted"])
    print(results_df.head())
    
    # import pdb; pdb.set_trace()
    if save_all == True:
        group_dir_state = f"{source_dir}/group/state_lin_regs"
        if not os.path.isdir(group_dir_state):
            os.mkdir(group_dir_state)
        if perms == True:
            name_result = f"{group_dir_state}/perm_state_rep_int_{trials}.csv"
        else:
            name_result = f"{group_dir_state}/state_rep_int_{trials}.csv"
        results_df.to_csv(name_result)

    import pdb; pdb.set_trace()
    
    # to combine them
    # perm_path = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/state_lin_regs/perm_state_rep_int_all_correct.csv'
    # perm_df = pd.read_csv(perm_path)
    # combined_df = pd.concat(
    #     [results_df, perm_df],
    #     ignore_index=True
    # )
    # combo_path = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/state_lin_regs/Combo_perm_emp_state_rep_int_all_correct.csv'
    # combined_df.to_csv(combo_path)
    # empirical_with_perm_p = compute_perm_pvalues(combined_df)
    # emp_sig_path = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/state_lin_regs/sign_state_rep_int_all_correct.csv'
    # empirical_with_perm_p.to_csv(emp_sig_path)

    print(f"saved cross-validated state tuning values in {name_result}")  
            
        
if __name__ == "__main__":
    # trials can be 'all', 'all_correct', 'early', 'late', 'all_minus_explore', 'residualised'
    # they can also be: 'first_correct', 'one_correct', ... 'nine_correct'
    compute_state_lin_mod_all(sessions=list(range(1,64)), trials = 'all_correct', perms = True, no_perms = 300, save_all = True)
    # compute_state_lin_mod_all(sessions=list(range(1,5)), trials = 'all_minus_explore', perms = True, save_all = True)
