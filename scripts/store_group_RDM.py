#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:06:56 2025

@author: Svenja Küchenhoff


This script takes as an input the MNI standard space coordinates and the 
result files it is supposed to look at.

It then loops through all subject folders, transforms the MNI coordinates into
subjects space, loads the data RDM, and takes the respective coordinate's RDM array
and concatenates it across subjects.

It then stores a) an average version of that vector and b) the concatenated version.

"""
import resource
import time
import sys
import numpy as np
import os
# from nilearn.image import load_img
import nibabel as nib
import sys
from fsl.data.image import Image
from fsl.transform import flirt
from matplotlib import pyplot as plt
from nilearn import plotting
import mc
import json
import fnmatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl


t0 = time.time()
# first step, take standard voxel coordinates as an input.
std_voxel = [70, 54, 63]

print("looking at data RDM in voxel", std_voxel)

RSA_version = 'DSR_stepwise_combos_14-01-2026'
glm_version = 'all_paths-stickrews-split_buttons'


# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
    RSA_version = 'old'
    glm_version = '03'
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
# second, define which subjects to summarise the data RDMs from,
# and which data RDMs to start with 
# Subjects
# # if any subject numbers are passed on the command line, use them
# if len(sys.argv) > 1:
#     subj_nos = sys.argv[1:]          # ['01', '02', '03', ...]
# else:
#     subj_nos = ['02']                # default

subj_nos = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '22', '23', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '35']


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

def report_usage(tag=""):
    r = resource.getrusage(resource.RUSAGE_SELF)

    # ru_maxrss is:
    #   - kilobytes on Linux
    #   - bytes on macOS
    if sys.platform == "darwin":
        mem_mb = r.ru_maxrss / (1024 * 1024)
    else:
        mem_mb = r.ru_maxrss / 1024

    elapsed = time.time() - t0
    print(
        f"[USAGE] {tag} | "
        f"time={elapsed:6.1f}s, "
        f"maxRSS={mem_mb:6.1f} MB, "
        f"userCPU={r.ru_utime:6.1f}s, "
        f"sysCPU={r.ru_stime:6.1f}s"
    )

def parse_paired_label(lbl: str):
    left, right = lbl.split(" with ")
    l = left.split("_")
    r = right.split("_")
    arrow = {"backw": "←", "forw": "→"}
    block = f"{l[0]}{arrow.get(l[1], '')}|{r[0]}{arrow.get(r[1], '')}"  # compact
    within = f"{l[2]}-{l[3]}".replace("reward", "rew")
    return block, within
    
def std_vox_to_func(std_vox):
    """
    std_vox: (i, j, k) voxel indices in FEAT's standard image (90x108x90)
    Returns:
      func_mm   - (x, y, z) in subject functional space (world/mm)
      func_vox  - (i, j, k) in subject functional voxel space (float)
    """
    std_vox = np.asarray(std_vox, dtype=float)

    # 1) standard voxel -> example_func world (mm)
    func_mm = nib.affines.apply_affine(stdvox_to_funcworld, std_vox)

    # 2) example_func world (mm) -> example_func voxel
    func_nib = func_img.nibImage               # underlying nibabel Nifti1Image
    func_vox = nib.affines.apply_affine(
        np.linalg.inv(func_nib.affine),
        func_mm
    )

    return func_mm, func_vox


# just in case I want to confirm that this indeed works
def plot_std_vox_on_func(std_vox):
    func_mm, func_vox = std_vox_to_func(std_vox)
    func_vox_rounded  = np.round(func_vox).astype(int)

    print("Standard voxel:          ", std_vox)
    print("Subject functional mm:   ", func_mm)
    print("Subject functional voxel:", func_vox, "(rounded:", func_vox_rounded, ")")

    func_nib = func_img.nibImage

    display = plotting.plot_epi(
        func_nib,
        display_mode="ortho",
        cut_coords=func_mm,  # mm coords in example_func space
        title=f"Std vox {std_vox} → subj vox {func_vox_rounded}"
    )
    display.add_markers([func_mm], marker_size=80)
    plotting.show()

    return func_mm, func_vox



subjects = [f"sub-{s}" for s in subj_nos]
print("current list included in data RDM average is", subjects)
print("this will be based on RSA", RSA_version, "and glm", glm_version)

res_dir = f"{source_dir}/data/derivatives/group/RDM_plots"
os.makedirs(res_dir, exist_ok=True)

      
# --- Load configuration ---
# config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_simple.json"
config_file = sys.argv[1] if len(sys.argv) > 2 else "rsa_config_state_Aones_and_combostate-pathandrew.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
EV_string = config.get("load_EVs_from")
regression_version = config.get("regression_version")
RSA_version = config.get("name_of_RSA")
# conditions selection
conditions = config.get("EV_condition_selection")
parts_to_use = conditions["parts"]



data_RDMs_per_sub = []
# third, loop through the subject folders.
for sub in subjects:
    data_dir = f"{source_dir}/data/derivatives/{sub}"
    print("Working on", sub, "in", data_dir)
    report_usage(f"start subject {sub}")
    
    # --- Paths ----
    reg_dir      = f"{data_dir}/func/preproc_clean_01.feat/reg"
    std_img_path = f"{reg_dir}/standard.nii.gz"                      # 90 x 108 x 90
    xfm_path     = f"{reg_dir}/standard2example_func.mat"           # FLIRT matrix

    func_img_path = f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz"  # 108 x 108 x 64
        
    # --- Load images via fslpy ---
    std_img  = Image(std_img_path)   # "source" of the transform (standard)
    func_img = Image(func_img_path)  # "reference" of the transform (example_func)

    # --- Load FLIRT matrix (standard -> example_func) ---
    flirt_mat = flirt.readFlirt(xfm_path)
    # Build an affine that maps:
    #   standard VOXELS  ->  example_func WORLD (mm)
    # i.e. src voxel coords -> ref world coords
    stdvox_to_funcworld = flirt.flirtMatrixToSform(flirt_mat, srcImage=std_img, refImage=func_img)
    # This function wraps the correct FSL-coordinate logic for us. :contentReference[oaicite:3]{index=3}
    
    func_mm, func_vox = std_vox_to_func(std_voxel)
    func_vox_rounded  = np.round(func_vox).astype(int)

    # data RDM will be stored in original format (108, 108, 64 x n-datapoints of the RDM)
    # ADD THE DATE HERE!!
    
    
    
    data_RDM_nifti = nib.load(f"{data_dir}/func/data_RDMs_{RSA_version}_glmbase_{glm_version}/data_RDM.nii.gz")
    # only load the current voxel: current voxel data RDM = 
    ROI_data_RDM = np.asarray(data_RDM_nifti.dataobj[func_vox_rounded[0], func_vox_rounded[1],func_vox_rounded[2], :], dtype=np.float32)
    # data_RDM = data_RDM_nifti.get_fdata()
    # standard space will have dimension 90 x 108 x 90
    # current voxel data RDM = 
    # ROI_data_RDM = data_RDM[func_vox_rounded[0], func_vox_rounded[1],func_vox_rounded[2], :]
    
    # store in a dictionary.
    data_RDMs_per_sub.append(ROI_data_RDM)
    report_usage(f"after subject {sub}")

  

# load the respective config file
data_RDM_avg = np.asarray(np.mean(data_RDMs_per_sub,axis =0))
file_name = f"vox_{std_voxel[0]}_{std_voxel[1]}_{std_voxel[2]}_data_RDM_{RSA_version}_glmbase_{glm_version}"
np.save(f"{res_dir}/{file_name}", data_RDM_avg)

print("now saving the averaged RDM in", f"{res_dir}/{file_name}")

L = data_RDM_avg.size
# recover n from L = n(n+1)/2
n = int((np.sqrt(1 + 8*L) - 1) // 2)
rdm = np.zeros((n, n), dtype=data_RDM_avg.dtype)
iu = np.triu_indices(n)
rdm[iu] = data_RDM_avg         # fill upper triangle
# rdm = rdm + rdm.T - np.diag(np.diag(rdm))   # make symmetric (optional)

plt.figure(); plt.imshow(rdm)
plt.savefig(f"{res_dir}/{file_name}.png")

# use the same plotting function as for the model RDMs.
# for that, I first need to call what determined the input of the data RDMs.
# loading the data EVs into dict
data_EVs, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(data_dir, regression_version=regression_version, only_load_labels = True)
# if you don't want all conditions created through FSL, exclude some here based on config

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

# data_th1, data_th2, paired_labels = pair_correct_tasks(EV_keys, EV_keys)
data_th1, data_th2, paired_labels = pair_correct_tasks(data_EVs, EV_keys)

# determine how long one block is.
first_block_str = paired_labels[0].split('_')[0]+'_'+paired_labels[0].split('_')[1]
for i, l in enumerate(paired_labels):
    if first_block_str in l:
        block_size = i+1


fig, ax = plt.subplots(figsize=(4.2, 4.2))
cmap_obj = plt.get_cmap("RdBu").copy()
cmap_obj.set_bad("white")
norm = (mpl.colors.Normalize(vmin=0, vmax=2))

im = ax.imshow(rdm, cmap=cmap_obj, norm=norm, interpolation="None", aspect="equal")

# block separators
for b in range(block_size, n, block_size):
    ax.axhline(b - 0.5, color="white", lw=1.2)
    ax.axvline(b - 0.5, color="white", lw=1.2)

# hierarchical block labels (one per 8/blocksize)
parsed = [parse_paired_label(l) for l in paired_labels]
block_labels = [parsed[i][0] for i in range(0, n, block_size)]
centers = np.arange(block_size / 2 - 0.5, n, block_size)

ax.set_xticks([])  # usually drop x labels for small figs
ax.set_yticks(centers)
ax.set_yticklabels(block_labels, fontsize=10)

# y labels on the right
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.tick_params(length=0, pad=1)

ax.set_title(file_name, fontsize=12)

within = [parsed[i][1] for i in range(0, block_size)]
ax.set_xlabel("Within-block: " + " | ".join(within), fontsize=9, labelpad=6)

# colorbar on the left (robust way)
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="4%", pad=0.08)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.tick_params(labelsize=10)

# import pdb; pdb.set_trace()

fig.savefig(f"{res_dir}/{file_name}.svg")
np.save(f"{res_dir}/avg_RDM_{file_name}", data_RDM_avg)
report_usage("after avergaging, storing and plotting:")

