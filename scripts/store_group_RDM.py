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

t0 = time.time()

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


import numpy as np
import os
# from nilearn.image import load_img
import nibabel as nib
import sys
from fsl.data.image import Image
from fsl.transform import flirt
from matplotlib import pyplot as plt
from nilearn import plotting

# first step, take standard voxel coordinates as an input.
std_voxel = [45, 22, 42]
print("looking at data RDM in voxel", std_voxel)

RSA_version = 'state_and_combo_11-12-2025'
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
# if any subject numbers are passed on the command line, use them
if len(sys.argv) > 1:
    subj_nos = sys.argv[1:]          # ['01', '02', '03', ...]
else:
    subj_nos = ['02']                # default

subjects = [f"sub-{s}" for s in subj_nos]
print("current list included in data RDM average is", subjects)
print("this will be based on RSA", RSA_version, "and glm", glm_version)

res_dir = f"{source_dir}/data/derivatives/group/RDM_plots"
os.makedirs(res_dir, exist_ok=True)

      
# # --- Load configuration ---
# # config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_simple.json"
# config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_state_Aones_and_combostate-pathandrew.json"
# with open(f"{config_path}/{config_file}", "r") as f:
#     config = json.load(f)

# # SETTINGS
# EV_string = config.get("load_EVs_from")
# regression_version = config.get("regression_version")
# name_RSA = config.get("name_of_RSA")


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

report_usage("after avergaging, storing and plotting:")

