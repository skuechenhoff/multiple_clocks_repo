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

import numpy as np
import os
from nilearn.image import load_img
import sys
import nibabel as nib
from fsl.data.image import Image
from fsl.transform import flirt
from matplotlib import pyplot as plt

# first step, take standard voxel coordinates as an input.
std_voxel = [45, 22, 42]

# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    print("Running on laptop.")
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
# second, define which subjects to summarise the data RDMs from,
# and which data RDMs to start with 
# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]

RSA_version = 'state_Aones_and_combo_10-12-2025'
glm_version = 'all_paths-stickrews-split_buttons'


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
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        # DONT FORGET TO COMMENT THIS OUT!!!!
        RSA_version = 'old'
        glm_version = '03'
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")


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
    data_RDM_nifti = load_img(f"{data_dir}/func/data_RDMs_{RSA_version}_glmbase_{glm_version}/data_RDM.nii.gz")
    data_RDM = data_RDM_nifti.get_fdata()
    # standard space will have dimension 90 x 108 x 90
    
    # current voxel data RDM = 
    ROI_data_RDM = data_RDM[func_vox_rounded[0], func_vox_rounded[1],func_vox_rounded[2], :]
    
    # store in a dictionary.
    data_RDMs_per_sub.append(ROI_data_RDM)
    
    
    
data_RDM_avg = np.asarray(np.mean(data_RDMs_per_sub))

L = data_RDM_avg.size

# recover n from L = n(n+1)/2
n = int((np.sqrt(1 + 8*L) - 1) // 2)

rdm = np.zeros((n, n), dtype=data_RDM_avg.dtype)
iu = np.triu_indices(n)
rdm[iu] = data_RDM_avg         # fill upper triangle
# rdm = rdm + rdm.T - np.diag(np.diag(rdm))   # make symmetric (optional)


plt.figure(); plt.imshow(rdm)



# finally, put it back into the dimensions.


    
#     # first: transform the MNI coordinates from standard space to subject space.
    
#     subj_example_func = image.load_img('/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-02/func/preproc_clean_01.feat/example_func.nii.gz')
    
#     # second: take the RDM that corresponds to these coordinates and transform it into standard space
#     # third: store as 3dim nifti in a group folder
#     # forth: concatenate and average.
    
#     import pdb; pdb.set_trace() 

#     func2std_mat = np.loadtxt(f"{data_dir}/func/preproc_clean_01.feat/reg/example_func2standard.mat")
#     std2func_mat = np.loadtxt(f"{data_dir}/func/preproc_clean_01.feat/reg/standard2example_func.mat")
#     std_img = nib.load(f"{data_dir}/func/preproc_clean_01.feat/reg/standard.nii.gz" )  # 90 x 108 x 90
#     func_img = nib.load(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz") # 108 x 108 x 64
    
    
#     # 1) Standard voxel -> standard mm
#     std_mm = nib.affines.apply_affine(std_img.affine, std_voxel)
    
#     # 2) Standard mm -> subject mm (using FSL transform)
#     func_mm = nib.affines.apply_affine(std2func_mat, std_mm)
    
#     # 3) Subject mm -> subject voxel
#     func_vox = nib.affines.apply_affine(np.linalg.inv(func_img.affine), func_mm)

    
    
#     x_subj_mm, y_subj_mm, z_subj_mm = image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], stan2func_mat)
    
    
#     affine = np.loadtxt(transform_mat)
#     subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
    
#     data_RDM_file_name = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-12/func/data_RDMs_state-only_masked_same_locinstate_27-11-2025_glmbase_03-4/data_RDM.nii.gz"
#     data_RDM = load_img(data_RDM_file_name)
#     RDM_affine = data_RDM.affine
#     subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
#     subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], np.linalg.inv(data_RDM.affine))

    
    
#     # mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [54, 63, 41], ref_img, condition_names)
#     # mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDMs, [54, 63, 41], ref_img, condition_names, centers = centers, no_rsa_toolbox=True)





# subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
# subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], data_RDM.affine)

# subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], np.linalg.inv(affine))
# subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], data_RDM.affine)

# subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
# subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], np.linalg.inv(data_RDM.affine))

# subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], np.linalg.inv(affine))
# subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], np.linalg.inv(data_RDM.affine))
    
    
    
    
    
    
    
    
    
