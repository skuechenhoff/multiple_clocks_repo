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
import matplotlib.pyplot as plt
import mc
import pickle
import sys
from datetime import date
import json
import nibabel as nib
import scipy
import nilearn
import nilearn.image


# MNI coordinates: 
MNI_coords = [10, -80, 13]

# import pdb; pdb.set_trace() 
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    print("Running on laptop.")
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    print(f"Running on Cluster, setting {source_dir} as data directory")
       
    
# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '02'  
subjects = [f"sub-{subj_no}"]

for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        # DONT FORGET TO COMMENT THIS OUT!!!!
        regression_version = '03-4'
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")
    # data_rdm_dir = f"{data_dir}/func/data_RDMs_{RDM_version}_glmbase_{regression_version}"
    transform_mat = f"{data_dir}/func/preproc_clean_01.feat/reg/standard2example_func.mat"
    
    affine = np.loadtxt(transform_mat)
    subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
    
    data_RDM_file_name = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-12/func/data_RDMs_state-only_masked_same_locinstate_27-11-2025_glmbase_03-4/data_RDM.nii.gz"
    data_RDM = load_img(data_RDM_file_name)
    RDM_affine = data_RDM.affine
    subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
    subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], np.linalg.inv(data_RDM.affine))
    
    
    import pdb; pdb.set_trace() 
    
     mc.plotting.deep_data_plt.plot_data_RDMconds_per_searchlight(data_RDM_file_2d, centers, neighbors, [54, 63, 41], ref_img, condition_names)
     mc.plotting.deep_data_plt.plot_dataRDM_by_voxel_coords(data_RDMs, [54, 63, 41], ref_img, condition_names, centers = centers, no_rsa_toolbox=True)





subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], data_RDM.affine)

subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], np.linalg.inv(affine))
subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], data_RDM.affine)

subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], affine)
subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], np.linalg.inv(data_RDM.affine))

subj_coords = nilearn.image.coord_transform(MNI_coords[0], MNI_coords[1], MNI_coords[2], np.linalg.inv(affine))
subj_indices = nilearn.image.coord_transform(subj_coords[0], subj_coords[1], subj_coords[2], np.linalg.inv(data_RDM.affine))
    
    
    
    
    
    
    
    
    
