#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:10:34 2024
any function that helps to deal with MRI files

@author: Svenja Küchenhoff
"""
import nibabel as nib
import numpy as np
import os
import rsatoolbox
import nilearn
from nilearn.image import load_img



def save_my_RSA_results(result_file, centers, file_path, file_name, mask, number_regr, ref_image_for_affine_path):
    x, y, z = mask.shape
    ref_img = load_img(ref_image_for_affine_path)
    affine_matrix = ref_img.affine
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    # results are est.tvalues[1:], est.params[1:], est.pvalues[1:]
    
    t_result_brain = np.zeros([x*y*z])
    t_result_brain[centers] = [vox[0][number_regr] for vox in result_file]
    t_result_brain = t_result_brain.reshape([x,y,z])
    
    t_result_brain_nifti = nib.Nifti1Image(t_result_brain, affine=affine_matrix)
    t_result_brain_file = f"{file_path}/{file_name}_t_val.nii.gz"
    nib.save(t_result_brain_nifti, t_result_brain_file)
    
    b_result_brain = np.zeros([x*y*z])
    b_result_brain[centers] = [vox[1][number_regr] for vox in result_file]
    b_result_brain = b_result_brain.reshape([x,y,z])
    
    b_result_brain_nifti = nib.Nifti1Image(b_result_brain, affine=affine_matrix)
    b_result_brain_file = f"{file_path}/{file_name}_beta.nii.gz"
    nib.save(b_result_brain_nifti, b_result_brain_file)
    
    p_result_brain = np.zeros([x*y*z])
    p_result_brain[centers] = [1 - vox[2][number_regr] for vox in result_file]
    p_result_brain = p_result_brain.reshape([x,y,z])
    
    p_result_brain_nifti = nib.Nifti1Image(p_result_brain, affine=affine_matrix)
    p_result_brain_file = f"{file_path}/{file_name}_p_val.nii.gz"
    nib.save(p_result_brain_nifti, p_result_brain_file)
    
    

def save_my_data_RDM_as_nifti(data_RDM_file, file_path, file_name, ref_image_for_affine_path):
    ref_img = load_img(ref_image_for_affine_path)
    x, y, z = ref_img.shape
    affine_matrix = ref_img.affine
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    import pdb; pdb.set_trace() 
    brain_4d = np.zeros([x,y,z,len(data_RDM_file[0].dissimilarities[0])])
    for i in range(0,len(data_RDM_file[0].dissimilarities[0])):
        curr_slice = np.zeros([x*y*z])
        curr_slice[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox.dissimilarities[0][i] for vox in data_RDM_file]
        brain_4d[:,:,:,i] = curr_slice.reshape([x,y,z])
    
    brain_4d_nifti = nib.Nifti1Image(brain_4d, affine=affine_matrix)
    brain_4d_file = f"{file_path}/{file_name}"
    nib.save(brain_4d_nifti, brain_4d_file)



def save_data_RDM_as_nifti(data_RDM_file, file_path, file_name, ref_image_for_affine_path, centers_for_voxel_index, rdm_toolbox = False):
    # import pdb; pdb.set_trace() 
    ref_img = load_img(ref_image_for_affine_path)
    x, y, z = ref_img.shape
    affine_matrix = ref_img.affine
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    brain_4d = np.zeros([x,y,z,len(data_RDM_file[0])])

    for i in range(0,len(data_RDM_file[0])):
        curr_slice = np.zeros([x*y*z])
        if rdm_toolbox == False:
            curr_slice[list(centers_for_voxel_index)] = [vox[i] for vox in data_RDM_file]
        elif rdm_toolbox == True:
            curr_slice[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox.dissimilarities[0][i] for vox in data_RDM_file]
        brain_4d[:,:,:,i] = curr_slice.reshape([x,y,z])
    
    brain_4d_nifti = nib.Nifti1Image(brain_4d, affine=affine_matrix)
    brain_4d_file = f"{file_path}/{file_name}.nii.gz"
    nib.save(brain_4d_nifti, brain_4d_file)
    
    print('careful! currently only saving as nifti. change this in mc.analyse.handle_MRI_files.py if wanted!')
    # np.save(f"{file_path}/{file_name}", data_RDM_file)
    


def smooth_RDMs(data_RDM_file, ref_img, fwhm, use_rsa_toolbox = True, path_to_save=None, centers = None):
    # note: include centers if not using RSA_toolbox!
    # import pdb; pdb.set_trace() 
    x, y, z = ref_img.shape
    affine_matrix = ref_img.affine
    header = ref_img.header
    
    if use_rsa_toolbox == False:
        voxel_indices = centers
        diss_matrix = data_RDM_file
        
    elif use_rsa_toolbox == True:
        voxel_indices = np.array(data_RDM_file.rdm_descriptors['voxel_index'])
        # Each row corresponds to a voxel and columns to conditions.
        diss_matrix = np.array([vox.dissimilarities[0] for vox in data_RDM_file])
    
    num_conditions = diss_matrix.shape[1]
    
    # Pre-allocate a flat array for the entire brain.
    brain_flat = np.zeros((x * y * z, num_conditions))
    brain_flat[voxel_indices, :] = diss_matrix  # set dissimilarities for the selected voxel indices
    brain_4d = brain_flat.reshape((x, y, z, num_conditions))
    nifti_RDM = nib.Nifti1Image(brain_4d, affine_matrix, header)
    
    # smooth the RDMs
    smoothed_RDM = nilearn.image.smooth_img(nifti_RDM, fwhm=fwhm)
    if path_to_save:
        # save it now that you're here
        nib.save(smoothed_RDM, path_to_save)
        
    # then perpare the RSA object again
    smoothed_RDM_file = data_RDM_file.copy()
    # Get the smoothed data as a numpy array and reshape it to 2D for fast indexing.
    smoothed_RDM_4d = smoothed_RDM.get_fdata() 
    smoothed_flat = smoothed_RDM_4d.reshape(-1, num_conditions)
    # Extract the dissimilarities from the smoothed data using the voxel indices.
    dissimilarities_array = smoothed_flat[voxel_indices, :]
    
            
    if use_rsa_toolbox == True:
        # Update the RSA object with the new dissimilarities.
        smoothed_RDM_file.dissimilarities = dissimilarities_array
    elif use_rsa_toolbox == False:
        smoothed_RDM_file = dissimilarities_array
        if path_to_save:
            path_to_save = os.path.splitext(os.path.splitext(path_to_save)[0])[0]
            np.save(path_to_save, smoothed_RDM_file)

    return smoothed_RDM_file



def apply_mask(niftis, mask):
    masked_niftis = {}
    # import pdb; pdb.set_trace() 
    for nii_data in niftis:
        #masked_niftis[nii_data] = niftis[nii_data] * mask
        masked_niftis[nii_data] = np.where(mask, niftis[nii_data], 0)
    return masked_niftis

def create_common_mask(niftis):
    # first create mask in the shape of the images
    common_mask = np.ones_like(niftis[next(iter(niftis))], dtype=bool)
    # then iteratively check if there are any nans or 0s and if so, add to mask
    for nii_data in niftis:
        common_mask &= np.isfinite(niftis[nii_data])  # Check for NaNs
        common_mask &= (niftis[nii_data] != 0)        # Check for 0s
    return common_mask

def save_niftis(nifti_dict, ref_nifti_path, output_file_name_extension, alternative_paths_to_save = None):
    ref_img = nib.load(ref_nifti_path)
    affine_matrix = ref_img.affine
    x,y,z = ref_img.shape    
    # make sure that dict keys are also new nifti paths!!!
    for nifti_path in nifti_dict:
        split_at = nifti_path.rfind('.nii.gz')
        if split_at != -1:
            out_path_file = nifti_path[:split_at] + '_' + output_file_name_extension + '.nii.gz'
        nifti_out = nib.Nifti1Image(nifti_dict[nifti_path], affine_matrix)
        nib.save(nifti_out, out_path_file)
        
        
def load_niftis(subject_dirs, nifti_filename_list):
    niftis = {}
    #niftis = []
    #file_path_list = []
    for subject in subject_dirs:
        for nifti_filename in nifti_filename_list:
            file_path = subject + nifti_filename
            if os.path.isfile(file_path):
                niftis[file_path] = nib.load(file_path).get_fdata()
                #file_path_list.append(file_path)
                #niftis.append(nib.load(file_path).get_fdata())
            else:
                print(f"Careful! {file_path} didn't exist! skipping this one")
    return niftis
