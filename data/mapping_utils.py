import glob 
import os
import shutil
import numpy as np
import json
import pydicom
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import re

def check_datset(root):
    """
    Check for missing contours, dicom files, subfolders, etc.

    Args:
        root (str, path): root folder of the dataset, whith Patient (...) subfolders
    """
    required_slices = [
        "T1_map_apex",
        "T1_map_base",
        "T1_map_mid_",
        "T1_Mapping_",
        "T2_map_apex",
        "T2_map_base",
        "T2_map_mid_",
        "T2_Mapping_"]
    allowed_slices = required_slices + [
        "T1_map_apex_uncorrected",
        "T1_map_apex_contrast",
        "T1_map_base_uncorrected",
        "T1_map_base_contrast",
        "T1_map_mid_uncorrected",
        "T1_map_mid_contrast",
        "T2_map_apex_uncorrected",
        "T2_map_base_uncorrected",
        "T2_map_mid_uncorrected"]

    patients = sorted(glob.glob(os.path.join(root,'*')))
    
    for patient_folder in patients:
        print(os.path.basename(patient_folder))

        slice_paths = glob.glob(os.path.join(patient_folder,'*'))
        slice_folders = [os.path.basename(slice_folder) for slice_folder in slice_paths]

        # Check if all required slices/subfolders are found for the patient
        for required_slice in required_slices:
            if required_slice not in slice_folders:
                raise FileNotFoundError(f"Slice folder {required_slice} missing from {os.path.basename(patient_folder)}")
        
        # Check for unexpected slices/folders
        # If a folder is allowed, pop it from the list. 
        # If folders remain in the list after the loop, they are not allowed.     
        for allowed_slice in allowed_slices:
            if allowed_slice in slice_folders:
                slice_folders.remove(allowed_slice)
                continue
        if slice_folders != []:
            print(f"  Unexpected subfolders found in {os.path.basename(patient_folder)}: {slice_folders}")
        
        for slice_path in slice_paths:
            # Check for empty slice folders and skip checking their content
            if os.listdir(slice_path) == []:
                continue
            # Skip _uncorrected folders from checking their contents
            if "_uncorrected" in os.path.basename(slice_path):
                continue
            # Skip _contrast folders from checking their contents, because these don't have any contours
            if "_contrast" in os.path.basename(slice_path):
                continue
            # Check if contours are found in ..._map_... folders
            if "_map_" in os.path.basename(slice_path):
                expected_contours_path = os.path.join(slice_path, "Contours.json")
                if not os.path.exists(expected_contours_path):
                    print(f"  Contours file missing: {expected_contours_path.replace(root+'/','')}")
            # Check for missing mapping files
            elif  "Mapping" in os.path.basename(slice_path):
                for mapping_dcm_path in glob.glob(os.path.join(slice_path, '*.dcm')):
                    # Mapping dcms should be named Apex.dcm, Base.dcm, Mid.dcm, Apex_2.dcm, Base_2.dcm, Mid_2.dcm
                    # Contours for _2 files are not loaded, but their presence is not reported as an error!
                    mapping_dcm_type = os.path.basename(mapping_dcm_path).replace("_2", "")
                    if mapping_dcm_type not in ["Apex.dcm", "Base.dcm", "Mid.dcm"]:
                        print(f"  Unexpected Mapping file: {mapping_dcm_path.replace(root+'/','')}")

                

def construct_samples_list(root, contours_filename="Contours.json"):
    """
    Search for image files and matching segmentation contour files.
    Construct a list of [[image path, segmentation contour file path], ...] pairs for each image. 
    The same contour file is used for multiple images.
    Only inlcudes images with valid contour files. Don't use this if you only need images. 

    Dicom path                            Contour path
    ------------------------------------  -------------------------------------
    Patient (1)/T1_map_mid_/...23102.dcm  Patient (1)/T1_map_mid_/Contours.json
    Patient (1)/T1_map_mid_/...23097.dcm  Patient (1)/T1_map_mid_/Contours.json
    Patient (1)/T1_map_mid_/...23100.dcm  Patient (1)/T1_map_mid_/Contours.json
    Patient (1)/T1_map_mid_/...23099.dcm  Patient (1)/T1_map_mid_/Contours.json
    Patient (1)/T1_map_mid_/...23101.dcm  Patient (1)/T1_map_mid_/Contours.json

    Args:
        root (str): dataset root folder
        contours_filename (str) : Filename for contour files, intended for use in the inter-observer analysis

    Returns:
        list of tuples (str, str): List of tuples, each with the following elements: dicon_path, contour_path
    """
    samples = []
    # No warnings for missing contours in this function. Simply skip incorrect samples. Run check_dataset to check for missing data.
    for patient_folder in sorted(glob.glob(os.path.join(root, '*', 'Patient (*'))):
        for slice_path in glob.glob(os.path.join(patient_folder,'*')):
            if "_uncorrected" in os.path.basename(slice_path):
                continue
            elif "_map_" in os.path.basename(slice_path):
                expected_contours_path = os.path.join(slice_path, contours_filename)
                if os.path.exists(expected_contours_path):
                    dcm_paths =  glob.glob(os.path.join(slice_path,'*.dcm'))
                    contour_paths = [expected_contours_path]*len(dcm_paths)
                    samples.extend(zip(dcm_paths, contour_paths))
            elif  "Mapping" in os.path.basename(slice_path):
                if "T1_Mapping" in os.path.basename(slice_path):
                    contours_folders = {"Apex.dcm": "T1_map_apex", 
                                        "Base.dcm": "T1_map_base", 
                                        "Mid.dcm": "T1_map_mid_"}
                elif "T2_Mapping" in os.path.basename(slice_path):
                    contours_folders = {"Apex.dcm": "T2_map_apex", 
                                        "Base.dcm": "T2_map_base", 
                                        "Mid.dcm": "T2_map_mid_"} 
                for mapping_dcm_path in glob.glob(os.path.join(slice_path, '*.dcm')):
                    # Mapping dcms should be named Apex.dcm, Base.dcm, Mid.dcm, Apex_2.dcm, Base_2.dcm, Mid_2.dcm
                    # Contours for _2 files are not loaded!
                    mapping_dcm_type = os.path.basename(mapping_dcm_path)
                    if mapping_dcm_type not in ["Apex.dcm", "Base.dcm", "Mid.dcm"]:
                        continue
                    conotours_folder = contours_folders[mapping_dcm_type]
                    expected_contours_path = os.path.join(patient_folder, conotours_folder, contours_filename)
                    if os.path.exists(expected_contours_path): 
                        samples.append((mapping_dcm_path, expected_contours_path))

    return samples


def consturct_unlabelled_samples_list(root):
    """
    Search for mapping dicom image files including unlabelled, contrast and uncorrected samples.
    
    Args:
        root (str): dataset root folder

    Returns:
        list of tuples (str, str): List containing paths for all images.
    """
    samples = sorted(glob.glob(os.path.join(root, '**', '*.dcm'), recursive=True))
    samples = [sample for sample in samples if "Localizer" not in sample] # @Máté: Minden betegnél van 3 Localizer nevű könyvtár, azok számotokra irreleváns adatok
    return samples


def split_samples_list(samples, split_patient_ids):
    """
    Select a subset of the samples list based on a list of patient ids. 

    Args:
        samples (list of [image path, segmentation contour file path] tuples): as returned by construct_samples_list
        split_patient_ids (list of ints): list of patient ids

    Returns:
        (list of [image path, segmentation contour file path] tuples): samples of patients whose id is in the split_patient_ids list
    """
    split_samples = []
    for sample in samples:
        if isinstance(sample, tuple) or isinstance(sample, list):
            path_parts = sample[0].split(os.path.sep)
        else:
            path_parts = sample.split(os.path.sep) # for the unlabeled sample list, wich is a list containing paths only
        # e.g. path_parts: Patient (1)/T1_map_base/Contours.json
        patient_folder = path_parts[-3]   
        # e.g. patient_folder: Patient (1)
        patient_id = int(patient_folder[patient_folder.index('(') + 1:patient_folder.index(')')])
        if patient_id in split_patient_ids:
            split_samples.append(sample)
    return split_samples


def print_diagnostics(root, samples, list_shapes=False, get_mean_std=False):
    contours_list = np.unique([sample[1] for sample in samples])
    print(f"Found {len(samples)} dcm files with contours and {len(contours_list)} contours files.")
    all_dcms =  glob.glob(os.path.join(root, "**/*.dcm"), recursive=True)
    uncorrected_dicoms = [dcm_path for dcm_path in all_dcms if "_uncorrected" in dcm_path]
    print(f"Found {len(uncorrected_dicoms)} uncorrected dcm files.")
    contrast_dicoms = [dcm_path for dcm_path in all_dcms if "_contrast" in dcm_path or "_2.dcm" in dcm_path]
    print(f"Found {len(contrast_dicoms)} contrast dcm files.")
    dcms_with_missing_contours = len(all_dcms) - len(samples) - len(uncorrected_dicoms)  - len(contrast_dicoms)
    print(f"Couldn't identify contours file for  {dcms_with_missing_contours} dcm files (not including uncorrected or contrast files).")
    
    if list_shapes or get_mean_std:
        shapes = []
        all_pixels = []
        # for dicom_path in tqdm(sorted(glob.glob(os.path.join(root,'**', '*.dcm'), recursive=True)), desc="Loading all dicoms for dataset stats"):
        for dicom_path in tqdm(sorted([sample[0] for sample in samples]), desc="Loading all dicoms for dataset stats"):
            image = pydicom.dcmread(dicom_path).pixel_array
            if image.shape not in shapes:
                shapes.append(image.shape)
            if get_mean_std:
                all_pixels.append(image.flatten())
        if list_shapes:
            print("Found dicom images with shapes:\n  ", sorted(shapes))
        if get_mean_std:
            all_pixels = np.concatenate(all_pixels)
            print(f"Pixel mean for all images {all_pixels.mean()}")
            print(f"Pixel std for all images {all_pixels.std()}")


def rename_Mapping_2_mapping(root: str):
    """ DON'T USE THIS
    First 68 patients has Mapping folder with capital M this function renames these folder to use lower case m

    >>> from data.mapping_utils import rename_Mapping_2_mapping
    >>> rename_Mapping_2_mapping("/home1/ssl-phd/data/mapping")
    
    Args:
        root (str): dataset root folder with Patient (xx) folders in it.
    """
    for patient_folder in sorted(glob.glob(os.path.join(root,'*'))):
        for slice_path in glob.glob(os.path.join(patient_folder,'*')):
            if "T1_Mapping_"  == os.path.basename(slice_path):
                new_name = os.path.join(os.path.dirname(slice_path), "T1_mapping_")
                print(f"{slice_path} -> {new_name}")
                shutil.move(slice_path, new_name)
            elif "T2_Mapping_" == os.path.basename(slice_path):
                new_name = os.path.join(os.path.dirname(slice_path), "T2_mapping_")
                print(f"{slice_path} -> {new_name}")
                shutil.move(slice_path, new_name)


def rename_mapping_2_Mapping(root: str, dry_run=True):
    """Initially we received some patients, who has Mapping folder with capital M, some with lower case m.
    This function renames folders consistently to capital M Mapping_

    >>> from data.mapping_utils import rename_mapping_2_Mapping
    >>> rename_mapping_2_Mapping("/mnt/hdd2/se", dry_run=False)
    
    Args:
        root (str): dataset root folder with Patient (xx) folders in it or in it's subfolders (1 level deep).
        dry_run (bool): If True only prints what 
    """
    for patient_folder in sorted(glob.glob(os.path.join(root,'**/Patient (*)'), recursive=True)):
        for slice_path in glob.glob(os.path.join(patient_folder,'*')):
            if "T1_mapping_"  == os.path.basename(slice_path):
                new_name = os.path.join(os.path.dirname(slice_path), "T1_Mapping_")
                print(f"{slice_path} -> {new_name}")
                if not dry_run:
                    shutil.move(slice_path, new_name)
            elif "T2_mapping_" == os.path.basename(slice_path):
                new_name = os.path.join(os.path.dirname(slice_path), "T2_Mapping_")
                print(f"{slice_path} -> {new_name}")
                if not dry_run:
                    shutil.move(slice_path, new_name)
    if dry_run: 
        print("Dry run completed. Listed all folders to be renamed, but didn't perform renaming.")


def get_dataset_mean_std(root: str):
    """
    Args:
        root (str): dataset root folder

    Returns:
        (float, float): mean, std
    """
    all_pixels = []
    for dcm_path in glob.glob(os.path.join(root, "**/*.dcm"), recursive=True):
        image = pydicom.dcmread(dcm_path).pixel_array
        all_pixels.append(image.flatten())
    all_pixels = np.concatenate(all_pixels)
    return all_pixels.mean(), all_pixels.std()


def load_dicom(path, mode='channels_first'):
    """
    Loads dicom files to a numpy array. 
    Args:
        path (str): 
        mode (str, optional): 'channels_first', 'channels_last', '2d' or None. Defaults to 'channels_first'.

    Returns:
        np.ndarray: image data from the dicom file
    """
    try:
        dcm = pydicom.dcmread(path)
    except pydicom.errors.InvalidDicomError as e:
        print(e)
        print("Incorrect dicom:", path)
        dcm = pydicom.dcmread(path, force=True)
        
    try:
        image = dcm.pixel_array
    except AttributeError as e:
        print(e)
        print("Incorrect dicom:", path)

    if mode == 'channels_first':
        image = np.expand_dims(image, 0)
    elif mode == 'channels_last': 
        image = np.expand_dims(image, -1)
    else:
        assert mode == '2d' or mode == None, f"Unrecognized loading mode: {mode}!" \
            "Allowed values \'channels_first\', \'channels_last\', \'2d\' or None"
  
    return image


def load_contours(labelfile):
    epicardial_contour = None
    endocardial_contour = None
    try:
        with open(labelfile) as f:
            contour_data = json.load(f)
            epicardial_contour = np.stack([contour_data['epicardial_contours_x'],
                                            contour_data['epicardial_contours_y']], 
                                        axis=1)
            endocardial_contour = np.stack([contour_data['endocardial_contours_x'],
                                            contour_data['endocardial_contours_y']], 
                                            axis=1)
    except ValueError as e:
        print(f"ERROR loading {labelfile}:\n{e}")
    # print("epicardial_contour.shape = ", epicardial_contour.shape)
    # print("endocardial_contour.shape = ", endocardial_contour.shape)
    return epicardial_contour, endocardial_contour


def contours_to_masks(contours, shape, contour_fp_precision_bits = 10):
    """Converts segmentation contours (epicardial_contour, endocardial_contour) to segmentation mask.
       Region between the two contours is foreground, rest is background. 

    Args:
        contours ([np.ndarray, np.ndarray]): Coordinates of epicardial_contour, endocardial_contour points as ndarrays.
        shape (tuple): Shape of the output mask, shape of the input image
        round_bits (int, optional): OpenCV's fillPoly accepts contours with fix point representation.
                                    contour_fp_precision_bits determines the number of fractional bits.
                                    Contours are rounded to this precision. Defaults to 10.

    Returns:
        np.ndarray: Segmentation mask
    """
    # OpenCV's fillPoly accepts contours with fix point representation, passed as int32,
    # whose last 'shift' bits are interpreted as fractional bits.
    # Here we multiply by 2^contour_fp_precision_bits to achieve this representation.
    rounded_conoturs = [np.around(contour*2**contour_fp_precision_bits).astype(np.int32) for contour in contours]

    # Rounded contours might have repeated points which could break filling betweeng the two contours.
    rounded_conoturs = [unique_consecutive(contour) for contour in rounded_conoturs]
    
    if len(shape) == 2:
        mask = np.zeros(shape)
    else:
        mask = np.zeros(shape[-2:])
    
    # Three class version
    cv2.fillPoly(mask, pts=[rounded_conoturs[0]], color=(1,1,1), shift=contour_fp_precision_bits)
    cv2.fillPoly(mask, pts=[rounded_conoturs[1]], color=(2,2,2), shift=contour_fp_precision_bits)

    # Foreground between contours only
    # cv2.fillPoly(mask, pts=[np.concatenate(rounded_conoturs)], color=(1,1,1), shift=contour_fp_precision_bits)
    return mask


def unique_consecutive(contour):
    """Removes repeated consecutive poitns from a contour array, leaving only one occurance of each point."""
    duplicate_pts = []
    for idx in range(contour.shape[0]-1):
        if np.array_equal(contour[idx], contour[idx+1]):
            duplicate_pts.append(idx)
    return np.delete(contour, duplicate_pts, axis=0)


def plot_contours(contours, ax, formats=('C0', 'C1'), labels=("Epicardial contour", "Endocardial contour"), **kwargs):
    epicardial_contour, endocardial_contour = contours
    if epicardial_contour is not None:
        ax.plot(epicardial_contour[:,0], epicardial_contour[:,1], formats[0], label=labels[0], **kwargs)
    if endocardial_contour is not None:
        ax.plot(endocardial_contour[:,0], endocardial_contour[:,1], formats[1], label=labels[1], **kwargs)
    ax.legend()

def plot_between_contours(contours, ax, format=('C0'), label=("Prediction"), **kwargs):
    epicardial_contour, endocardial_contour = contours
    if epicardial_contour is not None and endocardial_contour is not None:
        u = np.vstack([epicardial_contour, epicardial_contour[0], endocardial_contour[0], endocardial_contour[::-1]])
        ax.fill(u[:,0], u[:,1], format, label=label, **kwargs)
    ax.legend()


def plot_dcm(dcm_path, ax):
    ds = pydicom.dcmread(dcm_path)
    ax.imshow(ds.pixel_array, cmap=plt.cm.bone)
    ax.set_xticks([])
    ax.set_yticks([])



def check_contours(contours):
    if len(contours) !=2:
        print("Warning! Expecting contours as (epicardial_contour, endocardial_contour)",
              f"len(contours) = {len(contours)}")


def get_pixel_size(dcm: pydicom.dataset.FileDataset):
    """ Compute the size of a pixel/voxel in 
    Relevant fields in the dicom header
        (0051, 100c) [Unknown]                           LO: 'FoV 315*360'
        (0051, 100b) [AcquisitionMatrixText]             LO: '148p*256'
        (0018, 0050) Slice Thickness                     DS: '8.0'

    Args:
        dcm (pydicom.dataset.FileDataset): dicom file loaed with pydicom.dcmread

    Returns:
        float: size of a pixel/voxel in mm
    """
    # \D Matches any character which is not a decimal digit. -> removes non digit characters
    FoV = float(re.sub("\D", "", str.split(dcm[0x0051,0x100C].value, '*')[-1]))
    AcquisitionMatrixSize = float(re.sub("\D", "", str.split(dcm[0x0051,0x100B].value, '*')[-1]))
    voxelxy = FoV / AcquisitionMatrixSize
    return voxelxy


def hausdorff_dist(contours_pred, contours_gt, missing_value=np.NaN):
    """ Compute Hausdorf distance between ground truth and predicted contours.
        If a predicted contour is None, missing_value (np.NaN by default) is returned instead of a float.

    Args:
        contours_pred (np.ndarray, np.ndarray): Predicted contours as tuple or list of two arrays: (epicardial_contour, endocardial_contour)
        contours_gt (np.ndarray, np.ndarray): Ground truth contours as tuple or list of two arrays: (epicardial_contour, endocardial_contour)
    Returns:
        np.ndarray: Epicardial Hausdorf distance, Endocardial Hausdorf distance
    """

    check_contours(contours_pred)
    check_contours(contours_gt)

    results = []
    for idx in range(len(contours_gt)):
        if contours_pred[idx] is not None:
            dir_hausdorf_gt_pred, _, _ = scipy.spatial.distance.directed_hausdorff(contours_gt[idx],   contours_pred[idx])
            dir_hausdorf_pred_gt, _, _ = scipy.spatial.distance.directed_hausdorff(contours_pred[idx], contours_gt[idx])
            hausdorf_dist = max(dir_hausdorf_gt_pred, dir_hausdorf_pred_gt)
            results.append(hausdorf_dist)
        else:
            print("Missing predicted contour.")
            results.append(missing_value)
    
    return np.array(results)


def mean_surface_distance(contours_pred, contours_gt, missing_value=np.NaN):
    """ Compute Mean (surface) distance between ground truth and predicted contours

    Args:
        contours_pred (np.ndarray, np.ndarray): Predicted contours as tuple or list of two arrays: (epicardial_contour, endocardial_contour)
        contours_gt (np.ndarray, np.ndarray): Ground truth contours as tuple or list of two arrays: (epicardial_contour, endocardial_contour)

    Returns:
        np.ndarray: Epicardial Mean Surface distance, Endocardial Mean Surface distance
    """
    check_contours(contours_pred)
    check_contours(contours_gt)

    results = []
    for contour_idx in range(len(contours_gt)):
        if contours_pred[contour_idx] is not None:
            pairwise_dists = scipy.spatial.distance.cdist(contours_gt[contour_idx], contours_pred[contour_idx])
            h_mean_AB = np.mean(np.min(pairwise_dists, axis=0))
            h_mean_BA = np.mean(np.min(pairwise_dists, axis=1))
            MeanSurfaceDist = np.mean([h_mean_AB, h_mean_BA])
            results.append(MeanSurfaceDist)
        else:
            print("Missing predicted contour.")
            results.append(missing_value)

    return np.array(results)


def signed_mean_surface_distance(contours_pred, contours_gt, missing_value=np.NaN):
    """ Compute Mean (surface) distance between ground truth and predicted contours.
    Positive if the predicted contour (1st arg) is rather outside the ground truth contour than inside it. 

    Args:
        contours_pred (np.ndarray, np.ndarray): Predicted contours as tuple or list of two arrays: (epicardial_contour, endocardial_contour)
        contours_gt (np.ndarray, np.ndarray): Ground truth contours as tuple or list of two arrays: (epicardial_contour, endocardial_contour)

    Returns:
        np.ndarray: Epicardial Mean Surface distance, Endocardial Mean Surface distance
    """
    check_contours(contours_pred)
    check_contours(contours_gt)

    results = []
    for contour_idx in range(len(contours_gt)):
        if contours_pred[contour_idx] is not None:
            center = np.mean(contours_gt[contour_idx], axis=0, keepdims=True)
            pairwise_dists = scipy.spatial.distance.cdist(contours_gt[contour_idx], contours_pred[contour_idx])
            center_gt_dists = scipy.spatial.distance.cdist(contours_gt[contour_idx], center)
            center_pred_dists = scipy.spatial.distance.cdist(contours_pred[contour_idx], center)
            # pairwise_dists.shape = (len(gt) ,len(pred))
            # center_gt_dists.shape = (len(gt), 1)
            # center_pred_dists.shape = (len(pred), 1)

            min_dists_AB = np.min(pairwise_dists, axis=0) # shape: len(pred)
            closest_point_in_gt = np.argmin(pairwise_dists, axis=0) # shape: len(pred)
            signs_AB = np.sign(center_pred_dists - center_gt_dists[closest_point_in_gt])
            h_mean_AB = np.mean(signs_AB * min_dists_AB)

            min_dists_BA = np.min(pairwise_dists, axis=1) # shape: len(gt)
            closest_point_in_pred = np.argmin(pairwise_dists, axis=1) # shape: len(gt)
            signs_BA = np.sign(center_pred_dists[closest_point_in_pred] - center_gt_dists)
            h_mean_BA = np.mean(signs_BA * min_dists_BA)

            MeanSurfaceDist = np.mean([h_mean_AB, h_mean_BA])
            results.append(MeanSurfaceDist)
        else:
            print("Missing predicted contour.")
            results.append(missing_value)

    return np.array(results)


def fit_contours(mask:np.ndarray, num_classes=2, interp_factor=4):
    """Fit contour for segmentation masks. If multiple patches are found for a class only the largest one is returned.
    If interp_factor > 1 B-spline interpolation of contours is computed and returned.

    Args:
        mask (np.ndarray): Segmentation mask with integer labels
        num_classes (int, optional): Number of classes on the mask (assuming multi-class segmentation). Defaults to 2.
        interp_factor (int, optional): Multiplier of number of point for interporating contours. 1 means no interpolation. 

    Returns:
        [np.ndarray, np.ndarray]: List of contours for each class of the segmentation mask. E.g.: epicardial_contour, endocardial_contour
    """
    fitted_contours = []
    for i in range(1, num_classes + 1):
        class_mask = (mask == i).astype(np.uint8)
        fitted_contour, hierarchy = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        # Only keep the largest contour if multiple found
        if len(fitted_contour) > 1:
            # Only keep contours which have nice, square-ish bounding box
            squareish_contours = []
            for fc in fitted_contour:
                x,y,w,h = cv2.boundingRect(fc)
                if w/h < 3 and h/w < 3:
                    squareish_contours.append(fc)
            if len(squareish_contours) > 0:
                # If all contours are elongated, fall back to area based selection. (probably this contour is bad anyway...)
                fitted_contour = squareish_contours

            # Only keep the largest contour
            patch_sizes = [cv2.contourArea(fc) for fc in fitted_contour]
            fitted_contour = fitted_contour[np.argmax(patch_sizes)]
        elif len(fitted_contour) == 1:
            fitted_contour = fitted_contour[0]
        else:
            fitted_contours.append(None)
            continue

        # Compute convex hull and replace highly concave contours with the convex hull
        # This is to handle non-continuous, C shaped epicardial contours
        # For 'Patient (190)/T2_map_apex/...2169.dcm' the interpolation below failed, because it tried to interpolate a C shaped contour...
        convex_hull = cv2.convexHull(fitted_contour)
        if cv2.contourArea(convex_hull) / 2 > cv2.contourArea(fitted_contour):
            # print("Warning! Concave contour replaced by its convex hull!")
            # print(np.vstack(fitted_contour))
            fitted_contour = convex_hull

        # Ensure counter-clockwise orientation
        if cv2.contourArea(fitted_contour, oriented=True) < 0:
            fitted_contour = fitted_contour[::-1]
            
        fitted_contour = fitted_contour.squeeze(axis=1)
        num_pts = fitted_contour.shape[0]
        # splprep can't fit a curve to <= 3 points.
        if interp_factor > 1 and num_pts > 3:
            # Close the contour by repeating the first point (if not done here the interpolated curve won't go through one of the endpoints, presumably the last)
            fitted_contour = np.vstack([fitted_contour, fitted_contour[0]])
            # Based on: https://agniva.me/scipy/2016/10/25/contour-smoothing.html
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = scipy.interpolate.splprep(fitted_contour.T.tolist(), u=None, s=1.0, per=1)
            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), num_pts*interp_factor)
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = scipy.interpolate.splev(u_new, tck, der=0)
            fitted_contour = np.stack([x_new, y_new], axis=1)

        fitted_contours.append(fitted_contour)
    return fitted_contours
