import os
import numpy as np
import cv2
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix
from tqdm import tqdm
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
import albumentations as A
import pydicom
import torchmetrics
import pandas as pd

import data.mapping_utils as mapping_utils
from data.mapping_dataset import MappingDatasetAlbu, DATASET_MEAN, DATASET_STD
from supervised_segmentation.train import CardiacSegmentation
import utils
import time


PRED_FOLDER = "inference"

def generate_prediction_pdfs(checkpoint_path: str,
                             dataset_root = "/home1/ssl-phd/data/mapping",
                             num_samples = 10000,
                             output_folder=".",
                             merge_prediction_pdfs=True,
                             remove_individual_pdfs=False,
                             pad_to = 224,
                             spatial_unit = 'mm',
                             dataset_split = 'test',
                             observer_id = 1,
                             filename_suffix = ''):
    if observer_id != 1:
        assert dataset_split == 'interobserver'

    model = CardiacSegmentation.load_from_checkpoint(checkpoint_path)
    model.eval()

    norm_max_pixel_value = model.config["norm_max_pixel_value"]

    test_augs = []
    if pad_to is not None:
        test_augs.append(A.PadIfNeeded(pad_to,pad_to, border_mode=cv2.BORDER_CONSTANT, value=0, position='top_left'))
    test_augs.append(A.Normalize(mean=(DATASET_MEAN,), std=(DATASET_STD,), max_pixel_value=norm_max_pixel_value),)
    test_aug = A.Compose(test_augs)

    ds = MappingDatasetAlbu(dataset_root,
                            transforms=test_aug, 
                            split=dataset_split, 
                            check_dataset=False, 
                            observer_id=observer_id,
                            mapping_only=model.config.get("test_mapping_only", False))
    # Limit the number of samples considered in the test dataset. IF num_samples < len(ds). Otherwise use the full test set
    indeces = range(0, min(num_samples, len(ds))) 
    ds = Subset(ds, indeces)

    ious = []
    dice_scores = []
    per_class_ious = []
    img_paths = []
    confusion_matrix = np.zeros((model.num_classes, model.num_classes))
    hausdorff_distances = []
    mean_surface_distances = []
    signed_msds = []
    pdf_paths = []
    inference_times = []
    for idx, data in tqdm(enumerate(ds), total=len(ds), desc="Computing predictions"):
        t_inference_start = time.time()
        img, targets = data
        img_path, target_path = ds.dataset.samples[idx]
        img_paths.append(img_path)
        targets = torch.from_numpy(targets.astype(np.int64))

        input_ = utils.pad_to_next_multiple_of_32(img)
        
        input_ = torch.from_numpy(input_)
        input_ = input_.unsqueeze(dim=0)  # add minibatch dimension
        logits = model(input_)
        logits = logits[0, :, :img.shape[-2], :img.shape[-1]] # Remove padding and the minibatch dim
        logits = logits.detach().cpu()
        preds = torch.argmax(logits,dim=0)
        
        # Compute IoUs and confusion matrix  
        per_class_iou = torchmetrics.functional.jaccard_index(logits[None, ...], targets[None, ...], absent_score=np.NaN, 
                                                              num_classes=model.num_classes, average="none")
        iou = np.nanmean(per_class_iou)
        per_class_ious.append(per_class_iou)
        ious.append(iou)
        dice_scores.append(torchmetrics.functional.dice(preds, targets, num_classes=model.num_classes,
                                                         mdmc_average='global', average='none'))

        confusion_matrix += torchmetrics.functional.confusion_matrix(preds, targets, num_classes=model.num_classes).numpy()
        
        # Fit contours to mask and compute contour metrics
        preds = preds.numpy()
        fitted_contours = mapping_utils.fit_contours(preds)
        inference_times.append(time.time() - t_inference_start)
        gt_contours = mapping_utils.load_contours(target_path)
        dcm_file = pydicom.dcmread(img_path)
        spatial_scaler = mapping_utils.get_pixel_size(dcm_file) if spatial_unit == 'mm' else 1
        hausdorff_distances.append(mapping_utils.hausdorff_dist(fitted_contours, gt_contours) * spatial_scaler)
        mean_surface_distances.append(mapping_utils.mean_surface_distance(fitted_contours, gt_contours) * spatial_scaler)
        signed_msds.append(mapping_utils.signed_mean_surface_distance(fitted_contours, gt_contours) * spatial_scaler)
        
        original_img = dcm_file.pixel_array    
        preds = preds[:original_img.shape[0], :original_img.shape[1]]

        plot_title = img_path.replace(dataset_root+"/", "") + "\n" + target_path.replace(dataset_root+"/", "") + \
        f"\nIoU: Mean = {iou:.3f} | Epicardial = {per_class_ious[-1][1]:.3f} | Endocardial = {per_class_ious[-1][2]:.3f} | Background = {per_class_ious[-1][0]:.3f}" + \
        f"\n Dice: Epicardial = {dice_scores[-1][1]:.3f} | Endocardial = {dice_scores[-1][2]:.3f} | Background = {dice_scores[-1][0]:.3f}" + \
        f"\n DH: Epicardial = {hausdorff_distances[-1][0]:.3f} | Endocardial = {hausdorff_distances[-1][1]:.3f}" + \
        f"\n MSD: Epicardial = {mean_surface_distances[-1][0]:.3f} | Endocardial = {mean_surface_distances[-1][1]:.3f}"
        
        out_path = img_path.replace(".dcm", ".pdf")
        out_path = out_path.replace(dataset_root, PRED_FOLDER)
        out_path = os.path.join(output_folder, out_path)

        # utils.plot_mapping_prediction(original_img, preds, target_path, plot_title, out_path)
        utils.plot_mapping_predicted_contours(original_img, fitted_contours, target_path, plot_title, out_path)
        pdf_paths.append(out_path)

    print(f"Inference time = {np.mean(inference_times)} ± {np.std(inference_times)} for {len(inference_times)} samples")

    mean_iou =  np.mean(ious)
    std_iou = np.std(ious)
    per_class_ious_averaged_over_all_samples = np.nanmean(np.stack(per_class_ious), axis=0)
    np.save(os.path.join(output_folder,"per_class_ious"), np.stack(per_class_ious))
    print("=============================")
    print("Checkpoint", checkpoint_path)
    print(f"Evaluated {len(ious)} samples")
    print("Images padded to = ", pad_to)
    print("Mean IoU =", mean_iou)
    print("Stdev IoU =", std_iou)
    print("Per-class-IoUs", per_class_ious_averaged_over_all_samples)
    print("Mean of per-class-IoUs", np.mean(np.stack(per_class_ious_averaged_over_all_samples)))
    print("Confusion matrix: \n", confusion_matrix)
    print("=============================")

    # Compiling results for each sample to a dataframe and saving as a csv
    results = np.hstack([np.stack(per_class_ious), np.stack(dice_scores), np.stack(hausdorff_distances), np.stack(mean_surface_distances), np.stack(signed_msds)])
    columns = ["Background IoU",  "Epicardial(MYO) IoU", "Endocardial(LV) IoU", 
               "Background Dice", "Epicardial Dice", "Endocardial Dice", 
               f"Epicardial DH [{spatial_unit}]",   f"Endocardial DH [{spatial_unit}]", 
               f"Epicardial MSD [{spatial_unit}]",  f"Endocardial MSD [{spatial_unit}]",
               f"Epicardial sMSD [{spatial_unit}]", f"Endocardial sMSD [{spatial_unit}]"]
    results_df = pd.DataFrame(results, index=img_paths, columns=columns)
    results_df["model"] = checkpoint_path
    if pad_to != 224: 
        results_df["resolution"] = pad_to
    for m in ["IoU", "Dice"]:
        results_df[f"Mean {m}"]=results_df.filter(like=m).mean(axis=1)
    for m in ['DH', 'MSD', 'sMSD']:
        results_df[f"Mean {m} [{spatial_unit}]"]=results_df.filter(like=f" {m}").mean(axis=1) # " {m}" -> the space is necessary to distinguish MSD and sMSD
    per_sample_results_path = os.path.join(output_folder,f"inference_results_{spatial_unit}_{dataset_split}_obs{observer_id}{filename_suffix}.csv")
    print("Raw results saved to:", per_sample_results_path)
    results_df.to_csv(per_sample_results_path)

    # categories = ["T1_map_apex", "T1_map_base", "T1_map_mid_", "T1_Mapping_",
    #               "T2_map_apex", "T2_map_base", "T2_map_mid_", "T2_Mapping_"]
    # header = ["Category", "Mean IoU", "Background IoU", "Epicardial(MYO) IoU", "Endocardial(LV) IoU"]
    # utils.summarize_results(per_class_ious, img_paths, header, filter_values=categories)
    #
    # print("\nMean Hausdorf distance (DH) for each category for each class")
    # header = ["Category", "Mean DH","Epicardial DH", "Endocardial DH"]
    # utils.summarize_results(hausdorff_distances, img_paths, header, filter_values=categories)
    #
    # print("\nMean \"Mean Surface Distance\" (MSD) for each category for each class")
    # header = ["Category", "Mean MSD","Epicardial MSD", "Endocardial MSD"]
    # utils.summarize_results(mean_surface_distances, img_paths, header, filter_values=categories)
    #
    # print("\nMean \"Signed Mean Surface Distance\" (MSD) for each category for each class")
    # header = ["Category", "Mean sMSD","Epicardial sMSD", "Endocardial sMSD"]
    # utils.summarize_results(signed_msds, img_paths, header, filter_values=categories)

    labels = ["Background", "Epicarcical(MYO)", "Endocardial(LV)"]
    utils.plot_iou_histograms(np.stack(per_class_ious), labels, os.path.join(output_folder, PRED_FOLDER))
    utils.plot_confmat(confusion_matrix, labels, os.path.join(output_folder, PRED_FOLDER))

    if merge_prediction_pdfs:
        sort_by = results_df[f"Endocardial DH [{spatial_unit}]"].to_numpy()
        sorted_pdfs = np.array(pdf_paths)[np.argsort(sort_by)].tolist()
        utils.merge_pdfs(output_folder, pdf_path_list=sorted_pdfs, out_file_name=f"Predictions_{dataset_split}_obs{observer_id}.pdf", remove_files=remove_individual_pdfs)

    return results_df

def shorten_ckpt_str(ckpt_path:str):
    ckpt_path_parts = ckpt_path.split(os.path.sep)
    checkpoint = ckpt_path_parts[-1]
    version = ckpt_path_parts[-3]
    experiment = ckpt_path_parts[-4]
    if version != "version_0":
        ckpt_str = os.path.join(experiment, version, "...", checkpoint)
    else:
        ckpt_str = os.path.join(experiment, "...", checkpoint)
        ckpt_str = "`" + ckpt_str + "`"
    return ckpt_str

def multi_ckpt_eval(ckpt_paths):
    if not isinstance(ckpt_paths, list):
        ckpt_paths = [ckpt_paths]
    for ckpt_path in ckpt_paths:
        assert os.path.exists(ckpt_path), os.path.abspath(ckpt_path)
    ckpt_strs = []
    result_dfs = []
    for ckpt_path in ckpt_paths:
        results_df = generate_prediction_pdfs(
            ckpt_path,
            output_folder=os.path.dirname(os.path.dirname(ckpt_path)),
            merge_prediction_pdfs=True,
            remove_individual_pdfs=True,
            spatial_unit='mm'
            )
        result_dfs.append(results_df)
        ckpt_strs.append(shorten_ckpt_str(ckpt_path))

    result_dfs = pd.concat(result_dfs, keys=[df["model"][0] for df in result_dfs]) 
    
    header = ["Metric", *list(range(len(ckpt_strs)))]
    for i, ckpt in enumerate(ckpt_strs):
        print(f"{i}: {ckpt}")

    print("Mean metrics")
    summarized_resuts = result_dfs.groupby("model").mean().transpose()
    print(tabulate(summarized_resuts, header, tablefmt="pipe"))

    print("Median metrics")
    summarized_resuts = result_dfs.groupby("model").median().transpose()
    print(tabulate(summarized_resuts, header, tablefmt="pipe"))


def multi_resolution_eval(ckpt_path):
    result_dfs = []
    for resolution in [None, 160, 224, 256, 320, 512]:
        results_df = generate_prediction_pdfs(
            ckpt_path,
            output_folder=os.path.dirname(os.path.dirname(ckpt_path)),
            merge_prediction_pdfs=False,
            remove_individual_pdfs=True,
            pad_to=resolution
            )
        result_dfs.append(results_df)
    
    result_dfs = pd.concat(result_dfs, keys=[df["resolution"][0] for df in result_dfs]) 

    summarized_resuts = result_dfs.groupby("resolution").mean().transpose()
    print(tabulate(summarized_resuts, headers='keys', tablefmt="pipe"))

def multi_observer_eval(ckpt_path):
    result_dfs = []
    observer_ids = [1,2,3]
    for observer_id in observer_ids:
        results_df = generate_prediction_pdfs(
            ckpt_path,
            output_folder=os.path.dirname(os.path.dirname(ckpt_path)),
            merge_prediction_pdfs=False,
            remove_individual_pdfs=True,
            dataset_split='interobserver',
            observer_id=observer_id
            )
        results_df["observer_id"] = observer_id
        result_dfs.append(results_df)
    
    result_dfs = pd.concat(result_dfs, keys=observer_ids)

    summarized_resuts = result_dfs.groupby("observer_id").mean().transpose()
    print(tabulate(summarized_resuts, headers='keys', tablefmt="pipe"))


if __name__ == '__main__':
    # ckpt_paths = [
    #     "path_to_your_checkpoint.ckpt",
    #     ]
    
    # multi_ckpt_eval(ckpt_paths)

    multi_observer_eval("path_to_your_checkpoint.ckpt")
    