from operator import indexOf
import os
from typing import Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import data.mapping_utils as mapping_utils
import pandas as pd
import glob
from tabulate import tabulate
from tqdm import tqdm
import utils

DATASET_MEAN = 243.0248
DATASET_STD = 391.5486

# Test and validation samples are selected on a patient level (i.e. all data of a patient is either training, test, or validation)
# Test and validation patients were selected randomly
# from the list of patients without missing contours, or incorrect mapping files
# fmt: off
# fmt: off prevents black from autoformatting these lines
TEST_PATIENTS = [7, 21, 30, 33, 34, 37, 41, 58, 86, 110, 123, 135, 145, 148, 155, 163, 164, 172, 177, 183, 190, 191, 207, 212, 220]
VAL_PATIENTS  = [3, 4, 12, 14, 19, 23, 28, 35, 40, 46, 50, 55, 98, 107, 130, 137, 156, 162, 176, 182, 185, 197, 209, 213, 219]
TRAIN_PATIENTS = [id for id in range(1, 222+1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS]
INTEROBSERVER_PATIENTS = range(223, 262+1)

#fmt: off

class MappingDatasetAlbu(Dataset):
    """ Pytorch dataset for the SE dataset that supports Albumentations augmentations (including bounding box safe cropping).

        Args:
            root (str, Path): Root directory of the dataset containing the 'Patient (...)' folders.
            transforms (albumentations.Compose, optional): Albumentations augmentation pipeline. Defaults to None.
            split (str, optional):  The dataset split, supports `train`, `val`, `test` or `interobserver`. Defaults to 'train'.
            check_dataset (bool, optional): Check dataset for missing and unexpected files. Outdated... Defaults to False.
            observer_id (int, optional): Contours from 3 annotators are available for the same patient in the `interobserver` `split`. `observer_id` selects from these. Supports 1,2 or 3. Defaults to 1.
            mapping_only (bool, optional): Include T1 and T2 mappping images only or also include the map sequences that were used to construct these mappings. Defaults to False.
    """    
    def __init__(self, 
                 root, 
                 transforms=None, 
                 split='train', 
                 check_dataset=False, 
                 observer_id = 1,
                 mapping_only = False) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split
        if check_dataset:
            mapping_utils.check_datset(self.root)
        if split == "interobserver" and observer_id != 1:
            contours_filename = f"Contours_{observer_id}.json"
        else:
            contours_filename = f"Contours.json"
        self.all_samples = mapping_utils.construct_samples_list(
            self.root, contours_filename
        )
        mapping_utils.print_diagnostics(self.root, self.all_samples)

        segmentation_partitions = {
            "train": TRAIN_PATIENTS,
            "val": VAL_PATIENTS,
            "test": TEST_PATIENTS,
            "interobserver": INTEROBSERVER_PATIENTS,
        }

        self.samples = mapping_utils.split_samples_list(
            self.all_samples, segmentation_partitions[self.split]
        )
        if mapping_only:
            self.to_mapping_only()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target)
        """
        path, target_path = self.samples[index]
        sample = mapping_utils.load_dicom(path, mode=None)
        target_contours = mapping_utils.load_contours(target_path)
        target = mapping_utils.contours_to_masks(target_contours, sample.shape)
        if self.transforms is not None:
            if "bboxes" in self.transforms.processors.keys():
                bbox = self.compute_bbox(target)
                transformed = self.transforms(image=sample, mask=target, bboxes=[bbox])
            else:
                transformed = self.transforms(image=sample, mask=target)
            sample, target = transformed["image"], transformed["mask"]

        # Convert images to channels_first mode, from albumentations' 2d grayscale images
        sample = np.expand_dims(sample, 0)

        return sample, target

    def compute_bbox(self, mask):
        # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        bbox = [
            mask.nonzero()[1].min(),  # / mask.shape[0],
            mask.nonzero()[0].min(),  # / mask.shape[1],
            mask.nonzero()[1].max(),  # / mask.shape[0],
            mask.nonzero()[0].max(),  # / mask.shape[1],
            "dummy_label",
        ]  # x_min, y_min, x_max, y_max
        return bbox

    def __len__(self) -> int:
        return len(self.samples)

    def to_mapping_only(self):
        self.samples = [(x, t) for x, t in self.samples if "_Mapping_" in x]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose(
        [
            # A.LongestMaxSize(192),
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.Normalize(mean=(243.0248,), std=(391.5486,)),
            # ToTensorV2()
        ]
    )

    train_ds = MappingDatasetAlbu(
        "/home1/ssl-phd/data/mapping",
        transform,
        check_dataset=False,
        mapping_only=True,
    )
    print("Train samples =", len(train_ds))
    val_ds = MappingDatasetAlbu(
        "/home1/ssl-phd/data/mapping", split="val", mapping_only=True
    )
    print("Val samples =", len(val_ds))
    test_ds = MappingDatasetAlbu(
        "/home1/ssl-phd/data/mapping", split="test", mapping_only=True
    )
    print("Test samples =", len(test_ds))
    interobs_ds = MappingDatasetAlbu(
        "/home1/ssl-phd/data/mapping", split="interobserver", mapping_only=True
    )

    # from torch.utils.data import DataLoader

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=32,
    #     num_workers=2,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # print("Train dataloader = ", len(train_loader))

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=32,
    #     num_workers=2,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    # print("Val dataloader = ", len(val_loader))

    # train_ds.to_mapping_only()
    print("Train samples =", len(train_ds))
    print("Val samples =", len(val_ds))
    print("Test samples =", len(test_ds))
    print("Interobserver samples =", len(interobs_ds))
    # print("Train dataloader = ", len(train_loader))
    # print("Val dataloader = ", len(val_loader))

    # img, mask = train_ds[10]
    # print("Img shape", img.shape)
    # print("Mask shape", mask.shape)
    # img = img[0,...]#.detach().numpy()
    # cv2.imwrite(f"example_transformed.png", np.concatenate([img, np.ones((img.shape[0], 3))*255, mask], axis=1))
