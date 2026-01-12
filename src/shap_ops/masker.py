# src/shap_ops/masker.py

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def load_shap_masked_dataloader(
    shap_dir: str,
    percentile: float=90.0,
    batch_size: int=64,
    return_original: bool=False,
    split: str='train'
) -> DataLoader:
    """
    Load SHAP values and corresponding input images, apply top-k percentile masking, and return a DataLoader.

    Args:
        shap_dir (str): Path to directory containing `shap_class*_batch*.npy`, `images_batch*.npy`, and `labels_batch*.npy`.
        percentile (float): Top-k percentile for masking (e.g., 90.0 to keep top 10% important pixels).
        batch_size (int): DataLoader batch size.
        return_original (bool): If True, return original images along with masked ones.

    Returns:
        DataLoader containing (masked_images, labels) or (masked_images, original_images, labels).
    """
    shap_files = sorted([
        f for f in os.listdir(shap_dir)
        if f.startswith("shap_batch") and f.endswith(".npy")
    ])
    image_files = sorted([
        f for f in os.listdir(shap_dir)
        if f.startswith("images_batch") and f.endswith(".npy")
    ])
    label_files = sorted([
        f for f in os.listdir(shap_dir)
        if f.startswith("labels_batch") and f.endswith(".npy")
    ])

    masked_all = []
    original_all = []
    labels_all = []

    for shap_file, img_file, lbl_file in zip(shap_files, image_files, label_files):
        shap_vals = torch.tensor(np.load(os.path.join(shap_dir, shap_file))).float()
        images = torch.tensor(np.load(os.path.join(shap_dir, img_file))).float()
        labels = torch.tensor(np.load(os.path.join(shap_dir, lbl_file)), dtype=torch.long)

        if shap_vals.shape != images.shape:
            print(f"[WARN] Resizing SHAP mask {shap_file} from {shap_vals.shape[2:]} to {images.shape[2:]}")
            shap_vals = F.interpolate(shap_vals, size=images.shape[2:], mode='bilinear', align_corners=False)

        threshold = torch.quantile(shap_vals.abs().flatten(1), percentile / 100.0, dim=1)
        threshold = threshold.view(-1, 1, 1, 1)

        mask = (shap_vals.abs() >= threshold).float()
        masked = images * mask

        masked_all.append(masked)
        labels_all.append(labels)
        if return_original:
            original_all.append(images)

    masked_all = torch.cat(masked_all)
    labels_all = torch.cat(labels_all)

    if return_original:
        original_all = torch.cat(original_all)
        dataset = TensorDataset(original_all, labels_all)
    else:
        dataset = TensorDataset(masked_all, labels_all)
    shuffle = False
    if split == 'train':
        shuffle = True

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, persistent_workers=True,
                      prefetch_factor=4)
