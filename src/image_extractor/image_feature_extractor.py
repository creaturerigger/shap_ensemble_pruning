# src/image_extractor/image_feature_extractor.py

from src.trainer.CNN import CustomCNN
from torch.utils.data import DataLoader
import torch
import os
import numpy as np

def extract_and_save_features(model: CustomCNN, dataloader: DataLoader, dataset_name: str, out_dir: str, split: str) -> None:
        
        os.makedirs(out_dir, exist_ok=True)

        model.eval()
        features, labels = [], []

        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(model.latent.device)
                latent = model.get_latent_features(data)
                features.append(latent.cpu())
                labels.append(target.cpu())

        features = torch.cat(features)
        labels = torch.cat(labels)
        
        np.save(f"{out_dir}/{split}_features.npy", {    # type: ignore
            "features": features.numpy(),
            "labels": labels.numpy()
        })