# src/flows/training_flow.py

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
logging.getLogger("shap").setLevel(logging.ERROR)


from prefect import flow, task
from prefect.logging import get_run_logger
from src.trainer.CNN import CustomCNN
from src.trainer.trainer import Trainer
from src.image_extractor.image_feature_extractor import extract_and_save_features
from src.logger.logger import PostgresLogger
from src.ensemble_trainer.ensemble_trainer import CNNEnsembleTrainer
from src.pruning.sip_pruning import sip_pruning

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from shap import DeepExplainer
import numpy as np
import os
from tqdm import tqdm
import yaml
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import pandas as pd
from pathlib import Path

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

@task
def get_dataloaders(dataset_name: str, batch_size: int=128) -> tuple:
    

    if dataset_name == "mnist":
        transform = transforms.ToTensor()
        train_ds = datasets.MNIST(root="MNIST", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="MNIST", train=False, download=True, transform=transform) 
        input_channels = 1
        num_classes = 10
    elif dataset_name == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=int(32*0.125)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02,0.12))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491,0.482,0.447),
                                (0.247,0.243,0.262)),
        ])

        train_ds = datasets.CIFAR10(root="CIFAR10", train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(root="CIFAR10", train=False, download=True, transform=test_transform)
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported")
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int((len(test_ds) / len(train_ds)) * batch_size))
    
    for x, _ in train_loader:
        _, c, h, w = x.shape
        image_size = (h, w)
        break
    
    return (
        train_loader,
        test_loader,
        input_channels,
        num_classes,
        image_size    # type: ignore
    )

@task
def train_model(train_loader, test_loader, input_channels, num_classes, image_size, epochs: int,
                dataset_name):
    logger = PostgresLogger()
    prefect_logger = get_run_logger()

    model_root_dir = f"model_artefacts/without_shap/{dataset_name}"

    if logger.check_if_completed(dataset_name, step='train'):
        prefect_logger.info(f"🔁 Training already done for '{dataset_name}', skipping.")
        model = CustomCNN(input_channels, num_classes, image_size)
        model.load_state_dict(torch.load(f"{model_root_dir}/model_epoch_{epochs}.pt"))
        model.eval()
        return model
    model = CustomCNN(input_channels, num_classes, image_size)
    trainer = Trainer(train_loader, test_loader, cnn_in_ch=input_channels, epochs=epochs,
                      num_classes=num_classes, artefact_dir=model_root_dir)
    final_test_acc, best_model = trainer.fit()

    logger.log_run(
        dataset=dataset_name,
        step="train",
        status="completed",
        model_path=model_root_dir,
        accuracy=final_test_acc
    )

    return best_model

@task
def extract_features(model, dataloader, dataset_name: str, split: str='train'):
    logger = PostgresLogger()
    prefect_logger = get_run_logger()
    out_dir = f"extracted_features/{dataset_name}"
    feature_path = f"{out_dir}/{split}_features.npy"

    if logger.check_if_completed(dataset_name, f"extract_{split}"):
        prefect_logger.info(f"✅ Skipping feature extraction for {dataset_name} ({split})")
        return

    extract_and_save_features(model, dataloader, dataset_name, out_dir, split)

    logger.log_run(
        dataset=dataset_name,
        step=f"extract_{split}",
        status="completed",
        feature_path=feature_path
    )

@task
def apply_shap_and_save(model, dataloader, dataset_name: str, split: str='train'):
    import torch.nn as nn
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    logger = PostgresLogger()
    prefect_logger = get_run_logger()

    def save_shap_and_inputs(shap_values, images, labels, out_dir, batch_idx):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"images_batch{batch_idx}.npy"), images.cpu().numpy())
        np.save(os.path.join(out_dir, f"labels_batch{batch_idx}.npy"), labels.cpu().numpy())
        np.save(os.path.join(out_dir, f"shap_batch{batch_idx}.npy"), shap_values)

    out_dir = f"shap_values/without_shap/{dataset_name}/{split}"
    os.makedirs(out_dir, exist_ok=True)

    if logger.check_if_completed(dataset_name, f"shap_full_{split}"):
        prefect_logger.info(f"✅ SHAP already completed for {dataset_name} ({split})")
        return

    background_data = next(iter(dataloader))[0][:5].to(device)
    explainer = DeepExplainer(model, background_data)

    total_batches = len(dataloader)
    shap_p_bar = tqdm(dataloader, desc=f"Computing SHAP ({split})")
    for batch_idx, (images, labels) in enumerate(shap_p_bar):
        images = images[:4].to(device)
        labels = labels[:4].to(device)

        shap_vals, _ = explainer.shap_values(images, ranked_outputs=1, check_additivity=False)

        
        shap_vals = np.squeeze(np.stack(shap_vals), axis=-1)    # type: ignore
    
        save_shap_and_inputs(shap_vals, images, labels, out_dir, batch_idx)

        if batch_idx % 5 == 0:
            shap_p_bar.set_description(f"✅ Processed batch {batch_idx+1}/{total_batches}")

    logger.log_run(
        dataset=dataset_name,
        step=f"shap_full_{split}",
        status="completed",
        feature_path=out_dir
    )


@task
def train_ensemble(dataset_name: str):
    logger = PostgresLogger()
    prefect_logger = get_run_logger()

    if logger.check_if_completed(dataset_name, step='ensemble'):
        prefect_logger.info(f"✅ Ensemble already completed for {dataset_name}")
        return

    with open("config/ensemble_config.yaml", "r") as f:
        ensemble_config = yaml.safe_load(f)

    ensemble_models = list(ensemble_config.keys())
    fixed_rand = random.Random(42)
    ensemble_models = fixed_rand.sample(ensemble_models, 100)
    pbar = tqdm(ensemble_models, desc="Training ensemble models")

    for model_id in pbar:
        trainer = CNNEnsembleTrainer(
            model_id=model_id,
            top_k_percentile=90 if dataset_name == 'mnist' else 40,
            shap_dir="shap_values/without_shap",
            save_dir="ensemble_models/without_shap",
            dataset_name=dataset_name,
            early_stopping_patience=20,
            masked_data=False
        )
        result = trainer.train()

        pbar.set_description(
            f"Model: {model_id} | Best Acc: {result['best_val_acc']:.4f} | Epochs: {result['epochs_trained']}"
        )

    logger.log_run(
        dataset=dataset_name,
        step="ensemble",
        status="completed",
        model_path="ensemble_models"
    )

@task
def prune_and_evaluate_with_sip(prediction_dir: str):
    logger = get_run_logger()

    # Find all .pt model files
    model_files = [f for f in os.listdir(prediction_dir) if f.startswith("MODEL") and f.endswith("_best_model.pt")]

    if not model_files:
        raise FileNotFoundError("No model checkpoint files found.")

    model_preds = []
    model_ids = []

    for f in model_files:
        match = re.search(r"MODEL(\d+)", f)
        if not match:
            logger.warning(f"Skipping unrecognized model file: {f}")
            continue

        model_id = int(match.group(1))
        pred_file_npy = f"MODEL{model_id}_Test_Predictions_withProbabilities.npy"
        pred_file_csv = f"MODEL{model_id}_Test_Predictions_withProbabilities.csv"

        pred_path_npy = os.path.join(prediction_dir, pred_file_npy)
        pred_path_csv = os.path.join(prediction_dir, pred_file_csv)

        if os.path.exists(pred_path_npy):
            preds = np.load(pred_path_npy)
        elif os.path.exists(pred_path_csv):
            preds = pd.read_csv(pred_path_csv).values
        else:
            logger.warning(f"❌ No prediction file found for MODEL{model_id}, skipping.")
            continue

        model_preds.append(preds)
        model_ids.append(model_id)

    if not model_preds:
        raise RuntimeError("No valid predictions loaded for SIP pruning.")

    model_preds = np.array(model_preds)
    logger.info(f"✅ Loaded predictions from {len(model_preds)} models")

    # Load true labels
    label_file = next((f for f in os.listdir(prediction_dir)
                       if "Test_y_labels" in f and (f.endswith(".npy") or f.endswith(".csv"))), None)

    if label_file is None:
        raise FileNotFoundError("No test labels file found.")

    y_path = os.path.join(prediction_dir, label_file)
    if label_file.endswith(".npy"):
        y_true = np.load(y_path)
    else:
        y_true = pd.read_csv(y_path).values.squeeze()

    weights = sip_pruning(model_preds)
    ensemble_probs = np.average(model_preds, axis=0, weights=weights)
    y_pred = ensemble_probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    logger.info(f"🧪 Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    logger.info(f"✅ SIP used {np.count_nonzero(weights > 1e-6)} out of {len(model_ids)} models")

    return {
        "selected_models": model_ids,
        "weights": weights.tolist(),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "used_model_count": np.count_nonzero(weights > 1e-6)
    }


@task
def evaluate_pruned_models_on_original_testset(
    model_dir: str,
    model_ids: list[int],
    weights: list[float],
    dataset_name: str,
    batch_size: int = 128
):
    logger = PostgresLogger()
    prefect_logger = get_run_logger()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if dataset_name.lower() == "mnist":
        transform = transforms.ToTensor()
        test_ds = datasets.MNIST(root="MNIST", train=False, download=True, transform=transform)
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])
        test_ds = datasets.CIFAR10(root="CIFAR10", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    y_true = torch.cat([y for _, y in test_loader], dim=0).numpy()
    prefect_logger.info(f"🧪 Loaded {len(y_true)} test samples for evaluation.")

    preds_list = []

    for idx in model_ids:
        model_path = os.path.join(model_dir, f"MODEL{idx}_best_model.pt")
        if not os.path.exists(model_path):
            prefect_logger.warning(f"⚠️ Model file not found: {model_path}")
            continue

        trainer = CNNEnsembleTrainer(
            model_id=f"MODEL{idx}",
            top_k_percentile=100,
            shap_dir="",
            save_dir=model_dir,
            dataset_name=dataset_name,
            masked_data=False
        )
        model = trainer.model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        all_probs = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                outputs = model(xb)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())

        preds = torch.cat(all_probs, dim=0).numpy()
        preds_list.append(preds)

    if not preds_list:
        raise RuntimeError("❌ No models successfully evaluated on the original test set.")

    preds_array = np.array(preds_list)
    ensemble_probs = np.average(preds_array, axis=0, weights=np.array(weights))
    y_pred = np.argmax(ensemble_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    prefect_logger.info(
        f"✅ Evaluation on Original Test Set:\n"
        f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


@flow
def training_flow(dataset_name: str, num_epochs=5):
    train_loader, test_loader, input_channels, num_classes, image_size = get_dataloaders(dataset_name)
    model = train_model(train_loader, test_loader, input_channels, num_classes, image_size, num_epochs, dataset_name)
    apply_shap_and_save(model, train_loader, dataset_name)
    apply_shap_and_save(model, train_loader, dataset_name, split='test')
    train_ensemble(dataset_name)
    results = prune_and_evaluate_with_sip(prediction_dir=f"ensemble_models/with_shap/{dataset_name}")

    metrics_orig = evaluate_pruned_models_on_original_testset(
        model_dir=f"ensemble_models/with_shap/{dataset_name}",
        weights=results["weights"],
        model_ids=results["selected_models"],
        dataset_name=dataset_name,
    )

    df = pd.DataFrame([{
        "dataset_name": dataset_name,
        "num_models": len(results["selected_models"]),
        "accuracy_masked": results["accuracy"],
        "precision_masked": results["precision"],
        "recall_masked": results["recall"],
        "f1_masked": results["f1"],
        "accuracy_original": metrics_orig["accuracy"],
        "precision_original": metrics_orig["precision"],
        "recall_original": metrics_orig["recall"],
        "f1_original": metrics_orig["f1"],
    }])

    output_dir = Path("logs/shap_values_ensemble_logs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"ensemble_pruning_results_{dataset_name}_1.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Results saved to: {output_path.resolve()}")