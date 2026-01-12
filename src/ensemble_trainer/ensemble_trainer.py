# src/ensemble_trainer/ensemble_trainer.py

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import psycopg2
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error
)
from src.shap_ops.masker import load_shap_masked_dataloader
from dotenv import load_dotenv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

load_dotenv()

db_name_key = "DB_NAME"
db_user_key = "POSTGRES_USER"
db_user_password_key = "PSQL_PASSWORD"
db_name = os.getenv(db_name_key)
db_user = os.getenv(db_user_key)
db_user_password = os.getenv(db_user_password_key)

class CNNEnsembleTrainer:
    def __init__(self, model_id: str, top_k_percentile: float, shap_dir: str, dataset_name: str, batch_size=32, masked_data=True,
                 early_stopping_patience: int=5, config_path="config/ensemble_config.yaml", save_dir="ensemble_models",
                save_model_summary=True):
        self.model_id = model_id
        self.save_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self._load_config(config_path)
        self.top_k_percentile = top_k_percentile
        self.batch_size = batch_size
        self.masked_data = masked_data
        self.early_stopping_patient = early_stopping_patience
        self.dataset_name = dataset_name
        self.shap_dir = shap_dir
        self._prepare_data()
        self._build_model()
        if save_model_summary:
            self._save_model_summary()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.model_config = config[self.model_id]

    def _prepare_data(self):
        shap_dir = f"{self.shap_dir}/{self.dataset_name}"
        if self.masked_data:
            self.train_loader = load_shap_masked_dataloader(
                shap_dir=shap_dir + "/train",
                percentile=self.top_k_percentile,
                batch_size=self.batch_size
            )

            self.test_loader = load_shap_masked_dataloader(
                shap_dir=shap_dir + "/test",
                percentile=self.top_k_percentile,
                batch_size=self.batch_size,
                return_original=True,
                split='test'
            )
        else:
            transform = transforms.ToTensor()
            if self.dataset_name == 'mnist':
                train_ds = datasets.MNIST(root="MNIST", train=True, download=True, transform=transform)
                test_ds = datasets.MNIST(root="MNIST", train=False, download=True, transform=transform)
            elif self.dataset_name == 'cifar10':
                train_ds = datasets.CIFAR10(root='CIFAR10', train=True, download=True, transform=transform)
                test_ds = datasets.CIFAR10(root='CIFAR10', train=False, download=True, transform=transform)
            else:
                raise ValueError(f"Dataset {self.dataset_name} is not supported")
            self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                           num_workers=4, persistent_workers=True,
                                           prefetch_factor=4)
            
            self.test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                          num_workers=4, persistent_workers=True, prefetch_factor=4)

    def _build_model(self):
        layers = []
        in_channels = self.train_loader.dataset[0][0].shape[0]

        conv_filters = self.model_config['conv_filters']
        conv_layer_count = self.model_config['conv_layer_count']

        for i in range(conv_layer_count):
            out_channels = conv_filters[i]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            pooling_layer = nn.MaxPool2d(kernel_size=2) if self.model_config['pooling_types'][i] == 'max' else nn.AvgPool2d(kernel_size=2)
            layers.append(pooling_layer)
            in_channels = out_channels

        layers.append(nn.Flatten())
        dummy_input_shape = self.train_loader.dataset[0][0].shape
        dummy_input = torch.zeros(1, *dummy_input_shape)
        with torch.no_grad():
            dummy_output = nn.Sequential(*layers)(dummy_input)
        in_features = dummy_output.shape[1]

        dense_units = self.model_config['dense_units']
        dropout_rates = self.model_config.get('dropout_rates', [])

        for i, units in enumerate(dense_units[:-1]):
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            if i < len(dropout_rates):
                layers.append(nn.Dropout(float(dropout_rates[i])))
            in_features = units

        layers.append(nn.Linear(in_features, 10))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers).to(self.device)

    def _save_model_summary(self):
        os.makedirs(self.save_dir, exist_ok=True)
        summary_str = str(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        summary_str += f"\n\nTotal parameters: {total_params}\nTrainable parameters: {trainable_params}\n"

        with open(os.path.join(self.save_dir, f"{self.model_id}_summary.txt"), "w") as f:
            f.write(summary_str)

        try:
            db_name_key = "DB_NAME"
            db_user_key = "POSTGRES_USER"
            db_user_password_key = "PSQL_PASSWORD"
            db_name = os.getenv(db_name_key)
            db_user = os.getenv(db_user_key)
            db_user_password = os.getenv(db_user_password_key)
            conn = psycopg2.connect(
                dbname=db_name, user=db_user, password=db_user_password, host="localhost", port="5432"
            )
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ensemble_model_summaries (
                    model_id TEXT,
                    dataset TEXT,
                    summary TEXT,
                    total_params INT,
                    trainable_params INT,
                    CONSTRAINT ensemble_model_dataset_unique UNIQUE (model_id, dataset)
                )
            ''')
            cursor.execute('''
                INSERT INTO ensemble_model_summaries (model_id, dataset, summary, total_params, trainable_params)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (model_id, dataset) DO UPDATE SET summary = EXCLUDED.summary,
                                                    total_params = EXCLUDED.total_params,
                                                    trainable_params = EXCLUDED.trainable_params
            ''', (self.model_id, self.dataset_name, summary_str, total_params, trainable_params))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Failed to save model summary to PostgreSQL: {e}")


    def train(self):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_config['learning_rate'])

        train_acc, val_acc, train_loss, val_loss = [], [], [], []

        best_val_acc = 0
        best_model_state = None
        last_epoch_number = 0

        for epoch in range(self.model_config['epochs']):
            self.model.train()
            correct, total, running_loss = 0, 0, 0
            last_epoch_number = epoch

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_epoch_loss = running_loss / len(self.train_loader)
            train_epoch_acc = correct / total
            val_epoch_acc, val_epoch_loss = self.evaluate(epoch)

            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            self._log_epoch_to_postgres(epoch, train_epoch_acc, val_epoch_acc, train_epoch_loss, val_epoch_loss)

            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                best_model_state = copy.deepcopy(self.model.state_dict())

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        model_path = os.path.join(self.save_dir, f"{self.model_id}_best_model.pt")
        torch.save(self.model.state_dict(), model_path)

        self._save_metrics(train_acc, val_acc, train_loss, val_loss)

        return {
            "best_val_acc": best_val_acc,
            "epochs_trained": last_epoch_number + 1,
            "model_path": model_path
        }


    def evaluate(self, epoch):
        self.model.eval()
        correct, total, loss_total = 0, 0, 0
        all_labels = []
        all_preds = []
        all_probs = []
        train_labels = []
        train_probs = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_total += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                train_labels.extend(labels.cpu().numpy())
                train_probs.extend(outputs.cpu().numpy())

        self._save_evaluation_results(np.array(all_labels), np.array(all_preds), np.array(all_probs),
                                      np.array(train_labels), np.array(train_probs), epoch)

        return correct / total, loss_total / len(self.test_loader)

    def _save_metrics(self, train_acc, val_acc, train_loss, val_loss):
        df = pd.DataFrame({
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        df.to_csv(os.path.join(self.save_dir, f"{self.model_id}_metrics.csv"), index=False)

        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"{self.model_id}_accuracy.png"))
        plt.clf()

        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"{self.model_id}_loss.png"))
        plt.clf()

    def _log_epoch_to_postgres(self, epoch, train_acc, val_acc, train_loss, val_loss):
        try:
            conn = psycopg2.connect(
                dbname=db_name, user=db_user, password=db_user_password, host="localhost", port="5432"
            )
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ensemble_epoch_metrics (
                    model_id TEXT,
                    dataset TEXT,
                    epoch INT,
                    train_acc FLOAT,
                    val_acc FLOAT,
                    train_loss FLOAT,
                    val_loss FLOAT
                )
            ''')
            cursor.execute('''
                INSERT INTO ensemble_epoch_metrics (model_id, dataset, epoch, train_acc, val_acc, train_loss, val_loss)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (self.model_id, self.dataset_name, epoch, train_acc, val_acc, train_loss, val_loss))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Failed to save epoch metrics to PostgreSQL: {e}")

    def _save_evaluation_results(self, y_true, y_pred, y_prob, train_y, train_prob, epoch):
        model_prefix = os.path.join(self.save_dir, self.model_id)

        np.save(f"{model_prefix}_Train_y_labels.npy", train_y)
        np.save(f"{model_prefix}_Train_Predictions_withProbabilities.npy", train_prob)
        np.save(f"{model_prefix}_Test_y_labels.npy", y_true)
        np.save(f"{model_prefix}_Test_Predictions_withProbabilities.npy", y_prob)
        np.save(f"{model_prefix}_Test_Predictions_final.npy", y_pred)

        pd.DataFrame(train_y).to_csv(f"{model_prefix}_Train_y_labels.csv", index=False)
        pd.DataFrame(train_prob).to_csv(f"{model_prefix}_Train_Predictions_withProbabilities.csv", index=False)
        pd.DataFrame(y_true).to_csv(f"{model_prefix}_Test_y_labels.csv", index=False)
        pd.DataFrame(y_prob).to_csv(f"{model_prefix}_Test_Predictions_withProbabilities.csv", index=False)
        pd.DataFrame(y_pred).to_csv(f"{model_prefix}_Test_Predictions_final.csv", index=False)

        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        pd.DataFrame(cm).to_csv(f"{model_prefix}_CM.csv", index=False)
        pd.DataFrame([acc, mae, mape, mse]).to_csv(f"{model_prefix}_Performance_Scores.csv", index=False)

        with open(f"{model_prefix}_eval_metrics.txt", "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"MAPE: {mape:.4f}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

        try:
            
            conn = psycopg2.connect(
                dbname=db_name, user=db_user, password=db_user_password, host="localhost", port="5432"
            )
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ensemble_results (
                    model_id TEXT,
                    dataset TEXT,
                    accuracy FLOAT,
                    mae FLOAT,
                    mape FLOAT,
                    mse FLOAT,
                    confusion_matrix TEXT,
                    epoch INT
                )
            ''')
            cursor.execute('''
                INSERT INTO ensemble_results (model_id, dataset, accuracy, mae, mape, mse, confusion_matrix, epoch)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (self.model_id, self.dataset_name, acc, mae, mape, mse, cm.tolist().__str__(), epoch))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Failed to save to PostgreSQL: {e}")