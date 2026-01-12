# src/trainer/trainer.py

from __future__ import annotations
"""Generic training loop that can switch backbone automatically.

Usage
-----
>>> trainer = Trainer(train_loader, test_loader, artefacts_dir,
                      dataset_name="cifar10", epochs=30)
>>> best_acc = trainer.fit()

If you *really* want to supply your own model you still can:
>>> trainer = Trainer(train_loader, test_loader, artefacts_dir,
                      model=my_custom_net)
"""

import os
import contextlib
from pathlib import Path
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models
from torchvision.models import EfficientNet

def _make_default_model(dataset: str, custom_cnn_in_channels: int, num_classes: int = 10) -> nn.Module:  # noqa: D401
    """Return a reasonable default network for the given dataset.

    * **MNIST** → small CNN (the old `CustomCNN`).
    * **CIFAR‑10** → EfficientNet‑B0 (224×224) with final layer swapped.
    """
    if dataset.lower() == "cifar10":
        net = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        net._conv_stem = nn.Conv2d(custom_cnn_in_channels, 32, kernel_size=3, stride=1, bias=False)
        in_feat = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_feat, num_classes)    # type: ignore
        return net

    class _CustomCNN(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            self.in_ch = in_ch
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

        def forward(self, x):
            return self.fc_layers(self.conv_layers(x))

    return _CustomCNN(in_ch=custom_cnn_in_channels)

class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        artefact_dir: os.PathLike | str,
        cnn_in_ch: int,
        num_classes: int,
        dataset_name: str = "mnist",
        model: Optional[nn.Module] = None,
        device: torch.device | str | None = None,
        epochs: int = 20,
        save_every: int = 5,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
        lr: float = 1e-3,
    ) -> None:
        if not isinstance(train_dataloader, DataLoader) or not isinstance(test_dataloader, DataLoader):
            raise ValueError("train_dataloader and test_dataloader must be torch.utils.data.DataLoader instances")

        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.dataset = dataset_name.lower()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
        self.artefact_dir = Path(artefact_dir)
        self.artefact_dir.mkdir(parents=True, exist_ok=True)

        self.model: nn.Module = model or _make_default_model(self.dataset, cnn_in_ch, num_classes)
        self.model.to(self.device)

        self.epochs = epochs
        self.save_every = save_every
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimiser, mode="min", patience=scheduler_patience, factor=scheduler_factor, min_lr=min_lr
        )

        if self.dataset == "cifar10":
            self.optimiser = torch.optim.SGD(
                self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimiser, T_max=self.epochs
            )

    def _eval_once(self) -> tuple[float, float]:
        self.model.eval()
        correct = total = 0
        loss_total = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss_total += self.criterion(out, y).item()
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total, loss_total / total

    def fit(self) -> Tuple[float, nn.Module | EfficientNet]:

        use_amp = self.device.type == "cuda"
        null_ctx = contextlib.nullcontext()
        if use_amp:
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)     # type: ignore
            autocast = torch.amp.autocast     # type: ignore
        else:
            scaler = None
            autocast = lambda *a, **k: null_ctx

        progress = tqdm(range(1, self.epochs + 1), ncols=100)
        best_acc = 0.0
        best_model = None

        for epoch in progress:
            self.model.train()
            running_loss, correct, seen = 0.0, 0, 0

            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimiser.zero_grad()

                with autocast('cuda'):
                    out = self.model(xb)
                    loss = self.criterion(out, yb)

                if use_amp:
                    scaler.scale(loss).backward()    # type: ignore
                    scaler.step(self.optimiser)    # type: ignore
                    scaler.update()    # type: ignore
                else:
                    loss.backward()
                    self.optimiser.step()

                running_loss += loss.item()
                correct += (out.argmax(1) == yb).sum().item()
                seen += yb.size(0)

            train_acc = correct / seen
            val_acc, val_loss = self._eval_once()
            self.scheduler.step(val_loss)    # type: ignore

            progress.set_description(f"Ep {epoch}/{self.epochs}")
            progress.set_postfix(train=f"{train_acc:.3f}", val=f"{val_acc:.3f}", lr=f"{self.optimiser.param_groups[0]['lr']:.1e}")

            if epoch % self.save_every == 0:
                torch.save(self.model.state_dict(), self.artefact_dir / f"model_epoch_{epoch}.pt")
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = self.model
                torch.save(self.model.state_dict(), self.artefact_dir / "best_model.pt")

        return best_acc, best_model    # type: ignore