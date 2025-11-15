import logging
import os
import random
from datetime import datetime
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import REVERSE_LOOKUP


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # для GPU
    torch.cuda.manual_seed_all(seed)  # для многопроцессорности на GPU
    np.random.seed(seed)
    random.seed(seed)
    # Для DataLoader
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Callable,
        metrics: Callable,
        optimizer,
        num_classes: int,
        device: str,
        seed: int,
        log_dir="logs",
        save_dir="checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.num_classes = num_classes
        self.device = device

        if seed is not None:
            set_seed(seed)

        self.log_dir = log_dir
        self.save_dir = save_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # Логирование
        self.logger = self._setup_logger()
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        # self.reverse_lookup = np.zeros(256, dtype=np.int64)
        # for cls, gray in CLASS_TO_GRAY.items():
        #     self.reverse_lookup[cls] = gray  # индекс класса -> градация серого

    def _setup_logger(self):
        logger = logging.getLogger("Trainer")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.log_dir, "training.log"))
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_metrics = 0
        for images, masks in tqdm(self.train_loader, desc="Train", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            total_metrics += self.metrics(preds, masks, self.num_classes)

        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = total_metrics / len(self.train_loader)
        return avg_loss, avg_metrics

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        total_loss = 0
        total_metrics = 0
        for images, masks in tqdm(self.val_loader, desc="Validation", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            total_metrics += self.metrics(preds, masks, self.num_classes)

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = total_metrics / len(self.val_loader)
        return avg_loss, avg_metrics

    def visualize_samples(self, epoch):
        """Сохраняет визуализации предсказаний для нескольких примеров"""
        self.model.eval()

        with torch.no_grad():
            images, masks = next(iter(self.val_loader))

            batch_size = images.shape[0]
            examples_count = 4
            if batch_size < examples_count:
                examples_count = batch_size

            images, masks = images[:examples_count].to(self.device), masks[
                :examples_count
            ].to(self.device)
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        columns_count = 3
        fig, axes = plt.subplots(examples_count, columns_count, figsize=(12, 16))
        axes = axes.reshape(examples_count, columns_count)
        for i in range(examples_count):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            mask = masks[i].cpu().numpy()
            pred = preds[i]

            mask = REVERSE_LOOKUP[mask]
            pred = REVERSE_LOOKUP[pred]

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
            axes[i, 1].set_title("True Mask")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=255)
            axes[i, 2].set_title("Pred Mask")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"epoch_{epoch}_samples.png"))
        plt.close()

    def train(self, epochs):
        for epoch in tqdm(range(epochs), desc="Epoch", leave=False):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.val_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)

            self.logger.info(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Metrics: {train_metrics:.4f}, Val Metrics: {val_metrics:.4f}"
            )

            # Визуализация каждую эпоху
            self.visualize_samples(epoch)

            # Сохранение чекпоинта
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                    "train_metrics": self.train_metrics,
                    "val_metrics": self.val_metrics,
                },
                os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

        print("Training finished.")
