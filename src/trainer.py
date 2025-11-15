import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        device: torch.device = None,
        save_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x, y in tqdm(self.train_loader, desc="Train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        for x, y in tqdm(self.val_loader, desc="Validation", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, epochs: int = 10, save_best: bool = True):
        best_val_loss = float("inf")
        for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(f"Epoch {epoch:02d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

            # Сохраняем лучшую модель по валидации
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    ckpt_path,
                )
                print(f"Saved best model (val_loss={val_loss:.5f})")

        return best_val_loss
