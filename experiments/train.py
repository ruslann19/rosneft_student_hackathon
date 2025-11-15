import argparse
import sys
from pathlib import Path

src_path = str(Path(".").resolve())

if src_path not in sys.path:
    sys.path.append(src_path)
    print(f"Добавлен путь: {src_path}")
else:
    print(f"Путь уже в sys.path: {src_path}")

import torch
from torch.utils.data import DataLoader

from src.dataset import SegmentationDataset
from src.dice_loss import DiceLoss
from src.iou import compute_miou
from src.models import SegNet, UNet
from src.trainer import Trainer


def train(split_dir: str) -> None:
    print(f"split_dir: {split_dir}")

    train_img_path = f"data/{split_dir}/train/images"
    train_mask_path = f"data/{split_dir}/train/maskes"

    val_img_path = f"data/{split_dir}/val/images"
    val_mask_path = f"data/{split_dir}/val/maskes"

    train_dataset = SegmentationDataset(
        img_dir=train_img_path, mask_dir=train_mask_path
    )
    val_dataset = SegmentationDataset(img_dir=val_img_path, mask_dir=val_mask_path)

    batch_size = 2
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    # model = UNet(
    #     in_channels=3,
    #     out_channels=40,
    #     base_channels=32,
    # )

    model = UNet(
        in_channels=3,
        out_channels=40,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, 1e-4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics=compute_miou,
        criterion=DiceLoss(num_classes=40),
        optimizer=optimizer,
        num_classes=40,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )

    best_loss = trainer.train(epochs=50)


def main():
    parser = argparse.ArgumentParser(description="Обучение модели")
    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits",
        help="Папка с разделением на train/val",
    )

    args = parser.parse_args()

    train(split_dir=args.split_dir)


if __name__ == "__main__":
    main()
