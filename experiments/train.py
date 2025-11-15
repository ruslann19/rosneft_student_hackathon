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
from src.trainer import Trainer
from src.unet import UNet

train_img_path = "data/splits/train/input"
train_mask_path = "data/splits/train/target"

val_img_path = "data/splits/val/input"
val_mask_path = "data/splits/val/target"

train_dataset = SegmentationDataset(img_dir=train_img_path, mask_dir=train_mask_path)
val_dataset = SegmentationDataset(img_dir=val_img_path, mask_dir=val_mask_path)

batch_size = 2
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

model = UNet(
    in_channels=3,
    out_channels=40,
    base_channels=32,
)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, 1e-4)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=DiceLoss(),
    optimizer=optimizer,
)

best_loss = trainer.train(epochs=50)
