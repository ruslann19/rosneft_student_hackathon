import sys
from pathlib import Path

src_path = str(Path(".").resolve())

if src_path not in sys.path:
    sys.path.append(src_path)
    print(f"Добавлен путь: {src_path}")
else:
    print(f"Путь уже в sys.path: {src_path}")

import argparse
import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import REVERSE_LOOKUP, SegmentationDataset
from src.models import UNet


def predict(model, x) -> np.ndarray:
    outputs = model(x)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

    return preds


def resize_up(masks: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    # masks: [B, H, W]
    resized_masks_list = []
    for mask in masks:
        # mask: (H, W)
        resized_mask = cv2.resize(
            mask, dsize=new_size, interpolation=cv2.INTER_NEAREST
        )  # Используем INTER_NEAREST для масок
        resized_masks_list.append(resized_mask)

    # Собираем обратно в батч
    resized_masks_np = np.stack(resized_masks_list, axis=0)  # (B, H_new, W_new)
    return resized_masks_np


def run_prediction(predict_dir: str, model_name: str) -> None:
    model = UNet(
        in_channels=3,
        out_channels=40,
    )

    checkpoint_path = f"saved_models/{model_name}.pth"
    checkpoint = torch.load(
        f=checkpoint_path, map_location="cpu", weights_only=False
    )  # или 'cuda' если GPU
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset_mode = "test"
    test_img_path = f"data/{predict_dir}"

    test_dataset = SegmentationDataset(
        mode=dataset_mode,
        img_dir=test_img_path,
    )

    x_0 = test_dataset[0]
    # print("x_0.shape:", x_0.shape)

    source_shape = (x_0.shape[1], x_0.shape[2])
    working_shape = (640, 640)
    # print("source_shape:", source_shape)
    # print("working_shape:", working_shape)

    batch_size = 1
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    results_dir = "results"
    predicted_masks_dir = f"{results_dir}/predict_target"
    visualization_dir = f"{results_dir}/visualization"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predicted_masks_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    for i, x in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
        # print("x.shape:", x.shape)
        x = F.interpolate(x, size=working_shape, mode="bilinear", align_corners=False)
        # print("x.shape (working):", x.shape)

        preds = predict(
            model=model,
            x=x,
        )
        # print("preds.shape (working):", preds.shape)
        preds = resize_up(masks=preds, new_size=source_shape)
        # print("x.shape (source):", preds.shape)

        batch_size = len(x)
        for j in range(batch_size):
            img = x[j].cpu().permute(1, 2, 0).numpy()
            pred = preds[j]
            pred = REVERSE_LOOKUP[pred]
            pred = pred.clip(0, 255).astype(np.uint8)

            cv2.imwrite(filename=f"{predicted_masks_dir}/{i + j}.png", img=pred)

            columns_count = 2
            examples_count = 1
            fig, axes = plt.subplots(examples_count, columns_count, figsize=(12, 16))
            axes = axes.reshape(examples_count, columns_count)
            for k in range(examples_count):

                axes[k, 0].imshow(img)
                axes[k, 0].set_title("Image")
                axes[k, 0].axis("off")

                axes[k, 1].imshow(pred, cmap="gray", vmin=0, vmax=255)
                axes[k, 1].set_title("Pred Mask")
                axes[k, 1].axis("off")

            plt.tight_layout()
            plt.savefig(f"{visualization_dir}/{i + j}.png")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Обучение модели")
    parser.add_argument(
        "--predict_dir",
        type=str,
        default="predict_images",
        help="Папка с тестовыми данными",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Название файла, в котором сохранени обученная модель",
    )

    args = parser.parse_args()

    run_prediction(predict_dir=args.predict_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()
