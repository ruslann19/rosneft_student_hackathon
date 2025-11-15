#!/usr/bin/env python3
# tools/split_data.py
import argparse
import os
import random
from pathlib import Path


def split_data(
    images_dir,
    maskes_dir,
    train_size=None,
    val_size=None,
    val_ratio=0.2,
    seed=42,
    output_dir="data/splits",
):
    # (ваша существующая логика без изменений)
    images_dir = Path(images_dir)
    maskes_dir = Path(maskes_dir)

    # Собираем пары
    pairs = []
    for img_path in images_dir.glob("*.png"):
        mask_path = maskes_dir / img_path.name
        if mask_path.exists():
            pairs.append(img_path.stem)

    pairs.sort(key=int)
    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)

    # Определяем размеры (как в предыдущей версии)
    if train_size is not None and val_size is not None:
        if train_size + val_size > total:
            raise ValueError(
                f"train_size + val_size ({train_size}+{val_size}) > всего файлов ({total})"
            )
        train_pairs = set(pairs[:train_size])
        val_pairs = set(pairs[train_size : train_size + val_size])

    elif train_size is not None:
        max_val = int(total * val_ratio)
        val_n = min(total - train_size, max_val)
        if train_size + val_n > total:
            val_n = total - train_size
        train_pairs = set(pairs[:train_size])
        val_pairs = set(pairs[train_size : train_size + val_n])

    elif val_size is not None:
        min_train = int(total * (1 - val_ratio))
        train_n = max(min_train, total - val_size)
        train_pairs = set(pairs[:train_n])
        val_pairs = set(pairs[train_n : train_n + val_size])

    else:
        n_val = int(total * val_ratio)
        val_pairs = set(pairs[:n_val])
        train_pairs = set(pairs[n_val:])

    # Создаём структуру
    for split, pairs_set in [("train", train_pairs), ("val", val_pairs)]:
        split_input = Path(output_dir) / split / images_dir.name
        split_target = Path(output_dir) / split / maskes_dir.name
        split_input.mkdir(parents=True, exist_ok=True)
        split_target.mkdir(parents=True, exist_ok=True)

        for stem in pairs_set:
            # input
            src = images_dir / f"{stem}.png"
            dst = split_input / f"{stem}.png"
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    import shutil

                    shutil.copy2(src, dst)
            # target
            src = maskes_dir / f"{stem}.png"
            dst = split_target / f"{stem}.png"
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    import shutil

                    shutil.copy2(src, dst)

    print(
        f"✅ Разделение: {len(train_pairs)} train, {len(val_pairs)} val (всего {total})"
    )
    return train_pairs, val_pairs


def main():
    parser = argparse.ArgumentParser(description="Разделение данных на train/val")
    parser.add_argument(
        "--images_dir", type=str, default="data/images", help="Папка с изображениями"
    )
    parser.add_argument(
        "--maskes_dir", type=str, default="data/maskes", help="Папка с масками"
    )
    parser.add_argument("--train_size", type=int, help="Количество изображений в train")
    parser.add_argument("--val_size", type=int, help="Количество изображений в val")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Доля val (если train/val_size не заданы)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Сид для воспроизводимости"
    )
    parser.add_argument(
        "--split_dir", type=str, default="splits", help="Папка для результата"
    )

    args = parser.parse_args()

    split_data(
        images_dir=args.images_dir,
        maskes_dir=args.maskes_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        output_dir=f"data/{args.split_dir}",
    )


if __name__ == "__main__":
    main()
