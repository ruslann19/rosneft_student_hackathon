# split_data.py
import os
import random
from pathlib import Path


def split_data(input_dir, target_dir, val_ratio=0.2, seed=42):
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)

    # Получаем все пары (только по input, проверяя наличие маски)
    pairs = []
    for img_path in input_dir.glob("*.png"):
        mask_path = target_dir / img_path.name
        if mask_path.exists():
            pairs.append(img_path.stem)

    # Сортируем по числу (0,1,2,...,10)
    pairs.sort(key=int)
    random.seed(seed)
    random.shuffle(pairs)  # для случайного разделения

    n_val = int(len(pairs) * val_ratio)
    val_pairs = set(pairs[:n_val])
    train_pairs = set(pairs[n_val:])

    # Создаём структуру
    for split, pairs_set in [("train", train_pairs), ("val", val_pairs)]:
        os.makedirs(f"data/splits/{split}/input", exist_ok=True)
        os.makedirs(f"data/splits/{split}/target", exist_ok=True)

        # Символические ссылки
        for stem in pairs_set:
            # input
            src = input_dir / f"{stem}.png"
            dst = Path(f"data/splits/{split}/input/{stem}.png")
            if not dst.exists():
                dst.symlink_to(src.resolve())
            # target
            src = target_dir / f"{stem}.png"
            dst = Path(f"data/splits/{split}/target/{stem}.png")
            if not dst.exists():
                dst.symlink_to(src.resolve())

    print(f"Разделение завершено: {len(train_pairs)} train, {len(val_pairs)} val")


if __name__ == "__main__":
    split_data("data/input", "data/target", val_ratio=0.2)
