import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: [B, C, H, W], float — выход модели (без softmax!)
        target: [B, H, W], long — индексы классов
        """
        # 1. Вероятности через softmax (для многоклассового Dice)
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # 2. One-hot target (но НЕ сохраняем в память — создаём on-the-fly)
        if self.ignore_index is not None:
            # Маска валидных пикселей
            valid_mask = target != self.ignore_index  # [B, H, W]
            target = target.clone()
            target[~valid_mask] = 0  # временно заменяем на 0 (чтобы one_hot не упал)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        target_one_hot = F.one_hot(target, num_classes=probs.shape[1])  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        if self.ignore_index is not None:
            # Применяем маску: игнорируемые пиксели → 0 в one-hot
            target_one_hot *= valid_mask.unsqueeze(1)  # [B, C, H, W]

        # 3. Dice по каждому классу (кроме игнорируемых)
        intersection = (probs * target_one_hot).sum(dim=(2, 3))  # [B, C]
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # [B, C]

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Средний Dice по классам и батчу
        # Можно исключить фон (dice[:, 0]) — зависит от задачи
        return 1.0 - dice.mean()
