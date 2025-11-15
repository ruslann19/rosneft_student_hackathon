import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Логиты (B, C, H, W)
            target (torch.Tensor): Истинные метки (B, H, W) — индексы классов
        """
        # Применяем softmax к предсказаниям
        pred_soft = F.softmax(pred, dim=1)  # (B, C, H, W)

        # Создаём one-hot из target: (B, C, H, W)
        target_one_hot = (
            F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        )

        # Вычисляем пересечение и union для каждого класса
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))  # (B, C)
        total = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (B, C)

        dice = (2.0 * intersection + self.eps) / (total + self.eps)  # (B, C)

        # Усредняем по классам и батчу
        loss = 1.0 - dice.mean()
        return loss
