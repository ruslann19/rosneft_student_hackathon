import numpy as np
import torch


def compute_iou(pred_mask, true_mask, eps=1e-6):
    """
    Вычисляет IoU (Intersection over Union) между предсказанной и истинной масками.

    Args:
        pred_mask (np.ndarray): Предсказанная маска (bool или 0/1)
        true_mask (np.ndarray): Истинная маска (bool или 0/1)
        eps (float): Маленькое значение для избежания деления на 0

    Returns:
        float: IoU
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    iou = intersection / (union + eps)
    return iou


def compute_class_iou(pred_mask, true_mask, class_id, eps=1e-6):
    """
    Вычисляет IoU для конкретного класса.
    """
    pred_class = pred_mask == class_id
    true_class = true_mask == class_id

    intersection = np.logical_and(pred_class, true_class).sum()
    union = np.logical_or(pred_class, true_class).sum()

    return intersection / (union + eps)


@torch.no_grad()
def compute_miou(pred_mask, true_mask, num_classes, eps=1e-6):
    """
    Вычисляет средний IoU по всем классам (mIoU).
    """
    iou_sum = 0
    for class_id in range(num_classes):
        iou_sum += compute_class_iou(pred_mask, true_mask, class_id, eps)
    return iou_sum / num_classes
