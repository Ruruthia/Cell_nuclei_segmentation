import numpy as np

def calculate_dice(mask_true, mask_pred, eps=1e-5):
    # cast to binary
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    intersection = np.sum(mask_true & mask_pred)
    return (2 * intersection) / (np.sum(mask_true) + np.sum(mask_pred) + eps)

def calculate_iou(mask_true, mask_pred, eps=1e-5):
    # cast to binary
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    intersection = np.sum(mask_true & mask_pred)
    union = np.sum(mask_true | mask_pred)
    return intersection / (union + eps)

def calculate_map(mask_true, mask_pred, eps=1e-5):
    precision = 0
    classes = np.unique(np.concatenate((mask_true, mask_pred)))
    for class_idx in classes:
        precision += (
            np.sum((mask_true == class_idx) & (mask_pred == class_idx))
            / (np.sum(mask_true == class_idx) + eps)
        )
    return precision/len(classes)
