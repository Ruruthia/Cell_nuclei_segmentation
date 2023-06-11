import scipy.ndimage as ndi
import numpy as np

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from stardist.matching import matching


def apply_watershed(semantic_mask: np.ndarray, min_distance: int = 30) -> np.ndarray:
    distance = ndi.distance_transform_edt(semantic_mask)
    coords = peak_local_max(distance, labels=semantic_mask, min_distance=min_distance)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    return watershed(-distance, markers, mask=semantic_mask)


def map_mask(mask, pred):
    matches = matching(mask.astype(int), pred, report_matches=True)
    mapping = np.array(matches.matched_pairs)
    max_mask_class = np.max(mapping[:, 0])
    pred[pred != 0] += max_mask_class
    for mask_val, pred_val in mapping:
        pred[pred == pred_val + max_mask_class] = mask_val
    return pred