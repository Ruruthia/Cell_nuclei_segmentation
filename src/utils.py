import scipy.ndimage as ndi
import numpy as np

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def apply_watershed(semantic_mask: np.ndarray) -> np.ndarray:
    distance = ndi.distance_transform_edt(semantic_mask)
    coords = peak_local_max(distance, footprint=np.ones((3,3)), labels=semantic_mask)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    return watershed(-distance, markers, mask=semantic_mask)