from __future__ import annotations

import numpy as np
import imageio.v2 as imageio

from torch.utils.data import IterableDataset
from typing import Union, Optional
from pathlib import Path

class RandomPatchesDataset(IterableDataset):
    def __init__(
        self,
        images_dir: Union[Path, str],
        masks_dir: Union[Path, str],
        patch_size: tuple[int, int] = (256, 256),
        rng: Optional[Union[np.random.Generator, int]] = None,
        ) -> None:

        super().__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.image_names = [file.name for file in self.images_dir.glob('*.tif')]
        
        self.patch_size = patch_size
        self.rng = np.random.default_rng(rng)

    def __next__(self) -> np.ndarray:
        random_name = self.rng.choice(self.image_names)
        random_image = imageio.imread(self.images_dir / random_name)
        random_mask = imageio.imread(self.masks_dir / random_name)

        if len(random_image.shape) == 3:
            # Some images have three channels instead of one, but they all contain the same values
            assert(
                np.all(random_image[:, :, 0] == random_image[:, :, 1])
                and np.all(random_image[:, :, 0] == random_image[:, :, 2])
            )
            random_image = random_image[:, :, 0]
        
        crop_y = np.random.randint(0, random_image.shape[0] - self.patch_size[0])
        crop_x = np.random.randint(0, random_image.shape[1] - self.patch_size[1])

        random_image_crop = random_image[
            crop_y: crop_y + self.patch_size[0],
            crop_x: crop_x + self.patch_size[1],
            ]
        random_mask_crop = random_mask[
            crop_y: crop_y + self.patch_size[0],
            crop_x: crop_x + self.patch_size[1],
            ]

        random_mask_crop[random_mask_crop > 0] = 1
        
        return random_image_crop, random_mask_crop

    def __iter__(self) -> RandomPatchesDataset:
        return self
