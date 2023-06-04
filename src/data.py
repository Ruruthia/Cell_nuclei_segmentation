from __future__ import annotations

import numpy as np
import imageio.v2 as imageio

from torch.utils.data import IterableDataset, Dataset
from typing import Union, Optional
from pathlib import Path

class RandomPatchesDataset(IterableDataset):
    def __init__(
        self,
        images_dir: Union[Path, str],
        masks_dir: Union[Path, str],
        patch_size: tuple[int, int] = (256, 256),
        seed: Optional[Union[np.random.Generator, int]] = None,
        ) -> None:

        super().__init__()
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)

        image_names = [file.name for file in images_dir.glob('*.tif')]

        self.data = []
        for name in image_names:
            image = imageio.imread(images_dir / name)
            mask = imageio.imread(masks_dir / name)
            self.data.append((image, mask))

        print(f'Succesfully loaded {len(image_names)} images')
        
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)

    def __next__(self) -> np.ndarray:
        index = self.rng.choice(len(self.data))
        random_image, random_mask = self.data[index]
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
        
        return (random_image_crop[np.newaxis, ...].astype(float),
                random_mask_crop[np.newaxis, ...].astype(float))

    def __iter__(self) -> RandomPatchesDataset:
        return self

class FullImageDataset(Dataset):
    def __init__(
        self,
        images_dir: Union[Path, str],
        masks_dir: Union[Path, str],
        ) -> None:

        super().__init__()
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)

        image_names = [file.name for file in images_dir.glob('*.tif')]

        self.data = []
        for name in image_names:
            image = imageio.imread(images_dir / name)
            mask = imageio.imread(masks_dir / name)
            self.data.append((image, mask))

        print(f'Succesfully loaded {len(image_names)} images')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        img, mask = self.data[index]
        if len(img.shape) == 3:
            # Some images have three channels instead of one, but they all contain the same values
            assert(
                np.all(img[:, :, 0] == img[:, :, 1])
                and np.all(img[:, :, 0] == img[:, :, 2])
            )
            img = img[:, :, 0]

        mask[mask > 0] = 1
        
        return (img[np.newaxis, ...].astype(float),
                mask[np.newaxis, ...].astype(float))
    
    