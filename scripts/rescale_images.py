import click
import pandas as pd
import imageio
import cv2

from pathlib import Path

@click.command()
@click.argument('description_file', type=click.File('r'))
@click.argument('images_dir', type=click.Path(exists=True))
@click.argument('masks_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.option('--scale', type=int, default=20)
def main(description_file, images_dir, masks_dir, output_dir, scale):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    (output_dir / 'img').mkdir(parents=True, exist_ok=True)
    (output_dir / 'mask').mkdir(parents=True, exist_ok=True)

    description_pdf = pd.read_csv(description_file, delimiter=';')
    description_pdf = description_pdf.rename(columns={
        'Image_Name': 'image_name',
        'Magnification': 'magnification'
    })[['image_name', 'magnification']]

    for _, row in description_pdf.iterrows():
        image_name = row['image_name'] + '.tif'
        img = imageio.imread(images_dir / image_name)
        mask = imageio.imread(masks_dir / image_name)

        magnification = row['magnification']
        magnification = int(magnification[:-1])/scale

        new_shape = int(img.shape[1] / magnification), int(img.shape[0] / magnification)
        img_reshaped = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
        mask_reshaped = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)

        imageio.imsave(output_dir / 'img' / image_name, img_reshaped)
        imageio.imsave(output_dir / 'mask' / image_name, mask_reshaped)


if __name__ == '__main__':
    main()
