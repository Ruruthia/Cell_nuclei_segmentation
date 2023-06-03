import click
import pandas as pd
import shutil

from pathlib import Path

@click.command()
@click.argument('description_file', type=click.File('r'))
@click.argument('images_dir', type=click.Path(exists=True))
@click.argument('masks_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.option('--validation_fraction', type=float, default=0.1)
@click.option('--seed', type=int, default=42)
def main(description_file, images_dir, masks_dir, output_dir, validation_fraction, seed):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    description_pdf = pd.read_csv(description_file, delimiter=';')
    description_pdf = description_pdf.rename(columns={
        'Image_Name': 'image_name',
        'Train-/Testset split': 'dataset',
    })[['image_name', 'dataset']]

    train_pdf = description_pdf[description_pdf['dataset'] == 'train']
    test_pdf = description_pdf[description_pdf['dataset'] == 'test']

    val_pdf = train_pdf.sample(frac=validation_fraction, random_state=seed)
    train_pdf = train_pdf[~train_pdf['image_name'].isin(val_pdf['image_name'])]

    for dataset, pdf in zip(('train', 'test', 'val'), (train_pdf, test_pdf, val_pdf)):
        img_dest_dir = output_dir / dataset / 'img'
        mask_dest_dir = output_dir / dataset / 'mask'
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        mask_dest_dir.mkdir(parents=True, exist_ok=True)

        for image_name in pdf['image_name']:
            shutil.copy(images_dir / f'{image_name}.tif', img_dest_dir)
            shutil.copy(masks_dir / f'{image_name}.tif', mask_dest_dir)


if __name__ == '__main__':
    main()
