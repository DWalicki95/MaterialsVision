"""Example script demonstrating Cellpose prediction workflow."""

from pathlib import Path
from cellpose import models
from cellpose.io import imread

from materials_vision.cellpose.prediction import predict_and_save


def get_img_mask_pairs(
    dataset_path,
    img_extension: str = '.jpg',
    img_suffix='_image',
    mask_extension='.tif',
    mask_suffix='_masks',
    loaded: bool = False
):
    """
    Load image-mask pairs from a directory.

    Parameters
    ----------
    dataset_path : Path
        Directory containing images and masks.
    img_extension : str, optional
        Image file extension (default: '.jpg').
    img_suffix : str, optional
        Suffix before image extension (default: '_image').
    mask_extension : str, optional
        Mask file extension (default: '.tif').
    mask_suffix : str, optional
        Suffix before mask extension (default: '_masks').
    loaded : bool, optional
        If True, load images into memory; if False, return paths
        (default: False).

    Returns
    -------
    dict or list of dict
        If loaded=False: dict mapping image paths to mask paths.
        If loaded=True: list of dicts with 'image' and 'true_mask' arrays.
    """
    images = sorted(dataset_path.rglob(f'*{img_suffix}{img_extension}'))

    img_mask = {}

    for img_path in images:
        base_name = img_path.stem.replace(img_suffix, '')
        mask_path = img_path.parent / f'{base_name}{mask_suffix}{mask_extension}'

        if mask_path.exists():
            img_mask[img_path] = mask_path

    if loaded:
        img_mask = [
            {
                'image': imread(str(img)),
                'true_mask': imread(str(mask))
            }
            for img, mask in img_mask.items()
        ]

    return img_mask


def main():
    """Run Cellpose prediction example."""
    # Configuration
    MODEL_PATH = '/path/to/your/cellpose_model'
    DATASET_PATH = Path('/path/to/your/dataset')
    OUTPUT_DIR = Path('/path/to/output/predictions')

    # Load dataset
    print("Loading dataset...")
    dataset = get_img_mask_pairs(DATASET_PATH, loaded=True)
    print(f"Loaded {len(dataset)} image-mask pairs")

    # Load Cellpose model
    print("Loading Cellpose model...")
    model = models.CellposeModel(
        gpu=True,
        diam_mean=None,
        pretrained_model=MODEL_PATH
    )

    # Make predictions
    print("Running predictions...")
    dataset = predict_and_save(
        dataset=dataset,
        model=model,
        output_dir=OUTPUT_DIR,
        diameter=None,  # Auto-estimate diameter
        channels=[0, 0],  # Grayscale
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=15,
        batch_size=8,
        normalize=True,
        save_predictions=True
    )

    print("Predictions complete!")
    print(f"Results saved to: {OUTPUT_DIR}")

    # Access predictions from dataset
    for idx, sample in enumerate(dataset[:3]):  # Show first 3
        print(f"\nSample {idx}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  True mask shape: {sample['true_mask'].shape}")
        print(f"  Pred mask shape: {sample['pred_mask'].shape}")
        print(f"  Unique labels in prediction: {len(set(sample['pred_mask'].flatten()))}")


if __name__ == '__main__':
    main()
