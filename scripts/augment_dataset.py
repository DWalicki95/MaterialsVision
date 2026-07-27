import argparse
from materials_vision.image_preprocessing import augment_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Augment image dataset with various transformations'
    )

    # Required arguments
    parser.add_argument(
        '--input-dir',
        type=str,
        default="/workspace/dane/train",
        help='Input directory containing images and masks'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default="/workspace/dane/train_augmented",
        help='Output directory for augmented dataset'
    )

    # Optional arguments
    parser.add_argument(
        '--image-suffix',
        type=str,
        default='_image.jpg',
        help='Suffix for image files (default: _image.jpg)'
    )

    parser.add_argument(
        '--mask-suffix',
        type=str,
        default='_masks.tif',
        help='Suffix for mask files (default: _masks.tif)'
    )

    # Boolean flags for augmentation options
    parser.add_argument(
        '--no-save-dataset',
        action='store_false',
        dest='save_dataset',
        help='Do not save the augmented dataset'
    )

    parser.add_argument(
        '--no-rotation',
        action='store_false',
        dest='apply_rotation',
        help='Disable rotation augmentation'
    )

    parser.add_argument(
        '--no-flip',
        action='store_false',
        dest='apply_flip',
        help='Disable flip augmentation'
    )

    parser.add_argument(
        '--no-contrast',
        action='store_false',
        dest='apply_contrast',
        help='Disable contrast augmentation'
    )

    parser.add_argument(
        '--no-gamma',
        action='store_false',
        dest='apply_gamma',
        help='Disable gamma augmentation'
    )

    parser.add_argument(
        '--apply-poisson',
        action='store_true',
        dest='apply_poisson',
        help='Enable Poisson noise augmentation (disabled by default)'
    )

    # Set defaults for boolean flags
    parser.set_defaults(
        save_dataset=True,
        apply_rotation=True,
        apply_flip=True,
        apply_contrast=True,
        apply_gamma=True,
        apply_poisson=False
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_suffix=args.image_suffix,
        mask_suffix=args.mask_suffix,
        save_dataset=args.save_dataset,
        apply_rotation=args.apply_rotation,
        apply_flip=args.apply_flip,
        apply_contrast=args.apply_contrast,
        apply_gamma=args.apply_gamma,
        apply_poisson=args.apply_poisson
    )
