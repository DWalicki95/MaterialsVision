import argparse
from pathlib import Path

from materials_vision.artificial_dataset.synthetic_microstructures import \
    SyntheticMicrostructuresGenerator
from materials_vision.experiments.cellpose.utils import \
    split_into_train_and_test_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--dataset_name", type=str, default='')
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--split_output_root", type=Path, required=True)
    parser.add_argument("--split_dataset_name", type=str,
                        default='synthetic_dataset_4000')
    parser.add_argument("--split_subset_size", type=int, default=4000)
    parser.add_argument("--visualization_type", type=str, default='all')
    args = parser.parse_args()

    artificial_structures_manager = SyntheticMicrostructuresGenerator(
        n_samples=args.n_samples,
        dataset_name=args.dataset_name
    )
    img_data_dictionary = (
        artificial_structures_manager.generate_artificial_microstructures(
            save=args.save,
            save_path=args.save_path
        )
    )
    artificial_structures_manager.visualize_pores_mask(
        img_data_dictionary,
        visualization_type=args.visualization_type
    )

    # organize files for cellpose model training
    dataset_path = (
        args.save_path /
        f'synthetic_dataset_{args.dataset_name}'
    )
    split_into_train_and_test_directory(
        dataset_store_path=dataset_path,
        output_root=args.split_output_root,
        dataset_name=args.split_dataset_name,
        subset_size=args.split_subset_size
    )
