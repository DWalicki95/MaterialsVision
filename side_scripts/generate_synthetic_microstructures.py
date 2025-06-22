import argparse
from pathlib import Path

from materials_vision.artificial_dataset.synthetic_microstructures import \
    SyntheticMicrostructuresGenerator
from materials_vision.cellpose.utils import split_into_train_and_test_directory
from materials_vision.config import SYNTHETIC_DATASET_PATH_LOCAL_DRIVE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--dataset_name", type=str, default='')
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument(
        "--verbose", action="save_path", type=Path,
        default=SYNTHETIC_DATASET_PATH_LOCAL_DRIVE
    )
    parser.add_argument("--visualization_type", type=str, default='all')
    args = parser.parse_args()

    # choose number of samples you want to generate
    n_samples = 10000
    dataset_name = ''  # suffix could be changed
    save_path = SYNTHETIC_DATASET_PATH_LOCAL_DRIVE

    artificial_structures_manager = SyntheticMicrostructuresGenerator(
        n_samples=n_samples,
        dataset_name=dataset_name
    )
    img_data_dictionary = (
        artificial_structures_manager.generate_artificial_microstructures(
            save=True,
            save_path=save_path
        )
    )
    artificial_structures_manager.visualize_pores_mask(
        img_data_dictionary,
        visualization_type='all'
    )

    # organize files for cellpose modeel traning
    dataset_path = (
        save_path /
        f'synthetic_dataset_{dataset_name}'
    )
    split_into_train_and_test_directory(
        dataset_store_path=dataset_path,
        dataset_name='synthetic_dataset_4000',
        subset_size=4000
    )
