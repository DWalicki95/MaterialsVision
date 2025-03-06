from materials_vision.artificial_dataset.synthetic_microstructures import \
    SyntheticMicrostructuresGenerator
from materials_vision.cellpose.utils import split_into_train_and_test_directory
from materials_vision.config import SYNTHETIC_DATASET_PATH_LOCAL_DRIVE


if __name__ == '__main__':
    # choose number of samples you want to generate
    n_samples = 10000
    dataset_name = ''  # suffix could be changed

    artificial_structures_manager = SyntheticMicrostructuresGenerator(
        n_samples=n_samples,
        dataset_name=dataset_name
    )
    img_data_dictionary = (
        artificial_structures_manager.generate_artificial_microstructures(
            save=True
        )
    )
    artificial_structures_manager.visualize_pores_mask(
        img_data_dictionary,
        visualization_type='all'
    )

    # organize files for cellpose modeel traning
    dataset_path = (
        SYNTHETIC_DATASET_PATH_LOCAL_DRIVE /
        f'synthetic_dataset_{dataset_name}'
    )
    split_into_train_and_test_directory(dataset_path)
