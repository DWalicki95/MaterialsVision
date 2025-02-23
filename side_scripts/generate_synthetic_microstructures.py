from materials_vision.artificial_dataset.synthetic_microstructures import \
    SyntheticMicrostructuresGenerator

if __name__ == '__main__':
    # choose number of samples you want to generate
    n_samples = 10000

    artificial_structures_manager = SyntheticMicrostructuresGenerator(
        n_samples=n_samples
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
