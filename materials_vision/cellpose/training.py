from cellpose import io, models, train
from materials_vision.utils import (
    get_train_and_test_dir, create_current_time_output_directory
)
from typing import Optional
import logging


def retrain_cyto(
        train_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        model_name: str = 'cyto3_retrained.pth'
):
    """
    Allows to retrain cyto3 model.

    Parameters
    ----------
    train_dir : Optional[str], optional
        Path to train directory, by default None
    test_dir : Optional[str], optional
        Path to test directory, by default None
    model_name : str, optional
        Retrained model name, by default 'cyto3_retrained'
    """
    logging.root.handlers.clear()
    io.logger_setup()

    output_dir = create_current_time_output_directory()

    if train_dir is None or test_dir is None:
        train_dir, test_dir = get_train_and_test_dir('synthetic_dataset_')

    output = io.load_train_test_data(
        train_dir=train_dir,
        test_dir=test_dir,
        image_filter='_image',
        mask_filter='_masks',
        look_one_level_down=False
    )
    images, labels, image_names, test_images, test_labels, image_names_test = \
        output

    model = models.CellposeModel(gpu=True, model_type='cyto3')

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        channels=[0, 0],
        normalize=True,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=1e-4,
        learning_rate=0.1,
        n_epochs=100,
        batch_size=2,
        save_path=output_dir,
        SGD=False,
        compute_flows=False,
        save_every=100,
        model_name=model_name
    )
    output_dir = create_current_time_output_directory()
    return train_losses, test_losses
