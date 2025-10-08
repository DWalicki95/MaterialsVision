import logging
from pathlib import Path
from typing import Tuple

import mlflow
from cellpose import io, models, train

from materials_vision.experiments.helpers import load_experiment_config
from materials_vision.experiments.plots import plot_loss
from materials_vision.utils import create_current_time_output_directory


def retrain_cyto(config_path) -> Tuple[str, float, float, str]:
    """
    Allows to retrain cyto3 model.

    Returns
    -------
    Tuple
        (model_path, train_losses, test_losses, run_id)
    """
    # load config:
    config = load_experiment_config(config_path=config_path)

    # logging configuration
    logging.root.handlers.clear()
    io.logger_setup()

    # if any mlflow run is active, end it
    if mlflow.active_run():
        mlflow.end_run()

    # load variables
    output_dir_base = config['general']['output_dir']
    output_dir = create_current_time_output_directory(output_dir_base)

    gpu = config['general']['gpu']
    save_every = config['general']['save_every']

    train_dir = config['dataset']['train_dir']
    test_dir = config['dataset']['test_dir']
    image_filter = config['dataset']['image_filter']
    mask_filter = config['dataset']['mask_filter']
    dataset_name = config['dataset']['name']
    look_one_level_down = config['dataset']['look_one_level_down']
    ds_version = config['dataset']['version']

    model_name = config['training']['model_name']
    weight_decay = config['training']['weight_decay']
    normalize = config['training']['normalize']
    learning_rate = float(config['training']['learning_rate'])
    n_epochs = config['training']['epochs']
    batch_size = config['input']['batch_size']
    compute_flows = config['training']['compute_flows']

    # mlflow initialization
    mlflow.set_experiment(config["logging"]["experiment_name"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        experiment_name = mlflow.get_experiment(experiment_id).name

        print(f"Experiment Name: {experiment_name}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Run ID: {run_id}")
        print(f"Output directory: {output_dir}")

        mlflow.set_tag('Dataset Name', dataset_name)
        mlflow.set_tag('Dataset Version', ds_version)
        mlflow.set_tag('Model Name', model_name)
        mlflow.set_tag('Output Directory', str(output_dir))

        if config_path:
            mlflow.log_artifact(str(config_path), "config")
        else:
            mlflow.log_dict(
                config, "materials_vision/experiments/experiment_config.yaml")

        mlflow.log_param('actual_output_dir', str(output_dir))

        if config["logging"].get("log_system_metrics", False):
            mlflow.enable_system_metrics_logging()

        print('Loading training and testing data...')
        output = io.load_train_test_data(
            train_dir=train_dir,
            test_dir=test_dir,
            image_filter=image_filter,
            mask_filter=mask_filter,
            look_one_level_down=look_one_level_down
        )
        (
            images, labels, image_names, test_images,
            test_labels, image_names_test
        ) = output

        mlflow.log_param('n_train_images', len(images))
        mlflow.log_param('n_test_images', len(test_images))

        if len(images) > 0:
            mlflow.log_param('image_shape', str(images[0].shape))
        else:
            raise ValueError('No training images found!')

        print(f'Initializing {model_name} model')
        model = models.CellposeModel(gpu=gpu, model_type=model_name)

        print('Starting training...')
        model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=images,
            train_labels=labels,
            normalize=normalize,
            test_data=test_images,
            test_labels=test_labels,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            save_path=output_dir,
            compute_flows=compute_flows,
            save_every=save_every,
            model_name=model_name
        )

        mlflow.log_param('model_path', model_path)

        print('Logging metrics...')
        zipped_losses = zip(train_losses, test_losses)
        for epoch, (train_loss, test_loss) in enumerate(zipped_losses):
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('test_loss', test_loss, step=epoch)

        # log final losses
        mlflow.log_metric('final_train_loss', train_losses[-1])
        mlflow.log_metric('final_test_loss', test_losses[-1])
        mlflow.log_metric('best_train_loss', min(train_losses))
        mlflow.log_metric('best_test_loss', min(test_losses))

        print("Creating loss plots...")
        train_losses_plot = plot_loss(n_epochs, train_losses, 'Train')
        test_losses_plot = plot_loss(n_epochs, test_losses, 'Test')
        mlflow.log_figure(train_losses_plot, 'train_losses.png')
        mlflow.log_figure(test_losses_plot, 'test_losses.png')

        print("Logging model artifact...")
        if Path(model_path).exists():
            mlflow.log_artifact(model_path, artifact_path="model")

        print(f"Training complete. Run ID: {run_id}")
        return model_path, train_losses, test_losses, run_id
