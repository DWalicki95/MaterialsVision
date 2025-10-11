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
    Retrain cyto3 model with proper MLflow tracking.

    Returns
    -------
    Tuple
        (model_path, train_losses, test_losses, run_id)
    """
    # load config
    config = load_experiment_config(config_path=config_path)

    # logging configuration
    logging.root.handlers.clear()
    io.logger_setup()

    # end any active MLflow runs
    if mlflow.active_run():
        mlflow.end_run()

    # load variables
    output_dir_base = config['general']['output_dir']
    output_dir = create_current_time_output_directory(output_dir_base)
    output_dir = Path(output_dir)  # Convert to Path object
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # MLflow initialization - set tracking URI if needed
    # mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment(config["logging"]["experiment_name"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        experiment_name = mlflow.get_experiment(experiment_id).name

        print(f"Experiment Name: {experiment_name}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Run ID: {run_id}")
        print(f"Output directory: {output_dir}")

        # enable system metrics
        if config["logging"].get("log_system_metrics", False):
            mlflow.enable_system_metrics_logging()
            print("System metrics logging enabled")

        # log tags
        mlflow.set_tag('Dataset Name', dataset_name)
        mlflow.set_tag('Dataset Version', ds_version)
        mlflow.set_tag('Model Name', model_name)
        mlflow.set_tag('Output Directory', str(output_dir))

        # log all parameters from config
        mlflow.log_params({
            'gpu': gpu,
            'save_every': save_every,
            'weight_decay': weight_decay,
            'normalize': normalize,
            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'compute_flows': compute_flows,
            'look_one_level_down': look_one_level_down,
            'output_dir': str(output_dir)
        })

        # log config file as artifact
        if config_path and Path(config_path).exists():
            mlflow.log_artifact(str(config_path), artifact_path="config")
        else:
            # save config as YAML and log it
            config_file_path = output_dir / "experiment_config.yaml"
            import yaml
            with open(config_file_path, 'w') as f:
                yaml.dump(config, f)
            mlflow.log_artifact(str(config_file_path), artifact_path="config")

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

        # log dataset statistics
        mlflow.log_params({
            'n_train_images': len(images),
            'n_test_images': len(test_images),
        })

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
            save_path=str(output_dir),
            compute_flows=compute_flows,
            save_every=save_every,
            model_name=model_name
        )

        print('Logging metrics...')
        # log losses per epoch
        enumerated_zippped_losses = enumerate(zip(train_losses, test_losses))
        for epoch, (train_loss, test_loss) in enumerated_zippped_losses:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'loss_diff': abs(train_loss - test_loss)
            }, step=epoch)

        # log summary metrics
        mlflow.log_metrics({
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'best_train_loss': min(train_losses),
            'best_test_loss': min(test_losses),
            'avg_train_loss': sum(train_losses) / len(train_losses),
            'avg_test_loss': sum(test_losses) / len(test_losses),
        })

        print("Creating and logging loss plots...")
        # create plots
        train_losses_plot = plot_loss(n_epochs, train_losses, 'Train')
        test_losses_plot = plot_loss(n_epochs, test_losses, 'Test')

        # save plots to disk first (for backup)
        train_plot_path = output_dir / 'train_losses.png'
        test_plot_path = output_dir / 'test_losses.png'
        train_losses_plot.savefig(
            train_plot_path, dpi=150, bbox_inches='tight')
        test_losses_plot.savefig(test_plot_path, dpi=150, bbox_inches='tight')

        # log as figures
        mlflow.log_figure(train_losses_plot, 'train_losses.png')
        mlflow.log_figure(test_losses_plot, 'test_losses.png')

        # also log as artifacts for download
        mlflow.log_artifact(str(train_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(test_plot_path), artifact_path="plots")

        print("Logging model artifact...")
        model_path_obj = Path(model_path)
        if model_path_obj.exists():
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.log_param('model_path', str(model_path))

            # log checkpoints if exists
            checkpoint_dir = output_dir / "models"
            if checkpoint_dir.exists():
                for checkpoint in checkpoint_dir.glob("*.pth"):
                    if checkpoint != model_path_obj:
                        mlflow.log_artifact(
                            str(checkpoint),
                            artifact_path="models/checkpoints"
                        )
        else:
            logging.warning(f"Model path {model_path} does not exist!")

        # log all training outputs as artifacts
        print("Logging additional artifacts from output directory...")
        for file_path in output_dir.rglob('*'):
            if file_path.is_file() and file_path != model_path_obj:
                # clculate relative path for organized artifact storage
                relative_path = file_path.relative_to(output_dir)
                artifact_subpath = str(
                    relative_path.parent
                ) if relative_path.parent != Path('.') else "outputs"

                try:
                    mlflow.log_artifact(
                        str(file_path), artifact_path=artifact_subpath)
                except Exception as e:
                    logging.warning(f"Could not log artifact {file_path}: {e}")

        print(f"Training complete. Run ID: {run_id}")
        return str(model_path), train_losses, test_losses, run_id