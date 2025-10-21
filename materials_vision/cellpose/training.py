import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Tuple, Optional

import mlflow
import psutil
from cellpose import io, models, train

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning(
        "GPUtil not installed. GPU monitoring disabled. "
        "Install with: pip install gputil"
    )

from materials_vision.experiments.helpers import load_experiment_config
from materials_vision.experiments.plots import plot_loss
from materials_vision.utils import create_current_time_output_directory


class SystemMonitor:
    """Monitor system resources (CPU, RAM, GPU) during training."""

    def __init__(self, interval: float = 10.0, log_to_mlflow: bool = True):
        """
        Initialize system monitor.

        Parameters
        ----------
        interval : float
            Monitoring interval in seconds (default: 10s)
        log_to_mlflow : bool
            Whether to log metrics to MLflow
        """
        self.interval = interval
        self.log_to_mlflow = log_to_mlflow
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.metrics_history = {
            'cpu_percent': [],
            'ram_percent': [],
            'ram_used_gb': [],
            'gpu_percent': [],
            'gpu_memory_percent': [],
            'gpu_memory_used_gb': [],
            'timestamp': []
        }

    def _monitor_loop(self):
        """Internal monitoring loop."""
        step = 0
        while self.monitoring:
            try:
                timestamp = time.time()

                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)

                # RAM metrics
                ram = psutil.virtual_memory()
                ram_percent = ram.percent
                ram_used_gb = ram.used / (1024 ** 3)

                # Store metrics
                self.metrics_history['cpu_percent'].append(cpu_percent)
                self.metrics_history['ram_percent'].append(ram_percent)
                self.metrics_history['ram_used_gb'].append(ram_used_gb)
                self.metrics_history['timestamp'].append(timestamp)

                # GPU metrics (if available)
                gpu_percent = 0
                gpu_memory_percent = 0
                gpu_memory_used_gb = 0

                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # use first GPU
                            gpu_percent = gpu.load * 100
                            gpu_memory_percent = gpu.memoryUtil * 100
                            gpu_memory_used_gb = gpu.memoryUsed / 1024

                            self.metrics_history['gpu_percent'].append(
                                gpu_percent)
                            self.metrics_history['gpu_memory_percent'].append(
                                gpu_memory_percent)
                            self.metrics_history['gpu_memory_used_gb'].append(
                                gpu_memory_used_gb)
                    except Exception as e:
                        logging.debug(f"GPU monitoring error: {e}")

                # Log to MLflow
                if self.log_to_mlflow and mlflow.active_run():
                    mlflow.log_metrics({
                        'system/cpu_percent': cpu_percent,
                        'system/ram_percent': ram_percent,
                        'system/ram_used_gb': ram_used_gb,
                        'system/gpu_percent': gpu_percent,
                        'system/gpu_memory_percent': gpu_memory_percent,
                        'system/gpu_memory_used_gb': gpu_memory_used_gb,
                    }, step=step)

                step += 1
                time.sleep(self.interval)

            except Exception as e:
                logging.error(f"System monitoring error: {e}")
                time.sleep(self.interval)

    def start(self):
        """Start monitoring in background thread."""
        if self.monitoring:
            logging.warning("Monitor already running!")
            return

        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logging.info(f"System monitoring started (interval: {self.interval}s)")

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5)
        logging.info("System monitoring stopped")

    def get_summary_stats(self) -> dict:
        """Get summary statistics of monitored metrics."""
        stats = {}

        for metric_name, values in self.metrics_history.items():
            if metric_name == 'timestamp' or not values:
                continue

            stats[f'{metric_name}_avg'] = sum(values) / len(values)
            stats[f'{metric_name}_max'] = max(values)
            stats[f'{metric_name}_min'] = min(values)

        return stats


def retrain_cyto(config_path) -> Tuple[str, float, float, str]:
    """
    Retrain cyto3 model with proper MLflow tracking and system monitoring.

    Returns
    -------
    Tuple
        (cellpose_model_path, train_losses, test_losses, run_id)
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
    output_dir = Path(output_dir)
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

    # System monitoring configuration
    enable_monitoring = config.get("logging", {}).get(
        "enable_system_monitoring", True)
    monitoring_interval = config.get("logging", {}).get(
        "monitoring_interval", 10.0)

    mlflow.set_experiment(config["logging"]["experiment_name"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        experiment_name = mlflow.get_experiment(experiment_id).name

        logging.info(f"Experiment Name: {experiment_name}")
        logging.info(f"Experiment ID: {experiment_id}")
        logging.info(f"Run ID: {run_id}")
        logging.info(f"Output directory: {output_dir}")

        # initialize system monitor
        monitor = None
        if enable_monitoring:
            monitor = SystemMonitor(
                interval=monitoring_interval,
                log_to_mlflow=True
            )
            monitor.start()
            logging.info(
                f"System monitoring enabled (interval: {monitoring_interval}s)"
            )

        try:
            # log system info at start
            cpu_count = psutil.cpu_count()
            ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)

            mlflow.log_params({
                'system/cpu_count': cpu_count,
                'system/ram_total_gb': round(ram_total_gb, 2),
            })

            if GPU_AVAILABLE and gpu:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_info = gpus[0]
                        mlflow.log_params({
                            'system/gpu_name': gpu_info.name,
                            'system/gpu_memory_total_gb': round(
                                gpu_info.memoryTotal / 1024, 2),
                        })
                        logging.info(
                            f"GPU: {gpu_info.name} "
                            f"({gpu_info.memoryTotal / 1024:.1f} GB)")
                except Exception as e:
                    logging.warning(f"Could not log GPU info: {e}")

            # Enable MLflow system metrics (built-in)
            if config.get("logging", {}).get("log_system_metrics", False):
                mlflow.enable_system_metrics_logging()
                logging.info("MLflow system metrics logging enabled")

            # log tags
            mlflow.set_tag('Dataset Name', dataset_name)
            mlflow.set_tag('Dataset Version', ds_version)
            mlflow.set_tag('Model Name', model_name)
            mlflow.set_tag('Output Directory', str(output_dir))
            mlflow.set_tag('Run ID', run_id)

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
                config_file_path = output_dir / "experiment_config.yaml"
                import yaml
                with open(config_file_path, 'w') as f:
                    yaml.dump(config, f)
                mlflow.log_artifact(
                    str(config_file_path), artifact_path="config")

            logging.info('Loading training and testing data...')
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

            logging.info(f'Initializing {model_name} model')
            model = models.CellposeModel(gpu=gpu, model_type=model_name)

            logging.info('Starting training...')
            # Train and save to output_dir
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

            logging.info(f'Training complete. Model saved to: {model_path}')

            # CRITICAL: Copy model to .cellpose/models/ directory
            cellpose_models_dir = Path.home() / '.cellpose' / 'models'
            cellpose_models_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cellpose_model_name = f"{model_name}_{dataset_name}_{timestamp}"
            cellpose_model_path = cellpose_models_dir / cellpose_model_name

            if Path(model_path).exists():
                shutil.copy2(model_path, cellpose_model_path)
                logging.info(
                    'Model copied to .cellpose directory: '
                    f'{cellpose_model_path}'
                )
                mlflow.log_param(
                    'cellpose_model_path', str(cellpose_model_path))
            else:
                logging.warning(f"Model file {model_path} not found!")

            logging.info('Logging training metrics...')
            # log losses per epoch
            for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
                mlflow.log_metrics({
                    'training/train_loss': train_loss,
                    'training/test_loss': test_loss,
                    'training/loss_diff': abs(train_loss - test_loss)
                }, step=epoch)

            # log summary metrics
            mlflow.log_metrics({
                'summary/final_train_loss': train_losses[-1],
                'summary/final_test_loss': test_losses[-1],
                'summary/best_train_loss': min(train_losses),
                'summary/best_test_loss': min(test_losses),
                'summary/avg_train_loss': sum(
                    train_losses) / len(train_losses),
                'summary/avg_test_loss': sum(test_losses) / len(test_losses),
            })

            # Stop monitoring and log summary stats
            if monitor:
                monitor.stop()
                summary_stats = monitor.get_summary_stats()

                # Log system resource summary
                mlflow.log_metrics({
                    f'system_summary/{k}': v
                    for k, v in summary_stats.items()
                })

                logging.info("System Resource Summary:")
                logging.info(f"  CPU: avg="
                             f"{summary_stats.get('cpu_percent_avg', 0):.1f}%,"
                             "max="
                             f"{summary_stats.get('cpu_percent_max', 0):.1f}%")
                logging.info(f"RAM: avg="
                             f"{summary_stats.get('ram_percent_avg', 0):.1f}%,"
                             f"max="
                             f"{summary_stats.get('ram_percent_max', 0):.1f}%")
                if GPU_AVAILABLE:
                    logging.info(
                          "GPU: avg="
                          f"{summary_stats.get('gpu_percent_avg', 0):.1f}%, "
                          "max="
                          f"{summary_stats.get('gpu_percent_max', 0):.1f}%")
                    logging.info(
                       "GPU Memory: avg="
                       f"{summary_stats.get('gpu_memory_percent_avg', 0):.1f}%"
                       ",max="
                       f"{summary_stats.get('gpu_memory_percent_max', 0):.1f}%"
                    )

            logging.info("Creating and logging loss plots...")
            # create plots
            train_losses_plot = plot_loss(n_epochs, train_losses, 'Train')
            test_losses_plot = plot_loss(n_epochs, test_losses, 'Test')

            # save plots
            train_plot_path = output_dir / 'train_losses.png'
            test_plot_path = output_dir / 'test_losses.png'
            train_losses_plot.savefig(
                train_plot_path, dpi=150, bbox_inches='tight')
            test_losses_plot.savefig(
                test_plot_path, dpi=150, bbox_inches='tight')

            # log figures to MLflow
            mlflow.log_figure(train_losses_plot, 'train_losses.png')
            mlflow.log_figure(test_losses_plot, 'test_losses.png')
            mlflow.log_artifact(str(train_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(test_plot_path), artifact_path="plots")

            logging.info("Logging model artifacts...")
            model_path_obj = Path(model_path)

            # log the main model file (as artifact)
            if model_path_obj.exists():
                mlflow.log_artifact(str(model_path), artifact_path="models")
                mlflow.log_param(
                    'model_path', str(model_path))
                mlflow.log_param(
                    'model_size_mb',
                    round(model_path_obj.stat().st_size / 1024 / 1024, 2)
                )

                # log the .cellpose copy
                if cellpose_model_path.exists():
                    mlflow.log_artifact(
                        str(cellpose_model_path), artifact_path="models")

                # log checkpoints if they exist
                checkpoint_dir = output_dir / "models"
                if checkpoint_dir.exists():
                    for checkpoint in checkpoint_dir.glob("*.pth"):
                        if checkpoint != model_path_obj:
                            mlflow.log_artifact(
                                str(checkpoint),
                                artifact_path="models/checkpoints"
                            )
            else:
                logging.error(f"Model path {model_path} does not exist!")
                raise FileNotFoundError(
                    f"Trained model not found at {model_path}")

            # log additional artifacts from output directory
            logging.info("Logging additional artifacts...")
            for file_path in output_dir.rglob('*'):
                if file_path.is_file() and file_path != model_path_obj:
                    relative_path = file_path.relative_to(output_dir)
                    artifact_subpath = str(
                        relative_path.parent) if relative_path.parent != Path(
                            '.') else "outputs"

                    try:
                        mlflow.log_artifact(
                            str(file_path), artifact_path=artifact_subpath)
                    except Exception as e:
                        logging.warning(
                            f"Could not log artifact {file_path}: {e}")

            # Create a README with model info
            readme_path = output_dir / "MODEL_INFO.txt"
            with open(readme_path, 'w') as f:
                f.write("Cellpose Model Training Info\n")
                f.write("="*60 + "\n\n")
                f.write(f"Run ID: {run_id}\n")
                f.write(f"Model Name: {model_name}\n")
                f.write(f"Dataset: {dataset_name} (v{ds_version})\n")
                f.write(f"Training Images: {len(images)}\n")
                f.write(f"Test Images: {len(test_images)}\n")
                f.write(f"Epochs: {n_epochs}\n")
                f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
                f.write(f"Final Test Loss: {test_losses[-1]:.4f}\n\n")

                if monitor and summary_stats:
                    f.write("System Resource Usage:\n")
                    f.write(
                        "CPU: avg="
                        f"{summary_stats.get('cpu_percent_avg', 0):.1f}%,"
                        f"max={summary_stats.get('cpu_percent_max', 0):.1f}"
                        "%\n"
                    )
                    f.write(
                        "RAM: avg="
                        f"{summary_stats.get('ram_percent_avg', 0):.1f}%, "
                        "max="
                        f"{summary_stats.get('ram_percent_max', 0):.1f}%\n")
                    if GPU_AVAILABLE:
                        f.write(
                            "GPU: avg="
                            f"{summary_stats.get('gpu_percent_avg', 0):.1f}%, "
                            "max="
                            f"{summary_stats.get('gpu_percent_max', 0):.1f}%\n"
                        )
                        f.write(
                            "GPU Memory: avg="
                            f"{summary_stats.get('gpu_memory_percent_avg', 0):.1f}%, "
                            f"max={summary_stats.get('gpu_memory_percent_max', 0):.1f}%\n")

                f.write("\nModel Locations:\n")
                f.write(f" - Output Dir: {model_path}\n")
                f.write(f" - .cellpose Dir: {cellpose_model_path}\n\n")
                f.write("To load this model:\n")
                f.write("from cellpose import models\n")
                f.write("model = models.CellposeModel(\n")
                f.write("gpu=True,\n")
                f.write("pretrained_model='{cellpose_model_path}'\n")
                f.write(")\n")

            mlflow.log_artifact(str(readme_path), artifact_path="info")

            # log and print final summary
            summary_msg = (
                f"\n{'='*70}\n"
                f"Training Complete!\n"
                f"{'='*70}\n"
                f"Run ID: {run_id}\n"
                f"Model saved to:\n"
                f"  - Output: {model_path}\n"
                f"  - .cellpose: {cellpose_model_path}\n"
                f"Final Losses - Train: {train_losses[-1]:.4f}"
                f"Test: {test_losses[-1]:.4f}\n"
                f"{'='*70}\n"
            )

            print(summary_msg)
            logging.info(summary_msg)

            return str(cellpose_model_path), train_losses, test_losses, run_id

        except Exception as e:
            # ensure monitor stops even if training fails
            if monitor:
                monitor.stop()
            raise e

        finally:
            # eleanup
            if monitor and monitor.monitoring:
                monitor.stop()
