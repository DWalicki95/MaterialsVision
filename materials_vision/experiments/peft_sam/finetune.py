import argparse
import logging
import os

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model, export_custom_qlora_model
from peft_sam.util import get_default_peft_kwargs
from materials_vision.logging_config import setup_logging
from materials_vision.experiments.peft_sam.config import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, EARLY_STOPPING, LEARNING_RATE,
    N_EPOCHS, EXPORT_PATH, N_OBJECTS_PER_BATCH, RUN_NAME
)
from materials_vision.experiments.peft_sam.dataloader import get_data_loaders


logger = logging.getLogger(__name__)


def _log_cuda_info():
    logger.info("torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("CUDA device count: %d", torch.cuda.device_count())
    if torch.cuda.is_available():
        logger.info("Device name: %s", torch.cuda.get_device_name(0))


def finetune_sam(args):
    _log_cuda_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    model_type = "vit_b"
    checkpoint_path = None

    if args.peft_method == "freeze_encoder":
        freeze_parts = "image_encoder"
        peft_kwargs = {}
    else:
        freeze_parts = None
        peft_kwargs = get_default_peft_kwargs(args.peft_method)

    train_loader, val_loader = get_data_loaders(
        train_dir=TRAIN_DATA_PATH,
        val_dir=VAL_DATA_PATH,
    )
    logger.info("Train batches per epoch: %d", len(train_loader))
    logger.info("PEFT kwargs: %s", peft_kwargs)

    if args.peft_method is None:
        checkpoint_name = f"{model_type}/full_finetuning/foam_sam_{RUN_NAME}"
    else:
        checkpoint_name = (
            f"{model_type}/{args.peft_method}/foam_sam_{RUN_NAME}"
        )

    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=EARLY_STOPPING,
        lr=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        device=device,
        freeze=freeze_parts,
        peft_kwargs=peft_kwargs,
        n_objects_per_batch=N_OBJECTS_PER_BATCH,
        with_segmentation_decoder=True,
    )

    if EXPORT_PATH is not None:
        checkpoint_path = os.path.join(
            "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=EXPORT_PATH,
        )

    if args.peft_method == "qlora":
        checkpoint_path = os.path.join(
            "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_qlora_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=EXPORT_PATH,
        )


def main():
    setup_logging(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Finetune Segment Anything for microscopy data."
    )
    parser.add_argument(
        "--peft_method",
        type=str,
        default=None,
        help="The method to use for PEFT.",
        choices=[
            "freeze_encoder", "lora", "qlora", "fact", "attention_tuning",
            "adaptformer", "bias_tuning", "layernorm_tuning", "ssf",
            "late_lora", "late_ft",
        ],
    )
    args = parser.parse_args()
    finetune_sam(args)
