"""Cellpose model prediction utilities."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import tifffile
from tqdm import tqdm


log = logging.getLogger(__name__)


def predict_and_save(
    dataset: List[Dict[str, np.ndarray]],
    model,
    output_dir: Union[str, Path],
    diameter: Optional[float] = None,
    channels: List[int] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    batch_size: int = 8,
    normalize: bool = True,
    save_predictions: bool = True,
    prediction_suffix: str = '_pred_mask',
    prediction_extension: str = '.tif'
) -> List[Dict[str, np.ndarray]]:
    """
    Generate Cellpose predictions and save to output directory.

    Takes a dataset of image-mask pairs, runs Cellpose model inference in
    evaluation mode with default or specified parameters, adds predicted
    masks to the dataset, and optionally saves predictions as TIFF files.

    Parameters
    ----------
    dataset : list of dict
        List of dictionaries, each containing at minimum:
        - 'image' : np.ndarray
            Input image (2D grayscale or 3D RGB).
        - 'true_mask' : np.ndarray, optional
            Ground truth segmentation mask.
    model : cellpose.models.CellposeModel
        Pre-loaded Cellpose model instance (e.g., CellposeModel or Cellpose).
    output_dir : str or Path
        Directory path where prediction masks will be saved.
    diameter : float, optional
        Expected diameter of objects in pixels. If None, model estimates
        diameter automatically (default: None).
    channels : list of int, optional
        List of channels to use for segmentation. For grayscale, use [0, 0].
        For RGB with cytoplasm in green and nucleus in blue, use [2, 3].
        If None, defaults to [0, 0] for grayscale (default: None).
    flow_threshold : float, optional
        Maximum allowed error of flows for masks. Higher values result in
        fewer masks (default: 0.4).
    cellprob_threshold : float, optional
        Threshold for cell probability. Pixels with probability above this
        value are used to run dynamics and determine masks. Lower values
        result in more masks (default: 0.0).
    min_size : int, optional
        Minimum number of pixels per mask. Masks smaller than this are
        removed (default: 15).
    batch_size : int, optional
        Number of images to process simultaneously. Larger values are faster
        but require more GPU memory (default: 8).
    normalize : bool, optional
        Whether to normalize images before prediction. Recommended for
        best results (default: True).
    save_predictions : bool, optional
        Whether to save prediction masks to disk (default: True).
    prediction_suffix : str, optional
        Suffix to append to saved prediction filenames
        (default: '_pred_mask').
    prediction_extension : str, optional
        File extension for saved predictions (default: '.tif').

    Returns
    -------
    dataset : list of dict
        Input dataset augmented with 'pred_mask' key containing predicted
        segmentation masks as np.ndarray for each sample.

    Raises
    ------
    ValueError
        If dataset is empty or contains invalid entries.
    OSError
        If output_dir cannot be created or is not writable.

    Examples
    --------
    >>> from cellpose import models
    >>> from pathlib import Path
    >>>
    >>> # Load model
    >>> model = models.CellposeModel(
    ...     gpu=True,
    ...     pretrained_model='/path/to/model'
    ... )
    >>>
    >>> # Load dataset
    >>> dataset = get_img_mask_pairs(
    ...     Path('/data/test'),
    ...     loaded=True
    ... )
    >>>
    >>> # Make predictions
    >>> dataset = predict_and_save(
    ...     dataset=dataset,
    ...     model=model,
    ...     output_dir='/output/predictions',
    ...     diameter=30.0,
    ...     channels=[0, 0]
    ... )
    >>>
    >>> # Access predictions
    >>> pred_mask = dataset[0]['pred_mask']
    """
    if not dataset:
        raise ValueError("Dataset is empty")

    output_path = Path(output_dir)
    if save_predictions:
        output_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Created output directory: {output_path}")

    # Set default channels for grayscale
    if channels is None:
        channels = [0, 0]

    log.info(
        f"Starting prediction on {len(dataset)} images with model "
        f"(diameter={diameter}, channels={channels})"
    )

    # Process dataset
    for idx, sample in enumerate(tqdm(dataset, desc="Generating predictions")):
        if 'image' not in sample:
            raise ValueError(f"Sample {idx} missing 'image' key")

        image = sample['image']

        # Run model prediction
        pred_mask, flows, styles, diams = model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            batch_size=batch_size,
            normalize=normalize
        )

        # Add prediction to dataset
        sample['pred_mask'] = pred_mask

        # Optionally save to disk
        if save_predictions:
            filename = f"sample_{idx:05d}{prediction_suffix}{prediction_extension}"
            output_file = output_path / filename
            tifffile.imwrite(output_file, pred_mask.astype(np.uint16))

    log.info(f"Predictions completed and saved to {output_path}")

    return dataset
