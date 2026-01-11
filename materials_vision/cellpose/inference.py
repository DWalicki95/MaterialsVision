"""
Cellpose Inference Evaluation Module.

This module provides functionality for running inference with a pretrained
Cellpose model on image files and saving the results including masks, flows,
and style vectors.
"""
from pathlib import Path
from cellpose import models, io
import logging
from cellpose.io import imread
from config import MODEL_PATH_INFERENCE, OUTPUT_PATH_INFERENCE, PATH_TO_FILES_INFERENCE
from typing import List, Union, Optional, Tuple
import numpy as np
from tifffile import imwrite

io.logger_setup()

logger = logging.getLogger(__name__)


class CellposeInferenceEvaluation():
    """
    Cellpose model inference and evaluation handler.

    This class manages the complete inference pipeline including loading images,
    initializing the Cellpose model, running inference, and saving outputs.

    Parameters
    ----------
    path_to_files : str or Path, optional
        Path to directory containing input image files.
        Default is PATH_TO_FILES_INFERENCE from config.
    model_path : str or Path, optional
        Path to the pretrained Cellpose model file.
        Default is MODEL_PATH_INFERENCE from config.
    output_path : str or Path, optional
        Directory path where inference outputs will be saved.
        Default is OUTPUT_PATH_INFERENCE from config.

    Attributes
    ----------
    path_to_files : str or Path
        Path to input image files directory.
    model_path : str or Path
        Path to pretrained model.
    output_path : str or Path
        Output directory for inference results.
    """
    def __init__(
            self,
            path_to_files: Union[str, Path] = PATH_TO_FILES_INFERENCE,
            model_path: Union[str, Path] = MODEL_PATH_INFERENCE,
            output_path: Union[str, Path] = OUTPUT_PATH_INFERENCE
        ) -> None:
        self.path_to_files = path_to_files
        self.model_path = model_path
        self.output_path = output_path


    def _load_files(self) -> List[np.ndarray]:
        """
        Load all JPG image files from the input directory.

        Searches for all .jpg files in the path_to_files directory and loads
        them as numpy arrays using Cellpose's imread function.

        Returns
        -------
        List[np.ndarray]
            List of loaded images as numpy arrays. Each array represents
            an image with shape (height, width) for grayscale or
            (height, width, channels) for color images.

        Notes
        -----
        Only files with .jpg extension are loaded. Other image formats
        in the directory will be ignored.
        """
        files = Path(self.path_to_files).glob('*.jpg')
        files = [file for file in files if file.is_file()]
        return [imread(f) for f in files]


    def _init_model(self) -> models.CellposeModel:
        """
        Initialize the Cellpose model with GPU acceleration.

        Creates and returns a CellposeModel instance using the pretrained
        model specified in the model_path attribute. GPU acceleration is
        enabled by default.

        Returns
        -------
        models.CellposeModel
            Initialized Cellpose model ready for inference with GPU support.

        Notes
        -----
        Requires CUDA-compatible GPU for GPU acceleration. If GPU is not
        available, the model initialization may fail or fall back to CPU.
        """
        return models.CellposeModel(gpu=True, pretrained_model=self.model_path)

    def eval(
            self,
            imgs: Union[List[np.ndarray], np.ndarray],
            model: models.CellposeModel
        ) -> Tuple[Union[List[np.ndarray], np.ndarray], Union[List[np.ndarray], np.ndarray], Union[List[np.ndarray], np.ndarray]]:
        """
        Run inference on images using the Cellpose model.

        Performs segmentation on the provided images using the given Cellpose
        model, returning masks, flow fields, and style vectors.

        Parameters
        ----------
        imgs : List[np.ndarray] or np.ndarray
            Input image(s) to segment. Can be a single image array or a list
            of image arrays. Each image should have shape (height, width) for
            grayscale or (height, width, channels) for color images.
        model : models.CellposeModel
            Initialized Cellpose model to use for inference.

        Returns
        -------
        masks : List[np.ndarray] or np.ndarray
            Segmentation masks where each unique integer value represents
            a different segmented object. Background is 0.
        flows : List[np.ndarray] or np.ndarray
            Flow fields computed during segmentation. Contains gradient
            information used to generate the masks.
        styles : List[np.ndarray] or np.ndarray
            Style vectors extracted from the images. These represent
            learned feature representations from the model.

        Notes
        -----
        The output format (list vs single array) depends on the input format.
        Single images return single arrays, multiple images return lists.
        """
        masks, flows, styles = model.eval(imgs)
        return (masks, flows, styles)
    
    def run_pipeline(self, save_flows: bool = True, save_styles: bool = True) -> None:
        """
        Execute the complete inference pipeline.

        Runs the entire workflow from loading images, initializing the model,
        performing inference, to saving all outputs. This is the main entry
        point for running Cellpose inference end-to-end.

        Parameters
        ----------
        save_flows : bool, optional
            Whether to save flow field outputs. Default is True.
        save_styles : bool, optional
            Whether to save style vector outputs. Default is True.

        Notes
        -----
        The pipeline executes the following steps in order:
        1. Load all JPG images from path_to_files directory
        2. Initialize Cellpose model with GPU support
        3. Run inference on all loaded images
        4. Save masks (always), and optionally flows and styles

        All progress and errors are logged using the module logger.

        Examples
        --------
        >>> evaluator = CellposeInferenceEvaluation(
        ...     path_to_files="/path/to/images",
        ...     model_path="/path/to/model",
        ...     output_path="/path/to/output"
        ... )
        >>> evaluator.run_pipeline()

        >>> # Run pipeline without saving flows and styles
        >>> evaluator.run_pipeline(save_flows=False, save_styles=False)
        """
        logger.info("Starting Cellpose inference pipeline")

        # Step 1: Load images
        logger.info(f"Loading images from {self.path_to_files}")
        files = list(Path(self.path_to_files).glob('*.jpg'))
        files = [file for file in files if file.is_file()]

        if not files:
            logger.warning(f"No JPG files found in {self.path_to_files}")
            return

        logger.info(f"Found {len(files)} images to process")
        imgs = self._load_files()

        # Step 2: Initialize model
        logger.info(f"Initializing Cellpose model from {self.model_path}")
        model = self._init_model()

        # Step 3: Run inference
        logger.info("Running inference on images")
        masks, flows, styles = self.eval(imgs, model)
        logger.info("Inference completed successfully")

        # Step 4: Save outputs
        logger.info(f"Saving outputs to {self.output_path}")
        self.save_output(
            files,
            masks,
            flows if save_flows else None,
            styles if save_styles else None
        )

        logger.info("Pipeline completed successfully")

    def save_output(
            self,
            files: List[Path],
            masks: List[np.ndarray],
            flows: Optional[List[np.ndarray]] = None,
            styles: Optional[List[np.ndarray]] = None
        ) -> None:
        """
        Save inference outputs to disk.

        Saves segmentation masks as TIFF files and optionally saves flow fields
        and style vectors as NumPy .npy files. Output filenames are derived from
        input filenames with appropriate suffixes.

        Parameters
        ----------
        files : List[Path]
            List of Path objects representing the input files. Used to generate
            output filenames based on the input file stems.
        masks : List[np.ndarray]
            List of segmentation mask arrays to save. Each mask is saved as a
            separate TIFF file with suffix '_predicted_masks.tif'.
        flows : List[np.ndarray], optional
            List of flow field arrays to save. If provided, each flow is saved
            as a NumPy file with suffix '_flows.npy'. Default is None.
        styles : List[np.ndarray], optional
            List of style vector arrays to save. If provided, each style vector
            is saved as a NumPy file with suffix '_styles.npy'. Default is None.

        Notes
        -----
        - All outputs are saved to the directory specified in self.output_path
        - Mask files are saved in TIFF format for better compatibility with
          image analysis tools
        - Flow and style files are saved in NumPy format for efficient storage
          and loading
        - Errors during saving are logged but do not halt execution for
          remaining files

        Examples
        --------
        >>> evaluator = CellposeInferenceEvaluation()
        >>> files = list(Path("input").glob("*.jpg"))
        >>> masks, flows, styles = evaluator.eval(imgs, model)
        >>> evaluator.save_output(files, masks, flows, styles)
        """
        for i, (input_file, mask) in enumerate(zip(files, masks)):
            # Save masks
            mask_filename = f"{input_file.stem}_predicted_masks.tif"
            mask_path = Path(self.output_path) / mask_filename
            try:
                imwrite(mask_path, mask)
                logger.info(f"Saved: {mask_filename}")
            except Exception as e:
                logger.error(f"Error saving {mask_filename}: {e}")

            # Save flows
            if flows is not None:
                flow_filename = f"{input_file.stem}_flows.npy"
                flow_path = Path(self.output_path) / flow_filename
                try:
                    np.save(flow_path, flows[i])
                    logger.info(f"Saved: {flow_filename}")
                except Exception as e:
                    logger.error(f"Error saving {flow_filename}: {e}")

            # Save styles
            if styles is not None:
                style_filename = f"{input_file.stem}_styles.npy"
                style_path = Path(self.output_path) / style_filename
                try:
                    np.save(style_path, styles[i])
                    logger.info(f"Saved: {style_filename}")
                except Exception as e:
                    logger.error(f"Error saving {style_filename}: {e}")