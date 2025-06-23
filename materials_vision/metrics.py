import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose.metrics import boundary_scores
from skimage.segmentation import relabel_sequential


def get_bbox_from_mask(mask: np.ndarray, relabel_image: bool = True) -> dict:
    """
    Extract bounding boxes from a labeled mask.

    Parameters
    ----------
    mask : np.ndarray
        A 2D array where each unique non-zero value represents a
        labeled region
    relabel_image : bool, optional
        If True, relabels the mask to ensure sequential labeling
        starting from 1, by default True

    Returns
    -------
    dict
        A dictionary where keys are the labels and values are
        tuples representing the bounding box
        coordinates (x_min, y_min, x_max, y_max) for each label
    """
    if relabel_image:
        mask, _, _ = relabel_sequential(mask, offset=1)
    labels = np.unique(mask)
    bboxes = {}
    for label in labels[labels > 0]:
        rows, cols = np.where(mask == label)
        if rows.size == 0:
            continue
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        bboxes[int(label)] = int(x_min), int(y_min), int(x_max), int(y_max)
    return bboxes


def calculate_iou(
    bbox_true: Union[List[float], Tuple[float]],
    bbox_pred: Union[List[float], Tuple[float]],
    epsilon: float = 1e-5
) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes

    Parameters
    ----------
    bbox_true : list[float] or tuple[float, float, float, float]
        The ground truth bounding box in the format
        [x_min, y_min, x_max, y_max]
    bbox_pred : list[float] or tuple[float, float, float, float]
        The predicted bounding box in the format [x_min, y_min, x_max, y_max]
    epsilon : float, optional
        A small value to avoid division by zero, by default 1e-5

    Returns
    -------
    float
        The IoU value, a number between 0 and 1, where 0 means no overlap and 1
        means perfect overlap
    """
    # coordinates of the intersection box
    x1 = np.max([bbox_true[0], bbox_pred[0]])
    y1 = np.max([bbox_true[1], bbox_pred[1]])
    x2 = np.min([bbox_true[2], bbox_pred[2]])
    y2 = np.min([bbox_true[3], bbox_pred[3]])

    # area of overlap
    width = x2-x1
    height = y2-y1

    # if there is no overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_of_overlap = width * height
    # combined area
    area_bbox_true = (
      (bbox_true[2] - bbox_true[0]) * (bbox_true[3] - bbox_true[1])
    )
    area_bbox_pred = (
      (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
    )
    area_combined = area_bbox_true + area_bbox_pred - area_of_overlap
    # ratio of area of overlap over combined area
    iou = area_of_overlap / (area_combined + epsilon)
    return iou


def get_iou_matrix(
    bboxes_true: List[Tuple[float]],
    bboxes_pred: List[Tuple[float]]
) -> np.ndarray:
    """
    Calculate the IoU matrix between two lists of bounding boxes.

    Parameters
    ----------
    bboxes_true : List[Tuple[float]]
        A list of ground truth bounding boxes, each represented as a tuple
        (x_min, y_min, x_max, y_max).
    bboxes_pred : List[Tuple[float]
        A list of predicted bounding boxes, each represented as a tuple
        (x_min, y_min, x_max, y_max).

    Returns
    -------
    np.ndarray
        A 2D array where the element at position (i, j) represents the IoU
        between the i-th ground truth bounding box and the j-th predicted
        bounding box.
    """
    n_bboxes_true = len(bboxes_true)
    n_bboxes_pred = len(bboxes_pred)
    I = np.zeros((n_bboxes_true, n_bboxes_pred), dtype=np.float32)
    for i in range(n_bboxes_true):
        for j in range(n_bboxes_pred):
            I[i, j] = calculate_iou(bboxes_true[i], bboxes_pred[j])
    return I


def calculate_pores_difference(n_pores_true: int, n_pores_pred: int) -> float:
    '''
    Calculates differences in number of pores between true and predicted masks.
    '''
    return float(np.abs(n_pores_true - n_pores_pred) / n_pores_true)


def greedy_match_iou(
    iou_matrix: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Perform greedy matching based on the Intersection over Union (IoU) matrix
    and calculate various metrics

    Parameters
    ----------
    iou_matrix : np.ndarray
        A 2D array representing the IoU values between ground truth and
        predicted bounding boxes
    threshold : float
        The minimum IoU value required to consider a match

    Returns
    -------
    dict
        A dictionary containing the following metrics:
        - 'TP': int, the number of true positives
        - 'FP': int, the number of false positives
        - 'FN': int, the number of false negatives
        - 'precision': float, the precision of the matches
        - 'recall': float, the recall of the matches
        - 'f1_score': float, the F1 score of the matches
        - 'iou_mean': float, the mean IoU of the matches
        - 'pores_diff': float, the difference in the number of ground truth and
          predicted bounding boxes
    """
    I = iou_matrix.copy()
    all_triples_list = []
    matched_true, matched_pred = set(), set()
    matches = []
    I[I < threshold] = 0.0
    N_true, N_pred = I.shape
    # get pores difference
    pores_diff = calculate_pores_difference(N_true, N_pred)

    non_zero = np.argwhere(I > 0.0)
    for (i, j) in non_zero:
        all_triples_list.append((i, j, I[i, j]))

    sorted_all_triples_list = sorted(
        all_triples_list, key=lambda x: x[2], reverse=True
    )

    for triples in sorted_all_triples_list:
        if (triples[0] not in matched_true) & (triples[1] not in matched_pred):
            matched_true.add(triples[0])
            matched_pred.add(triples[1])
            matches.append((triples[0], triples[1], triples[2]))

    TP = len(matches)
    FN = N_true - len(matched_true)
    FP = N_pred - len(matched_pred)

    precision = (TP / (TP + FP) if (TP + FP) > 0 else 0.0)
    recall = (TP / (TP + FN) if (TP + FN) > 0 else 0.0)
    f1_score = (2 * (precision * recall) / (precision + recall) if
                (precision + recall) > 0 else 0.0)
    iou_mean = float(
      sum(score for (_, _, score) in matches) / TP if TP > 0 else 0.0
    )

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou_mean': iou_mean,
        'pores_diff': pores_diff
    }


def iou_scores_batch(
    true_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    greedy_match_threshold: float = 0.5
) -> Dict[int, Dict[str, float]]:
    """
    Calculate Intersection over Union (IoU) scores for a batch of true and
    predicted masks.

    Parameters
    ----------
    true_masks : List[np.ndarray]
        A list of 2D arrays representing the ground truth binary masks for
        each image in the batch
    pred_masks : List[np.ndarray]
        A list of 2D arrays representing the predicted binary masks for each
        image in the batch
    greedy_match_threshold : float, optional
        The IoU threshold for greedy matching of predicted and true bounding
        boxes, by default 0.5.

    Returns
    -------
    Dict[int, Dict[str, float]]
        A dictionary where the key is the index of the image in the batch, and
        the value is another dictionary mapping the index of the true bounding
        box to its IoU score with the matched predicted bounding box.
    """
    iou_results = {}
    for idx, (true_mask, pred_mask) in enumerate(zip(true_masks, pred_masks)):
        pred_bboxes = get_bbox_from_mask(pred_mask)
        true_bboxes = get_bbox_from_mask(true_mask)
        iou_matrix = get_iou_matrix(
            list(pred_bboxes.values()), list(true_bboxes.values())
        )
        one_img_output = greedy_match_iou(iou_matrix, greedy_match_threshold)
        iou_results[idx] = one_img_output
    return iou_results


def boundary_scores_batched(
    masks_true: Sequence[np.ndarray],
    masks_pred: Sequence[np.ndarray],
    scales: Sequence[float],
    batch_size: int = 2
) -> Tuple[np.ndarray]:
    """
    Compute boundary precision, recall, and F-score in batches.

    Parameters
    ----------
    masks_true : Sequence[np.ndarray]
        List of ground-truth binary mask arrays.
    masks_pred : Sequence[np.ndarray]
        List of predicted binary mask arrays.
    scales : Sequence[float]
        Boundary tolerance scales for score computation.
    batch_size : int, optional
        Number of mask pairs to process per batch, by default 2.

    Returns
    -------
    Tuple[np.ndarray]
        Three arrays of shape (len(scales), len(masks_true))
        containing precision, recall, and F-score values.
    """
    N = len(masks_true)
    M = len(scales)

    precision_all = np.zeros((M, N), dtype=float)
    recall_all = np.zeros((M, N), dtype=float)
    fscore_all = np.zeros((M, N), dtype=float)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # part of lists (batched solution)
        sub_true = masks_true[start:end]
        sub_pred = masks_pred[start:end]
        p_sub, r_sub, f_sub = boundary_scores(sub_true, sub_pred, scales)

        precision_all[:, start:end] = p_sub
        recall_all[:, start:end] = r_sub
        fscore_all[:, start:end] = f_sub

        # clean memory
        gc.collect()

    return precision_all, recall_all, fscore_all


def summarize_evaluation_iou(
    iou_results: Dict[int, Dict[str, float]]
) -> pd.DataFrame:
    """
    Summarize IoU evaluation metrics with macro and micro averaging.

    Parameters
    ----------
    iou_results : Dict[int, Dict[str, float]]
        Mapping from image indices to IoU metric values. Each inner dict
        contains floats for keys 'TP', 'FP', 'FN', and other metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by metric name with column 'value'. Contains
        macro-averaged metrics for each IoU measure and micro-averaged
        precision, recall, and F1-score.
    """
    TP_all, FP_all, FN_all = [], [], []
    macro_metric_results = {}
    # macro evaluation
    for metric_dict in iou_results.values():
        for metric, value in metric_dict.items():
            if metric not in ['TP', 'FP', 'FN']:
                macro_metric_results[f'iou_mean_{metric}'] = np.mean(value)
            # micro evaluation
            elif metric == 'TP':
                TP_all.append(value)
            elif metric == 'FP':
                FP_all.append(value)
            elif metric == 'FN':
                FN_all.append(value)
    TP_all_sum = np.sum(TP_all)
    FP_all_sum = np.sum(FP_all)
    FN_all_sum = np.sum(FN_all)
    precision_micro = TP_all_sum / (TP_all_sum + FP_all_sum)
    recall_micro = TP_all_sum / (TP_all_sum + FN_all_sum)
    f1_micro = (
        2 * precision_micro * recall_micro
      ) / (precision_micro + recall_micro)
    # prepare final report
    report = pd.DataFrame.from_dict(
        macro_metric_results, orient='index', columns=['value']
      )
    report.loc['iou_precision_micro'] = precision_micro
    report.loc['iou_recall_micro'] = recall_micro
    report.loc['iou_f1_micro'] = f1_micro
    return report


def summarize_evaluation_boundary_score(
    precision_all: List[float],
    recall_all: List[float],
    fscore_all: List[float]
) -> pd.DataFrame:
    """
    Summarize evaluation boundary score metrics by calculating their
    mean values.

    Parameters
    ----------
    precision_all : List[float]
        A list of precision values for each boundary evaluation
        instance.
    recall_all : List[float]
        A list of recall values for each boundary evaluation instance.
    fscore_all : List[float]
        A list of f1-scores for each boundary evaluation instance.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by metric names (
        'boundary_score_mean_precision',
        'boundary_score_mean_recall',
        'boundary_score_mean_f1'
        ) with a single column 'value' containing the computed
        mean scores.
    """
    bs_precision_mean = np.mean(precision_all)
    bs_recall_mean = np.mean(recall_all)
    bs_fscore_mean = np.mean(fscore_all)
    bs_metrics_dict = {
          'boundary_score_mean_precision': bs_precision_mean,
          'boundary_score_mean_recall':    bs_recall_mean,
          'boundary_score_mean_f1':        bs_fscore_mean,
    }
    report = pd.DataFrame.from_dict(
        bs_metrics_dict, orient='index', columns=['value']
    )
    return report


def get_today_datetime_str():
    '''Returns current datetime as a string formatted for filenames.'''
    today_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    today_datetime = today_datetime[:-3]  # skip seconds
    today_datetime = (
        today_datetime
        .replace('-', '_')
        .replace(':', '_')
        .replace('/', '_')
        .replace(' ', '__')
    )
    return today_datetime


def summarize_evaluation(
    output_filename: str,
    iou_results: Dict[int, Dict[str, float]],
    boundary_score_precision_all: List[float],
    boundary_score_recall_all: List[float],
    boundary_score_fscore_all: List[float],
    evaluation_comment: str = '',
    save_raw_iou_results: bool = True,
    save_final_results: bool = True,
    output_directory_path: Optional[Path] = None,
    add_today_datetime_to_filename: bool = True
) -> pd.DataFrame:
    """
    Summarize evaluation metrics for IOU and boundary scores, with
    optional saving of raw and final results.

    Parameters
    ----------
    output_filename : str
        Base name for the output files (without extension).
    iou_results : Dict[int, Dict[str, float]]
        Dictionary of IOU results from evaluation.
    boundary_score_precision_all : List[float]
        Precision values for boundary score evaluation instances.
    boundary_score_recall_all : List[float]
        Recall values for boundary score evaluation instances.
    boundary_score_fscore_all : List[float]
        F1-score values for boundary evaluation instances.
    evaluation_comment : str, optional
        A comment to include with the results, by default ''.
    save_raw_iou_results : bool, optional
        Whether to save the raw IOU results to JSON, by default True.
    save_final_results : bool, optional
        Whether to save the final report to CSV, by default True.
    output_directory_path : Optional[Path], optional
        Directory path for saving outputs; defaults to a preconfigured
        metrics folder if None.
    add_today_datetime_to_filename : bool, optional
        Whether to append current date-time to filenames, by default True.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame of IOU and boundary score metrics,
        including any evaluation comments.
    """
    report_iou = summarize_evaluation_iou(iou_results)
    report_boundary_score = summarize_evaluation_boundary_score(
        precision_all=boundary_score_precision_all,
        recall_all=boundary_score_recall_all,
        fscore_all=boundary_score_fscore_all
    )
    report_final = pd.concat([report_iou, report_boundary_score])
    report_final['comment'] = evaluation_comment
    # save files
    today_datetime_str = (
        get_today_datetime_str() if add_today_datetime_to_filename else ''
    )
    if not output_directory_path:
        output_directory_path = Path(
            '/content/drive/MyDrive/evaluation_metrics'
        )
    # save raw iou
    if save_raw_iou_results:
        output_directory_path.mkdir(parents=True, exist_ok=True)
        output_filename = (
            f'raw_iou_results_{evaluation_comment}_{today_datetime_str}.json'
        )
        output_filepath = output_directory_path / output_filename
        with open(output_filepath, 'w') as f:
            json.dump(iou_results, f)
        print('Raw iou test dictionary saved.')
    # save final results
    if save_final_results:
        output_directory_path.mkdir(parents=True, exist_ok=True)
        output_filename = (
            f'evaluation_results_{today_datetime_str}.csv'
        )
        output_filepath = output_directory_path / output_filename
        report_final.to_csv(output_filepath)
        print('Final evaluation metrics report saved.')
    return report_final


def visualize_results(
    pred_masks: List[np.ndarray],
    img_idx_to_visualize: int = 0,
    save_subsample: bool = True,
    output_directory_path=None,
    evaluation_comment: str = ''
):
    """
    Visualizes and optionally saves segmentation results.

    Parameters
    ----------
    pred_masks : List[np.ndarray]
        List of predicted segmentation masks.
    img_idx_to_visualize : int, optional
        Index of the image to visualize, by default 0.
    save_subsample : bool, optional
        Whether to save a subsample of the segmentation results as images, by
        default True.
    output_directory_path : Path or str, optional
        Directory path where the subsampled images will be saved. If None, a
        default path is used, by default None.
    evaluation_comment : str, optional
        A comment to include in the filenames of saved images, by default ''.
    """
    # save images
    if save_subsample:
        output_directory_path.mkdir(parents=True, exist_ok=True)
        num_images = 10
        for img_idx in range(num_images):
            fig = plt.figure(figsize=(20, 12))
            plot.show_segmentation(
              fig=fig,
              maski=pred_masks[img_idx],
              img=imgs[img_idx],
              flowi=flows[img_idx][0]
            )
            plt.savefig(
                f'{output_directory_path}/{evaluation_comment}_figure_'
                f'{img_idx}.jpg'
            )
            plt.close(fig)
        # show_figure
        fig = plt.figure(figsize=(20, 12))
        plot.show_segmentation(
          fig=fig,
          maski=pred_masks[img_idx_to_visualize],
          img=imgs[img_idx_to_visualize],
          flowi=flows[img_idx_to_visualize][0]
        )
        plt.show()
