"""This module contains the functions for calculating the metrics."""

import numpy as np
import pandas as pd


def get_true_positive(groundtruth: np.ndarray, pred: np.ndarray) -> int:
    """Utility function that calculates true positive."""
    return np.count_nonzero((groundtruth == pred) & (groundtruth == 1))


def get_groundtruth_positive(groundtruth: np.ndarray) -> int:
    """Utility function that calculates the total positive in groundtruth."""
    return np.count_nonzero(groundtruth == 1)


def get_predicted_positive(pred: np.ndarray) -> int:
    """Utility function that calculates the total predicted positive."""
    return np.count_nonzero(pred == 1)


def get_precision(groundtruth: np.ndarray, pred: np.ndarray) -> float:
    """This function calculates the precision.

    Args:
        groundtruth (np.array): A numpy array representing the ground truth data.
        pred (np.array): A numpy array representing the predicted data.

    Returns:
        float: The precision score.
    """
    correct = get_true_positive(groundtruth, pred)
    positive = get_predicted_positive(pred)
    return correct / positive if positive else 0


def get_recall(groundtruth: np.ndarray, pred: np.ndarray) -> float:
    """This function calculates the recall.

    Args:
        groundtruth (np.array): A numpy array representing the ground truth data.
        pred (np.array): A numpy array representing the predicted data.

    Returns:
        float: The recall score.
    """
    correct = get_true_positive(groundtruth, pred)
    truth = get_groundtruth_positive(groundtruth)
    return correct / truth if truth else 0


def get_fscore(groundtruth: np.ndarray, pred: np.ndarray) -> float:
    """This function calculates the F-score.

    Args:
        groundtruth (np.array): A numpy array representing the ground truth data.
        pred (np.array): A numpy array representing the predicted data.

    Returns:
        float: The F-score.
    """
    precision = get_precision(groundtruth, pred)
    recall = get_recall(groundtruth, pred)

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


def get_ndcg(groundtruth: np.array, pred_rank_list: np.array, k: int) -> float:
    """This function calculates the Normalized Discounted Cumulative Gain (NDCG).

    Args:
        groundtruth (np.array): A numpy array representing the ground truth data.
        pred_rank_list (np.array): A numpy array representing the predicted ranking.
        k (int): The number of items to consider from the top of the predicted ranking.


    Returns:
        float: The NDCG score.
    """
    pred_rank_list = pred_rank_list[:k]
    relevant_scores = groundtruth[pred_rank_list]

    dcg = np.sum(
        (relevant_scores == 1) / np.log2(np.arange(2, len(pred_rank_list) + 2))
    )

    num_real_item = np.sum(groundtruth)
    num_item = int(num_real_item)

    idcg = np.sum(1 / np.log2(np.arange(2, num_item + 2)))

    return dcg / idcg if idcg > 0 else 0

def calculate_metrics(
    future_df: pd.DataFrame,
    test_ids: np.ndarray,
    merged_his_vecs: np.ndarray,
    output_size: int,
    top_k: int,
) -> tuple[float, float, float, float, float]:
    """Calculate the metrics for the given test set.

    Args:
        future_df (pd.DataFrame): The future dataframe.
        test_ids (np.ndarray): The customer IDs of the test set.
        merged_his_vecs (np.ndarray): The merged history vectors.
        output_size (int): The number of unique material numbers.
        top_k (int): The number of recommendations to make.

    Returns:
        tuple[float, float, float, float, float]: The precision, recall,
            F-score, NDCG, and PHR.
    """
    # Check if test_ids and merged_his_vecs are of the same length
    assert len(test_ids) == len(
        merged_his_vecs
    ), "Mismatch in length of test_ids and merged_his_vecs"

    target_variables = future_df.loc[test_ids].to_numpy()
    output_vectors = merged_his_vecs

    # Initializing metrics
    recall, precision, f_score, ndcg = [], [], [], []

    for target_variable, output_vector in zip(target_variables, output_vectors):
        # Get the indices of the top_k elements
        target_top_k = output_vector.argsort()[-top_k:][::-1]

        # Create boolean masks
        output_mask = np.isin(range(output_size), target_top_k)
        target_mask = np.isin(range(output_size), np.array(target_variable) - 1)

        # Calculate metrics
        recall.append(get_recall(target_mask, output_mask))
        precision.append(get_precision(target_mask, output_mask))
        f_score.append(get_fscore(target_mask, output_mask))
        ndcg.append(get_ndcg(target_mask, target_top_k, top_k))

    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f_score),
        np.mean(ndcg),
    )
