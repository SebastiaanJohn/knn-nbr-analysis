"""This module contains the functions for calculating the metrics."""


from itertools import islice

import numpy as np
import pandas as pd


def get_precision_recall_fscore(
    groundtruth: list[int],
    pred: list[int],
) -> tuple[float, float, float, int]:
    """This function calculates the precision, recall, and F-score.

    Args:
        groundtruth (list): A list representing the ground truth data.
        pred (list): A list representing the predicted data.

    Returns:
        tuple: A tuple containing the precision, recall, F-score, and correct count.
    """
    assert len(groundtruth) == len(pred), "Both groundtruth and pred should have the same length."

    correct = np.count_nonzero((groundtruth == pred) & (groundtruth == 1))
    truth = np.count_nonzero(groundtruth == 1)
    positive = np.count_nonzero(pred == 1)

    precision = correct / positive if positive else 0
    recall = correct / truth if truth else 0

    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0

    return precision, recall, f_score, correct

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

    dcg = np.sum((relevant_scores == 1) / \
                 np.log2(np.arange(2, len(pred_rank_list) + 2)))

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
) -> tuple[float, float, float, float]:
    """Calculate the metrics for the given test set.

    Args:
        future_df (pd.DataFrame): The future dataframe.
        test_ids (np.ndarray): The customer IDs of the test set.
        merged_his_vecs (np.ndarray): The merged history vectors.
        output_size (int): The number of unique material numbers.
        top_k (int): The number of recommendations to make.

    Returns:
        tuple[float, float, float, float]: The precision, recall, F-score, and NDCG.
    """
    recalls, precisions, f_scores, ndcg = [], [], [], []
    for idx, test_id in enumerate(test_ids):
        target_variable = future_df.loc[test_id].to_numpy()[0]
        output_vector = merged_his_vecs[idx]
        output = np.zeros(output_size)
        target_top_k = output_vector.argsort()[::-1]
        for value in islice(target_top_k, top_k):
            output[value] = 1
        vectorized_target = np.zeros(output_size)

        for target in target_variable:
            vectorized_target[target - 1] = 1

        precision, recall, f_score, _ = get_precision_recall_fscore(
            vectorized_target, output,
        )

        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
        ndcg.append(get_ndcg(vectorized_target, target_top_k, top_k))

    return np.mean(recalls), np.mean(ndcg), np.mean(f_scores)
