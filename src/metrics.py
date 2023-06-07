"""This module contains the functions for calculating the metrics."""


import numpy as np


def get_precision_recall_fscore(groundtruth: list[int], pred: list[int]) -> tuple[float, float, float, int]:
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

    dcg = np.sum((relevant_scores == 1) / np.log2(np.arange(2, len(pred_rank_list) + 2)))

    num_real_item = np.sum(groundtruth)
    num_item = int(num_real_item)

    idcg = np.sum(1 / np.log2(np.arange(2, num_item + 2)))

    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg

