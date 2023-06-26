"""This module contains the functions for calculating the metrics."""



import numpy as np
import pandas as pd


def get_true_positive(groundtruth: np.ndarray, pred: np.ndarray) -> int:
    """This function calculates the number of true positives."""
    return np.count_nonzero((groundtruth == pred) & (groundtruth == 1))

def get_true_negative(groundtruth: np.ndarray, pred: np.ndarray) -> int:
    """This function calculates the number of true negatives."""
    return np.count_nonzero((groundtruth == pred) & (groundtruth == 0))

def get_false_positive(groundtruth: np.ndarray, pred: np.ndarray) -> int:
    """This function calculates the number of false positives."""
    return np.count_nonzero((groundtruth != pred) & (groundtruth == 0))

def get_false_negative(groundtruth: np.ndarray, pred: np.ndarray) -> int:
    """This function calculates the number of false negatives."""
    return np.count_nonzero((groundtruth != pred) & (groundtruth == 1))


def get_recall(groundtruth: np.ndarray, pred: np.ndarray) -> float:
    """This function calculates the recall.

    Args:
        groundtruth (np.ndarray): A numpy array representing the ground truth data.
        pred (np.ndarray): A numpy array representing the predicted data.

    Returns:
        float: The recall score.
    """
    true_positive = get_true_positive(groundtruth, pred)
    false_negative = get_false_negative(groundtruth, pred)

    return true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0

def get_precision(groundtruth: np.ndarray, pred: np.ndarray) -> float:
    """This function calculates the precision.

    Args:
        groundtruth (np.ndarray): A numpy array representing the ground truth data.
        pred (np.ndarray): A numpy array representing the predicted data.

    Returns:
        float: The precision score.
    """
    true_positive = get_true_positive(groundtruth, pred)
    false_positive = get_false_positive(groundtruth, pred)

    return true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0

def get_fscore(groundtruth: np.ndarray, pred: np.ndarray) -> float:
    """This function calculates the F-score.

    Args:
        groundtruth (np.ndarray): A numpy array representing the ground truth data.
        pred (np.ndarray): A numpy array representing the predicted data.

    Returns:
        float: The F-score.
    """
    precision = get_precision(groundtruth, pred)
    recall = get_recall(groundtruth, pred)

    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

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
        merged_his_vecs (np.ndarray): The merged history vectors
        output_size (int): The number of unique material numbers.
        top_k (int): The number of recommendations to make.

    Returns:
        dict[str, float]: A dictionary containing the metrics.
    """
    recall, precision, f_score, ndcg = [], [], [], []
    for idx, test_id in enumerate(test_ids):
        target_variable = future_df.loc[test_id].to_numpy()[0]
        output_vector = merged_his_vecs[idx]

        # Get the top K indices sorted by value in descending order
        target_top_k = output_vector.argsort()[::-1][:top_k]

        # Initialize the output vector and set top K positions to 1
        output = np.zeros(output_size)
        output[target_top_k] = 1

        # Vectorize target variable
        vectorized_target = np.zeros(output_size)
        for target in target_variable:
            vectorized_target[target - 1] = 1

        precision.append(get_precision(vectorized_target, output))
        recall.append(get_recall(vectorized_target, output))
        f_score.append(get_fscore(vectorized_target, output))
        ndcg.append(get_ndcg(vectorized_target, target_top_k, top_k))

    return {
        f"Precision@{top_k}": round(np.mean(precision), 4),
        f"Recall@{top_k}": round(np.mean(recall), 4),
        f"F1@{top_k}": round(np.mean(f_score), 4),
        f"NDCG@{top_k}": round(np.mean(ndcg), 4),
    }
