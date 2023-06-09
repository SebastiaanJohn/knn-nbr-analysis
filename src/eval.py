"""This module implements the TIFUKNN algorithm."""


import argparse
import logging

import numpy as np
import pandas as pd
from metrics import calculate_metrics
from utils import get_total_materials

from data.dataset import load_data, partition_data_ids
from models.knn import knn


def group_history(hist_arr: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    """Group the history vectors into groups of size `group_size`.

    Args:
        hist_arr (np.ndarray): The history vectors to group.
        group_size (int): The size of each group.

    Returns:
        A tuple of two values:
            - The grouped vectors.
            - The number of vectors in the grouped array.
    """
    num_vectors = hist_arr.shape[0]

    if num_vectors < group_size:
        return hist_arr, num_vectors

    base_vectors_per_group = num_vectors // group_size
    leftover_vectors = num_vectors % group_size
    groups_with_extra_vector = leftover_vectors

    indices_base = np.arange(
        0,
        (group_size - groups_with_extra_vector) * base_vectors_per_group,
        base_vectors_per_group,
    )

    indices_extra = np.arange(
        (group_size - groups_with_extra_vector) * base_vectors_per_group,
        num_vectors,
        base_vectors_per_group + 1,
    )

    indices = np.concatenate((indices_base, indices_extra))

    estimated_vectors_per_group = np.concatenate(
        (
            np.full(group_size - groups_with_extra_vector, base_vectors_per_group),
            np.full(groups_with_extra_vector, base_vectors_per_group + 1),
        ),
    )

    grouped_vectors = (
        np.add.reduceat(hist_arr, indices, axis=0)
        / estimated_vectors_per_group[:, None]
    )

    return grouped_vectors, group_size


def temporal_decay(
    customer: pd.Series, r_b: float, r_g: float, m: int, output_size: int
) -> np.ndarray:
    """Calculate the time decayed history vector for a given customer.

    Args:
        customer (pd.Series): Series of lists of material numbers for a given customer.
        r_b (float): The time-decayed ratio within group.
        r_g (float): The time-decayed ratio across the groups.
        m (int): The number of groups to split the history into.
        output_size (int): The number of unique material numbers.


    Returns:
        np.ndarray: The time-decayed history vector for the given customer.
    """
    n = len(customer)
    decayed_vals = np.power(r_b, np.arange(n - 1, -1, -1))

    hist_arr = np.zeros((n, output_size))

    for idx, elements in enumerate(customer):
        hist_arr[idx, np.array(elements) - 1] = decayed_vals[idx]

    grouped_vectors, real_group_size = group_history(hist_arr, m)

    idx = np.arange(real_group_size)
    decayed_vals_group = np.power(r_g, m - 1 - idx)
    his_vec = np.sum(
        grouped_vectors[:real_group_size] * decayed_vals_group[:, np.newaxis], axis=0
    )

    return his_vec / real_group_size

def merge_history(
    train_his_vecs: np.ndarray,
    test_his_vecs: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Merge the history vectors of the train and test sets.

    Args:
        train_his_vecs (np.ndarray): The history vectors of the train set.
        test_his_vecs (np.ndarray): The history vectors of the test set.
        train_ids (np.ndarray): The customer IDs of the train set.
        test_ids (np.ndarray): The customer IDs of the test set.
        indices (np.ndarray): The indices of the nearest neighbors.
        alpha (float): The weight of the history vector of the query vector.
    """
    # Create a dictionary mapping the customer IDs to their history vectors
    train_his_dict = dict(zip(train_ids, train_his_vecs))
    test_his_dict = dict(zip(test_ids, test_his_vecs))

    # Initialize a list to store the merged history vectors
    merged_his_vecs = []

    # Iterate over the indices of the nearest neighbors
    for idx, row in enumerate(indices):
        # Get the customer ID of the query vector
        query_id = test_ids[idx]
        # Get the customer ID of the k nearest neighbors
        nn_ids = train_ids[row]
        # Get the history vectors of the k nearest neighbors
        nn_his_vecs = np.array([train_his_dict[nn_id] for nn_id in nn_ids])
        # Get the history vector of the query vector
        query_his_vec = test_his_dict[query_id]
        # Calculate the merged history vector
        merged_his_vec = alpha * query_his_vec + (1 - alpha) * np.mean(
            nn_his_vecs, axis=0,
        )
        # Append the merged history vector to the list
        merged_his_vecs.append(merged_his_vec)

    return np.array(merged_his_vecs)


def evaluate(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    m: int = 7,
    r_b: float = 0.9,
    r_g: float = 0.7,
    k: int = 300,
    alpha: float = 0.7,
    top_k: int = 10,
    distance_metric: str = "cosine",
) -> tuple[float, float, float]:
    """Trains a KNN model and evaluates it on the validation and test sets.

    Args:
        history_df (pd.DataFrame): The dataframe containing the customer purchase history.
        future_df (pd.DataFrame): The dataframe containing the customer future purchases.
        train_ids (np.ndarray): The customer IDs in the training set.
        val_ids (np.ndarray): The customer IDs in the validation set.
        test_ids (np.ndarray): The customer IDs in the test set.
        m (int): The number of groups to split the history into.
            Defaults to 7.
        r_b (float): The decay rate within a group.
            Defaults to 0.9.
        r_g (float): The decay rate across groups.
            Defaults to 0.7.
        k (int): The number of neighbors to find for each customer.
            Defaults to 300.
        alpha (float): The alpha value for the TIFUKNN model.
            Defaults to 0.7.
        top_k (int): The number of recommendations to make for each customer.
            Defaults to 10.
        distance_metric (str): The distance metric to use for the KNN model.
            Defaults to "cosine".

    Returns:
        Returns the recall@k, NDGC@k, and hr@k.
    """
    # Get the total number of unique materials
    output_size = get_total_materials(history_df)

    # Calculate the history vectors for each customer in the training set
    train_his_vecs = np.array(
        [
            temporal_decay(history_df.loc[customer_id], r_b, r_g, m, output_size)
            for customer_id in train_ids
        ],
    )

    # Calculate the history vectors for each customer in the test set
    test_his_vecs = np.array(
        [
            temporal_decay(history_df.loc[customer_id], r_b, r_g, m, output_size)
            for customer_id in test_ids
        ],
    )
    # TODO: add validation set

    # Calculate the future vectors for each customer in the training set
    logging.info(
        "Calculating the future vectors for each customer in the training set..."
    )
    indices, _ = knn(test_his_vecs, train_his_vecs, k, distance_metric)

    # Merge the history vectors of the train and test sets
    logging.info("Merging the history vectors of the train and test sets...")
    merged_his_vecs = merge_history(
        train_his_vecs, test_his_vecs, train_ids, test_ids, indices, alpha,
    )

    return calculate_metrics(future_df, test_ids, merged_his_vecs, output_size, top_k)

def main(args: argparse.Namespace) -> None:
    """Runs the TIFUKNN model on the specified dataset."""
    # Load the datasets.
    logging.info("Loading the datasets...")
    history_df, future_df = load_data(
        args.history_file, args.future_file, args.min_orders,
    )

    # Create training, validation, and test sets.
    logging.info("Creating the training, validation, and test sets...")
    train_ids, val_ids, test_ids = partition_data_ids(
        history_df, args.train_pct, args.val_pct, args.test_pct, args.seed, args.shuffle,
    )
    logging.info(
        f"Number of training customers: {len(train_ids)} "
        f"({len(train_ids) / len(history_df):.2%})"
    )
    logging.info(
        f"Number of validation customers: {len(val_ids)} "
        f"({len(val_ids) / len(history_df):.2%})"
    )
    logging.info(
        f"Number of test customers: {len(test_ids)} "
        f"({len(test_ids) / len(history_df):.2%})"
    )

    # Evaluate the model.
    logging.info("Evaluating the model...")
    recall, ndcg, f_score = evaluate(
        history_df,
        future_df,
        train_ids,
        val_ids,
        test_ids,
        args.m,
        args.r_b,
        args.r_g,
        args.k,
        args.alpha,
        args.top_k,
        args.distance_metric,
    )
    # Print the results.
    logging.info("Results:")
    logging.info(f"Recall@{args.top_k}: {recall:.4f}")
    logging.info(f"NDCG@{args.top_k}: {ndcg:.4f}")
    logging.info(f"F1@{args.top_k}: {f_score:.4f}")


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments.
    parser.add_argument(
        "history_file", help="The file containing the customer purchase history.",
    )
    parser.add_argument(
        "future_file", help="The file containing the customer future purchases.",
    )

    # Evaluation arguments.
    parser.add_argument(
        "--k", help="The number of nearest neighbors.", type=int, default=300,
    )
    parser.add_argument(
        "--r_b", help="The decay rate within a group.", type=float, default=0.9,
    )
    parser.add_argument(
        "--r_g", help="The decay rate between groups.", type=float, default=0.7,
    )
    parser.add_argument(
        "--alpha", help="The weight of the current group.", type=float, default=0.7,
    )
    parser.add_argument(
        "--m", help="The size of a group.", type=int, default=7,
    )
    parser.add_argument(
        "--top_k", help="The number of top elements.", type=int, default=10,
    )
    parser.add_argument(
        "--distance_metric",
        help="The distance metric to use.",
        choices=["euclidean", "cosine", "manhattan"],
        default="euclidean",
    )

    # Dataset arguments.
    parser.add_argument(
        "--attributes",
        help="The list of attributes to use.",
        nargs="+",
        default=["MATERIAL_NUMBER"],
    )
    parser.add_argument(
        "--min_orders",
        help="The minimum number of orders for a customer to be included.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train_pct",
        help="The percentage of the training set.",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--val_pct",
        help="The percentage of the validation set.",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--test_pct",
        help="The percentage of the test set.",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--shuffle",
        help="Whether to shuffle the data before splitting.",
        action="store_true",
    )

    # Logging arguments.
    parser.add_argument(
        "--logging_level", help="The logging level.", type=int, default=logging.INFO,
    )

    # Other arguments.
    parser.add_argument(
        "--seed", help="The random seed.", type=int, default=42,
    )

    # Parse the arguments.
    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Print command line arguments.
    logging.info(f"{args=}")

    main(args)
