"""This module implements the TIFUKNN algorithm."""


import argparse
import logging

import numpy as np
import pandas as pd
from data.dataset import load_data, partition_data_ids
from merge_history import merge_customer_history_vectors_dbscan, merge_history
from metrics import calculate_metrics
from tabulate import tabulate
from temporal_decay import temporal_decay
from utils import create_hist_dict, get_total_materials

from models import dbscan, hdbscan, knn


def evaluate(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    m: int = 7,
    r_b: float = 0.9,
    r_g: float = 0.7,
    k: int = 300,
    alpha: float = 0.7,
    top_k: int = 10,
    eps: float = 0.5,
    min_samples: int = 5,
    distance_metric: str = "cosine",
    model: str = "knn",
) -> tuple[float, float, float]:
    """Trains a KNN model and evaluates it on the validation and test sets.

    Args:
        history_df (pd.DataFrame): The dataframe containing the customer purchase history.
        future_df (pd.DataFrame): The dataframe containing the customer future purchases.
        train_ids (np.ndarray): The customer IDs in the training set.
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
        eps (float): The epsilon value for hdbscan/dbscan clustering.
            Defaults to 0.5.
        min_samples (int): The minimum number of samples for hdbscan/dbscan clustering.
            Defaults to 5.
        distance_metric (str): The distance metric to use for the KNN model.
            Defaults to "cosine".
        model (str): The model to use for the recommendations.
            Defaults to "knn".

    Returns:
        Returns the recall@k, NDGC@k, and hr@k.
    """
    # Get the total number of unique materials
    history_size = get_total_materials(history_df)
    future_size = get_total_materials(future_df)
    output_size = max(history_size, future_size)

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

    # Calculate the future vectors for each customer in the training set
    logging.info(f"Calculating the future vectors using {model}...")
    if model == "knn":
        neighbor_indices, _ = knn(test_his_vecs, train_his_vecs, k, distance_metric)
    elif model == "dbscan":
        cluster_labels = dbscan(
            test_his_vecs, eps=eps, min_samples=min_samples, metric=distance_metric
        )
    elif model == "hdbscan":
        cluster_labels = hdbscan(
            test_his_vecs, min_samples=min_samples, metric=distance_metric
        )
    else:
        raise ValueError(
            f"Invalid clustering method: {model}. "
            "Please choose between 'knn', 'dbscan', and 'hdbscan'.",
        )

    # Merge the history vectors of the train and test sets
    logging.info("Merging the history vectors of the train and test sets...")
    if model in {"dbscan", "hdbscan"}:
        train_history_mapping = create_hist_dict(train_ids, train_his_vecs)
        test_history_mapping = create_hist_dict(test_ids, test_his_vecs)
        merged_his_vecs = merge_customer_history_vectors_dbscan(
            train_history_mapping, test_history_mapping, cluster_labels, train_ids, test_ids, alpha
        )
    else:
        merged_his_vecs = merge_history(
            train_his_vecs,
            test_his_vecs,
            train_ids,
            test_ids,
            neighbor_indices,
            alpha,
        )

    return calculate_metrics(future_df, test_ids, merged_his_vecs, output_size, top_k)


def main(args: argparse.Namespace) -> None:
    """Runs the TIFUKNN model on the specified dataset."""
    # Load the datasets.
    logging.info("Loading the datasets...")
    history_df, future_df = load_data(
        args.history_file,
        args.future_file,
        args.min_orders,
    )

    # Create training, validation, and test sets.
    logging.info("Creating the training, validation, and test sets...")
    train_ids, val_ids, test_ids = partition_data_ids(
        history_df,
        args.train_pct,
        args.val_pct,
        args.test_pct,
        args.seed,
        args.shuffle,
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
    metrics = evaluate(
        history_df,
        future_df,
        train_ids,
        test_ids,
        args.m,
        args.r_b,
        args.r_g,
        args.k,
        args.alpha,
        args.top_k,
        args.eps,
        args.min_samples,
        args.distance_metric,
        args.model,
    )
    # Print the results.
    table = tabulate(
        metrics.items(),
        headers=["Metric", "Value"],
        tablefmt="pretty",
        colalign=("left", "left")
    )
    logging.info(f'\n{table}')

if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments.
    parser.add_argument(
        "history_file",
        help="The file containing the customer purchase history.",
    )
    parser.add_argument(
        "future_file",
        help="The file containing the customer future purchases.",
    )

    # Evaluation arguments.
    parser.add_argument(
        "--model",
        help="The model to use.",
        choices=["knn", "dbscan", "hdbscan"],
        default="knn",
    )
    parser.add_argument(
        "--k",
        help="The number of nearest neighbors.",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--r_b",
        help="The decay rate within a group.",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--r_g",
        help="The decay rate between groups.",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--alpha",
        help="The weight of the current group.",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--m",
        help="The size of a group.",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--top_k",
        help="The number of top elements.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eps",
        help="The maximum distance between two samples.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min_samples",
        help="The number of samples in a neighborhood for a point to be considered as a core point.",
        type=int,
        default=5,
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
        "--logging_level",
        help="The logging level.",
        type=int,
        default=logging.INFO,
    )

    # Other arguments.
    parser.add_argument(
        "--seed",
        help="The random seed.",
        type=int,
        default=42,
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
