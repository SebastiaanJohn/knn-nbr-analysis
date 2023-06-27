import argparse
import functools
import logging

import numpy as np
import pandas as pd
from eval import evaluate
from tabulate import tabulate

from data.dataset import load_data, partition_data_ids


class DatasetNotFoundError(Exception):
    """Raised when the dataset is not found in the BEST_PARAMS dictionary."""
    pass

# Parameters from the paper
PAPER_PARAMS = {
    "VS_history_order": {"k": 300, "m": 7, "r_b": 1, "r_g": 0.6, "alpha": 0.7},
    "Instacart_history": {"k": 900, "m": 3, "r_b": 0.9, "r_g": 0.7, "alpha": 0.9},
    "Dunnhumby_history": {"k": 900, "m": 3, "r_b": 0.9, "r_g": 0.6, "alpha": 0.2},
    "TaFang_history_NB": {"k": 300, "m": 7, "r_b": 0.9, "r_g": 0.7, "alpha": 0.7},
}


@functools.lru_cache
def get_data_for_search(
    history_file: str,
    future_file: str,
    min_orders: int,
    split_percents: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    shuffle: bool = False,
) -> tuple:
    """Get necessary data for grid search.

    Returns:
        history_df: history dataframe
        future_df: future dataframe
        train_ids: training set ids
        val_ids: validation set ids
        test_ids: test set ids
        dataset_name: name of the dataset
    """
    logging.info("Loading the datasets...")
    history_df, future_df = load_data(
        history_file, future_file, min_orders
    )
    dataset_name = history_file.split("/")[-1].split(".")[0]

    # Create training, validation, and test sets
    logging.info("Creating the training and validation sets...")
    train_pct, val_pct, test_pct = split_percents
    train_ids, val_ids, test_ids = partition_data_ids(
        history_df, train_pct, val_pct, test_pct, seed, shuffle
    )

    return history_df, future_df, train_ids, val_ids, test_ids, dataset_name


def evaluate_params(
    args: argparse.Namespace,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    dataset_params: dict,
    extra_params: dict,
    dataset_name: str,
) -> dict[str, float]:
    """Evaluate current parameter combination.

    Returns:
        metrics: dictionary of metrics for the current parameter combination
    """
    logging.info(f"Evaluating with {extra_params}, data={dataset_name}")
    metrics = evaluate(
        history_df,
        future_df,
        train_ids,
        val_ids,
        m=dataset_params["m"],
        r_b=dataset_params["r_b"],
        r_g=dataset_params["r_g"],
        k=dataset_params["k"],
        alpha=dataset_params["alpha"],
        top_k=args.top_k,
        distance_metric=args.distance_metric,
        model=args.model,
        **extra_params,
    )
    return metrics


def perform_grid_search(
    args: argparse.Namespace,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    dataset_name: str,
) -> tuple[float, dict]:
    """Perform grid search based on various parameters for dbscan model and hdbscan model.

    Returns:
        best_score: best recall@k score
        best_params: best parameters
    """
    # Assuming that metrics is a dictionary with a key 'score' representing the evaluation score
    dataset_params = PAPER_PARAMS.get(dataset_name, None)
    if not dataset_params:
        raise DatasetNotFoundError(f"Dataset {dataset_name} not found in PAPER_PARAMS")

    best_score = -np.inf
    best_params = None
    params_iter = {
        "dbscan": [
            {"eps": eps, "min_samples": min_samples}
            for eps in args.eps_values
            for min_samples in args.min_samples_values
        ],
        "hdbscan": [
            {"min_cluster_size": min_cluster_size, "min_samples": min_samples}
            for min_cluster_size in args.min_cluster_size_values
            for min_samples in args.min_samples_values
        ],
    }

    for extra_params in params_iter[args.model]:
        metrics = evaluate_params(
            args,
            history_df,
            future_df,
            train_ids,
            val_ids,
            dataset_params,
            extra_params,
            dataset_name,
        )
        score = metrics.get(f"Recall@{args.top_k}")
        if score > best_score:
            logging.info(
                f"New best score: {score}, using parameters: {extra_params}, {dataset_name}"
            )
            best_score = score
            best_params = extra_params

    return best_score, best_params


def grid_search(args: argparse.Namespace) -> None:
    """Performs grid search on a specified dataset."""
    split_percents = (args.train_pct, args.val_pct, args.test_pct)
    assert sum(split_percents) == 1.0, "Split percentages must sum to 1.0"
    (
        history_df,
        future_df,
        train_ids,
        val_ids,
        test_ids,
        dataset_name,
    ) = get_data_for_search(
        args.history_file,
        args.future_file,
        args.min_orders,
        split_percents,
        args.seed,
        args.shuffle,
    )

    best_score, best_params = perform_grid_search(
        args, history_df, future_df, train_ids, val_ids, dataset_name
    )

    # Print the best parameters
    logging.info(
        f"Best parameters: {best_params} with Recall@{args.top_k}: {best_score}"
    )

    # Evaluate on the test set
    logging.info("Evaluating on the test set...")
    metrics = evaluate_params(
        args, history_df, future_df, train_ids, test_ids, dataset_name, best_params
    )

    # Print the results.
    table = tabulate(
        metrics.items(),
        headers=["Metric", "Value"],
        tablefmt="pretty",
        colalign=("left", "left"),
    )
    logging.info(f"\n{table}")


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
        choices=["dbscan", "hdbscan"],
        default="dbscan",
    )
    parser.add_argument(
        "--top_k",
        help="The number of top elements.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eps_values",
        type=list,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument(
        "--min_cluster_size_values",
        help="The minimum size of clusters.",
        type=list,
        default=[2, 3, 4, 5, 10, 15, 20, 25, 30],
    )
    parser.add_argument(
        "--min_samples_values",
        type=list,
        default=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
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

    grid_search(args)
