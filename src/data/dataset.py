"""This module contains functions for loading and partitioning the dataset."""

import logging

import numpy as np
import pandas as pd


def load_data(
    history_file: str,
    future_file: str,
    min_orders: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for the given file.

    We remove customers who have less then k orders in one of the two dataframes.
    If a customer has less than k orders in the history dataframe, we remove
    all of their orders from the future dataframe as well.

    Args:
        history_file (str): CSV file to read the data from.
            The file should have the following columns:
            - CUSTOMER_ID
            - ORDER_NUMBER
            - MATERIAL_NUMBER
        future_file (str): CSV file to read the data from with
            the same columns as history_file.
        min_orders (int): The minimum number of orders a customer must have
            in both the history and future dataframes. Defaults to 1.

    Returns:
        A tuple of two dataframes. The first dataframe is the history
        dataframe and the second is the future dataframe, both with
        the following structure:
            The dataframe is grouped by CUSTOMER_ID and ORDER_NUMBER.
            The MATERIAL_NUMBER column is converted to a list of values.
            e.g.,
                CUSTOMER_ID ORDER_NUMBER MATERIAL_NUMBER
                1           1            [1, 2, 3]
                            2            [4, 5, 6, 7, 8]
    """
    def filter_data(file: str, is_future: bool = False) -> tuple[pd.DataFrame, set]:
        df = (
            pd.read_csv(file)
            .groupby(["CUSTOMER_ID", "ORDER_NUMBER"])
            .apply(lambda x: sorted(x["MATERIAL_NUMBER"].values))
        )
        customers_to_keep = df.reset_index()["CUSTOMER_ID"].value_counts()
        if is_future:
            return df, set(customers_to_keep[customers_to_keep >= min_orders].index)
        return df, set(customers_to_keep[customers_to_keep > min_orders].index)

    # Load and filter data
    history_df, past_customers_to_keep = filter_data(history_file)
    future_df, future_customers_to_keep = filter_data(future_file, is_future=True)

    # Find the intersection of customers who have at least k transactions in both
    # the history and future
    customers_to_keep = past_customers_to_keep.intersection(future_customers_to_keep)

    # Log the number of customers we deleted
    total = len(history_df.index.get_level_values("CUSTOMER_ID").unique())
    logging.info(
        f"Removed {total - len(customers_to_keep)} of {total} customers with <{min_orders} orders.")

     # Filter both DataFrames to keep only the customers in `customers_to_keep`
    history_df = history_df[
        history_df.index.get_level_values("CUSTOMER_ID").isin(customers_to_keep)
    ]
    future_df = future_df[
        future_df.index.get_level_values("CUSTOMER_ID").isin(customers_to_keep)
    ]

    # Return the processed dataframes
    return history_df, future_df


def partition_data_ids(
    history_df: pd.DataFrame,
    train_pct: float = 0.72,
    val_pct: float = 0.08,
    test_pct: float = 0.20,
    seed: int = 42,
    shuffle: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition the customer IDs into training, validation, and test datasets.

    Args:
        history_df (pd.DataFrame): The dataframe with the history data.
        train_pct (float): The proportion of data to use for training. Defaults to 0.72.
        val_pct (float): The proportion of data to use for validation. Defaults to 0.08.
        test_pct (float): The proportion of data to use for testing. Defaults to 0.20.
        seed (int): The random seed to use for shuffling the data. Defaults to 42.
        shuffle (bool): Whether to shuffle the data. Defaults to True.

    Returns:
        Three numpy arrays: train_ids, val_ids, test_ids.
    """
    rng = np.random.default_rng(seed)
    assert train_pct + val_pct + test_pct == 1, "The percentages must add up to 1."

    # Extract all unique customer IDs
    customer_ids = history_df.index.get_level_values("CUSTOMER_ID").unique().to_numpy()

    # Shuffle the customer IDs
    if shuffle:
        rng.shuffle(customer_ids)

    # Determine the cutoffs for the training and validation sets
    total_customers = len(customer_ids)
    train_end_idx = int(train_pct * total_customers)
    val_end_idx = train_end_idx + int(val_pct * total_customers)

    train_ids = customer_ids[:train_end_idx]
    val_ids = customer_ids[train_end_idx:val_end_idx]
    test_ids = customer_ids[val_end_idx:]

    return train_ids, val_ids, test_ids