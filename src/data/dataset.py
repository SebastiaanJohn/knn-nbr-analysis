"""This module contains functions for loading and partitioning the dataset."""

import argparse
import logging
from datetime import datetime

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


def handle_lastfm(
    path: str,
    months_for_baskets: int = 6,
    listen_threshold: int = 10
) -> None:
    """Handle the LastFM dataset.

    Args:
        path (str): The path to the LastFM dataset.
        months_for_baskets (int): The number of months to use for each basket.
            Defaults to 6.
    """
    # Load the data
    df = pd.read_csv(path, delimiter="\t", header=None, error_bad_lines=False)
    df = df.dropna(subset=[5]) # Drop rows with missing song names
    
    #Get all the songnames into a list, map each songname to a unique id
    song_names = df[5].unique()
    
    # Find the counts of each song
    song_counts = df[5].value_counts()
    
    # Remove songs that are played less than threshold
    logging.info(f"Number of unique songs before removing: {len(song_names)}")
    song_names = song_names[song_counts > listen_threshold]
    logging.info(f"Number of unique songs after removing: {len(song_names)}")
    
    song_ids = {}
    for idx, song_name in enumerate(song_names):
        song_ids[song_name] = idx + 1

    # Read line by line and create a dictionary of users
    users = {}
    current_user = 0
    for index, row in df.iterrows():
        song_name = row[5]
        # Check if the song is in the keys of the dictionary
        if song_name not in song_ids:
            continue
        user = int(row[0][-5::])
        if user > 10:
            break
        if user % 2 == 0 and user != current_user:
            current_user = user
            logging.info(f"Reading current user: {user}/1000")
        date = datetime.strptime(row[1], "%Y-%m-%dT%H:%M:%SZ")
        song_id = song_ids[song_name]
        if user not in users:
            users[user] = []
        users[user].append((date, song_id))
    
    # For every user, sort the songs by date
    for user in users:
        users[user].sort(key=lambda x: x[0])
        
    # For every user, create baskets
    for user in users:
        first_date = users[user][0][0]
        # Fix first date to beginning of the month
        first_date = datetime(first_date.year, first_date.month, 1)
        for i in range(len(users[user])):
            difference = int((users[user][i][0] - first_date).days / 30)
            basket_id = int(difference / months_for_baskets) + 1
            users[user][i] = (basket_id, users[user][i][1])
    
    # Create the future and history dataframes
    future_df = pd.DataFrame(columns=["CUSTOMER_ID", "ORDER_NUMBER", "MATERIAL_NUMBER"])
    history_df = pd.DataFrame(columns=["CUSTOMER_ID", "ORDER_NUMBER", "MATERIAL_NUMBER"])
    for user in users:
        max_basket_id = max([x[0] for x in users[user]])
        half = int((max_basket_id + 1) / 2)
        history = []
        future = []
        # Split the baskets into history and future
        for basket_id, song_id in users[user]:
            if basket_id <= half:
                history.append((basket_id, song_id))
            else:
                future.append((basket_id, song_id))

        # Create the history dataframe
        history_df = pd.concat(
            [
                history_df,
                pd.DataFrame(
                    {
                        "CUSTOMER_ID": user,
                        "ORDER_NUMBER": [x[0] for x in history],
                        "MATERIAL_NUMBER": [x[1] for x in history],
                    }
                ),
            ]
        )
        
        # Create the future dataframe
        future_df = pd.concat(
            [
                future_df,
                pd.DataFrame(
                    {
                        "CUSTOMER_ID": user,
                        "ORDER_NUMBER": [x[0] for x in future],
                        "MATERIAL_NUMBER": [x[1] for x in future],
                    }
                ),
            ]
        )
    
    # Save the dataframes
    history_df.to_csv("data/lastfm_history.csv", index=False)
    future_df.to_csv("data/lastfm_future.csv", index=False)
        
        
    
def main(args: argparse.Namespace) -> None:
    # Give error if dataset is not lastfm or mmtd
    if args.dataset not in ["lastfm", "mmtd"]:
        raise ValueError("Dataset must be either 'lastfm' or 'mmtd'.")
    
    # Handle the LastFM dataset
    if args.dataset == "lastfm":
        handle_lastfm(args.path, args.months_for_baskets, args.listen_threshold)
        
    # Handle the MMTD dataset
    elif args.dataset == "mmtd":
        pass


if __name__ == "__main__":
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset", type=str, help="The dataset to use.", default="lastfm"
    )
    
    parser.add_argument(
        "--path", type=str, help="The path to read the data from.", default="data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
    )
    
    parser.add_argument(
        "--months_for_baskets", help="Time intervals to create baskets.", type=int, default=6
    )
    
    parser.add_argument(
        "--listen_threshold", help="If a song is listened less then the amount, it is removed.", type=int, default=10
    )
    
    # Parse the arguments.
    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Print command line arguments.
    logging.info(f"{args=}")

    main(args)