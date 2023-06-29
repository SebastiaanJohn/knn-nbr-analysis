import pandas as pd
import numpy as np

import argparse
import logging


def load_merged_data(history_file: str, future_file: str):
    """ Load the history and future files, merge them and print statistics of datasets.

    Args:
        history_file (str): Path to history file.
        future_file (str): Path to future file.
    """
    #Read both history and future files, merge them and return the merged data
    history_df = pd.read_csv(history_file)
    future_df = pd.read_csv(future_file)
    merged_df = pd.concat([history_df, future_df])
    
    #Sort the data by customer_id and order_date
    merged_df.sort_values(by=['CUSTOMER_ID', 'ORDER_NUMBER'], inplace=True)
    
    total_users = merged_df.CUSTOMER_ID.nunique()
    total_unique_items = merged_df.MATERIAL_NUMBER.nunique()
    
    #Log the number of users and items
    logging.info(f"Number of users: {total_users}")
    logging.info(f"Number of items: {total_unique_items}")
    
    total_items = merged_df.groupby('CUSTOMER_ID').size().sum()
    total_orders = merged_df.groupby('CUSTOMER_ID')['ORDER_NUMBER'].max().sum()
    
    #Log average items per basket
    logging.info(f"Average items per basket: {total_items/total_orders}")
    
    #Log average baskets per user
    logging.info(f"Average baskets per user: {total_orders/total_users}")
    
    #Get the counts of each item by each user
    item_counts = merged_df.groupby(['CUSTOMER_ID', 'MATERIAL_NUMBER'])
    item_counts = item_counts.size().reset_index(name='COUNT')
    
    # Calculate the counts of unique materials for each customer
    material_counts = merged_df.groupby(['CUSTOMER_ID', 'MATERIAL_NUMBER']).size().reset_index(name='COUNT')
    # Calculate the average counts of unique materials for each customer
    average_counts = material_counts.groupby('CUSTOMER_ID')['COUNT'].mean().reset_index(name='AVERAGE_COUNT')
    # Calculate the mean of the average counts for all customers
    mean_average_count = average_counts['AVERAGE_COUNT'].mean()

    # Log the mean of the average counts
    logging.info(f"Mean of the repetitions of same item by a user: {mean_average_count}")
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "history_dataset", type=str, help="The history dataset to use."
    )
    
    parser.add_argument(
        "future_dataset", type=str, help="The future dataset to use."
    )

    
    # Parse the arguments.
    args = parser.parse_args()

    load_merged_data(args.history_dataset, args.future_dataset)