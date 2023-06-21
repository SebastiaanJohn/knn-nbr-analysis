"""Utility functions for the project."""

import numpy as np
import pandas as pd


def get_total_materials(df: pd.DataFrame) -> int:
    """Get total number of unique materials in the dataframe."""
    return max(df.explode().unique())


def get_total_customers(df: pd.DataFrame) -> int:
    """Get total number of unique customers in the dataframe."""
    return len(df.index.get_level_values("CUSTOMER_ID").unique())

def create_hist_dict(train_ids: list[int], train_his_vecs: np.ndarray) -> dict[int, np.ndarray]:
    """Create a dictionary of customer ids and their corresponding historical vectors."""
    return dict(zip(train_ids, train_his_vecs))
