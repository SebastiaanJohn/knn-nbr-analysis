"""Utility functions for the project."""

import pandas as pd


def get_total_materials(df: pd.DataFrame) -> int:
    """Get total number of unique materials in the dataframe."""
    return max(df.explode().unique())


def get_total_customers(df: pd.DataFrame) -> int:
    """Get total number of unique customers in the dataframe."""
    return len(df.index.get_level_values("CUSTOMER_ID").unique())
