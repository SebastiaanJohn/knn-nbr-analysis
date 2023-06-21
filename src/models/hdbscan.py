"""HDBSCAN model."""

import numpy as np
from hdbscan import HDBSCAN


def hdbscan(
    query_vec: np.ndarray, min_samples: int = 5, metric: str = "cosine"
) -> np.ndarray:
    """Cluster the query vectors using HDBSCAN.

    Args:
        query_vec (np.ndarray): The query vectors.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        metric (str): The metric to use when calculating distance between instances in a feature array.

    Returns:
        np.ndarray: The cluster labels for each query vector.
    """
    clustering = HDBSCAN(min_samples=min_samples, metric=metric).fit(query_vec)

    return clustering.labels_
