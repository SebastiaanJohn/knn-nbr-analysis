"""DBSCAN model."""

import numpy as np
from sklearn.cluster import DBSCAN


def dbscan(
    query_vec: np.ndarray, eps: float = 0.5, min_samples: int = 5, metric: str = "cosine"
) -> np.ndarray:
    """Cluster the query vectors using DBSCAN.

    Args:
        query_vec (np.ndarray): The query vectors.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        metric (str): The metric to use when calculating distance between instances in a feature array.

    Returns:
        np.ndarray: The cluster labels for each query vector.
    """
    clustering = DBSCAN(eps=eps, metric=metric, min_samples=min_samples).fit(query_vec)

    return clustering.labels_
