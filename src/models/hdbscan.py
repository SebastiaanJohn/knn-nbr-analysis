"""HDBSCAN model."""

import numpy as np
from hdbscan import HDBSCAN


def hdbscan(
    query_vec: np.ndarray,
    min_samples: int = 5,
    min_cluster_size: int = 5,
    metric: str = "euclidean",
) -> np.ndarray:
    """Cluster the query vectors using HDBSCAN.

    Args:
        query_vec (np.ndarray): The query vectors.
        min_samples (int): The number of samples in a neighborhood for a
            point to be considered as a core point. Defaults to 5.
        min_cluster_size (int): The minimum number of samples in a cluster.
            Defaults to 5.
        metric (str): The metric to use when calculating distance between instances
            in a feature array. Defaults to "euclidean".

    Returns:
        np.ndarray: The cluster labels for each query vector.
    """
    clustering = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        metric=metric
    ).fit(query_vec)

    return clustering.labels_
