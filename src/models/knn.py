"""k-Nearest Neighbors (kNN) model for finding the k nearest neighbors for a given query set in a target set."""


import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn(
    query_vec: np.ndarray, target_vec: np.ndarray, k: int, distance_metric: str = "euclidean"
) -> tuple[np.ndarray, np.ndarray]:
    """Find the k nearest neighbors of a query vector within a target vector.

    Args:
        query_vec (np.ndarray): The query vector.
        target_vec (np.ndarray): The target vector.
        k (int): The number of nearest neighbors to find.
        distance_metric (str, optional): The distance metric to use.
            Defaults to "euclidean".

    Returns:
        tuple[np.ndarray, np.ndarray]: The indices and distances of the k nearest neighbors.
    """
    # Create a NearestNeighbors object and fit the target vector
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=distance_metric).fit(
        target_vec
    )
    # Find the k nearest neighbors of the query vector
    distances, indices = nbrs.kneighbors(query_vec)
    return indices, distances
