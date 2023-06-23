"""Merge the history vectors of the train and test sets."""

import numpy as np


def merge_history(
    train_history_vectors: np.ndarray,
    test_history_vectors: np.ndarray,
    train_customer_ids: np.ndarray,
    test_customer_ids: np.ndarray,
    nearest_neighbor_indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Merge the history vectors of the train and test sets.

    Args:
        train_history_vectors (np.ndarray): The history vectors of the train set.
        test_history_vectors (np.ndarray): The history vectors of the test set.
        train_customer_ids (np.ndarray): The customer IDs of the train set.
        test_customer_ids (np.ndarray): The customer IDs of the test set.
        nearest_neighbor_indices (np.ndarray): The indices of the nearest neighbors.
        alpha (float): The weight of the history vector of the query vector.
    """
    # Create dictionaries mapping customer IDs to their history vectors
    train_history_dict = dict(zip(train_customer_ids, train_history_vectors))
    test_history_dict = dict(zip(test_customer_ids, test_history_vectors))

    # Initialize a list to store the merged history vectors
    merged_history_vectors = []

    # Iterate over the indices of the nearest neighbors
    for index, row in enumerate(nearest_neighbor_indices):
        # Get the customer ID of the current query vector
        query_customer_id = test_customer_ids[index]
        # Get the customer IDs of the k nearest neighbors
        nearest_neighbor_ids = train_customer_ids[row]
        # Get the history vectors of the k nearest neighbors
        nearest_neighbor_history_vectors = np.array(
            [train_history_dict[neighbor_id] for neighbor_id in nearest_neighbor_ids]
        )
        # Get the history vector of the query vector
        query_history_vector = test_history_dict[query_customer_id]
        # Calculate the merged history vector
        merged_history_vector = alpha * query_history_vector + (1 - alpha) * np.mean(
            nearest_neighbor_history_vectors,
            axis=0,
        )
        # Append the merged history vector to the list
        merged_history_vectors.append(merged_history_vector)

    return np.array(merged_history_vectors)


def merge_customer_history_vectors_dbscan(
    train_history_mapping: dict[str, np.ndarray],
    test_history_mapping: dict[str, np.ndarray],
    dbscan_cluster_labels: np.ndarray,
    train_customer_ids: np.ndarray,
    test_customer_ids: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Merge the history vectors of the train and test sets using DBSCAN clustering.

    Args:
        train_history_mapping (dict[str, np.ndarray]): Mapping of customer IDs to
            history vectors for the train set.
        test_history_mapping (dict[str, np.ndarray]): Mapping of customer IDs to
            history vectors for the test set.
        dbscan_cluster_labels (np.ndarray): The DBSCAN cluster labels for each point in the test set.
        train_customer_ids (np.ndarray): The customer IDs of the train set.
        test_customer_ids (np.ndarray): The customer IDs of the test set.
        alpha (float): The weight of the history vector of the query vector.

    Returns:
        np.ndarray: The merged history vectors.
    """
    merged_history_vectors = []

    # Precompute indices for each cluster
    cluster_to_indices_mapping = {}
    for cluster_label in set(dbscan_cluster_labels):
        if cluster_label != -1:
            cluster_to_indices_mapping[cluster_label] = np.where(
                dbscan_cluster_labels == cluster_label
            )[0]

    # Iterate over the test customer IDs and associated cluster labels
    for index, cluster_label in enumerate(dbscan_cluster_labels):
        # Get the customer ID of the current query vector
        query_customer_id = test_customer_ids[index]

        # Get the history vector of the current query vector
        query_history_vector = test_history_mapping[query_customer_id]

        # Initialize nearest_neighbor_history_vectors as empty
        nearest_neighbor_history_vectors = np.empty((0, query_history_vector.shape[0]))

        # If valid cluster, retrieve history vectors of train customers in the same cluster
        if cluster_label in cluster_to_indices_mapping:
            train_cluster_customer_ids = train_customer_ids[
                cluster_to_indices_mapping[cluster_label]
            ]
            if train_cluster_customer_ids.size > 0:
                nearest_neighbor_history_vectors = np.array(
                    [
                        train_history_mapping[train_customer_id]
                        for train_customer_id in train_cluster_customer_ids
                    ]
                )

        # Calculate the merged history vector
        if nearest_neighbor_history_vectors.size > 0:
            merged_history_vector = alpha * query_history_vector + (
                1 - alpha
            ) * np.mean(nearest_neighbor_history_vectors, axis=0)
        else:
            merged_history_vector = query_history_vector

        # Append the merged history vector to the list
        merged_history_vectors.append(merged_history_vector)

    return np.array(merged_history_vectors)

