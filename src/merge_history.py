"""Merge the history vectors of the train and test sets."""

import numpy as np


def merge_history(
    train_his_vecs: np.ndarray,
    test_his_vecs: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Merge the history vectors of the train and test sets.

    Args:
        train_his_vecs (np.ndarray): The history vectors of the train set.
        test_his_vecs (np.ndarray): The history vectors of the test set.
        train_ids (np.ndarray): The customer IDs of the train set.
        test_ids (np.ndarray): The customer IDs of the test set.
        indices (np.ndarray): The indices of the nearest neighbors.
        alpha (float): The weight of the history vector of the query vector.
    """
    # Create a dictionary mapping the customer IDs to their history vectors
    train_his_dict = dict(zip(train_ids, train_his_vecs))
    test_his_dict = dict(zip(test_ids, test_his_vecs))

    # Initialize a list to store the merged history vectors
    merged_his_vecs = []

    # Iterate over the indices of the nearest neighbors
    for idx, row in enumerate(indices):
        # Get the customer ID of the query vector
        query_id = test_ids[idx]
        # Get the customer ID of the k nearest neighbors
        nn_ids = train_ids[row]
        # Get the history vectors of the k nearest neighbors
        nn_his_vecs = np.array([train_his_dict[nn_id] for nn_id in nn_ids])
        # Get the history vector of the query vector
        query_his_vec = test_his_dict[query_id]
        # Calculate the merged history vector
        merged_his_vec = alpha * query_his_vec + (1 - alpha) * np.mean(
            nn_his_vecs, axis=0,
        )
        # Append the merged history vector to the list
        merged_his_vecs.append(merged_his_vec)

    return np.array(merged_his_vecs)


def merge_history_dbscan(
    train_his_dict: dict[str, np.ndarray],
    test_his_dict: dict[str, np.ndarray],
    indices: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Merge the history vectors of the train and test sets using DBSCAN.

    Args:
        train_his_dict (dict[str, np.ndarray]): Mapping of customer IDs to
            history vectors for the train set.
        test_his_dict (dict[str, np.ndarray]): Mapping of customer IDs to
            history vectors for the test set.
        indices (np.ndarray): The indices of the clustering method.
        train_ids (np.ndarray): The customer IDs of the train set.
        test_ids (np.ndarray): The customer IDs of the test set.
        alpha (float): The weight of the history vector of the query vector.

    Returns:
        np.ndarray: The merged history vectors.
    """
    merged_his_vecs = []

    # Iterate over the test_ids and associated cluster labels
    for idx, cluster_label in enumerate(indices):
        # Get the customer ID of the query vector
        query_id = test_ids[idx]

        # Get the history vector of the query vector
        query_his_vec = test_his_dict[query_id]

        if cluster_label != -1:  # Valid cluster
            # Get the IDs of the customers in the same cluster as the test point
            train_cluster_ids = train_ids[np.where(indices == cluster_label)[0]]
            if train_cluster_ids.size > 0:
                # Get the history vectors of the train customers in the same cluster
                nn_his_vecs = np.array(
                    [train_his_dict[train_id] for train_id in train_cluster_ids]
                )
            else:
                nn_his_vecs = np.empty((0, query_his_vec.shape[0]))
        else:    # No valid cluster (noisy point)
            nn_his_vecs = np.empty((0, query_his_vec.shape[0]))

        if nn_his_vecs.size > 0:
            merged_his_vec = alpha * query_his_vec + (1 - alpha) * np.mean(
                nn_his_vecs, axis=0,
            )
            merged_his_vecs.append(merged_his_vec)
        else:
            merged_his_vecs.append(query_his_vec)

    return np.array(merged_his_vecs)
