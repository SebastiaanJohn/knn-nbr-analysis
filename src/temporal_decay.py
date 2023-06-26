"""Functions for calculating the time-decayed history vector for a given customer."""

import numpy as np
import pandas as pd


def group_history(hist_arr: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    """Group the history vectors into groups of size `group_size`.

    Args:
        hist_arr (np.ndarray): The history vectors to group.
        group_size (int): The size of each group.

    Returns:
        A tuple of two values:
            - The grouped vectors.
            - The number of vectors in the grouped array.
    """
    num_vectors = hist_arr.shape[0]

    if num_vectors < group_size:
        return hist_arr, num_vectors

    base_vectors_per_group = num_vectors // group_size
    leftover_vectors = num_vectors % group_size
    groups_with_extra_vector = leftover_vectors

    indices_base = np.arange(
        0,
        (group_size - groups_with_extra_vector) * base_vectors_per_group,
        base_vectors_per_group,
    )

    indices_extra = np.arange(
        (group_size - groups_with_extra_vector) * base_vectors_per_group,
        num_vectors,
        base_vectors_per_group + 1,
    )

    indices = np.concatenate((indices_base, indices_extra))

    estimated_vectors_per_group = np.concatenate(
        (
            np.full(group_size - groups_with_extra_vector, base_vectors_per_group),
            np.full(groups_with_extra_vector, base_vectors_per_group + 1),
        ),
    )

    grouped_vectors = (
        np.add.reduceat(hist_arr, indices, axis=0)
        / estimated_vectors_per_group[:, None]
    )

    return grouped_vectors, group_size


def temporal_decay(
    customer: pd.Series, r_b: float, r_g: float, m: int, output_size: int
) -> np.ndarray:
    """Calculate the time decayed history vector for a given customer.

    Args:
        customer (pd.Series): Series of lists of material numbers for a given customer.
        r_b (float): The time-decayed ratio within group.
        r_g (float): The time-decayed ratio across the groups.
        m (int): The number of groups to split the history into.
        output_size (int): The number of unique material numbers.


    Returns:
        np.ndarray: The time-decayed history vector for the given customer.
    """
    n = len(customer)
    decayed_vals = np.power(r_b, np.arange(n - 1, -1, -1))

    hist_arr = np.zeros((n, output_size))

    for idx, elements in enumerate(customer):
        hist_arr[idx, np.array(elements) - 1] = decayed_vals[idx]

    grouped_vectors, real_group_size = group_history(hist_arr, m)

    idx = np.arange(real_group_size)
    decayed_vals_group = np.power(r_g, m - 1 - idx)
    his_vec = np.sum(
        grouped_vectors[:real_group_size] * decayed_vals_group[:, np.newaxis], axis=0
    )

    return his_vec / real_group_size
