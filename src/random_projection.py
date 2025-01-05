"""Functions for random projection"""

import numpy as np
from collections import defaultdict


def generate_random_mask(dimensionality: int, proba: float = 0.8) -> np.ndarray:
    """Generate a random mask for random projection.

    Args:
        dimensionality (int): The dimensionality of the mask.
        proba (float): The probability of a 1 in the mask. Default is 0.8.

    Returns:
        np.ndarray: A numpy array representing the random mask.
    """
    if dimensionality == 1:
        mask = np.array([1])
    else:
        mask = np.random.choice([0, 1], size=dimensionality, p=[1 - proba, proba])
    return mask


def all_sax_representations_to_array(
    map_sax_representations: dict[int, str],
) -> np.ndarray:
    """Convert all SAX representations to a 2D numpy array.

    Args:
        map_sax_representations (dict[int, str]): A dictionary mapping indices to SAX representations.

    Returns:
        np.ndarray: A 2D numpy array representing all SAX representations.
    """
    values = list(map_sax_representations.values())
    return np.array([list(value) for value in values], dtype=str)


def generate_random_projection(
    array_sax_representations: np.ndarray, mask: np.ndarray
) -> list[str]:
    """Generate a random projection of SAX representations.

    Args:
        array_sax_representations (np.ndarray): A 2D numpy array representing SAX representations.
        mask (np.ndarray): A numpy array representing the random mask.

    Returns:
        list[str]: A list of strings representing the random projection of SAX representations.
    """
    if array_sax_representations.shape[1] != mask.shape[0]:
        raise ValueError("Incompatible dimensions")
    mask = mask.reshape(-1, 1)
    mask = np.tile(mask, array_sax_representations.shape[0])
    mask = mask.T
    projection = np.where(mask == 1, array_sax_representations, "")
    return ["".join(row) for row in projection]


def create_collision_tables(
    map_sax_representations: dict[int, str], map_series: dict[int, int]
) -> np.ndarray:
    """Create a collision table for SAX representations and series.

    Args:
        map_sax_representations (dict[int, str]): A dictionary mapping indices to SAX representations.
        map_series (dict[int, int]): A dictionary mapping indices to series.

    Returns:
        np.ndarray: A 2D numpy array representing the collision table.
    """
    n_series = len(set(map_series.values()))
    collision_table = np.zeros((len(map_sax_representations), n_series), dtype=int)
    return collision_table


def update_collision_table(
    collision_table: np.ndarray, projection: list[str], map_series: dict[int, int]
) -> np.ndarray:
    """Update the collision table based on the random projection.

    Args:
        collision_table (np.ndarray): A 2D numpy array representing the collision table.
        projection (list[str]): A list of strings representing the random projection of SAX representations.
        map_series (dict[int, int]): A dictionary mapping indices to series.

    Returns:
        np.ndarray: A 2D numpy array representing the updated collision table.
    """
    dict_collisions = defaultdict(lambda: {"indexes": set(), "series": set()})
    for i, projection_sax in enumerate(projection):
        series_id = map_series[i]
        dict_collisions[projection_sax]["indexes"].add(i)
        dict_collisions[projection_sax]["series"].add(series_id)
    for sax in dict_collisions.keys():
        collision_indexes = dict_collisions[sax]["indexes"]
        collision_series = dict_collisions[sax]["series"]
        for idx in collision_indexes:
            for series in collision_series:
                collision_table[idx][series] += 1
    return collision_table


def get_array_occ_by_class(dict_occ_by_class: dict[int, int]) -> np.ndarray:
    """Get the number of occurrences per class as a numpy array.

    Args:
        dict_occ_by_class (dict[int, int]): A dictionary mapping class indices to the number of occurrences.

    Returns:
        np.ndarray: A numpy array representing the number of occurrences per class.
    """
    array_occ_by_class = np.array(
        [dict_occ_by_class[i] for i in range(len(dict_occ_by_class))]
    )
    return array_occ_by_class


def transpose_collision_table_to_classes(
    collision_table: np.ndarray, dict_series_per_class: dict[int, list[int]]
) -> np.ndarray:
    """Transpose the collision table to classes.

    Args:
        collision_table (np.ndarray): A 2D numpy array representing the collision table.
        dict_series_per_class (dict[int, list[int]]): A dictionary mapping class indices to series indices.

    Returns:
        np.ndarray: A 2D numpy array representing the transposed collision table.
    """
    close_table = np.zeros((collision_table.shape[0], len(dict_series_per_class)))
    for class_id, indices in dict_series_per_class.items():
        close_table[:, class_id] = np.sum(collision_table[:, indices], axis=1)
    return close_table


def compute_distinguish_power(
    close_table: np.ndarray, r: int, array_occ_by_class: np.ndarray
) -> np.ndarray:
    """Compute the distinguish power for each SAX representation.

    Args:
        close_table (np.ndarray): A 2D numpy array representing the transposed collision table.
        r (int): The number of random projections.
        array_occ_by_class (np.ndarray): A numpy array representing the number of occurrences per class.

    Returns:
        np.ndarray: A numpy array representing the distinguish power for each SAX representation.
    """
    array_occ = r * array_occ_by_class
    full_table = np.tile(array_occ, (close_table.shape[0], 1))
    far_table = full_table - close_table
    distinguish_table = np.abs(close_table - far_table)
    distinguish_power = np.sum(distinguish_table, axis=1)
    return distinguish_power


def get_top_k_candidates_indexes(distinguish_power: np.ndarray, k: int) -> list[int]:
    """Get the top k candidates based on distinguish power.

    Args:
        distinguish_power (np.ndarray): A numpy array representing the distinguish power for each SAX representation.
        k (int): The number of top candidates to retrieve.

    Returns:
        list[int]: A list of integers representing the indices of the top k candidates.
    """
    top_k_indices = np.argpartition(distinguish_power, -k)[-k:]
    return top_k_indices[::-1].tolist()


def get_candidates_subsequences(
    candidates_indexes: list[int], map_subsequences: dict[int, np.ndarray]
) -> np.ndarray:
    """Get the subsequences of the top k candidates.

    Args:
        candidates_indexes (list[int]): A list of integers representing the indices of the top k candidates.
        map_subsequences (dict[int, np.ndarray]): A dictionary mapping indices to subsequences.

    Returns:
        np.ndarray: A 2D numpy array representing the subsequences of the top k candidates.
    """
    candidates_subsequences = np.array(
        [map_subsequences[idx_candidate] for idx_candidate in candidates_indexes]
    )
    return candidates_subsequences


def compute_candidates(
    params: dict[str, float],
    map_sax_representations: dict[int, str],
    map_series: dict[int, int],
    map_subsequences: dict[int, np.ndarray],
    dict_occ_per_class: dict[int, int],
    dict_series_per_class: dict[int, list[int]],
) -> np.ndarray:
    """Compute the top k candidates based on distinguish power.

    Args:
        params (dict[str, float]): A dictionary containing the parameters for the computation.
        map_sax_representations (dict[int, str]): A dictionary mapping indices to SAX representations.
        map_series (dict[int, int]): A dictionary mapping indices to series.
        map_subsequences (dict[int, np.ndarray]): A dictionary mapping indices to subsequences.
        dict_occ_per_class (dict[int, int]): A dictionary mapping class indices to the number of occurrences.
        dict_series_per_class (dict[int, list[int]]): A dictionary mapping class indices to series indices.

    Returns:
        np.ndarray: A 2D numpy array representing the subsequences of the top k candidates.
    """
    mask = generate_random_mask(
        dimensionality=params["dimensionality"], proba=params["proba"]
    )
    array_sax_representations = all_sax_representations_to_array(
        map_sax_representations=map_sax_representations
    )
    projection = generate_random_projection(
        array_sax_representations=array_sax_representations, mask=mask
    )
    collision_table = create_collision_tables(
        map_sax_representations=map_sax_representations, map_series=map_series
    )
    for _ in range(params["r"]):
        collision_table = update_collision_table(
            collision_table=collision_table,
            projection=projection,
            map_series=map_series,
        )
    array_occ_by_class = get_array_occ_by_class(dict_occ_by_class=dict_occ_per_class)
    close_table = transpose_collision_table_to_classes(
        collision_table=collision_table, dict_series_per_class=dict_series_per_class
    )
    distinguish_power = compute_distinguish_power(
        close_table=close_table, r=params["r"], array_occ_by_class=array_occ_by_class
    )
    candidates_indexes = get_top_k_candidates_indexes(
        distinguish_power=distinguish_power, k=params["k"]
    )
    candidates = get_candidates_subsequences(
        candidates_indexes=candidates_indexes, map_subsequences=map_subsequences
    )
    return candidates
