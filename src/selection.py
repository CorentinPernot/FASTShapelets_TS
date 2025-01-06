"""Functions to select the best candidate"""

import numpy as np
from fastdtw import fastdtw
from scipy.optimize import minimize_scalar

from src.preprocessing import (
    extract_subsequences,
    get_occ_per_class,
)


def euclidean_distance(series_1: np.ndarray, series_2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two series.

    Args:
        series_1 (np.ndarray): The first series.
        series_2 (np.ndarray): The second series.

    Returns:
        float: The Euclidean distance between the two series.
    """
    return float((1 / series_1.shape[0]) * np.linalg.norm(series_1 - series_2))


def dtw_distance(series_1: np.ndarray, series_2: np.ndarray) -> float:
    """
    Compute the DTW distance between two series.

    Args:
        series_1 (np.ndarray): The first series.
        series_2 (np.ndarray): The second series.

    Returns:
        float: The DTW distance between the two series.
    """
    distance, _ = fastdtw(series_1, series_2, dist=2)
    return float(distance)


def distance_to_series(
    target: np.ndarray, series: np.ndarray, distance_method: str = "euclidean"
) -> float:
    """
    Compute the distance between a target series and a series.

    Args:
        target (np.ndarray): The target series.
        series (np.ndarray): The series to compare with the target.
        distance_method (str, optional): The method to compute the distance. Defaults to "euclidean".

    Returns:
        float: The distance between the target series and the series.
    """
    subsequences = extract_subsequences(
        series=series, subsequence_length=target.shape[0]
    )
    if distance_method == "euclidean":
        distances = np.array(
            [euclidean_distance(target, subsequence) for subsequence in subsequences]
        )
    elif distance_method == "dtw":
        distances = np.array(
            [dtw_distance(target, subsequence) for subsequence in subsequences]
        )
    else:
        raise ValueError("Invalid distance method. Choose either 'euclidean' or 'dtw'.")
    return float(np.min(distances))


def distance_to_all_series(
    target: np.ndarray, X: np.ndarray, distance_method: str = "euclidean"
) -> np.ndarray:
    """
    Compute the distance between a target series and all series in a dataset.

    Args:
        target (np.ndarray): The target series.
        X (np.ndarray): The dataset of series.
        distance_method (str, optional): The method to compute the distance. Defaults to "euclidean".

    Returns:
        np.ndarray: The distances between the target series and all series in the dataset.
    """
    if len(X.shape) > 1:
        return np.array(
            [
                distance_to_series(
                    target=target, series=series, distance_method=distance_method
                )
                for series in X
            ]
        )
    else:
        return np.array(
            [
                distance_to_series(
                    target=target, series=X, distance_method=distance_method
                )
            ]
        )


def split(distances: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the indices of distances based on a threshold.

    Args:
        distances (np.ndarray): The distances.
        threshold (float): The threshold to split the indices.

    Returns:
        tuple: The indices of distances below the threshold and the indices of distances above the threshold.
    """
    below_threshold_indices = distances < threshold
    above_threshold_indices = distances >= threshold
    return below_threshold_indices, above_threshold_indices


def compute_entropy(dict_occ_by_class: dict[int, int]) -> float:
    """
    Compute the entropy of a dictionary of occurrences by class.

    Args:
        dict_occ_by_class (dict[int, int]): The dictionary of occurrences by class.

    Returns:
        float: The entropy of the dictionary.
    """
    total_occurrences = sum(dict_occ_by_class.values())
    probabilities = [occ / total_occurrences for occ in dict_occ_by_class.values()]
    return -float(np.sum(probabilities * np.log2(probabilities)))


def compute_information_gain(
    y: np.ndarray, below_indices: np.ndarray, above_indices: np.ndarray
) -> float:
    """
    Compute the information gain of splitting a dataset based on a threshold.

    Args:
        y (np.ndarray): The target variable.
        below_indices (np.ndarray): The indices of the dataset below the threshold.
        above_indices (np.ndarray): The indices of the dataset above the threshold.

    Returns:
        float: The information gain of the split.
    """
    y_below = y[below_indices]
    y_above = y[above_indices]
    n = y.shape[0]
    n_below = y_below.shape[0]
    n_above = y_above.shape[0]
    if n_below == 0 or n_above == 0:
        return 0
    entropy_total = compute_entropy(get_occ_per_class(y=y))
    entropy_below = compute_entropy(get_occ_per_class(y=y_below))
    entropy_above = compute_entropy(get_occ_per_class(y=y_above))
    return entropy_total - (n_below / n) * entropy_below - (n_above / n) * entropy_above


def function_to_minimize(
    threshold: float, distances: np.ndarray, y: np.ndarray
) -> float:
    """
    Function to minimize for finding the threshold that maximizes information gain.

    Args:
        threshold (float): The threshold to minimize.
        distances (np.ndarray): The distances.
        y (np.ndarray): The target variable.

    Returns:
        float: The negative information gain of the split.
    """
    below_indices, above_indices = split(distances=distances, threshold=threshold)
    return -compute_information_gain(
        y=y, below_indices=below_indices, above_indices=above_indices
    )


def maximize_information_gain(distances: np.ndarray, y: np.ndarray) -> float:
    """
    Find the threshold that maximizes information gain.

    Args:
        distances (np.ndarray): The distances.
        y (np.ndarray): The target variable.

    Returns:
        float: The threshold that maximizes information gain.
    """
    bounds = (0, distances.max())
    result = minimize_scalar(
        function_to_minimize,
        args=(distances, y),
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def compute_gap(
    distances: np.ndarray, below_indices: np.ndarray, above_indices: np.ndarray
) -> float:
    """
    Compute the gap between the average distances of two sets of indices.

    Args:
        distances (np.ndarray): The distances.
        below_indices (np.ndarray): The indices of the dataset below the threshold.
        above_indices (np.ndarray): The indices of the dataset above the threshold.

    Returns:
        float: The gap between the average distances of the two sets.
    """
    n_below = below_indices.shape[0]
    n_above = above_indices.shape[0]
    if n_below == 0 or n_above == 0:
        return 0
    below_distances = distances[below_indices]
    above_distances = distances[above_indices]
    return float(np.abs(np.mean(above_distances) - np.mean(below_distances)))


def select_best_candidate(
    array_candidates: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    params_best_candidate: dict[str, float],
    distance_method: str = "euclidean",
) -> tuple[int, int | float | np.ndarray]:
    """
    Select the best candidate from an array of candidates based on maximizing information gain and minimizing gap.

    Args:
        array_candidates (np.ndarray): The array of candidates.
        X (np.ndarray): The dataset of series.
        y (np.ndarray): The target variable.
        params_best_candidate (dict[str, float]): The parameters of the best candidate found so far.
        distance_method (str, optional): The method to compute the distance. Defaults to "euclidean".

    Returns:
        tuple: The index of the best candidate in the array, the best candidate, the threshold, and the information gain and gap.
    """
    for i, candidate in enumerate(array_candidates):
        distances = distance_to_all_series(
            target=candidate, X=X, distance_method=distance_method
        )
        threshold = maximize_information_gain(distances=distances, y=y)
        below_indices, above_indices = split(distances=distances, threshold=threshold)
        info_gain = compute_information_gain(
            y=y, below_indices=below_indices, above_indices=above_indices
        )
        gap = compute_gap(
            distances=distances,
            below_indices=below_indices,
            above_indices=above_indices,
        )
        if info_gain > params_best_candidate["info_gain"] or (
            info_gain == params_best_candidate["info_gain"]
            and gap > params_best_candidate["gap"]
        ):
            params_best_candidate["info_gain"] = info_gain
            params_best_candidate["gap"] = gap
            params_best_candidate["shapelet"] = candidate
            params_best_candidate["threshold"] = threshold
    return params_best_candidate
