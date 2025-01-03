"""Functions to select the best candidate"""

import numpy as np
from scipy.optimize import minimize_scalar


from src.preprocessing import (
    extract_subsequences,
    get_occ_per_class,
)


def euclidian_distance(series_1: np.ndarray, series_2: np.ndarray) -> float:
    return float((1 / series_1.shape[0]) * np.linalg.norm(series_1 - series_2))


def distance_to_series(target: np.ndarray, series: np.ndarray) -> float:
    subsequences = extract_subsequences(
        series=series, subsequence_length=target.shape[0]
    )
    distances = np.array(
        [euclidian_distance(target, subsequence) for subsequence in subsequences]
    )
    return float(np.min(distances))


def distance_to_all_series(target: np.ndarray, X: np.ndarray) -> np.ndarray:
    if len(X.shape) > 1:
        return np.array([distance_to_series(target, series) for series in X])
    else:
        return np.array([distance_to_series(target, X)])


def split(distances: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    below_threshold_indices = distances < threshold
    above_threshold_indices = distances >= threshold
    return below_threshold_indices, above_threshold_indices


def compute_entropy(dict_occ_by_class: dict[int, int]) -> float:
    total_occurrences = sum(dict_occ_by_class.values())
    probabilities = [occ / total_occurrences for occ in dict_occ_by_class.values()]
    return -float(np.sum(probabilities * np.log2(probabilities)))


def compute_information_gain(
    y: np.ndarray, below_indices: np.ndarray, above_indices: np.ndarray
) -> float:
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
    below_indices, above_indices = split(distances=distances, threshold=threshold)
    return -compute_information_gain(
        y=y, below_indices=below_indices, above_indices=above_indices
    )


def maximize_information_gain(distances: np.ndarray, y: np.ndarray) -> float:
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
) -> tuple[int, int | float | np.ndarray]:
    for i, candidate in enumerate(array_candidates):
        distances = distance_to_all_series(target=candidate, X=X)
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
