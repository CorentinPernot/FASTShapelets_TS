"""Functions for bruteforce shapelets"""

import numpy as np

from src.preprocessing import extract_subsequences, z_normalize_2d

from src.selection import select_best_candidate


def get_candidates(X_train: np.ndarray, subsequence_length: int) -> np.ndarray:
    """
    Get all candidates of a given length.

    Args:
        X_train (np.ndarray): Train data.
        subsequence_length (int): Length to look for.

    Returns:
        np.ndarray: Candidates of selected length.
    """
    list_candidates = []
    for i in range(X_train.shape[0]):
        series = X_train[i]
        candidates = extract_subsequences(
            series=series, subsequence_length=subsequence_length
        )
        list_candidates.append(candidates)
    return np.concatenate(list_candidates)


def compute_bf_shapelets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subsequence_length: int,
    distance_method: str = "euclidean",
) -> dict:
    """
    Compute bruteforce shapelets algorithm.

    Args:
        X_train (np.ndarray): Train data.
        y_train (np.ndarray): Train labels.
        subsequence_length (int): Length of shapelet to look for.
        distance_method (str, optional): Method to calculate distances. Defaults to "euclidean".

    Returns:
        dict: Parameters of the selected shapelet.
    """
    X_train = z_normalize_2d(X=X_train)
    params_best_candidate = {"info_gain": 0.0, "gap": 0.0}
    candidates = get_candidates(X_train=X_train, subsequence_length=subsequence_length)
    params_best_candidate = select_best_candidate(
        array_candidates=candidates,
        X=X_train,
        y=y_train,
        params_best_candidate=params_best_candidate,
        distance_method=distance_method,
    )
    return params_best_candidate
