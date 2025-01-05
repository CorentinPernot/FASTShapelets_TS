"""Functions for bruteforce shapelets"""

import numpy as np

from src.preprocessing import extract_subsequences, z_normalize_2d

from src.selection import select_best_candidate


def get_candidates(X_train: np.ndarray, subsequence_length: int) -> np.ndarray:
    list_candidates = []
    for i in range(X_train.shape[0]):
        series = X_train[i]
        candidates = extract_subsequences(
            series=series, subsequence_length=subsequence_length
        )
        list_candidates.append(candidates)
    return np.concatenate(list_candidates)


def compute_bf_shapelets(
    X_train: np.ndarray, y_train: np.ndarray, subsequence_length: int
) -> dict:
    X_train = z_normalize_2d(X=X_train)
    params_best_candidate = {"info_gain": 0.0, "gap": 0.0}
    candidates = get_candidates(X_train=X_train, subsequence_length=subsequence_length)
    params_best_candidate = select_best_candidate(
        array_candidates=candidates,
        X=X_train,
        y=y_train,
        params_best_candidate=params_best_candidate,
    )
    return params_best_candidate
