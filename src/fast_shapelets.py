"""Function to compute fast shapelets"""

import numpy as np
from joblib import Parallel, delayed
from src.preprocessing import (
    load_data_from_hf,
    z_normalize_2d,
    get_occ_per_class,
    get_series_per_class,
)
from src.sax import get_all_sax_representations
from src.random_projection import compute_candidates
from src.selection import select_best_candidate


def compute_fast_shapelets(
    X_train: np.ndarray, y_train: np.ndarray, params: dict[str, float]
) -> dict:
    """
    Compute fast shapelets using random projection and SAX.

    Args:
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels.
        params (dict[str, float]): The parameters for computing fast shapelets.

    Returns:
        dict: The parameters of the best candidate shapelet.
    """
    X_train = z_normalize_2d(X=X_train)
    dict_occ_per_class = get_occ_per_class(y=y_train)
    dict_series_per_class = get_series_per_class(y=y_train)
    params_best_candidate = {"info_gain": 0.0, "gap": 0.0}
    current_params = params.copy()
    for subsequence_length in range(X_train.shape[1]):
        current_params["subsequence_length"] = subsequence_length + 1
        current_params["dimensionality"] = params["dimensionality"]
        map_sax_representations, map_series, map_classes, map_subsequences = (
            get_all_sax_representations(X=X_train, y=y_train, params=current_params)
        )
        candidates = compute_candidates(
            params=current_params,
            map_sax_representations=map_sax_representations,
            map_series=map_series,
            map_subsequences=map_subsequences,
            dict_occ_per_class=dict_occ_per_class,
            dict_series_per_class=dict_series_per_class,
        )
        params_best_candidate = select_best_candidate(
            array_candidates=candidates,
            X=X_train,
            y=y_train,
            params_best_candidate=params_best_candidate,
        )
    return params_best_candidate


def function_to_parallelize(
    subsequence_length: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    current_params: dict,
    dict_occ_per_class: dict,
    dict_series_per_class: dict,
    params: dict[str, int | float],
) -> dict:
    """
    Function to parallelize the computation of fast shapelets.

    Args:
        subsequence_length (int): The length of the subsequence.
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels.
        current_params (dict): The current parameters for computing fast shapelets.
        dict_occ_per_class (dict): The occurrence per class.
        dict_series_per_class (dict): The series per class.
        params (dict[str, int | float]): The parameters for computing fast shapelets.

    Returns:
        dict: The parameters of the best candidate shapelet for the current subsequence length.
    """
    current_params["subsequence_length"] = subsequence_length + 1
    current_params["dimensionality"] = params["dimensionality"]
    map_sax_representations, map_series, map_classes, map_subsequences = (
        get_all_sax_representations(X=X_train, y=y_train, params=current_params)
    )
    candidates = compute_candidates(
        params=current_params,
        map_sax_representations=map_sax_representations,
        map_series=map_series,
        map_subsequences=map_subsequences,
        dict_occ_per_class=dict_occ_per_class,
        dict_series_per_class=dict_series_per_class,
    )
    params_best_candidate = {"info_gain": 0.0, "gap": 0.0}
    params_best_candidate = select_best_candidate(
        array_candidates=candidates,
        X=X_train,
        y=y_train,
        params_best_candidate=params_best_candidate,
    )
    return params_best_candidate


def compute_fast_shapelets_parallelized(
    X_train: np.ndarray, y_train: np.ndarray, params: dict[str, int | float]
) -> dict:
    """
    Compute fast shapelets using random projection and SAX in parallel.

    Args:
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels.
        params (dict[str, int | float]): The parameters for computing fast shapelets.

    Returns:
        dict: The parameters of the best candidate shapelet.
    """
    X_train = z_normalize_2d(X=X_train)
    dict_occ_per_class = get_occ_per_class(y=y_train)
    dict_series_per_class = get_series_per_class(y=y_train)
    results = Parallel(n_jobs=-1)(
        delayed(function_to_parallelize)(
            subsequence_length=subsequence_length,
            X_train=X_train,
            y_train=y_train,
            current_params=params.copy(),
            dict_occ_per_class=dict_occ_per_class,
            dict_series_per_class=dict_series_per_class,
            params=params,
        )
        for subsequence_length in range(X_train.shape[1])
    )
    return max(results, key=lambda d: (d["info_gain"], d["gap"]))


if __name__ == "__main__":
    X_train, y_train = load_data_from_hf("train")
    X_test, y_test = load_data_from_hf("test")
    X_train = z_normalize_2d(X_train)
    X_test = z_normalize_2d(X_test)
    params = {"dimensionality": 16, "cardinality": 4, "r": 10, "k": 10, "proba": 0.8}
    shapelet_params = compute_fast_shapelets_parallelized(
        X_train=X_train, y_train=y_train, params=params
    )
