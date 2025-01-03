"""Functions to evaluate the model"""

import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.preprocessing import extract_subsequences

from src.selection import euclidian_distance, distance_to_all_series, split

from src.fast_shapelets import (
    compute_fast_shapelets_parallelized,
)


def find_shapelet_position(
    series: np.ndarray, shapelet_params: dict[str, int | float | np.ndarray]
) -> int:
    subsequences = extract_subsequences(
        series=series, subsequence_length=shapelet_params["shapelet"].shape[0]
    )
    distances = np.array(
        [
            euclidian_distance(shapelet_params["shapelet"], subsequence)
            for subsequence in subsequences
        ]
    )
    return np.argmin(distances)


def find_classification_criterion(
    X_train: np.ndarray,
    y_train: np.ndarray,
    shapelet_params: dict[str, int | float | np.ndarray],
) -> None:
    classes = np.unique(y_train)
    distances = distance_to_all_series(target=shapelet_params["shapelet"], X=X_train)
    below_threshold_indices, above_threshold_indices = split(
        distances=distances, threshold=shapelet_params["threshold"]
    )
    y_below = y_train[below_threshold_indices]
    y_above = y_train[above_threshold_indices]
    class_count = {
        "below": {int(c): int(np.sum([y_below == c])) for c in classes},
        "above": {int(c): int(np.sum([y_above == c])) for c in classes},
    }
    class_below = max(class_count["below"], key=class_count["below"].get)
    class_above = max(class_count["above"], key=class_count["above"].get)
    if class_below != class_above:
        shapelet_params["class_below"] = class_below
        shapelet_params["class_above"] = class_above
        return None
    else:
        raise ValueError("This shapelet cannot split classes correctly.")


def predict(
    X_test: np.ndarray,
    shapelet_params: dict[str, int | float | np.ndarray],
) -> np.ndarray:
    distances = distance_to_all_series(target=shapelet_params["shapelet"], X=X_test)
    below_threshold_indices, above_threshold_indices = split(
        distances, shapelet_params["threshold"]
    )
    y_pred = np.where(
        distances < shapelet_params["threshold"],
        shapelet_params["class_below"],
        shapelet_params["class_above"],
    )
    return y_pred


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def run_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, float],
    param_to_test: str,
    values: np.ndarray,
    nb_times: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accuracies = []
    times = []
    for value in tqdm(values):
        params[param_to_test] = value
        accuracies_temp = []
        times_temp = []
        for i in range(nb_times):
            start_time = time.time()
            shapelet_params = compute_fast_shapelets_parallelized(
                X_train=X_train, y_train=y_train, params=params
            )
            end_time = time.time()
            find_classification_criterion(
                X_train=X_train, y_train=y_train, shapelet_params=shapelet_params
            )
            y_pred = predict(X_test=X_test, shapelet_params=shapelet_params)
            accuracy = compute_accuracy(y_true=y_test, y_pred=y_pred)
            accuracies_temp.append(accuracy)
            times_temp.append(end_time - start_time)
        accuracies.append(
            (
                np.mean(accuracies_temp),
                np.std(accuracies_temp, ddof=1) / np.sqrt(len(accuracies_temp)),
            )
        )
        times.append(
            (
                np.mean(times_temp),
                np.std(times_temp, ddof=1) / np.sqrt(len(times_temp)),
            )
        )
    return values, np.array(accuracies), np.array(times)


def compute_randomness(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, int | float],
    nb_times: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_accuracies = np.zeros(nb_times)
    test_accuracies = np.zeros(nb_times)
    for i in range(nb_times):
        shapelet_params = compute_fast_shapelets_parallelized(
            X_train=X_train, y_train=y_train, params=params
        )
        find_classification_criterion(
            X_train=X_train, y_train=y_train, shapelet_params=shapelet_params
        )
        y_pred_train = predict(X_test=X_train, shapelet_params=shapelet_params)
        y_pred_test = predict(X_test=X_test, shapelet_params=shapelet_params)
        train_accuracies[i] = compute_accuracy(y_true=y_train, y_pred=y_pred_train)
        test_accuracies[i] = compute_accuracy(y_true=y_test, y_pred=y_pred_test)
    return np.array(range(nb_times)), train_accuracies, test_accuracies
