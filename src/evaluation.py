"""Functions to evaluate the model"""

import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.preprocessing import extract_subsequences, add_noise, z_normalize_2d

from src.selection import euclidian_distance, distance_to_all_series, split

from src.fast_shapelets import (
    compute_fast_shapelets_parallelized,
)

from src.knn import knn_train, knn_predict


def find_shapelet_position(
    series: np.ndarray, shapelet_params: dict[str, int | float | np.ndarray]
) -> int:
    """
    Find the position of the shapelet in the series.

    Args:
        series (np.ndarray): The series where to find the shapelet.
        shapelet_params (dict): The parameters of the shapelet.

    Returns:
        int: The position of the shapelet in the series.
    """
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
    """
    Find the classification criterion for the shapelet.

    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The labels of the training data.
        shapelet_params (dict): The parameters of the shapelet.
    """
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
    else:
        raise ValueError("This shapelet cannot split classes correctly.")


def predict(
    X_test: np.ndarray,
    shapelet_params: dict[str, int | float | np.ndarray],
) -> np.ndarray:
    """
    Predict the labels of the test data using the shapelet.

    Args:
        X_test (np.ndarray): The test data.
        shapelet_params (dict): The parameters of the shapelet.

    Returns:
        np.ndarray: The predicted labels of the test data.
    """
    distances = distance_to_all_series(target=shapelet_params["shapelet"], X=X_test)
    below_threshold_indices, above_threshold_indices = split(
        distances=distances, threshold=shapelet_params["threshold"]
    )
    y_pred = np.where(
        distances < shapelet_params["threshold"],
        shapelet_params["class_below"],
        shapelet_params["class_above"],
    )
    return y_pred


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy of the predictions.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        float: The accuracy of the predictions.
    """
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
    """
    Run experiments to find the impact of a parameter on the accuracy.

    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The labels of the training data.
        X_test (np.ndarray): The test data.
        y_test (np.ndarray): The labels of the test data.
        params (dict): The parameters of the shapelet.
        param_to_test (str): The parameter to test.
        values (np.ndarray): The values to test for the parameter.
        nb_times (int): The number of times to repeat the experiment.

    Returns:
        tuple: The values tested, the mean accuracies, and the standard deviations of the accuracies.
    """
    accuracies = []
    times = []
    for value in tqdm(values):
        params[param_to_test] = value
        accuracies_temp = []
        times_temp = []
        for _ in range(nb_times):
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
    """
    Compute the randomness of the shapelet classifier.

    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The labels of the training data.
        X_test (np.ndarray): The test data.
        y_test (np.ndarray): The labels of the test data.
        params (dict): The parameters of the shapelet.
        nb_times (int): The number of times to repeat the experiment.

    Returns:
        tuple: The indices tested, the mean train accuracies, and the mean test accuracies.
    """
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


def compute_noise_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, float],
    sigmas: np.ndarray,
    noise: str,
    nb_times: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the accuracy of the shapelet classifier in the presence of noise.

    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The labels of the training data.
        X_test (np.ndarray): The test data.
        y_test (np.ndarray): The labels of the test data.
        params (dict): The parameters of the shapelet.
        sigmas (np.ndarray): The noise levels to test.
        noise (str): The type of noise to add.
        nb_times (int): The number of times to repeat the experiment.

    Returns:
        tuple: The noise levels tested, the mean KNN accuracies, and the mean shapelet accuracies.
    """
    knn_accuracies = []
    shapelet_accuracies = []
    for sigma in tqdm(sigmas):
        knn_temp = []
        shapelet_temp = []
        for _ in range(nb_times):
            # Data
            if noise == "both":
                X_train_noisy = add_noise(X=X_train, sigma=sigma)
                X_train_final = z_normalize_2d(X=X_train_noisy)
            else:
                X_train_final = z_normalize_2d(X=X_train)
            X_test_noisy = add_noise(X=X_test, sigma=sigma)
            X_test_final = z_normalize_2d(X=X_test_noisy)
            # KNN
            knn = knn_train(X_train=X_train_final, y_train=y_train)
            knn_pred = knn_predict(knn=knn, X_test=X_test_final)
            knn_accuracy = compute_accuracy(y_true=y_test, y_pred=knn_pred)
            knn_temp.append(knn_accuracy)
            # Shapelets
            shapelet_params = compute_fast_shapelets_parallelized(
                X_train=X_train_final, y_train=y_train, params=params
            )
            find_classification_criterion(
                X_train=X_train_final, y_train=y_train, shapelet_params=shapelet_params
            )
            shapelet_pred = predict(
                X_test=X_test_final, shapelet_params=shapelet_params
            )
            shapelet_accuracy = compute_accuracy(y_true=y_test, y_pred=shapelet_pred)
            shapelet_temp.append(shapelet_accuracy)
        knn_accuracies.append(
            (
                np.mean(knn_temp),
                np.std(knn_temp, ddof=1) / np.sqrt(len(knn_temp)),
            )
        )
        shapelet_accuracies.append(
            (
                np.mean(shapelet_temp),
                np.std(shapelet_temp, ddof=1) / np.sqrt(len(shapelet_temp)),
            )
        )
    return sigmas, np.array(knn_accuracies), np.array(shapelet_accuracies)
