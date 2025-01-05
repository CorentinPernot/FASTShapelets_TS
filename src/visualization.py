"""Functions for visualization"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.evaluation import find_shapelet_position, predict


def plot_shapelet(shapelet_params: dict[str, int | float | np.ndarray]) -> None:
    """
    Plots the selected shapelet.

    Args:
        shapelet_params (dict): A dictionary containing the shapelet parameters.
    """
    plt.plot(shapelet_params["shapelet"], c="red", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Signal value")
    plt.grid(visible=True, which="major", axis="y")
    plt.title("Selected shapelet")
    plt.xlim((-5, 90))
    plt.ylim((-6, 3))
    plt.show()


def plot_series_with_shapelet(
    X: np.ndarray,
    y: np.ndarray,
    shapelet_params: dict[str, int | float | np.ndarray],
    index: int,
) -> None:
    """
    Plots a series with the selected shapelet.

    Args:
        X (np.ndarray): The input series.
        y (np.ndarray): The true classes of the series.
        shapelet_params (dict): A dictionary containing the shapelet parameters.
        index (int): The index of the series to plot.
    """
    series = X[index]
    true_class = y[index]
    predicted_class = int(predict(X_test=series, shapelet_params=shapelet_params))
    position = find_shapelet_position(series=series, shapelet_params=shapelet_params)
    indices_shapelet = range(position, position + shapelet_params["shapelet"].shape[0])
    plt.plot(series, c="blue", linewidth=2)
    plt.plot(indices_shapelet, shapelet_params["shapelet"], c="red", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Signal value")
    plt.grid(visible=True, which="major", axis="y")
    plt.title(
        f"Series with selected shapelet. True class {true_class}. Predicted class {predicted_class}."
    )
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots the confusion matrix.

    Args:
        y_true (np.ndarray): The true classes of the series.
        y_pred (np.ndarray): The predicted classes of the series.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
    )
    plt.title("Confusion matrix")
    plt.xlabel("Predicted classes")
    plt.ylabel("True classes")
    plt.show()


def plot_results_experiments(
    values: np.ndarray,
    accuracies: np.ndarray,
    times: np.ndarray,
    param_to_test: str,
    objective: str = "accuracy",
) -> None:
    """
    Plot the results of an experiment.

    Args:
        values (np.ndarray): Values of the parameter.
        accuracies (np.ndarray): Accuracies.
        times (np.ndarray): Training times.
        param_to_test (str): Parameter to test.
        objective (str, optional): Data to plot. Defaults to "accuracy".
    """
    if objective == "accuracy":
        y = accuracies
    else:
        y = times
    y_mean = [o[0] for o in y]
    y_ci = [1.96 * o[1] for o in y]
    plt.errorbar(values, y_mean, yerr=y_ci, fmt="o-", c="orange")
    plt.xlabel(f"Value of {param_to_test}")
    plt.ylabel(
        objective.capitalize()
        if objective == "accuracy"
        else f"{objective.capitalize()} (s)"
    )
    plt.xticks(values)
    plt.ylim((0.5, 1) if objective == "accuracy" else (0, 40))
    plt.grid(True, which="major", axis="y")
    plt.title(f"Evolution of {objective} given {param_to_test}")


def plot_results_noise(
    sigmas: np.ndarray,
    knn_accuracies: np.ndarray,
    shapelet_accuracies: np.ndarray,
) -> None:
    """
    Plot results of noise experiments.

    Args:
        sigmas (np.ndarray): Values of sigma.
        knn_accuracies (np.ndarray): KNN accuracies.
        shapelet_accuracies (np.ndarray): Shapelet accuracies.
    """
    knn_mean = [o[0] for o in knn_accuracies]
    knn_ci = [1.96 * o[1] for o in knn_accuracies]
    shapelet_mean = [o[0] for o in shapelet_accuracies]
    shapelet_ci = [1.96 * o[1] for o in shapelet_accuracies]
    plt.errorbar(sigmas, knn_mean, yerr=knn_ci, fmt="o-", c="blue", label="1NN")
    plt.errorbar(
        sigmas,
        shapelet_mean,
        yerr=shapelet_ci,
        fmt="o-",
        c="red",
        label="Fast Shapelets",
    )
    plt.xlabel("Sigma")
    plt.ylabel("Accuracy")
    plt.xticks(sigmas)
    plt.ylim((0.3, 1))
    plt.grid(True, which="major", axis="y")
    plt.legend(loc="upper right")
    plt.title("Evolution of accuracy given sigma")


def plot_results_randomness(
    iterations: np.ndarray,
    train_accuracies: np.ndarray,
    test_accuracies: np.ndarray,
    objective: str = "train",
) -> None:
    """
    Plot the results of experiments on randmness.

    Args:
        iterations (np.ndarray): Iterations.
        train_accuracies (np.ndarray): Accuracies on the training set.
        test_accuracies (np.ndarray): Accuracies on the test set.
        objective (str, optional): Which accuracies to plot. Defaults to "train".
    """
    if objective == "train":
        y = train_accuracies
    else:
        y = test_accuracies
    mean = np.mean(y)
    ci = 1.96 * np.std(y, ddof=1) / np.sqrt(y.shape[0])
    plt.fill_between(
        iterations,
        mean - ci,
        mean + ci,
        color="red",
        alpha=0.2,
        label=f"95% CI [{mean - ci:.2f}, {mean + ci:.2f}]",
    )
    plt.axhline(mean, color="red", linestyle="--", label=f"Mean = {mean:.2f}")
    plt.scatter(iterations, y, color="blue", label="Data")
    plt.xlabel("Iteration")
    plt.ylabel(f"{objective.capitalize()} accuracy")
    plt.xticks(iterations)
    plt.ylim((0.5, 1))
    plt.grid(True, which="major", axis="y")
    plt.legend(loc="lower left")
    plt.title(f"Evolution of {objective} accuracy for each iteration")
