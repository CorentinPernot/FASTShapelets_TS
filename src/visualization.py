"""Functions for visualization"""

import numpy as np
import matplotlib.pyplot as plt

from src.evaluation import find_shapelet_position, predict


def plot_shapelet(shapelet_params: dict[str, int | float | np.ndarray]) -> None:
    plt.plot(shapelet_params["shapelet"], c="red", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Signal value")
    plt.grid(visible=True, which="major", axis="y")
    plt.title("Selected shapelet")
    plt.show()


def plot_series_with_shapelet(
    X: np.ndarray,
    y: np.ndarray,
    shapelet_params: dict[str, int | float | np.ndarray],
    index: int,
) -> None:
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


def plot_results_experiment(
    values: np.ndarray,
    accuracies: np.ndarray,
    times: np.ndarray,
    param_to_test: str,
    objective: str = "Accuracy",
) -> None:
    if objective == "Accuracy":
        y = accuracies
    else:
        y = times
    plt.plot(values, y, marker="o")
    plt.xlabel(f"Value of {param_to_test}")
    plt.ylabel(objective if objective == "Accuracy" else f"{objective} (s)")
    plt.xticks(values)
    plt.grid(True, which="major", axis="y")
    plt.title(f"Evolution of {objective} given {param_to_test}")


def plot_results_experiments(
    values: np.ndarray,
    accuracies: np.ndarray,
    times: np.ndarray,
    param_to_test: str,
    objective: str = "accuracy",
) -> None:
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
    train_accuracies: np.ndarray,
    test_accuracies: np.ndarray,
    objective: str = "train",
) -> None:
    if objective == "train":
        y = train_accuracies
    else:
        y = test_accuracies
    plt.plot(sigmas, y, marker="o", c="green")
    plt.xlabel("Sigma")
    plt.ylabel(f"{objective.capitalize()} accuracy")
    plt.xticks(sigmas)
    plt.ylim((0.5, 1))
    plt.grid(True, which="major", axis="y")
    plt.title(f"Evolution of {objective} accuracy given sigma")


def plot_results_randomness(
    iterations: np.ndarray,
    train_accuracies: np.ndarray,
    test_accuracies: np.ndarray,
    objective: str = "train",
) -> None:
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
    return None
