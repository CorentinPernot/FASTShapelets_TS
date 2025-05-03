"""Functions for 1NN (K-Nearest Neighbors)"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.selection import dtw_distance


def knn_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    distance_method: str = "euclidean",
    k: int = 1,
) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors classifier.

    Args:
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels.
        distance_method (str): The distance metric to use. Choices are 'euclidean' and 'dtw'. Default is 'euclidean'.
        k (int): The number of neighbors to consider. Default is 1.

    Returns:
        KNeighborsClassifier: The trained KNN classifier.
    """
    if distance_method == "euclidean":
        knn = KNeighborsClassifier(n_neighbors=k)
    elif distance_method == "dtw":
        knn = KNeighborsClassifier(n_neighbors=k, metric=dtw_distance)
    else:
        raise ValueError("Invalid distance method. Choose either 'euclidean' or 'dtw'.")
    knn.fit(X_train, y_train)
    return knn


def knn_predict(knn: KNeighborsClassifier, X_test: np.ndarray) -> np.ndarray:
    """
    Predict the labels for the test data using a trained KNN classifier.

    Args:
        knn (KNeighborsClassifier): The trained KNN classifier.
        X_test (np.ndarray): The test data features.

    Returns:
        np.ndarray: The predicted labels for the test data.
    """
    return knn.predict(X_test)
