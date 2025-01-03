"""Functions to load data"""

from scipy.io import arff
import numpy as np
import pandas as pd
from collections import defaultdict


def load_data(type: str = "TRAIN"):
    """
    Load the ECGFiveDays dataset from an ARFF file.

    Args:
        type (str, optional): Specifies which dataset to load.
                            Options are "TRAIN" or "TEST". Defaults to "TRAIN".

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The feature matrix where each row represents a sample
            and each column represents a feature.
            - y (numpy.ndarray): The array of labels corresponding to each sample.
    """
    data, _ = arff.loadarff(f"data/ECGFiveDays/ECGFiveDays_{type}.arff")
    df = pd.DataFrame(data)
    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].apply(lambda x: x.decode("utf-8")).values
    y = y.astype(int)
    y = y - 1
    print(f"X_{type} shape", X.shape)
    return X, y


def z_normalize(series: np.ndarray) -> np.ndarray:
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return series
    else:
        return (series - mean) / std


def z_normalize_2d(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std


def extract_subsequences(series: np.ndarray, subsequence_length: int) -> np.ndarray:
    n_subs = len(series) - subsequence_length + 1
    strides = series.strides[0]
    return np.lib.stride_tricks.as_strided(
        series, shape=(n_subs, subsequence_length), strides=(strides, strides)
    )


def get_occ_per_class(y: np.ndarray) -> dict:
    classes = np.unique(y)
    return {int(cls): int(np.sum(y == cls)) for cls in classes}


def get_series_per_class(y: np.ndarray) -> dict:
    dict_series_per_class = defaultdict(list)
    for i, cls in enumerate(y):
        dict_series_per_class[cls].append(i)
    return dict_series_per_class


def add_noise(X: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(scale=sigma, size=X.shape)
    return X + noise
