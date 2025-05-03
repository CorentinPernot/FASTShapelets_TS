"""Functions to load data"""

from scipy.io import arff
import numpy as np
import pandas as pd
from collections import defaultdict
from datasets import load_dataset


def load_data(type: str = "TRAIN") -> tuple[np.ndarray, np.ndarray]:
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


def load_data_from_hf(type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from Hugging Face.

    Args:
        type (str): Whether the dataset is for training or testing.

    Returns:
        tuple[np.ndarray, np.ndarray]: Features, targets.
    """
    data = load_dataset("jules-chapon/ECGFiveDays")
    df = data[type].to_pandas()
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].apply(lambda x: x.replace("b", "").replace("'", "")).values
    y = y.astype(int)
    y = y - 1
    print(f"X_{type} shape", X.shape)
    return X, y


def z_normalize(series: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D series using z-score.

    Args:
        series (np.ndarray): The 1D series to normalize.

    Returns:
        np.ndarray: The normalized series.
    """
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return series
    else:
        return (series - mean) / std


def z_normalize_2d(X: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D array using z-score along the rows.

    Args:
        X (np.ndarray): The 2D array to normalize.

    Returns:
        np.ndarray: The normalized array.
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std


def extract_subsequences(series: np.ndarray, subsequence_length: int) -> np.ndarray:
    """
    Extract subsequences of a given length from a 1D series.

    Args:
        series (np.ndarray): The 1D series from which to extract subsequences.
        subsequence_length (int): The length of the subsequences to extract.

    Returns:
        np.ndarray: The extracted subsequences.
    """
    n_subs = len(series) - subsequence_length + 1
    strides = series.strides[0]
    return np.lib.stride_tricks.as_strided(
        series, shape=(n_subs, subsequence_length), strides=(strides, strides)
    )


def get_occ_per_class(y: np.ndarray) -> dict:
    """
    Count the occurrences of each class in a target array.

    Args:
        y (np.ndarray): The target array.

    Returns:
        dict: A dictionary where keys are class labels and values are the corresponding counts.
    """
    classes = np.unique(y)
    return {int(cls): int(np.sum(y == cls)) for cls in classes}


def get_series_per_class(y: np.ndarray) -> dict:
    """
    Get the indices of series for each class in a target array.

    Args:
        y (np.ndarray): The target array.

    Returns:
        dict: A dictionary where keys are class labels and values are lists of indices corresponding to each class.
    """
    dict_series_per_class = defaultdict(list)
    for i, cls in enumerate(y):
        dict_series_per_class[cls].append(i)
    return dict_series_per_class


def add_noise(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Add Gaussian noise to a 2D array.

    Args:
        X (np.ndarray): The 2D array to which to add noise.
        sigma (float): The standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: The array with added noise.
    """
    noise = np.random.normal(scale=sigma, size=X.shape)
    return X + noise
