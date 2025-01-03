"""SAX representation functions"""

import numpy as np
from scipy.stats import norm
from collections import defaultdict

from src.preprocessing import extract_subsequences

ALPHABET = list(map(chr, range(97, 97 + 26)))


def mean_by_segment(series: np.ndarray, dimensionality: int) -> np.ndarray:
    m = len(series)
    segment_size = m / dimensionality
    segments_means = []
    for i in range(dimensionality):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        segment = series[start:end]
        segments_means.append(np.mean(segment))
    return np.array(segments_means)


def check_dimensionality(m: int, dimensionality: int) -> int:
    if m >= dimensionality:
        return dimensionality
    else:
        return m


def get_breakpoints(cardinality: int) -> np.ndarray:
    return norm.ppf(np.linspace(0, 1, cardinality + 1)[1:-1])


def convert_mean_to_sax(mean: float, breakpoints: np.ndarray) -> np.ndarray:
    sax_idx = np.digitize(mean, breakpoints)
    return ALPHABET[sax_idx]


def convert_segments_to_sax(segments: np.ndarray, breakpoints: np.ndarray) -> str:
    sax_segments = [convert_mean_to_sax(mean, breakpoints) for mean in segments]
    return "".join(sax_char for sax_char in sax_segments)


def get_subsequence_sax_representation(
    subsequence: np.ndarray, dimensionality: int, cardinality: int
) -> str:
    dimensionality = check_dimensionality(subsequence.shape[0], dimensionality)
    breakpoints = get_breakpoints(cardinality)
    segments_means = mean_by_segment(subsequence, dimensionality)
    sax_segments = convert_segments_to_sax(segments_means, breakpoints)
    return sax_segments


def get_series_sax_representations(
    series: np.ndarray, subsequence_length: int, dimensionality: int, cardinality: int
) -> dict[str, np.ndarray]:
    dict_representations = {}
    segments = extract_subsequences(
        series=series, subsequence_length=subsequence_length
    )
    dimensionality = check_dimensionality(subsequence_length, dimensionality)
    breakpoints = get_breakpoints(cardinality)
    for segment in segments:
        segments_means = mean_by_segment(segment, dimensionality)
        representation = convert_segments_to_sax(segments_means, breakpoints)
        if representation not in dict_representations:
            dict_representations[representation] = segment
    return dict_representations


def get_all_sax_representations(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, float],
) -> tuple[dict[int, str], dict[int, int], dict[int, int], dict[int, np.ndarray]]:
    map_sax_representations = defaultdict(str)
    map_series = defaultdict(int)
    map_classes = defaultdict(int)
    map_subsequences = defaultdict(np.ndarray)
    idx = 0
    params["dimensionality"] = check_dimensionality(
        params["subsequence_length"], params["dimensionality"]
    )
    breakpoints = get_breakpoints(params["cardinality"])
    for i in range(X.shape[0]):
        series = X[i]
        segments = extract_subsequences(
            series=series, subsequence_length=params["subsequence_length"]
        )
        series_representations = set()
        for segment in segments:
            segments_means = mean_by_segment(segment, params["dimensionality"])
            representation = convert_segments_to_sax(segments_means, breakpoints)
            if representation not in series_representations:
                series_representations.add(representation)
                map_sax_representations[idx] = representation
                map_series[idx] = i
                map_classes[idx] = int(y[i])
                map_subsequences[idx] = segment
                idx += 1
    return map_sax_representations, map_series, map_classes, map_subsequences
