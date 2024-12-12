### This script aims to compute the information gain for each candidate, and also allows us to find the best shapelet for a given length

import numpy as np
from scipy.stats import entropy
from fastdtw import fastdtw 

def compute_distance(s1, s2, metric='eucl'):
    if metric == 'eucl':
        return np.linalg.norm(np.array(s1) - np.array(s2))
    elif metric == 'dtw':
        distance, _ = fastdtw(s1, s2)
        return distance
    

def min_dist(time_series, cand, metric):
    sub_len = len(cand)
    subsequences = extract_subsequences(time_series, sub_len)
    
    if metric == 'eucl':
        # Calcul matriciel des distances euclidiennes
        distances = np.linalg.norm(subsequences - np.array(cand), axis=1)
    elif metric == 'dtw':
        # Calcul des distances DTW pour chaque sous-séquence (non vectorisable)
        distances = [compute_distance(cand, sub, metric=metric) for sub in subsequences]
    else:
        raise ValueError("Metric must be 'eucl' or 'dtw'")
    
    return np.min(distances)


def extract_subsequences(time_series, sub_len):
    """
    Extrait toutes les sous-séquences de taille sub_len pour une time_series
    Vectorise l'extraction de sous-séquences à l'aide de numpy.
    """
    time_series = np.array(time_series)
    n_subs = len(time_series) - sub_len + 1
    strides = time_series.strides[0]
    return np.lib.stride_tricks.as_strided(
        time_series,
        shape=(n_subs, sub_len),
        strides=(strides, strides)
    )


def calculate_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=2)


def information_gain(cand, X, y, metric='eucl'):
    """Objectif : trouver le meilleur threshold pour le candidat cand

    Args:
        cand (array): shapelet candidat
        X (liste de listes): observations
        y (list): labels
        metric (str, optional): distance à utiliser. Defaults to 'eucl'.

    Returns:
        best_information_gain, best_separation_gap, best_threshold
    """
    sub_len = len(cand)
    distances = []
    
    # calcul des distances min pour chaque serie
    for time_series in X:
        dist = min_dist(time_series, cand, metric)
        distances.append(dist)
    
    distances = np.array(distances)
    y = np.array(y)
    
    best_threshold = None
    best_information_gain = -float('inf')
    best_separation_gap = 0
    
    # on parcourt tous les seuils possibles
    all_thresholds = np.unique(distances)

    for threshold in all_thresholds:
        # on divise en 2 groupes
        group1 = y[distances <= threshold]
        group2 = y[distances > threshold]
        
        # entropie avant et après la séparation
        initial_entropy = calculate_entropy(y)
        entropy_group1 = calculate_entropy(group1) if len(group1) > 0 else 0
        entropy_group2 = calculate_entropy(group2) if len(group2) > 0 else 0
        
        weighted_entropy = (len(group1) / len(y)) * entropy_group1 + (len(group2) / len(y)) * entropy_group2
        
        # Information gain
        information_gain_value = initial_entropy - weighted_entropy
        
        # Separation gap
        group1_distances = distances[distances <= threshold]
        group2_distances = distances[distances > threshold]
        mean_group1 = np.mean(group1_distances) if len(group1_distances) > 0 else 0
        mean_group2 = np.mean(group2_distances) if len(group2_distances) > 0 else 0
        separation_gap = mean_group2 - mean_group1
        
        # Mise à jour
        if (information_gain_value > best_information_gain or
            (information_gain_value == best_information_gain and separation_gap > best_separation_gap)):
            best_information_gain = information_gain_value
            best_separation_gap = separation_gap
            best_threshold = threshold
    
    return best_information_gain, best_separation_gap, best_threshold


def eval_candidates(top_k_TS, X, y, metric="eucl"):
    "Parmi un ensemble de candidats, cette fonction renvoie celui avec le maximum information_gain_value"
    max_gain = 0 
    min_gap = 0
    shapelet = None
    for cand in top_k_TS:
        information_gain_value, separation_gap, best_threshold = information_gain(cand, X, y, metric=metric)
        if information_gain_value > max_gain or (information_gain_value==max_gain)&(separation_gap>min_gap):
            min_gap = separation_gap
            max_gain = information_gain_value
            shapelet = cand 
    return shapelet, min_gap, max_gain







