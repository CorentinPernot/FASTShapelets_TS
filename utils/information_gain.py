import numpy as np
from scipy.stats import entropy
from fastdtw import fastdtw 

def compute_distance(s1, s2, metric='eucl'):
    if metric == 'eucl':
        return np.linalg.norm(np.array(s1) - np.array(s2))
    elif metric == 'dtw':
        distance, _ = fastdtw(s1, s2)
        return distance

def extract_subsequences(time_series, sub_len):
    "Découper des sous-séquences de taille len(cand) dans une série temporelle"
    return [time_series[i:i+sub_len] for i in range(len(time_series) - sub_len + 1)]

def calculate_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=2)


def information_gain(cand, X, y, metric='eucl'):
    sub_len = len(cand)
    distances = []
    
    # Calculer les distances pour chaque série dans X
    for time_series in X:
        subsequences = extract_subsequences(time_series, sub_len)
        dist = min(compute_distance(cand, sub, metric=metric) for sub in subsequences)
        distances.append(dist)
    
    # Associer distances et classes
    distances = np.array(distances)
    y = np.array(y)
    
    # Fixer la médiane comme seuil
    threshold = np.median(distances)
    
    # Diviser en 2 groupes selon le seuil
    group1_distances = distances[distances <= threshold]
    group2_distances = distances[distances > threshold]
    group1 = y[distances <= threshold]
    group2 = y[distances > threshold]
    
    # Calculer l'entropie avant et après la séparation
    initial_entropy = calculate_entropy(y)
    entropy_group1 = calculate_entropy(group1) if len(group1) > 0 else 0
    entropy_group2 = calculate_entropy(group2) if len(group2) > 0 else 0
    
    # Ponderer les entropies
    weighted_entropy = (len(group1) / len(y)) * entropy_group1 + (len(group2) / len(y)) * entropy_group2
    
    # Gain d'information
    information_gain_value = initial_entropy - weighted_entropy
    
    # Calculer le separation gap
    mean_group1 = np.mean(group1_distances) if len(group1_distances) > 0 else 0
    mean_group2 = np.mean(group2_distances) if len(group2_distances) > 0 else 0
    separation_gap = mean_group2 - mean_group1
    
    return information_gain_value, separation_gap


def eval_candidates(top_k_TS, X, y, metric="eucl"):
    max_gain = 0 
    min_gap = 0
    for cand in top_k_TS:
        information_gain_value, separation_gap = information_gain(cand, X, y, metric=metric)
        if information_gain_value > max_gain or (information_gain_value==max_gain)&(separation_gap>min_gap):
            min_gap = separation_gap
            max_gain = information_gain_value
            shapelet = cand 
    print("Max gain :" + str(max_gain))
    print("Min Gap :" + str(min_gap))
    return shapelet, min_gap, max_gain


#shapelet = eval_candidates(top_k_TS, X, y, metric="eucl")






