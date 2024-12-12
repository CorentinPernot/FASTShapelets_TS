### This script computes the SAX representation for signals 

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt

### 2nd method by using subsequences and the sliding window technique

def discretize(value, breakpoints):
    """Discrétiser une valeur en fonction des breakpoints (quantiles d'une distribution normale)."""
    for i, threshold in enumerate(breakpoints):
        if value <= threshold:
            return i
    return len(breakpoints)  


def split_window(window, dim_window):
    " Permet de gérer le cas où len_window n'est pas divisible par dim_window"
    len_window = len(window)
    
    base_segment_size = len_window // dim_window
    remainder = len_window % dim_window
    
    segments = []
    start = 0
    
    for i in range(dim_window):
        # Calcul de la taille du segment (certains segments auront un élément supplémentaire)
        segment_size = base_segment_size + 1 if i < remainder else base_segment_size
        end = start + segment_size
        segments.append(window[start:end])
        start = end  # MAJ pour le prochain segment
    return segments

def sax_bf(series, subsequence_length, word_length, alphabet_size):
    """
    SAX representation en utilisant la slinding window method décrite dans [14]
    """
    
    T = len(series)

    # 1. on normalise la série temp avec StandardScaler
    scaler = StandardScaler()
    normalized_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()  # Normalisation par ligne

    # 2. on calcule les breakpoints basés sur la distribution gaussienne (quantiles)
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1]) 
    
    # 3. on discrétise les valeurs de chaque fenêtre
    sax_words = [] 
    subsequences = []
    for i in range(T-subsequence_length):
        subsequence = normalized_series[i : i+subsequence_length]
        subsequences.append(subsequence)
        segments = split_window(subsequence, word_length)

        means = [np.mean(segment) for segment in segments]
        
        symbols = [discretize(mean, breakpoints) for mean in means]

        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][:alphabet_size]
        word = ''.join([alphabet[symbol] for symbol in symbols])  # Mot SAX pour cette fenêtre
        sax_words.append(word)

    return sax_words, subsequences, breakpoints


def sax_for_set_bf(X, subsequence_length, word_length, alphabet_size):
    " Cette fonction itère la précédente sur tous les signaux de l'ensemble "
    sax_results = []
    all_subsequences = []
    for signal in X:
        sax_result, subsequences, breakpoints = sax_bf(signal, subsequence_length, word_length, alphabet_size)
        sax_results.append(sax_result)
        all_subsequences.append(subsequences)
    return sax_results, all_subsequences 


def filtered_sax_for_set_bf(X, subsequence_length, word_length, alphabet_size):
    "Cette fonction utilise la précédente mais enlève les doublons successifs, comme cela est fait dans [14] en 3.4."
    sax_results, all_subsequences  = sax_for_set_bf(X, subsequence_length, word_length, alphabet_size)

    filtered_sax_results = []
    filtered_all_subsequences = []

    for i, sax_result in enumerate(sax_results):
        filtered_sax_result = []
        filtered_subsequence = []
        for j, word in enumerate(sax_result):
            # ajouter le mot si c'est le premier ou s'il est différent du précédent
            if j == 0 or word != sax_result[j - 1]:
                filtered_sax_result.append(word)
                filtered_subsequence.append(all_subsequences[i][j])
        # ajouter le résultat filtré pour ce sax_result à la liste globale
        filtered_sax_results.append(filtered_sax_result)
        filtered_all_subsequences.append(filtered_subsequence)

    return filtered_sax_results, filtered_all_subsequences


