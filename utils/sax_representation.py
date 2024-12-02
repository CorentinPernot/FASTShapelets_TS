import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

def discretize(value, breakpoints):
    """Discrétiser une valeur en fonction des breakpoints (quantiles d'une distribution normale)."""
    for i, threshold in enumerate(breakpoints):
        if value <= threshold:
            return i
    return len(breakpoints)  # Si la valeur dépasse tous les seuils, on retourne le dernier symbole.

def sax(series, len_window, dim_window, alphabet_size, stride):
    """
    Applique la transformation SAX à une série temporelle en utilisant une distribution gaussienne pour la discrétisation.
    Cette fonction génère un mot SAX de 4 lettres pour chaque fenêtre de 16 points de la série.
    
    :param series: La série temporelle (1D numpy array ou liste)
    :param len_window: Le nombre de points par fenêtre (16 dans votre cas)
    :param alphabet_size: Le nombre de symboles distincts dans l'alphabet (ici, 4)
    :return: Une liste de mots SAX représentant la série
    """
    # 1. on normalise la série temp avec StandardScaler
    scaler = StandardScaler()
    normalized_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()  # Normalisation par ligne
    
    # 2. on découpe la en fenêtres de taille 'len_window' : sliding window
    n = len(normalized_series)
    windows = [normalized_series[i:i + len_window] for i in range(0, n - len_window + 1,stride)] 

    # 3. on calcule les breakpoints basés sur la distribution gaussienne (quantiles)
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1]) 
    
    # 4. on discrétise les valeurs de chaque fenêtre
    sax_words = [] 
    
    for window in windows:
        segment_length = len_window // dim_window
        segments = [window[i * segment_length:(i + 1) * segment_length] for i in range(dim_window)]
        
        means = [np.mean(segment) for segment in segments]
        
        symbols = [discretize(mean, breakpoints) for mean in means]

        alphabet = ['a', 'b', 'c', 'd'][:alphabet_size]
        word = ''.join([alphabet[symbol] for symbol in symbols])  # Mot SAX pour cette fenêtre
        sax_words.append(word)
    
    return sax_words, windows


















