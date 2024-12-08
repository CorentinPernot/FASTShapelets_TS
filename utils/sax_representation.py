import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt

### 1st method by using windows : we divide the signals into windows 

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


def sax(series, len_window, dim_window, alphabet_size, stride):
    """
    Applique la transformation SAX à une série temporelle en utilisant une distribution gaussienne pour la discrétisation.
    Cette fonction génère un mot SAX de 4 lettres pour chaque fenêtre de 16 points de la série.

    Args:
        series (np.ndarray ou list): La série temporelle à analyser.
        len_window (int): Le nombre de points par fenêtre (par exemple, 16 points).
        segment_length (int): longueur des segments
        alphabet_size (int): Le nombre de symboles distincts dans l'alphabet SAX (par exemple, 4).
        stride (int): Le pas de décalage entre les fenêtres (par exemple, 1).

    Returns:
        tuple: 
            - sax_words (list of str): Liste des mots SAX représentant la série.
            - windows (list of np.ndarray): Liste des fenêtres de la série temporelle utilisées pour calculer les mots SAX.
            - breakpoints (list): liste de séparation de zones 
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
        segments = split_window(window, dim_window)
        means = [np.mean(segment) for segment in segments]
        
        symbols = [discretize(mean, breakpoints) for mean in means]

        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][:alphabet_size]
        word = ''.join([alphabet[symbol] for symbol in symbols])  # Mot SAX pour cette fenêtre
        sax_words.append(word)
    
    return sax_words, windows, breakpoints


def sax_for_set(X, len_window, dim_window, alphabet_size, stride ):
    " Cette fonction itère la précédente sur tous les signaux de l'ensemble "
    sax_results = []
    for signal in X:
        sax_result, windows, breakpoints = sax(signal, len_window, dim_window, alphabet_size, stride)
        sax_results.append(sax_result)
    return sax_results




def filtered_sax_for_set(X, len_window, dim_window, alphabet_size, stride):
    "Cette fonction utilise la précédente mais enlève les doublons successifs, comme cela est fait dans [14] en 3.4."
    sax_results = sax_for_set(X, len_window, dim_window, alphabet_size, stride)
    filtered_sax_results = []

    for sax_result in sax_results:
        filtered_sax_result = []
        for i, word in enumerate(sax_result):
            # ajouter le mot si c'est le premier ou s'il est différent du précédent
            if i == 0 or word != sax_result[i - 1]:
                filtered_sax_result.append(word)
        # ajouter le résultat filtré pour ce sax_result à la liste globale
        filtered_sax_results.append(filtered_sax_result)

    return filtered_sax_results



def plot_sax_with_breakpoints(signal, sax_words, windows, len_window, dim_window, stride, breakpoints, alphabet):
    """
    Affiche le signal original, les segments de SAX, les mots SAX, et les breakpoints sur l'axe des ordonnées.
    """
    plt.figure(figsize=(15, 6))
    
    # tracer le signal
    plt.plot(signal, label="Signal", color='blue', marker="x")
    
    colors = ['red', 'green', 'orange', 'purple']
    
    # tracer les breakpoints
    for i, bp in enumerate(breakpoints):
        if i == len(breakpoints)-1:
            plt.text(-2, bp+0.3, f'{alphabet[-1]}', color='black', va='center', ha='left', fontsize=10)

        plt.axhline(y=bp, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.text(-2, bp-0.3, f'{alphabet[i]}', color='black', va='center', ha='left', fontsize=10)
    
    # tracer les segments et les mots SAX
    for idx, (word, window) in enumerate(zip(sax_words, windows)):
        start_idx = idx * stride
        if start_idx + len_window > len(signal):
            break
        
        # tracer les segments pour chaque fenêtre 
        segments = split_window(window, dim_window)
        
        start_segment = start_idx
        for i, segment in enumerate(segments):
            segment_end = start_segment + len(segment)
            segment_mean = np.mean(segment)
            plt.hlines(segment_mean, start_segment, segment_end, colors[i % len(colors)], linewidth=2)
            start_segment = segment_end
        
        # annoter le mot SAX au centre de la fenêtre
        window_center = start_idx + len_window//2
        plt.text(window_center, max(signal) * 0.9, word, color='black', ha='center', fontsize=10)
        plt.vlines(start_idx, ymin=min(signal), ymax=max(signal), color='gray', linestyle='--', alpha=0.3)    
        
    plt.title("Signal avec segments SAX, mots SAX et breakpoints")
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/sax_rep.png")
    plt.show()







