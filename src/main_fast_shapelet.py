### This script aims to find the shapelets with the FAST shapelets algorithm using functions from other scripts

from utils.sax_representations_bruteforce import filtered_sax_for_set_bf
from utils.random_mask import compute_collision_matrix, compute_distinguish_power, find_top_k, remap_SAX_to_TS, SAX_to_TS
from utils.information_gain import eval_candidates
from tqdm import tqdm
from joblib import Parallel, delayed



def process_subsequence_length(subsequence_length, X, y, dimensionality, cardinality, r, k, metric):
    """
    Effectue les calculs pour une longueur de sous-séquence spécifique.
    """
    word_length = dimensionality
    filtered_sax_results, filtered_all_subsequences = filtered_sax_for_set_bf(
        X, subsequence_length, word_length, cardinality
    )

    dict_sax = {i: liste for i, liste in enumerate(filtered_sax_results)}

    mots, collision_matrix = compute_collision_matrix(r, dict_sax)

    distinguish_power = compute_distinguish_power(r, collision_matrix, y)
    top_k = find_top_k(distinguish_power, mots, k)
    
    sax_to_ts_dict = SAX_to_TS(filtered_sax_results, filtered_all_subsequences)
    top_k_TS = remap_SAX_to_TS(top_k, sax_to_ts_dict)

    shapelet, min_gap, max_gain = eval_candidates(top_k_TS, X, y, metric=metric)

    return subsequence_length, shapelet, min_gap, max_gain


def main_fast_shapelets(X, y, min_length=5, max_length=None, dimensionality=5, cardinality=4, r=10, k=10, metric='eucl', n_jobs=-1):
    """Version parallélisée de main_fast_shapelets utilisant Joblib.

    Args:
        X (list of list): signaux 
        y (list): labels
        min_length (int, optional): min length of shapelets. Defaults to 5.
        max_length (int, optional): max length of shapelets. Defaults to None.
        dimensionality (int, optional): new size for the SAX representation. Defaults to 5.
        cardinality (int, optional): size of the alphabet. Defaults to 4.
        r (int, optional): nb of iterations/random projection . Defaults to 10.
        k (int, optional): nb of top_k SAX for each size. Defaults to 10.
        metric (str, optional): 'eucl' or 'dtw'. Defaults to 'eucl'.
        n_jobs (int, optional): Defaults to -1.

    Returns:
        final_shapelets (dict), max_gain_shapelets(dict), min_gap_shapelets(dict): dicts where the lengths of the shapelets are the keys 
    """
    max_gain_shapelets = {}
    min_gap_shapelets = {}
    final_shapelets = {}
    
    # Définir max_length si non spécifié
    if max_length is None:
        max_length = len(X[0])
    
    # Parallélisation sur les longueurs de sous-séquences
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subsequence_length)(
            subsequence_length, X, y, dimensionality, cardinality, r, k, metric
        ) for subsequence_length in tqdm(range(min_length, max_length))
    )
    
    # Stocker les résultats
    for subsequence_length, shapelet, min_gap, max_gain in results:
        max_gain_shapelets[subsequence_length] = max_gain
        min_gap_shapelets[subsequence_length] = min_gap
        final_shapelets[subsequence_length] = shapelet

    return final_shapelets, max_gain_shapelets, min_gap_shapelets