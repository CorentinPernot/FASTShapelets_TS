### This script aims to compute shapelets with the bruteforce algorithm in O(n^2m^4)
# where n is the number of time series in the dataset 
# and m is the length of the longest time series in the dataset

from tqdm import tqdm
from utils.information_gain import eval_candidates
from joblib import Parallel, delayed



def generate_candidates(ts, subsequence_length):
    length_ts = len(ts)
    subsequences = []
    for i in range(length_ts - subsequence_length):
        subsequences.append(ts[i:i + subsequence_length])
    return subsequences

def main_shapelets_bf(X, y, min_length=5, max_length=None, metric='eucl'):
    max_gain_shapelets = {}
    min_gap_shapelets = {}
    final_shapelets = {}

    length_ts = len(X[0])

    if max_length is None:
        max_length = length_ts

    # Paralléliser la génération des candidates pour chaque longueur de sous-séquence
    for subsequence_length in tqdm(range(min_length, max_length)):
        # Utilisation de joblib pour paralléliser la création des sous-séquences
        candidates = Parallel(n_jobs=-1)(delayed(generate_candidates)(ts, subsequence_length) for ts in X)

        # Aplatir la liste des candidats
        candidates = [item for sublist in candidates for item in sublist]

        # Évaluation des candidats
        shapelet, min_gap, max_gain = eval_candidates(candidates, X, y, metric=metric)

        # Stockage des résultats
        max_gain_shapelets[subsequence_length] = max_gain
        min_gap_shapelets[subsequence_length] = min_gap
        final_shapelets[subsequence_length] = shapelet

    return final_shapelets, max_gain_shapelets, min_gap_shapelets