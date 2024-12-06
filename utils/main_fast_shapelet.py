import numpy as np 
from utils.sax_representations_bruteforce import sax_bf, sax_for_set_bf, filtered_sax_for_set_bf
from utils.random_mask import compute_collision_matrix, compute_distinguish_power, find_top_k, remap_SAX_to_TS, SAX_to_TS
from utils.information_gain import eval_candidates
from utils.load_data import load_data
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


X_train, y_train = load_data(type="TRAIN")
X_test, y_test = load_data(type="TEST")

scaler = StandardScaler()
X_train_scaled = [scaler.fit_transform(x_train.reshape(-1, 1)).flatten() for x_train in X_train]
X_test_scaled = [scaler.fit_transform(x_test.reshape(-1, 1)).flatten() for x_test in X_test]

# params
subsequence_length = 20
word_length = 10
alphabet_size = 4
nb_iterations = 10

X = X_train_scaled
y = y_train
word_length_ref=10
alphabet_size=4
nb_iterations=10
k=10
metric='dtw'

def main_shapelets(X, y, word_length_ref=10, alphabet_size=4, nb_iterations=10, k=10, metric='dtw'): 
    final_shapelet_max_gain = 0
    final_shapelet_min_gap = 0
    for subsequence_length in tqdm(range(1,len(X[0]))):

        word_length = min(word_length_ref,subsequence_length)
        filtered_sax_results, filtered_all_subsequences = filtered_sax_for_set_bf(X, subsequence_length, word_length, alphabet_size)

        dict_sax = {i: liste for i, liste in enumerate(filtered_sax_results)}

        mots, collision_matrix = compute_collision_matrix(nb_iterations, dict_sax)

        distinguish_power = compute_distinguish_power(nb_iterations, collision_matrix, y)
        top_k = find_top_k(distinguish_power, mots, k)
        
        sax_to_ts_dict = SAX_to_TS(filtered_sax_results, filtered_all_subsequences)
        top_k_TS = remap_SAX_to_TS(top_k, sax_to_ts_dict)

        shapelet, min_gap, max_gain = eval_candidates(top_k_TS, X, y, metric=metric)
        if max_gain > final_shapelet_max_gain or (max_gain == final_shapelet_max_gain)&(min_gap>final_shapelet_min_gap):
            final_shapelet = shapelet

    return final_shapelet






