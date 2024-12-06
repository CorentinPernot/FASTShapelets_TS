import numpy as np 

def generate_random_masking(n, len_word):
    """
    Generates a table of n random binary masks, each of length len_word.
    
    Parameters:
        n (int): The number of masks.
        len_word (int): The length of each mask.
    
    Returns:
        np.ndarray: A 2D binary array of shape (n, len_word), 
                    where each row is a binary mask with elements 0 or 1.
    """
    return np.random.choice([0, 1], size=(n, len_word), p=[0.3, 0.7])


def apply_mask(word, mask):
    """
    Applies a binary mask to a word, masking characters where the mask is 0.
    
    Parameters:
        word (str): The word to be masked.
        mask (list or np.ndarray): A binary mask of the same length as the word.
    
    Returns:
        str: The masked word.
    """
    if len(word) != len(mask):
        raise ValueError("The length of the word and the mask must be the same.")
    return ''.join(char for char, m in zip(word, mask) if m == 1)



def compute_collision_matrix(nb_iterations, dict_sax):
    # pour éviter de calculer des distances sur des doublons
    mots = sorted(set(word for words in dict_sax.values() for word in words))
    nb_mots = len(mots)
    nb_objets = len(dict_sax)

    # initialisation
    matrice_collusion = np.zeros((nb_mots, nb_objets), dtype=int)
    mot_to_index = {mot: idx for idx, mot in enumerate(mots)} # pour accéder facilement aux indices des mots
    
    # générer les masks
    masks = generate_random_masking(nb_iterations, len(mots[0]))

    for iteration in range(nb_iterations):
        mask = masks[iteration]

        iteration_collusion = np.zeros((nb_mots, nb_objets), dtype=int)
        
        new_mot_index = {} # nouveau dictionnaire avec les mots masqués
        for idx, mot in enumerate(mots):
            masked_mot = apply_mask(mot, mask)
            if masked_mot not in new_mot_index:
                new_mot_index[masked_mot] = []
            new_mot_index[masked_mot].append(idx)  
        
        for idx_objet, (key, words) in enumerate(dict_sax.items()): # on parcourt les objets et les mots de chaque objet
            already_updated = []
            for word in words: # pour chaque mot 
                idx_word = mot_to_index[word]
                for key, value_list in new_mot_index.items(): # on trouve les autres mots masqués associés et leurs indices (correspondant aux mots non masqués)
                    if idx_word in value_list:
                        found_list = value_list
                        break
                for idx_word in found_list:
                    if idx_word not in already_updated: # si deja updated alors on n'update pas
                        iteration_collusion[idx_word, idx_objet] += 1
                        already_updated.append(idx_word)
        
        matrice_collusion += iteration_collusion

    return mots, matrice_collusion


def compute_distinguish_power(nb_iterations, collision_matrix, y):
    indices_class_1 = np.where(y == 1)[0]
    indices_class_2 = np.where(y == 2)[0]

    # Initialiser la matrice des sommes par classe
    close_to_ref = np.zeros((collision_matrix.shape[0], 2))

    for i in range(collision_matrix.shape[0]):
        close_to_ref[i, 0] = np.sum(collision_matrix[i, indices_class_1])
        close_to_ref[i, 1] = np.sum(collision_matrix[i, indices_class_2])

    far_from_ref = np.zeros((collision_matrix.shape[0], 2))
    far_from_ref[:, 0] = nb_iterations * len(indices_class_1)
    far_from_ref[:, 1] = nb_iterations * len(indices_class_2)

    distinguish_power = np.abs(close_to_ref[:, 0]-far_from_ref[:, 0]) + np.abs(close_to_ref[:, 1]-far_from_ref[:, 1]) 

    return distinguish_power

def find_top_k(distinguish_power,mots,k=10):
    indices_best = np.argsort(distinguish_power)[-k:][::-1]
    return [mots[i] for i in indices_best]


def SAX_to_TS(filtered_sax_results, filtered_all_subsequences):
    sax_to_ts = {}
    for i in range(len(filtered_sax_results)):
        for j in range(len(filtered_sax_results[i])):
            if filtered_sax_results[i][j] not in sax_to_ts.keys():
                sax_to_ts[filtered_sax_results[i][j]] = [filtered_all_subsequences[i][j]]
            else :
                sax_to_ts[filtered_sax_results[i][j]].append(filtered_all_subsequences[i][j])
    return sax_to_ts


def remap_SAX_to_TS(sax_words, sax_to_ts):
    #return [cand for cand in sax_to_ts[word] for word in sax_words]
    return [ts for word in sax_words for ts in sax_to_ts[word]]



# res = SAX_to_TS(filtered_sax_results, filtered_all_subsequences)
# for key in res.keys():
#     print(len(res[key]))

# top_k = find_top_k(distinguish_power,mots,k=10)
# len(remap_SAX_to_TS(top_k, res))

