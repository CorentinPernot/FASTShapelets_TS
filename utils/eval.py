import numpy as np 
import pandas as pd 
from tqdm import tqdm
from utils.information_gain import min_dist
from utils.plot import plot_shapelet
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from utils.load_data import load_data
from utils.main_fast_shapelet import main_fast_shapelets
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed



# def create_dataframe(signals, shapelets, metric='eucl'):
#     """
#     Crée un DataFrame où chaque colonne correspond à la distance minimale
#     entre les signaux et un shapelet donné.

#     Parameters:
#         signals (dict): Dictionnaire des signaux, où chaque clé est l'identifiant de l'observation
#                         et chaque valeur est une liste représentant le signal.
#         shapelets (dict): Dictionnaire des shapelets, où chaque clé est la longueur du shapelet
#                           et chaque valeur est une liste représentant le shapelet.
#     Returns:
#         pd.DataFrame: DataFrame contenant les distances minimales par shapelet.
#     """
#     data = {}

#     for obs, signal in tqdm(enumerate(signals)):
#         # Pour chaque observation, calculer les distances pour tous les shapelets
#         distances = {}
#         for length, shapelet in shapelets.items():
#             column_name = f"shapelet_{length}"
#             distances[column_name] = min_dist(signal, shapelet, metric=metric)
#         data[obs] = distances

#     # Conversion en DataFrame pandas
#     df = pd.DataFrame.from_dict(data , orient="index")

#     # Ajout de la colonne des observations
#     df.reset_index(inplace=True)
#     df.rename(columns={'index': 'Signal_nb'}, inplace=True)

#     return df


def compute_distances(signal, shapelets, metric):
    distances = {}
    for length, shapelet in shapelets.items():
        column_name = f"shapelet_{length}"
        distances[column_name] = min_dist(signal, shapelet, metric)
    return distances

def create_dataframe(signals, shapelets, metric='eucl', n_jobs=-1):
    """
    Crée un DataFrame avec les distances minimales pour chaque signal et shapelet.
    Utilise Joblib pour la parallélisation.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_distances)(signal, shapelets, metric)
        for signal in tqdm(signals)  # Parcourt directement les listes dans signals
    )  
    # Construction du DataFrame final
    df = pd.DataFrame(results)
    df['Signal_nb'] = list(range(len(signals)))  # Ajout de l'index des signaux
    return df


def create_decision_tree(df_train, y_train, df_test, y_test, max_depth=2):
    model = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
    model.fit(df_train, y_train)

    train_accuracy = model.score(df_train, y_train)
    test_accuracy = model.score(df_test, y_test)

    # Évaluer les performances de l'arbre
    y_pred = model.predict(df_test)
    report = classification_report(y_test, y_pred, target_names=["Classe 1", "Classe 2"])

    print(f"Accuracy sur le train : {train_accuracy:.4f}")
    print(f"Accuracy sur le test : {test_accuracy:.4f}")    
    return model, report
