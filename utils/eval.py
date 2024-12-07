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


def create_dataframe(signals, shapelets, metric='eucl'):
    """
    Crée un DataFrame où chaque colonne correspond à la distance minimale
    entre les signaux et un shapelet donné.

    Parameters:
        signals (dict): Dictionnaire des signaux, où chaque clé est l'identifiant de l'observation
                        et chaque valeur est une liste représentant le signal.
        shapelets (dict): Dictionnaire des shapelets, où chaque clé est la longueur du shapelet
                          et chaque valeur est une liste représentant le shapelet.
    Returns:
        pd.DataFrame: DataFrame contenant les distances minimales par shapelet.
    """
    data = {}

    for obs, signal in tqdm(enumerate(signals)):
        # Pour chaque observation, calculer les distances pour tous les shapelets
        distances = {}
        for length, shapelet in shapelets.items():
            column_name = f"shapelet_{length}"
            distances[column_name] = min_dist(signal, shapelet, metric=metric)
        data[obs] = distances

    # Conversion en DataFrame pandas
    df = pd.DataFrame.from_dict(data , orient="index")

    # Ajout de la colonne des observations
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Signal_nb'}, inplace=True)

    return df


def create_decision_tree(df_train, y_train, df_test, y_test, max_depth=2):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(df_train, y_train)

    # Évaluer les performances de l'arbre
    y_pred = model.predict(df_test)
    report = classification_report(y_test, y_pred, target_names=["Classe 1", "Classe 2"])
    return model, report 
