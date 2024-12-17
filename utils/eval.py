### This script aims to evaluate que quality of different shapelets (e.g. shapelets of different sizes)
# It creates a new dataframe with the observation (=signals) in rows and the minimal distance between all the subsequences of this signal
# It also create a decision tree based on this new features in order to find the most discriminant shapelet 

import pandas as pd 
from tqdm import tqdm
from utils.information_gain import min_dist
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from joblib import Parallel, delayed



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


def create_decision_tree(df_train, y_train, df_test, y_test, max_depth=2, experiments=False):
    model = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
    model.fit(df_train, y_train)

    train_accuracy = model.score(df_train, y_train)
    test_accuracy = model.score(df_test, y_test)

    # Évaluer les performances de l'arbre
    y_pred = model.predict(df_test)
    report = classification_report(y_test, y_pred, target_names=["Classe 1", "Classe 2"])

    if experiments:
        return model, report, test_accuracy
    
    print(f"Accuracy sur le train : {train_accuracy:.4f}")
    print(f"Accuracy sur le test : {test_accuracy:.4f}") 
    return model, report
