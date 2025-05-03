


import os
import requests
import zipfile
import tarfile
import gzip

# Chemin du dossier "data2" où enregistrer le dataset
data_folder = 'data2'

# URL du dataset
url = "https://www.timeseriesclassification.com/Downloads/ECGFiveDays.zip"

# Nom du fichier téléchargé
filename = url.split("/")[-1]
filepath = os.path.join(data_folder, filename)

# Créer le dossier "data2" s'il n'existe pas
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Télécharger le fichier
print(f"Téléchargement du fichier depuis {url}...")
response = requests.get(url)
response.content

# Vérifier si le téléchargement a réussi
if response.status_code == 200:
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f"Fichier téléchargé sous {filepath}")
else:
    print(f"Erreur lors du téléchargement: {response.status_code}")

# Vérifier si le téléchargement a réussi
if response.status_code == 200:
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f"Fichier téléchargé sous {filepath}")
else:
    print(f"Erreur lors du téléchargement: {response.status_code}")

# Décompresser le fichier en fonction du type de format
def decompress_file(filepath):
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(data_folder)
        print(f"Fichier ZIP décompressé dans {data_folder}")
    elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(data_folder)
        print(f"Fichier TAR.GZ décompressé dans {data_folder}")
    elif filepath.endswith('.tar'):
        with tarfile.open(filepath, 'r') as tar_ref:
            tar_ref.extractall(data_folder)
        print(f"Fichier TAR décompressé dans {data_folder}")
    elif filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[:-3], 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"Fichier GZ décompressé dans {data_folder}")
    else:
        print("Format de fichier non pris en charge pour la décompression")

# Décompresser le fichier téléchargé
decompress_file(filepath)

print("Opération terminée.")



from pyts.datasets.ucr import ucr_dataset_list, _load_ucr_dataset, fetch_ucr_dataset

fetch_ucr_dataset("ECGFiveDays", use_cache=True, data_home='data2')


if not os.path.exists('data2'):
    os.makedirs('data2')

fetch_ucr_dataset("ECGFiveDays", use_cache=True, data_home='/Users/Corentin/Desktop/MVA 2024-2025/ML for Time Series/Project/FASTShapelets_TS/data2')