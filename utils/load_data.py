from scipy.io import arff
import pandas as pd

def load_data(type="TRAIN"):
    """
    Load the ECGFiveDays dataset from an ARFF file.

    Args:
        type (str, optional): Specifies which dataset to load. 
                              Options are "TRAIN" or "TEST". Defaults to "TRAIN".

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The feature matrix where each row represents a sample 
              and each column represents a feature.
            - y (numpy.ndarray): The array of labels corresponding to each sample.
    """
    data, meta = arff.loadarff(f'data/ECGFiveDays/ECGFiveDays_{type}.arff')
    df = pd.DataFrame(data)
    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].apply(lambda x: x.decode('utf-8')).values
    y = y.astype(int)
    print(f"X_{type} shape", X.shape)
    return X, y 

