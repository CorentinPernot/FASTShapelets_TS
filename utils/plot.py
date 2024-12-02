import matplotlib.pyplot as plt
import numpy as np

def plot_ecg_time_series(X, y, num_series_per_class=2):
    """
    Plot ECG time series for signals with label 0 and label 1 on the same plot.
    
    Parameters:
    - X: The time series data (e.g., X_train or X_test).
    - y: The corresponding labels for the time series (e.g., y_train or y_test).
    - num_series_per_class: The number of signals to plot per label.
    """
    # Sélectionner les indices des signaux pour chaque label
    label_1_indices = [i for i in range(len(y)) if y[i] == 1][:num_series_per_class]
    label_2_indices = [i for i in range(len(y)) if y[i] == 2][:num_series_per_class]
    
    plt.figure(figsize=(12, 8))
    
    for i, idx in enumerate(label_1_indices):
        plt.plot(X[idx], label=f'Label 1 - Signal {i+1}', color='royalblue')
    
    for i, idx in enumerate(label_2_indices):
        plt.plot(X[idx], label=f'Label 2 - Signal {i+1}', color='tomato')
    
    # Personnalisation du plot
    plt.title(f'ECG Signals for Label 1 and Label 2')
    plt.xlabel('Time')
    plt.ylabel('ECG signal amplitude')
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    # Sauvegarder et afficher le plot
    plt.savefig("figures/ecg_signals.png")
    plt.show()
