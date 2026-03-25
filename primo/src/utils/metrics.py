import numpy as np

def compute_relative_error(x_true, x_comp):
    """Calcola l'errore relativo: ||x_true - x_comp|| / ||x_true||"""
    num = np.linalg.norm(x_true - x_comp)
    den = np.linalg.norm(x_true)
    return num / den