import numpy as np

def _compute_relative_error(x_true, x_comp):
    """Calcola l'errore relativo: ||x_true - x_comp|| / ||x_true||"""
    if x_comp is None:
        return float("NaN")
    num = np.linalg.norm(x_true - x_comp)
    den = np.linalg.norm(x_true)
    return num / den if den else float("NaN")