"""
dct_analysis.py
---------------
Funzioni di analisi delle frequenze DCT2 sull'intera immagine.

Separate dalla logica di compressione (dct_compression.py) perché
riguardano esclusivamente la visualizzazione e l'analisi statistica
dei coefficienti, non la ricostruzione dell'immagine.
"""

import numpy as np

from dct_compression import apply_dct2, build_frequency_cutoff_mask


def build_dct_frequency_map(
    image: np.ndarray,
    block_size: int,
    threshold_d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola le frequenze DCT2 su tutti i blocchi dell'immagine.

    Parameters
    ----------
    image       : array 2D uint8 (altezza × larghezza)
    block_size  : ampiezza F dei blocchi
    threshold_d : soglia di taglio frequenze d

    Returns
    -------
    freq_full  : mappa F×F con la media dei |coefficienti DCT| su tutti i blocchi
    freq_trunc : stessa mappa con i coefficienti azzerati secondo la maschera diagonale
    """
    h, w = image.shape
    blocks_r = h // block_size
    blocks_c = w // block_size

    accum = np.zeros((block_size, block_size), dtype=float)
    count = 0

    for r in range(blocks_r):
        for c in range(blocks_c):
            block = image[
                r * block_size:(r + 1) * block_size,
                c * block_size:(c + 1) * block_size,
            ].astype(float)
            accum += np.abs(apply_dct2(block))
            count += 1

    if count == 0:
        return accum, accum.copy()

    freq_full = accum / count

    mask = build_frequency_cutoff_mask(block_size, threshold_d)
    freq_trunc = freq_full.copy()
    freq_trunc[~mask] = 0.0

    return freq_full, freq_trunc
