"""
dct_compression.py
------------------
Logica di compressione JPEG-like tramite DCT2.

Algoritmo per ogni blocco F×F:
  1. Applica DCT2 (scipy)                    → coefficienti frequenza
  2. Azzera i coefficienti con k+l >= d      → taglio frequenze alte
  3. Applica IDCT2                           → blocco ricostruito (float)
  4. Arrotonda, clamp in [0, 255]            → valori ammissibili a 1 byte
"""

import numpy as np
from scipy.fft import dctn, idctn


# ---------------------------------------------------------------------------
# DCT2 / IDCT2 bidimensionali
# ---------------------------------------------------------------------------

def apply_dct2(block: np.ndarray) -> np.ndarray:
    """
    Calcola la DCT-II 2D ortogonale (norm='ortho') su un blocco quadrato.

    Usare norm='ortho' garantisce che IDCT2(DCT2(x)) == x
    e corrisponde allo scaling visto a lezione (funzioni di base ortonormali).
    """
    return dctn(block, type=2, norm="ortho")


def apply_idct2(coefficients: np.ndarray) -> np.ndarray:
    """
    Calcola la DCT-II inversa 2D ortogonale su un array di coefficienti.
    """
    return idctn(coefficients, type=2, norm="ortho")


# ---------------------------------------------------------------------------
# Maschera di taglio frequenze
# ---------------------------------------------------------------------------

def build_frequency_cutoff_mask(block_size: int, threshold_d: int) -> np.ndarray:
    """
    Costruisce una maschera booleana FxF dove True = coefficiente da mantenere.

    Un coefficiente c[k, l] viene mantenuto se e solo se  k + l < d.
    Con d=0 si eliminano tutte le frequenze; con d=2F-2 si elimina solo
    l'angolo in basso a destra (k=F-1, l=F-1).

    Parameters
    ----------
    block_size  : dimensione F del blocco
    threshold_d : soglia di taglio (intero in [0, 2F-2])

    Returns
    -------
    mask : array booleano FxF  (True = mantieni, False = azzera)
    """
    row_indices, col_indices = np.indices((block_size, block_size))
    mask = (row_indices + col_indices) < threshold_d
    return mask


# ---------------------------------------------------------------------------
# Compressione di un singolo blocco
# ---------------------------------------------------------------------------

def compress_block(
    pixel_block: np.ndarray,
    frequency_mask: np.ndarray,
) -> np.ndarray:
    """
    Applica la compressione DCT a un singolo blocco F×F di pixel.

    Steps:
      1. DCT2 → dominio frequenza
      2. Azzera i coefficienti fuori dalla maschera
      3. IDCT2 → dominio spaziale (float)
      4. Arrotonda e clamp in [0, 255]

    Parameters
    ----------
    pixel_block     : blocco F×F di valori uint8 (o float)
    frequency_mask  : maschera booleana F×F (True = mantieni)

    Returns
    -------
    reconstructed_block : array F×F di uint8
    """
    dct_coefficients = apply_dct2(pixel_block.astype(float))

    # Azzera le frequenze al di là della diagonale k+l < d
    dct_coefficients[~frequency_mask] = 0.0

    reconstructed_float = apply_idct2(dct_coefficients)

    # Arrotonda all'intero più vicino e clamp in [0, 255]
    reconstructed_block = np.round(reconstructed_float).clip(0, 255).astype(np.uint8)
    return reconstructed_block


# ---------------------------------------------------------------------------
# Compressione dell'intera immagine
# ---------------------------------------------------------------------------

def compress_image(
    grayscale_image: np.ndarray,
    block_size: int,
    threshold_d: int,
) -> np.ndarray:
    """
    Applica la compressione DCT JPEG-like all'intera immagine in toni di grigio.

    L'immagine viene suddivisa in blocchi quadrati F×F partendo dall'angolo
    in alto a sinistra; i pixel rimanenti sul bordo destro/inferiore vengono
    scartati (come richiesto dalla specifica).

    Parameters
    ----------
    grayscale_image : array 2D di uint8 (altezza × larghezza)
    block_size      : ampiezza F dei macro-blocchi
    threshold_d     : soglia di taglio frequenze d ∈ [0, 2F-2]

    Returns
    -------
    compressed_image : array 2D di uint8, stesse dimensioni della parte
                       utilizzata dell'immagine originale (multiplo di F)
    """
    image_height, image_width = grayscale_image.shape

    # Numero di blocchi interi nella direzione verticale e orizzontale
    num_blocks_vertical   = image_height // block_size
    num_blocks_horizontal = image_width  // block_size

    # Dimensioni della porzione dell'immagine effettivamente processata
    used_height = num_blocks_vertical   * block_size
    used_width  = num_blocks_horizontal * block_size

    # L'immagine compressa ha le stesse dimensioni della porzione usata
    compressed_image = np.zeros((used_height, used_width), dtype=np.uint8)

    frequency_mask = build_frequency_cutoff_mask(block_size, threshold_d)

    for row_block_index in range(num_blocks_vertical):
        for col_block_index in range(num_blocks_horizontal):
            # Coordinate pixel del blocco corrente
            row_start = row_block_index * block_size
            row_end   = row_start + block_size
            col_start = col_block_index * block_size
            col_end   = col_start + block_size

            pixel_block = grayscale_image[row_start:row_end, col_start:col_end]

            compressed_block = compress_block(pixel_block, frequency_mask)

            compressed_image[row_start:row_end, col_start:col_end] = compressed_block

    return compressed_image
