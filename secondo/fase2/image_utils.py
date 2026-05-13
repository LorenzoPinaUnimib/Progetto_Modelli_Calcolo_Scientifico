"""
image_utils.py
--------------
Utilità per il caricamento e il salvataggio di immagini BMP in toni di grigio.

Usa PIL (Pillow) per leggere e scrivere i file immagine.
"""

import numpy as np
from PIL import Image


def load_grayscale_bmp(file_path: str) -> np.ndarray:
    """
    Carica un'immagine BMP dal percorso indicato e la converte in scala di grigi.

    Se l'immagine è già in scala di grigi ('L') viene usata direttamente;
    se è a colori viene convertita tramite la formula di luminanza standard.

    Parameters
    ----------
    file_path : percorso al file .bmp

    Returns
    -------
    grayscale_array : array 2D numpy di uint8 (altezza × larghezza)

    Raises
    ------
    FileNotFoundError : se il file non esiste
    ValueError        : se il file non è un'immagine BMP valida
    """
    image = Image.open(file_path).convert("L")  # "L" = 8-bit grayscale
    grayscale_array = np.array(image, dtype=np.uint8)
    return grayscale_array


def numpy_array_to_pil_image(grayscale_array: np.ndarray) -> Image.Image:
    """
    Converte un array 2D uint8 in un oggetto PIL Image in scala di grigi.

    Parameters
    ----------
    grayscale_array : array 2D di uint8

    Returns
    -------
    pil_image : PIL Image in modalità "L"
    """
    return Image.fromarray(grayscale_array, mode="L")
