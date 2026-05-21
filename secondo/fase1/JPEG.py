
import math
import os
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — assicura che la root 'secondo/' sia importabile
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Funzioni di elaborazione immagine (invariate dalla versione originale)
# ---------------------------------------------------------------------------

def apri_immagine(name):
    percorso = os.path.join(__file__[:-14], "dati/", name)
    if not os.path.exists(percorso):
        print(f"Errore: File immagine non trovato: {percorso}")
        return None
    img = np.array(Image.open(percorso).convert("L"), dtype=np.uint8)
    return img


def split(immagine, dimensione):
    righe = len(immagine) // dimensione
    colonne = len(immagine[0]) // dimensione
    blocchi = np.zeros((righe * colonne, dimensione, dimensione))
    for i in range(righe):
        for j in range(colonne):
            blocchi[i * colonne + j] = immagine[
                i * dimensione:(i + 1) * dimensione,
                j * dimensione:(j + 1) * dimensione,
            ]
    blocchi = np.round(blocchi) - 128
    blocchi[blocchi < -128] = -128
    blocchi[blocchi > 127] = 127
    return blocchi, righe, colonne


def desplit(blocchi, righe, colonne):
    dim_blocco = len(blocchi[0])
    blocchi = np.round(blocchi) + 128
    blocchi[blocchi < 0] = 0
    blocchi[blocchi > 255] = 255
    immagine = np.zeros((righe * dim_blocco, colonne * dim_blocco))
    for i in range(righe):
        for j in range(colonne):
            immagine[
                i * dim_blocco:(i + 1) * dim_blocco,
                j * dim_blocco:(j + 1) * dim_blocco,
            ] = blocchi[i * colonne + j]
    return immagine


def calcola_D(N):
    D = [[1 / math.sqrt(N) * math.cos(0) for j in range(N)]]
    for i in range(1, N):
        D.append([
            math.sqrt(2 / N) * math.cos(i * math.pi * (2 * j + 1) / (2 * N))
            for j in range(N)
        ])
    return D


def tronca(c, n, triang=False):
    if triang:
        N = c.shape[1]
        k, l = np.indices((N, N))
        mask = (k + l) < n
        c[:, ~mask] = 0
    else:
        c[:, n:, :] = 0
        c[:, :, n:] = 0


def DCT1(f, D):
    return D @ f


def DCT2(blocchi, D):
    b = []
    for blocco in blocchi:
        tmp = np.array([DCT1(row, D) for row in blocco])
        tmp = np.array([DCT1(col, D) for col in tmp.T])
        b.append(np.asarray(tmp.T))
    return np.stack(b, axis=0)


def IDCT1(c, D_tr):
    return D_tr @ c


def IDCT2(c, D_tr):
    b = []
    for blocco in c:
        tmp = np.array([IDCT1(row, D_tr) for row in blocco])
        tmp = np.array([IDCT1(col, D_tr) for col in tmp.T])
        b.append(np.asarray(tmp.T))
    return np.stack(b, axis=0)


# ---------------------------------------------------------------------------
# GUI Tkinter — usa ZoomableImageCanvas condiviso con fase2
# ---------------------------------------------------------------------------

def _show_comparison_window(img_orig: np.ndarray, img_rec: np.ndarray,
                             title1: str, title2: str) -> None:
    """
    Apre una finestra Tkinter con le due immagini affiancate, zoom/pan sincronizzato,
    usando ZoomableImageCanvas (widget condiviso con fase2).
    """
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image as PilImage

    from widgets import ZoomableImageCanvas
    from constants import ZOOM_FACTOR_IN, ZOOM_FACTOR_OUT

    def _np_to_pil(arr: np.ndarray) -> PilImage.Image:
        return PilImage.fromarray(arr.astype(np.uint8), mode="L")

    root = tk.Tk()
    root.title("Visualizzatore risultato compressione JPEG")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    win_w = min(1400, int(sw * 0.90))
    win_h = min(700,  int(sh * 0.85))
    root.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")
    root.minsize(800, 400)

    # Layout principale
    main_frame = ttk.Frame(root, padding=8)
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(1, weight=1)

    # Titoli
    ttk.Label(main_frame, text=title1, font=("", 11, "bold"), anchor="center").grid(
        row=0, column=0, sticky=tk.EW, pady=(0, 4)
    )
    ttk.Label(main_frame, text=title2, font=("", 11, "bold"), anchor="center").grid(
        row=0, column=1, sticky=tk.EW, pady=(0, 4)
    )

    # Canvas immagini
    canvas_orig = ZoomableImageCanvas(
        main_frame, background="#2b2b2b", cursor="fleur",
    )
    canvas_orig.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 4))

    canvas_rec = ZoomableImageCanvas(
        main_frame, background="#2b2b2b", cursor="fleur",
    )
    canvas_rec.grid(row=1, column=1, sticky=tk.NSEW, padx=(4, 0))

    # Suggerimento controlli
    ttk.Label(
        main_frame,
        text="\U0001f50d Rotella: zoom  \u2022  \U0001f5b1 Trascina: pan  "
             "\u2022  \u2b1b Doppio clic: reimposta vista",
        foreground="gray",
    ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

    # Sincronizzazione zoom/pan bidirezionale
    canvas_orig.sync_with(canvas_rec)

    # Carica immagini (dopo che la finestra è pronta)
    pil_orig = _np_to_pil(img_orig)
    pil_rec  = _np_to_pil(img_rec)

    def _after_ready():
        canvas_orig.set_image(pil_orig)
        canvas_rec.set_image(pil_rec)

    root.after(50, _after_ready)
    root.mainloop()


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def JPEG(img, N, M, grafico=True, triangolare=False):
    if img is None or M > N or N <= 0 or M < 0:
        print("I parametri passati non sono una configurazione valida")
        return

    blocchi, righe, colonne = split(img, N)

    if len(blocchi) == 0:
        print("Il numero di blocchi creato è 0, riprovare con N più piccolo")
        return

    D = calcola_D(N)

    c = DCT2(blocchi, D)
    tronca(c, M, triangolare)

    blocchi = IDCT2(c, np.transpose(D))
    img_rec = desplit(blocchi, righe, colonne)

    if grafico:
        _show_comparison_window(
            img,
            img_rec.astype(np.uint8),
            title1="Originale",
            title2=f"Ricostruita (troncando da {N} a {M})",
        )

    return img_rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    img = apri_immagine("tonypitony.bmp")
    JPEG(img, 5, 5, True, False)
