"""
gui.py
------
Entry point dell'applicazione di compressione JPEG-like tramite DCT2.

Utilizzo
--------
Avvio normale (GUI)::

    python fase2/run.py           ← raccomandato
    python fase2/gui.py           ← funziona anche direttamente

Modalità test numerici (DCT 1D e DCT2 su blocco 8x8)::

    python fase2/run.py --test

Dipendenze esterne: tkinter (stdlib), Pillow, numpy, scipy, matplotlib.

Struttura del progetto
----------------------
::

    secondo/
    ├── constants.py            ← costanti condivise tra le fasi
    ├── widgets/                ← widget UI condivisi tra le fasi
    │   ├── __init__.py
    │   ├── zoomable_canvas.py
    │   ├── zoomable_chart_canvas.py
    │   ├── linked_axes.py
    │   └── chart_panel.py
    ├── fase1/
    │   ├── 1D_base.py
    │   ├── JPEG.py
    │   └── run.py              ← launcher fase1
    └── fase2/
        ├── gui.py              ← entry point fase2 (questo file)
        ├── run.py              ← launcher fase2 (raccomandato)
        ├── app.py
        ├── dct_compression.py
        ├── dct_analysis.py
        ├── image_utils.py
        └── tests.py
"""

import sys
import os

# Assicura che la root 'secondo/' sia nel path per importare widgets/ e constants.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Assicura che fase2/ sia nel path per importare i moduli locali
_FASE2 = os.path.dirname(os.path.abspath(__file__))
if _FASE2 not in sys.path:
    sys.path.insert(0, _FASE2)

import argparse
import tkinter as tk

from constants import PARAM_F_MIN, PARAM_D_MIN, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT


# ---------------------------------------------------------------------------
# Validazione parametri (usata anche da app.py)
# ---------------------------------------------------------------------------

def validate_compression_parameters(block_size: int, threshold_d: int) -> str | None:
    """
    Controlla che F e d siano nei range ammissibili.

    Returns
    -------
    str  : messaggio di errore se la validazione fallisce
    None : se i parametri sono validi
    """
    if block_size < PARAM_F_MIN:
        return f"F deve essere almeno {PARAM_F_MIN}."
    if threshold_d < PARAM_D_MIN or threshold_d > 2 * block_size - 2:
        return f"d deve essere compreso tra 0 e {2 * block_size - 2} (con F={block_size})."
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compressione immagini DCT2 — GUI o modalità test."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Esegue i test numerici della specifica (DCT 1D e DCT2 8x8) e termina.",
    )
    args = parser.parse_args()

    if args.test:
        from tests import run_tests
        run_tests()
        sys.exit(0)

    # Import pesanti solo quando serve la GUI
    from app import DctCompressionApp
    root = tk.Tk()

    # Adatta il minsize allo schermo disponibile (importante su macOS Retina)
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    min_w = min(WINDOW_MIN_WIDTH,  int(sw * 0.85))
    min_h = min(WINDOW_MIN_HEIGHT, int(sh * 0.85))
    root.minsize(min_w, min_h)
    # Centra la finestra allo startup
    root.geometry(f"{min_w}x{min_h}+{(sw - min_w) // 2}+{(sh - min_h) // 2}")

    DctCompressionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
