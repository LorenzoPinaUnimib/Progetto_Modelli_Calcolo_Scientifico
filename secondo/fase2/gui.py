"""
gui.py
------
Entry point dell'applicazione di compressione JPEG-like tramite DCT2.

Utilizzo
--------
Avvio normale (GUI)::

    python gui.py

Modalità test numerici (DCT 1D e DCT2 su blocco 8×8)::

    python gui.py --test

Dipendenze esterne: tkinter (stdlib), Pillow, numpy, scipy, matplotlib.

Struttura del progetto
----------------------
::

    fase2/
    ├── gui.py              ← entry point (questo file)
    ├── app.py              ← DctCompressionApp (finestra principale)
    ├── constants.py        ← costanti di layout e testo
    ├── dct_compression.py  ← algoritmo DCT block-by-block
    ├── dct_analysis.py     ← analisi frequenze DCT per i grafici
    ├── image_utils.py      ← caricamento/salvataggio BMP
    ├── tests.py            ← test numerici della specifica
    └── widgets/
        ├── __init__.py
        ├── zoomable_canvas.py  ← Canvas con zoom e pan
        ├── linked_axes.py      ← sincronizzazione assi matplotlib
        └── chart_panel.py      ← factory pannello grafico + toolbar
"""

import sys
import argparse
import tkinter as tk

from constants import PARAM_F_MIN, PARAM_D_MIN


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
        help="Esegue i test numerici della specifica (DCT 1D e DCT2 8×8) e termina.",
    )
    args = parser.parse_args()

    if args.test:
        from tests import run_tests
        run_tests()
        sys.exit(0)

    # Import pesanti solo quando serve la GUI
    from app import DctCompressionApp
    root = tk.Tk()
    DctCompressionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
