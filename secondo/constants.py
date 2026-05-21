"""
constants.py
------------
Costanti di layout, testo e comportamento per la GUI di compressione DCT2.
"""

import platform

# ---------------------------------------------------------------------------
# Finestra principale
# ---------------------------------------------------------------------------

WINDOW_TITLE = "Compressione Immagini tramite DCT2"

# Dimensioni minime adattate alla piattaforma:
# su macOS i display Retina piccoli (1280×800) non reggono 1200×900 pt.
_SCREEN_FACTOR = 0.85  # usa al più l'85 % dello schermo logico
_PLATFORM = platform.system()

if _PLATFORM == "Darwin":
    WINDOW_MIN_WIDTH  = 900
    WINDOW_MIN_HEIGHT = 700
else:
    WINDOW_MIN_WIDTH  = 1200
    WINDOW_MIN_HEIGHT = 900

# ---------------------------------------------------------------------------
# Etichette e testo dei widget
# ---------------------------------------------------------------------------

LABEL_SELECT_IMAGE   = "Nessun file selezionato"
LABEL_ORIGINAL       = "Immagine originale"
LABEL_COMPRESSED     = "Immagine compressa"
BUTTON_SELECT_TEXT   = "Scegli immagine BMP\u2026"
BUTTON_COMPRESS_TEXT = "Comprimi"

PARAM_F_LABEL = "F  (dimensione blocco)"
PARAM_D_LABEL = "d  (soglia taglio frequenze)"

# ---------------------------------------------------------------------------
# Limiti dei parametri F e d
# ---------------------------------------------------------------------------

PARAM_F_MIN = 1
PARAM_F_MAX = 512
PARAM_D_MIN = 0

# ---------------------------------------------------------------------------
# Dialogo di selezione file
# ---------------------------------------------------------------------------

FILE_TYPES = [("Immagini BMP", "*.bmp"), ("Tutti i file", "*.*")]

# ---------------------------------------------------------------------------
# Zoom e pan nel canvas immagine
# ---------------------------------------------------------------------------

ZOOM_FACTOR_IN  = 1.25
ZOOM_FACTOR_OUT = 0.8
ZOOM_MIN        = 0.05
ZOOM_MAX        = 20.0

# ---------------------------------------------------------------------------
# Grafici DCT
# ---------------------------------------------------------------------------

# Numero massimo di blocchi campionati per la visualizzazione DCT
DCT_SAMPLE_BLOCKS = 6
