"""
widgets/zoomable_canvas.py
--------------------------
Widget Canvas Tkinter con zoom (rotella mouse) e pan (drag tasto sinistro).

Funzionalità:
  - zoom centrato sul puntatore con la rotella del mouse
  - pan trascinando con il tasto sinistro
  - doppio clic per tornare alla vista fit-to-canvas
  - sincronizzazione bidirezionale con un canvas gemello (metodo sync_with)
  - supporto Windows (MouseWheel) e Linux/macOS (Button-4 / Button-5)
"""

import tkinter as tk
from PIL import ImageTk

from constants import ZOOM_FACTOR_IN, ZOOM_FACTOR_OUT, ZOOM_MIN, ZOOM_MAX


class ZoomableImageCanvas(tk.Canvas):
    """
    Canvas Tkinter che mostra un'immagine PIL con supporto a zoom e pan.
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)

        self._pil_image: "Image.Image | None" = None
        self._zoom_level: float = 1.0
        self._image_offset_x: float = 0.0
        self._image_offset_y: float = 0.0

        self._drag_start_x: int = 0
        self._drag_start_y: int = 0

        self._tk_image: ImageTk.PhotoImage | None = None
        self._synced_canvas: "ZoomableImageCanvas | None" = None
        self._syncing: bool = False

        self.bind("<ButtonPress-1>",   self._on_drag_start)
        self.bind("<B1-Motion>",       self._on_drag_move)
        self.bind("<MouseWheel>",      self._on_mousewheel_windows)
        self.bind("<Button-4>",        self._on_scroll_up_linux)
        self.bind("<Button-5>",        self._on_scroll_down_linux)
        self.bind("<Double-Button-1>", self._on_reset_view)
        self.bind("<Configure>",       self._on_resize)

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def set_image(self, pil_image: "Image.Image") -> None:
        """Imposta una nuova immagine PIL e ne adatta la vista al canvas."""
        self._pil_image = pil_image
        self._reset_fit()

    def clear(self) -> None:
        """Rimuove l'immagine e pulisce il canvas."""
        self._pil_image = None
        self._tk_image  = None
        self.delete("all")

    def sync_with(self, other: "ZoomableImageCanvas") -> None:
        """Collega questo canvas con un altro per sincronizzare zoom e pan."""
        self._synced_canvas  = other
        other._synced_canvas = self

    # ------------------------------------------------------------------
    # Rendering interno
    # ------------------------------------------------------------------

    def _reset_fit(self) -> None:
        """Adatta l'immagine alla dimensione del canvas mantenendo le proporzioni."""
        self.update_idletasks()
        if self._pil_image is None:
            return
        cw = self.winfo_width()  or 400
        ch = self.winfo_height() or 400
        iw, ih = self._pil_image.size
        self._zoom_level = min(cw / iw, ch / ih, 1.0)
        scaled_w = iw * self._zoom_level
        scaled_h = ih * self._zoom_level
        self._image_offset_x = (cw - scaled_w) / 2.0
        self._image_offset_y = (ch - scaled_h) / 2.0
        self._render()

    def _render(self) -> None:
        """Ridisegna l'immagine con il livello di zoom e l'offset correnti."""
        if self._pil_image is None:
            return
        iw, ih = self._pil_image.size
        new_w = max(1, int(iw * self._zoom_level))
        new_h = max(1, int(ih * self._zoom_level))
        scaled = self._pil_image.resize((new_w, new_h))
        self._tk_image = ImageTk.PhotoImage(scaled)
        self.delete("all")
        self.create_image(
            int(self._image_offset_x), int(self._image_offset_y),
            anchor=tk.NW, image=self._tk_image,
        )

    # ------------------------------------------------------------------
    # Calcolo coordinate
    # ------------------------------------------------------------------

    def _image_point_from_canvas(self, cx: float, cy: float) -> tuple[float, float]:
        """Converte coordinate canvas in coordinate normalizzate sull'immagine [0, 1]."""
        if self._pil_image is None or self._zoom_level == 0:
            return 0.0, 0.0
        iw, ih = self._pil_image.size
        return (
            (cx - self._image_offset_x) / (iw * self._zoom_level),
            (cy - self._image_offset_y) / (ih * self._zoom_level),
        )

    # ------------------------------------------------------------------
    # Sincronizzazione
    # ------------------------------------------------------------------

    def _apply_sync(self, zoom: float, img_px: float, img_py: float) -> None:
        """Aggiorna questo canvas con i parametri di vista provenienti dal canvas gemello."""
        if self._syncing:
            return
        self._syncing = True
        try:
            cw = self.winfo_width()  or 400
            ch = self.winfo_height() or 400
            new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
            self._zoom_level = new_zoom
            if self._pil_image is not None:
                iw, ih = self._pil_image.size
                self._image_offset_x = cw / 2.0 - img_px * iw * new_zoom
                self._image_offset_y = ch / 2.0 - img_py * ih * new_zoom
            self._render()
        finally:
            self._syncing = False

    def _propagate_view(self, anchor_cx: float, anchor_cy: float) -> None:
        """Propaga la vista corrente al canvas gemello centrandola su anchor."""
        if self._synced_canvas is None or self._synced_canvas._syncing:
            return
        if self._pil_image is None:
            return
        img_px, img_py = self._image_point_from_canvas(anchor_cx, anchor_cy)
        self._synced_canvas._apply_sync(self._zoom_level, img_px, img_py)

    # ------------------------------------------------------------------
    # Handler eventi
    # ------------------------------------------------------------------

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_move(self, event: tk.Event) -> None:
        self._image_offset_x += event.x - self._drag_start_x
        self._image_offset_y += event.y - self._drag_start_y
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._render()
        cw = self.winfo_width()  or 400
        ch = self.winfo_height() or 400
        self._propagate_view(cw / 2.0, ch / 2.0)

    def _zoom_at(self, x: int, y: int, factor: float) -> None:
        """Applica uno zoom di fattore `factor` centrato sul punto (x, y) del canvas."""
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom_level * factor))
        actual_factor = new_zoom / self._zoom_level
        self._image_offset_x = x - actual_factor * (x - self._image_offset_x)
        self._image_offset_y = y - actual_factor * (y - self._image_offset_y)
        self._zoom_level = new_zoom
        self._render()
        self._propagate_view(x, y)

    def _on_mousewheel_windows(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_IN if event.delta > 0 else ZOOM_FACTOR_OUT)

    def _on_scroll_up_linux(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_IN)

    def _on_scroll_down_linux(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_OUT)

    def _on_reset_view(self, _event: tk.Event) -> None:
        self._reset_fit()
        cw = self.winfo_width()  or 400
        ch = self.winfo_height() or 400
        self._propagate_view(cw / 2.0, ch / 2.0)

    def _on_resize(self, _event: tk.Event) -> None:
        self._render()
