"""
widgets/zoomable_canvas.py
--------------------------
Widget Canvas Tkinter con zoom (rotella mouse) e pan (drag tasto sinistro).

Funzionalità:
  - zoom centrato sul puntatore con la rotella del mouse
  - pan trascinando con il tasto sinistro
  - doppio clic per tornare alla vista fit-to-canvas
  - sincronizzazione N-vie tramite sync_with(*others) — identica a ZoomableChartCanvas
  - supporto cross-platform: Windows (MouseWheel delta), macOS (MouseWheel
    units), Linux (Button-4 / Button-5)

Note macOS
----------
Su macOS il Cocoa event loop richiede che TUTTE le operazioni UI avvengano
sul thread principale. Questa classe è progettata per essere usata solo dal
main thread. Non chiamare metodi di questa classe da thread secondari.

Propagazione eventi scroll
--------------------------
Gli handler di scroll restituiscono "break" per impedire che l'evento venga
propagato al widget padre (il canvas principale di scorrimento della finestra).
Quando il mouse è su questo canvas, la rotella fa SOLO zoom — non scrolla la
pagina. Lo scroll della pagina avviene solo quando il mouse è in aree libere.
"""

import platform
import tkinter as tk
from PIL import Image, ImageTk

from constants import ZOOM_FACTOR_IN, ZOOM_FACTOR_OUT, ZOOM_MIN, ZOOM_MAX

_PLATFORM = platform.system()  # "Windows" | "Darwin" | "Linux"


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
        self._synced_canvases: "list[ZoomableImageCanvas]" = []
        self._syncing: bool = False

        self._resize_after_id: str | None = None

        self.bind("<ButtonPress-1>",   self._on_drag_start)
        self.bind("<B1-Motion>",       self._on_drag_move)
        self.bind("<Double-Button-1>", self._on_reset_view)
        self.bind("<Configure>",       self._on_resize)

        # Cross-platform scroll — restituiscono "break" per bloccare la propagazione
        if _PLATFORM == "Darwin":
            self.bind("<MouseWheel>", self._on_mousewheel_macos)
        elif _PLATFORM == "Windows":
            self.bind("<MouseWheel>", self._on_mousewheel_windows)
        else:
            self.bind("<Button-4>", self._on_scroll_up_linux)
            self.bind("<Button-5>", self._on_scroll_down_linux)

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def set_image(self, pil_image: "Image.Image") -> None:
        """Imposta una nuova immagine PIL e ne adatta la vista al canvas."""
        self._pil_image = pil_image
        self.reset_fit()

    def clear(self) -> None:
        """Rimuove l'immagine e pulisce il canvas."""
        self._pil_image = None
        self._tk_image  = None
        self.delete("all")

    def sync_with(self, *others: "ZoomableImageCanvas") -> None:
        """Collega questo canvas con altri per sincronizzare zoom e pan (N-vie)."""
        for other in others:
            if other not in self._synced_canvases:
                self._synced_canvases.append(other)
            if self not in other._synced_canvases:
                other._synced_canvases.append(self)

    def reset_fit(self) -> None:
        """Adatta l'immagine alla dimensione del canvas mantenendo le proporzioni."""
        if self._pil_image is None:
            return
        self.after_idle(self._do_reset_fit)

    # ------------------------------------------------------------------
    # Rendering interno
    # ------------------------------------------------------------------

    def _do_reset_fit(self) -> None:
        """Calcolo effettivo del fit — chiamato sempre nell'idle loop."""
        if self._pil_image is None:
            return
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw <= 1 or ch <= 1:
            self.after(50, self._do_reset_fit)
            return
        iw, ih = self._pil_image.size
        self._zoom_level = min(cw / iw, ch / ih, 1.0)
        scaled_w = iw * self._zoom_level
        scaled_h = ih * self._zoom_level
        self._image_offset_x = (cw - scaled_w) / 2.0
        self._image_offset_y = (ch - scaled_h) / 2.0
        self._render()

    _reset_fit = reset_fit

    def _render(self) -> None:
        if self._pil_image is None:
            return

        # 1. Dimensioni del Canvas e dell'Immagine Originale
        cw = self.winfo_width()
        ch = self.winfo_height()
        iw, ih = self._pil_image.size

        # 2. Calcoliamo i bordi dell'area visibile (in pixel dell'immagine originale)
        # Troviamo dove si trova il "punto 0,0" del canvas rispetto all'immagine
        x0 = -self._image_offset_x / self._zoom_level
        y0 = -self._image_offset_y / self._zoom_level
        x1 = x0 + cw / self._zoom_level
        y1 = y0 + ch / self._zoom_level

        # 3. Applichiamo i limiti per non andare fuori dall'immagine
        left   = max(0, int(x0))
        top    = max(0, int(y0))
        right  = min(iw, int(x1) + 1)
        bottom = min(ih, int(y1) + 1)

        if left >= right or top >= bottom:
            self.delete("all")
            return

        # 4. RITAGLIO: Prendiamo solo la parte visibile
        crop = self._pil_image.crop((left, top, right, bottom))

        # 5. RESIZE: Ingrandiamo solo il ritaglio
        # Calcoliamo quanto deve essere grande il pezzetto sul canvas
        display_w = int((right - left) * self._zoom_level)
        display_h = int((bottom - top) * self._zoom_level)
        
        if display_w <= 0 or display_h <= 0:
            return

        scaled = crop.resize((display_w, display_h), Image.NEAREST)
        self._tk_image = ImageTk.PhotoImage(scaled)

        # 6. DISEGNO: Calcoliamo la posizione corretta sul canvas
        # Dobbiamo compensare il fatto che il ritaglio potrebbe non partire da (0,0)
        canvas_x = int(left * self._zoom_level + self._image_offset_x)
        canvas_y = int(top * self._zoom_level + self._image_offset_y)

        self.delete("all")
        self.create_image(canvas_x, canvas_y, anchor=tk.NW, image=self._tk_image)

    # ------------------------------------------------------------------
    # Coordinate
    # ------------------------------------------------------------------

    def _image_point_from_canvas(self, cx: float, cy: float) -> tuple[float, float]:
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

    def _apply_sync(self, zoom: float, img_px: float, img_py: float,
                    anchor_cx: float, anchor_cy: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
            self._zoom_level = new_zoom
            if self._pil_image is not None:
                iw, ih = self._pil_image.size
                self._image_offset_x = anchor_cx - img_px * iw * new_zoom
                self._image_offset_y = anchor_cy - img_py * ih * new_zoom
            self._render()
        finally:
            self._syncing = False

    def _propagate_view(self, anchor_cx: float, anchor_cy: float) -> None:
        if not self._synced_canvases or self._pil_image is None:
            return
        img_px, img_py = self._image_point_from_canvas(anchor_cx, anchor_cy)
        for other in self._synced_canvases:
            if not other._syncing:
                other._apply_sync(self._zoom_level, img_px, img_py,
                                  anchor_cx, anchor_cy)

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
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom_level * factor))
        actual_factor = new_zoom / self._zoom_level
        self._image_offset_x = x - actual_factor * (x - self._image_offset_x)
        self._image_offset_y = y - actual_factor * (y - self._image_offset_y)
        self._zoom_level = new_zoom
        self._render()
        self._propagate_view(x, y)

    def _on_mousewheel_windows(self, event: tk.Event) -> str:
        """Windows: event.delta multiplo di 120. Ritorna 'break' per bloccare lo scroll pagina."""
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_IN if event.delta > 0 else ZOOM_FACTOR_OUT)
        return "break"

    def _on_mousewheel_macos(self, event: tk.Event) -> str:
        """macOS: event.delta in unità. Ritorna 'break' per bloccare lo scroll pagina."""
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_IN if event.delta > 0 else ZOOM_FACTOR_OUT)
        return "break"

    def _on_scroll_up_linux(self, event: tk.Event) -> str:
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_IN)
        return "break"

    def _on_scroll_down_linux(self, event: tk.Event) -> str:
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_OUT)
        return "break"

    def _on_reset_view(self, _event: tk.Event) -> None:
        """Doppio clic: reimposta la vista su tutti i canvas sincronizzati."""
        # Reset su se stesso
        self._do_reset_fit_and_propagate()
        # Reset anche su tutti i canvas collegati
        for other in self._synced_canvases:
            if not other._syncing:
                other._syncing = True
                try:
                    other.reset_fit()
                finally:
                    other._syncing = False

    def _do_reset_fit_and_propagate(self) -> None:
        """Esegue reset_fit locale e poi propaga ai peer."""
        self.reset_fit()

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(30, self._render)
