"""
widgets/zoomable_chart_canvas.py
--------------------------------
Canvas Tkinter con zoom (rotella mouse) e pan (drag) che mostra un grafico
matplotlib renderizzato come immagine in memoria — senza embedding del pannello
FigureCanvasTkAgg.

Funzionalità
------------
- zoom centrato sul puntatore con la rotella del mouse
- pan trascinando con il tasto sinistro
- doppio clic per tornare alla vista fit-to-canvas
- collegamento bidirezionale con altri ZoomableChartCanvas (metodo sync_with)
- supporto cross-platform: Windows, macOS (Darwin), Linux
- debounce del resize per evitare freeze su macOS

Compatibilità macOS — threading
---------------------------------
Il render matplotlib avviene in un thread secondario (daemon).
Il risultato (PIL Image) viene consegnato al main thread via widget.after(0, ...)
e solo lì viene aggiornato il Tk PhotoImage e ridisegnato il canvas.

Propagazione eventi scroll
--------------------------
Gli handler di scroll restituiscono "break" per impedire che l'evento venga
propagato al widget padre. Quando il mouse è su questo canvas, la rotella
fa SOLO zoom — non scrolla la pagina.
"""

import io
import platform
import threading
import tkinter as tk
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import ZOOM_FACTOR_IN, ZOOM_FACTOR_OUT, ZOOM_MIN, ZOOM_MAX

_PLATFORM = platform.system()  # "Windows" | "Darwin" | "Linux"

_RENDER_DPI_BASE = 96


class ZoomableChartCanvas(tk.Canvas):
    """
    Canvas Tkinter che mostra un grafico matplotlib con zoom e pan.

    Parameters
    ----------
    parent      : widget padre Tkinter
    draw_fn     : callable(fig, ax) che popola il grafico
    fig_width   : larghezza della figura matplotlib in pollici
    fig_height  : altezza della figura matplotlib in pollici
    **kwargs    : argomenti passati a tk.Canvas
    """

    def __init__(
        self,
        parent: tk.Widget,
        draw_fn,
        fig_width: float  = 5.5,
        fig_height: float = 3.6,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)

        self._draw_fn    = draw_fn
        self._fig_width  = fig_width
        self._fig_height = fig_height

        self._fig: plt.Figure | None = None

        self._pil_image:  Image.Image       | None = None
        self._tk_image:   ImageTk.PhotoImage | None = None

        self._zoom_level:     float = 1.0
        self._image_offset_x: float = 0.0
        self._image_offset_y: float = 0.0

        self._drag_start_x: int = 0
        self._drag_start_y: int = 0

        self._synced_canvases: list["ZoomableChartCanvas"] = []
        self._syncing: bool = False

        self._resize_after_id: str | None = None

        self._render_lock = threading.Lock()
        self._render_pending: bool = False

        self.bind("<ButtonPress-1>",   self._on_drag_start)
        self.bind("<B1-Motion>",       self._on_drag_move)
        self.bind("<Double-Button-1>", self._on_reset_view)
        self.bind("<Configure>",       self._on_resize)

        # Scroll — restituiscono "break" per bloccare la propagazione verso la pagina
        if _PLATFORM == "Darwin":
            self.bind("<MouseWheel>", self._on_mousewheel_macos)
        elif _PLATFORM == "Windows":
            self.bind("<MouseWheel>", self._on_mousewheel_windows)
        else:
            self.bind("<Button-4>", self._on_scroll_up_linux)
            self.bind("<Button-5>", self._on_scroll_down_linux)

        self.after_idle(self._initial_render)

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    @property
    def fig(self) -> "plt.Figure | None":
        return self._fig

    def sync_with(self, *others: "ZoomableChartCanvas") -> None:
        for other in others:
            if other not in self._synced_canvases:
                self._synced_canvases.append(other)
            if self not in other._synced_canvases:
                other._synced_canvases.append(self)

    def redraw(self) -> None:
        self._launch_render(reset_fit_after=True)

    # ------------------------------------------------------------------
    # Render interno — asincrono
    # ------------------------------------------------------------------

    def _initial_render(self) -> None:
        self._launch_render(reset_fit_after=True)

    def _launch_render(self, reset_fit_after: bool = False) -> None:
        if not self._render_lock.acquire(blocking=False):
            self._render_pending = True
            return

        draw_fn    = self._draw_fn
        fig_width  = self._fig_width
        fig_height = self._fig_height
        old_fig    = self._fig

        def _worker():
            try:
                new_fig = plt.Figure(
                    figsize=(fig_width, fig_height),
                    dpi=_RENDER_DPI_BASE,
                    tight_layout=True,
                )
                new_fig.patch.set_facecolor("#f5f5f5")
                ax = new_fig.add_subplot(111)
                draw_fn(new_fig, ax)

                buf = io.BytesIO()
                new_fig.savefig(buf, format="png", dpi=_RENDER_DPI_BASE, bbox_inches="tight")
                buf.seek(0)
                pil_img = Image.open(buf).copy()
                buf.close()

                if old_fig is not None:
                    plt.close(old_fig)

                return new_fig, pil_img
            except Exception:
                return None, None

        def _on_done(result):
            new_fig, pil_img = result
            if new_fig is not None and pil_img is not None:
                self._fig       = new_fig
                self._pil_image = pil_img
                if reset_fit_after:
                    self._reset_fit()
                else:
                    self._display()

            self._render_lock.release()

            if self._render_pending:
                self._render_pending = False
                self._launch_render(reset_fit_after=False)

        def _thread_body():
            result = _worker()
            try:
                self.after(0, lambda: _on_done(result))
            except tk.TclError:
                self._render_lock.release()

        t = threading.Thread(target=_thread_body, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Fit e display
    # ------------------------------------------------------------------

    def _reset_fit(self) -> None:
        if self._pil_image is None:
            return
        self.after_idle(self._do_reset_fit)

    def _do_reset_fit(self) -> None:
        if self._pil_image is None:
            return
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw <= 1 or ch <= 1:
            self.after(50, self._do_reset_fit)
            return
        iw, ih = self._pil_image.size
        self._zoom_level = min(cw / iw, ch / ih, 1.0)
        self._image_offset_x = (cw - iw * self._zoom_level) / 2.0
        self._image_offset_y = (ch - ih * self._zoom_level) / 2.0
        self._display()

    def _display(self) -> None:
        if self._pil_image is None:
            return
        iw, ih = self._pil_image.size
        new_w = max(1, int(iw * self._zoom_level))
        new_h = max(1, int(ih * self._zoom_level))
        scaled = self._pil_image.resize((new_w, new_h), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(scaled)
        self.delete("all")
        self.create_image(
            int(self._image_offset_x), int(self._image_offset_y),
            anchor=tk.NW, image=self._tk_image,
        )

    # ------------------------------------------------------------------
    # Coordinate
    # ------------------------------------------------------------------

    def _canvas_to_image_norm(self, cx: float, cy: float) -> tuple[float, float]:
        if self._pil_image is None or self._zoom_level == 0:
            return 0.5, 0.5
        iw, ih = self._pil_image.size
        return (
            (cx - self._image_offset_x) / (iw * self._zoom_level),
            (cy - self._image_offset_y) / (ih * self._zoom_level),
        )

    # ------------------------------------------------------------------
    # Sincronizzazione
    # ------------------------------------------------------------------

    def _apply_sync(self, zoom: float, img_px: float, img_py: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            cw = self.winfo_width()  or 400
            ch = self.winfo_height() or 400
            self._zoom_level = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
            if self._pil_image is not None:
                iw, ih = self._pil_image.size
                self._image_offset_x = cw / 2.0 - img_px * iw * self._zoom_level
                self._image_offset_y = ch / 2.0 - img_py * ih * self._zoom_level
            self._display()
        finally:
            self._syncing = False

    def _propagate_view(self, anchor_cx: float, anchor_cy: float) -> None:
        if not self._synced_canvases or self._pil_image is None:
            return
        img_px, img_py = self._canvas_to_image_norm(anchor_cx, anchor_cy)
        for other in self._synced_canvases:
            if not other._syncing:
                other._apply_sync(self._zoom_level, img_px, img_py)

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _zoom_at(self, x: int, y: int, factor: float) -> None:
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom_level * factor))
        actual   = new_zoom / self._zoom_level
        self._image_offset_x = x - actual * (x - self._image_offset_x)
        self._image_offset_y = y - actual * (y - self._image_offset_y)
        self._zoom_level = new_zoom
        self._display()
        self._propagate_view(x, y)

    # ------------------------------------------------------------------
    # Handler eventi — tutti restituiscono "break" per bloccare lo scroll pagina
    # ------------------------------------------------------------------

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_move(self, event: tk.Event) -> None:
        self._image_offset_x += event.x - self._drag_start_x
        self._image_offset_y += event.y - self._drag_start_y
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._display()
        cw = self.winfo_width()  or 400
        ch = self.winfo_height() or 400
        self._propagate_view(cw / 2.0, ch / 2.0)

    def _on_mousewheel_windows(self, event: tk.Event) -> str:
        """Windows: delta multiplo di 120. Ritorna 'break' → solo zoom, no scroll pagina."""
        factor = ZOOM_FACTOR_IN if event.delta > 0 else ZOOM_FACTOR_OUT
        self._zoom_at(event.x, event.y, factor)
        return "break"

    def _on_mousewheel_macos(self, event: tk.Event) -> str:
        """macOS: delta in unità. Ritorna 'break' → solo zoom, no scroll pagina."""
        factor = ZOOM_FACTOR_IN if event.delta > 0 else ZOOM_FACTOR_OUT
        self._zoom_at(event.x, event.y, factor)
        return "break"

    def _on_scroll_up_linux(self, event: tk.Event) -> str:
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_IN)
        return "break"

    def _on_scroll_down_linux(self, event: tk.Event) -> str:
        self._zoom_at(event.x, event.y, ZOOM_FACTOR_OUT)
        return "break"

    def _on_reset_view(self, _event: tk.Event) -> None:
        self._reset_fit()
        cw = self.winfo_width()  or 400
        ch = self.winfo_height() or 400
        self._propagate_view(cw / 2.0, ch / 2.0)

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(30, self._display)
