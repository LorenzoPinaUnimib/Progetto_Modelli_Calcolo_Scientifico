"""
widgets/zoomable_chart_canvas.py
--------------------------------
Widget Tkinter che embeds una figura matplotlib tramite FigureCanvasTkAgg.
Lo zoom e il pan aggiornano direttamente i limiti degli assi matplotlib
(xlim / ylim), esattamente come imshow interattivo — senza toolbar e senza
dover cliccare sulla mano.

Funzionalità
------------
- scroll rotella  : zoom centrato sul punto del mouse (aggiorna xlim/ylim)
- drag tasto sin. : pan che sposta i limiti degli assi
- doppio clic     : reimposta la vista originale (home)
- sync_with()     : sincronizzazione bidirezionale tra più canvas (stessi limiti)
- supporto cross-platform: Windows (MouseWheel delta ×120), macOS (Darwin),
  Linux (Button-4 / Button-5)
- nessuna NavigationToolbar2Tk — nessun bottone da cliccare

Note implementative
-------------------
FigureCanvasTkAgg è un widget Tk nativo (Frame interno + Canvas Tk).
Tutti gli event binding vengono attaccati al widget TK interno
(canvas.get_tk_widget()) in modo da intercettare scroll e drag prima che
matplotlib possa usare i propri handler interni (che aprirebbero la toolbar).

Il blocco "break" sugli handler di scroll impedisce la propagazione
al canvas di scorrimento principale della finestra.
"""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk
from typing import Callable

import matplotlib
matplotlib.use("TkAgg")          # backend interattivo Tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

_PLATFORM = platform.system()  # "Windows" | "Darwin" | "Linux"

# Fattori di zoom per rotella mouse
_ZOOM_IN  = 1.25
_ZOOM_OUT = 0.80


class ZoomableChartCanvas(tk.Frame):
    """
    Frame Tkinter che contiene una figura matplotlib con zoom e pan sugli assi.

    Parameters
    ----------
    parent    : widget padre Tkinter
    draw_fn   : callable(fig, ax) che popola il grafico
    fig_width : larghezza figura in pollici
    fig_height: altezza figura in pollici
    **kwargs  : argomenti extra passati al Frame contenitore
    """

    def __init__(
        self,
        parent: tk.Widget,
        draw_fn: Callable,
        fig_width: float = 5.5,
        fig_height: float = 3.6,
        **kwargs,
    ) -> None:
        # Estraiamo 'background' e 'cursor' prima di passarli al Frame
        bg = kwargs.pop("background", kwargs.pop("bg", "#2b2b2b"))
        cursor = kwargs.pop("cursor", "fleur")
        super().__init__(parent, background=bg, **kwargs)

        self._draw_fn    = draw_fn
        self._fig_width  = fig_width
        self._fig_height = fig_height

        self._synced_canvases: list[ZoomableChartCanvas] = []
        self._syncing: bool = False

        # Limiti originali (salvati al primo draw per il reset)
        self._home_xlim: tuple[float, float] | None = None
        self._home_ylim: tuple[float, float] | None = None

        # Dati di drag
        self._drag_start_data: tuple[float, float] | None = None  # (xdata, ydata)

        # --- Crea figura e canvas matplotlib ---
        self._fig = Figure(
            figsize=(fig_width, fig_height),
            dpi=96,
            tight_layout=True,
            facecolor="#f5f5f5",
        )
        self._ax = self._fig.add_subplot(111)

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._tk_widget  = self._mpl_canvas.get_tk_widget()
        self._tk_widget.configure(cursor=cursor)
        self._tk_widget.pack(fill=tk.BOTH, expand=True)

        # Disegna il grafico iniziale
        draw_fn(self._fig, self._ax)
        self._mpl_canvas.draw()
        self._save_home()

        # Binding eventi (sul widget Tk interno del canvas matplotlib)
        self._tk_widget.bind("<ButtonPress-1>",   self._on_drag_start)
        self._tk_widget.bind("<B1-Motion>",       self._on_drag_move)
        self._tk_widget.bind("<ButtonRelease-1>", self._on_drag_end)
        self._tk_widget.bind("<Double-Button-1>", self._on_reset_view)

        if _PLATFORM == "Darwin":
            self._tk_widget.bind("<MouseWheel>", self._on_scroll_macos)
        elif _PLATFORM == "Windows":
            self._tk_widget.bind("<MouseWheel>", self._on_scroll_windows)
        else:
            self._tk_widget.bind("<Button-4>", self._on_scroll_up_linux)
            self._tk_widget.bind("<Button-5>", self._on_scroll_down_linux)

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    @property
    def fig(self) -> Figure:
        return self._fig

    @property
    def ax(self):
        return self._ax

    def sync_with(self, *others: ZoomableChartCanvas) -> None:
        """Collega questo canvas ad altri per sincronizzare zoom e pan."""
        for other in others:
            if other not in self._synced_canvases:
                self._synced_canvases.append(other)
            if self not in other._synced_canvases:
                other._synced_canvases.append(self)

    def redraw(self) -> None:
        """Ridisegna il grafico da zero (utile se draw_fn dipende da dati aggiornati)."""
        self._ax.cla()
        self._draw_fn(self._fig, self._ax)
        self._mpl_canvas.draw_idle()
        self._save_home()

    # ------------------------------------------------------------------
    # Limiti degli assi — helper interni
    # ------------------------------------------------------------------

    def _save_home(self) -> None:
        """Salva i limiti iniziali per poter fare reset con doppio clic."""
        self._home_xlim = self._ax.get_xlim()
        self._home_ylim = self._ax.get_ylim()

    def _get_lims(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self._ax.get_xlim(), self._ax.get_ylim()

    def _set_lims(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        redraw: bool = True,
    ) -> None:
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)
        if redraw:
            self._mpl_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Conversione coordinate canvas → dati matplotlib
    # ------------------------------------------------------------------

    def _canvas_to_data(self, event_x: int, event_y: int) -> tuple[float, float] | None:
        """
        Converte le coordinate pixel del widget Tk in coordinate dei dati matplotlib.
        Restituisce None se il punto è fuori dall'area degli assi.
        """
        # FigureCanvasTkAgg usa y invertita rispetto a Tk
        fig_h = self._tk_widget.winfo_height()
        mpl_x = event_x
        mpl_y = fig_h - event_y   # flip asse y

        inv = self._ax.transData.inverted()
        try:
            xd, yd = inv.transform((mpl_x, mpl_y))
        except Exception:
            return None
        return xd, yd

    # ------------------------------------------------------------------
    # Zoom centrato sul puntatore
    # ------------------------------------------------------------------

    def _zoom_at(self, event_x: int, event_y: int, factor: float) -> None:
        """
        Zoom che mantiene fisso il punto del mouse nello spazio dei dati.
        factor > 1 → zoom in (ingrandisce), < 1 → zoom out (rimpicciolisce).
        """
        pt = self._canvas_to_data(event_x, event_y)
        if pt is None:
            return
        xm, ym = pt

        xlim, ylim = self._get_lims()
        x0, x1 = xlim
        y0, y1 = ylim

        # Mantieni il punto (xm, ym) fisso: scala i range attorno ad esso
        new_x0 = xm - (xm - x0) / factor
        new_x1 = xm + (x1 - xm) / factor
        new_y0 = ym - (ym - y0) / factor
        new_y1 = ym + (y1 - ym) / factor

        self._set_lims((new_x0, new_x1), (new_y0, new_y1))
        self._propagate_lims()

    # ------------------------------------------------------------------
    # Pan
    # ------------------------------------------------------------------

    def _on_drag_start(self, event: tk.Event) -> None:
        self._drag_start_data = self._canvas_to_data(event.x, event.y)

    def _on_drag_move(self, event: tk.Event) -> None:
        if self._drag_start_data is None:
            return
        pt = self._canvas_to_data(event.x, event.y)
        if pt is None:
            return
        dx = pt[0] - self._drag_start_data[0]
        dy = pt[1] - self._drag_start_data[1]

        xlim, ylim = self._get_lims()
        new_xlim = (xlim[0] - dx, xlim[1] - dx)
        new_ylim = (ylim[0] - dy, ylim[1] - dy)
        self._set_lims(new_xlim, new_ylim)
        self._propagate_lims()
        # Aggiorna il punto di partenza in coordinate dati AGGIORNATE
        self._drag_start_data = self._canvas_to_data(event.x, event.y)

    def _on_drag_end(self, _event: tk.Event) -> None:
        self._drag_start_data = None

    # ------------------------------------------------------------------
    # Reset vista
    # ------------------------------------------------------------------

    def _on_reset_view(self, _event: tk.Event) -> None:
        if self._home_xlim and self._home_ylim:
            self._set_lims(self._home_xlim, self._home_ylim)
            self._propagate_lims()

    # ------------------------------------------------------------------
    # Sincronizzazione tra canvas
    # ------------------------------------------------------------------

    def _propagate_lims(self) -> None:
        """Invia i limiti correnti a tutti i canvas sincronizzati."""
        if self._syncing or not self._synced_canvases:
            return
        xlim, ylim = self._get_lims()
        for other in self._synced_canvases:
            if not other._syncing:
                other._apply_lims(xlim, ylim)

    def _apply_lims(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        """Riceve limiti da un canvas fratello e li applica senza propagare oltre."""
        if self._syncing:
            return
        self._syncing = True
        try:
            self._set_lims(xlim, ylim)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Handler scroll — tutti restituiscono "break" per bloccare scroll pagina
    # ------------------------------------------------------------------

    def _on_scroll_windows(self, event: tk.Event) -> str:
        factor = _ZOOM_IN if event.delta > 0 else _ZOOM_OUT
        self._zoom_at(event.x, event.y, factor)
        return "break"

    def _on_scroll_macos(self, event: tk.Event) -> str:
        factor = _ZOOM_IN if event.delta > 0 else _ZOOM_OUT
        self._zoom_at(event.x, event.y, factor)
        return "break"

    def _on_scroll_up_linux(self, event: tk.Event) -> str:
        self._zoom_at(event.x, event.y, _ZOOM_IN)
        return "break"

    def _on_scroll_down_linux(self, event: tk.Event) -> str:
        self._zoom_at(event.x, event.y, _ZOOM_OUT)
        return "break"
