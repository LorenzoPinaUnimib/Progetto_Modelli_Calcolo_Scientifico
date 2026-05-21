"""
widgets/chart_panel.py
----------------------
Factory function per creare un pannello grafico Tkinter con ZoomableChartCanvas.

Ogni pannello è un LabelFrame contenente un ZoomableChartCanvas che usa
FigureCanvasTkAgg embedded con zoom/pan sugli assi matplotlib (xlim/ylim).
Nessuna NavigationToolbar2Tk — la toolbar non viene mai creata.
"""

import tkinter as tk
from tkinter import ttk

from .zoomable_chart_canvas import ZoomableChartCanvas


def make_chart_panel(
    parent: tk.Widget,
    title: str,
    draw_fn,
    fig_width: float = 5.5,
    fig_height: float = 3.5,
) -> tuple[ttk.LabelFrame, "ZoomableChartCanvas"]:
    """
    Crea un LabelFrame con un ZoomableChartCanvas che mostra il grafico.

    Parameters
    ----------
    parent     : widget Tkinter padre
    title      : testo del LabelFrame
    draw_fn    : callable(fig, ax) che popola il grafico
    fig_width  : larghezza figura in pollici
    fig_height : altezza figura in pollici

    Returns
    -------
    (frame, chart_canvas)
      - frame        : il LabelFrame contenitore
      - chart_canvas : ZoomableChartCanvas con zoom/pan nativi e proprietà .fig
    """
    frame = ttk.LabelFrame(parent, text=title, padding=4)

    chart_canvas = ZoomableChartCanvas(
        frame,
        draw_fn=draw_fn,
        fig_width=fig_width,
        fig_height=fig_height,
        background="#2b2b2b",
        cursor="fleur",
    )
    chart_canvas.pack(fill=tk.BOTH, expand=True)

    return frame, chart_canvas
