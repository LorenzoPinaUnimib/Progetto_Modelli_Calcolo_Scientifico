"""
widgets/chart_panel.py
----------------------
Factory function per creare un pannello grafico Tkinter + matplotlib.

Ogni pannello è un LabelFrame contenente:
  - una Figure matplotlib con un singolo Axes
  - la NavigationToolbar2Tk (zoom a rettangolo, pan, reset, salvataggio)
"""

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def make_chart_panel(
    parent: tk.Widget,
    title: str,
    draw_fn,
    fig_width: float = 5.5,
    fig_height: float = 3.5,
) -> tuple[ttk.LabelFrame, FigureCanvasTkAgg, "plt.Axes"]:
    """
    Crea un LabelFrame con figura matplotlib e toolbar integrata.

    Parameters
    ----------
    parent     : widget Tkinter padre
    title      : testo del LabelFrame
    draw_fn    : callable(fig, ax) che popola il grafico
    fig_width  : larghezza figura in pollici
    fig_height : altezza figura in pollici

    Returns
    -------
    (frame, canvas_mpl, ax)
      - frame      : il LabelFrame contenitore
      - canvas_mpl : FigureCanvasTkAgg per gestire il ciclo di vita della figura
      - ax         : l'Axes matplotlib per eventuali operazioni successive (es. linking)
    """
    frame = ttk.LabelFrame(parent, text=title, padding=4)

    fig = plt.Figure(figsize=(fig_width, fig_height), tight_layout=True)
    fig.patch.set_facecolor("#f5f5f5")
    ax = fig.add_subplot(111)

    draw_fn(fig, ax)

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()

    # La toolbar richiede come parent il widget Tkinter (non la figura)
    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
    toolbar.update()

    toolbar.pack(side=tk.TOP, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    return frame, canvas, ax
