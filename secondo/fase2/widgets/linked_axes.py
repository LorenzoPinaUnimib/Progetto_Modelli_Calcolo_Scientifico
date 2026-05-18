"""
widgets/linked_axes.py
----------------------
Collega gruppi di ZoomableChartCanvas per sincronizzare zoom e pan
in modo bidirezionale.

A differenza del precedente approccio basato sui callback xlim_changed /
ylim_changed di matplotlib (che richiedeva FigureCanvasTkAgg), questa classe
opera direttamente sui canvas Tkinter tramite il loro meccanismo sync_with,
garantendo una sincronizzazione pulita senza embedding matplotlib.

L'API pubblica è rimasta invariata (LinkedChartGroup può essere usato esattamente
come prima) ma internamente non dipende più da Figure/Axes matplotlib.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zoomable_chart_canvas import ZoomableChartCanvas


class LinkedChartGroup:
    """
    Collega più ZoomableChartCanvas in modo che zoom e pan siano sincronizzati.

    Chiama semplicemente .sync_with() su tutti i canvas del gruppo, che
    implementa la propagazione bidirezionale.

    Parameters
    ----------
    canvases : lista di ZoomableChartCanvas da sincronizzare
    """

    def __init__(self, canvases: list["ZoomableChartCanvas"]) -> None:
        self._canvases = list(canvases)
        # Collega ogni canvas con tutti gli altri
        for i, canvas in enumerate(self._canvases):
            peers = [c for j, c in enumerate(self._canvases) if j != i]
            if peers:
                canvas.sync_with(*peers)

    def reset_all(self) -> None:
        """Ripristina la vista fit-to-canvas su tutti i canvas del gruppo."""
        for canvas in self._canvases:
            canvas._reset_fit()


# ---------------------------------------------------------------------------
# Alias di retrocompatibilità: il vecchio nome LinkedAxesGroup viene mantenuto
# per non richiedere modifiche ad altri moduli che potrebbero importarlo.
# ---------------------------------------------------------------------------
LinkedAxesGroup = LinkedChartGroup
