# widgets/__init__.py
# Rende la cartella un package Python e riesporta i widget principali.

from .zoomable_canvas import ZoomableImageCanvas
from .zoomable_chart_canvas import ZoomableChartCanvas
from .linked_axes import LinkedChartGroup, LinkedAxesGroup   # LinkedAxesGroup = alias
from .chart_panel import make_chart_panel

__all__ = [
    "ZoomableImageCanvas",
    "ZoomableChartCanvas",
    "LinkedChartGroup",
    "LinkedAxesGroup",
    "make_chart_panel",
]
