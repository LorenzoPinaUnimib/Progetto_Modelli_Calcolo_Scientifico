# widgets/__init__.py
# Rende la cartella un package Python e riesporta i widget principali.

from .zoomable_canvas import ZoomableImageCanvas
from .linked_axes import LinkedAxesGroup
from .chart_panel import make_chart_panel

__all__ = ["ZoomableImageCanvas", "LinkedAxesGroup", "make_chart_panel"]
