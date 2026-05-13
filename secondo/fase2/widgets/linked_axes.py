"""
widgets/linked_axes.py
----------------------
Collega gruppi di Axes matplotlib per sincronizzare zoom e pan
in modo bidirezionale tramite i callback xlim_changed / ylim_changed.

A differenza di sharex/sharey (solo in fase di creazione), questa classe
può essere applicata ad Axes già esistenti e appartenenti a figure diverse.
"""

import matplotlib.pyplot as plt


class LinkedAxesGroup:
    """
    Sincronizza i limiti di visualizzazione (xlim / ylim) tra più Axes matplotlib.

    Parameters
    ----------
    axes_list : lista di Axes da collegare
    sync_x    : se True, propaga i cambiamenti sull'asse X
    sync_y    : se True, propaga i cambiamenti sull'asse Y
    """

    def __init__(
        self,
        axes_list: list,
        sync_x: bool = True,
        sync_y: bool = True,
    ) -> None:
        self._axes    = list(axes_list)
        self._sync_x  = sync_x
        self._sync_y  = sync_y
        self._updating = False

        for ax in self._axes:
            if sync_x:
                ax.callbacks.connect("xlim_changed", self._on_xlim_changed)
            if sync_y:
                ax.callbacks.connect("ylim_changed", self._on_ylim_changed)

    # ------------------------------------------------------------------
    # Callback interni
    # ------------------------------------------------------------------

    def _on_xlim_changed(self, changed_ax) -> None:
        if self._updating or not self._sync_x:
            return
        self._updating = True
        try:
            xlim = changed_ax.get_xlim()
            for ax in self._axes:
                if ax is not changed_ax:
                    ax.set_xlim(xlim, emit=False)
                    ax.figure.canvas.draw_idle()
        finally:
            self._updating = False

    def _on_ylim_changed(self, changed_ax) -> None:
        if self._updating or not self._sync_y:
            return
        self._updating = True
        try:
            ylim = changed_ax.get_ylim()
            for ax in self._axes:
                if ax is not changed_ax:
                    ax.set_ylim(ylim, emit=False)
                    ax.figure.canvas.draw_idle()
        finally:
            self._updating = False

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """Ripristina la vista autoscale su tutti gli assi del gruppo."""
        for ax in self._axes:
            ax.autoscale()
            ax.figure.canvas.draw_idle()
