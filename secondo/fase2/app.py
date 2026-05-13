"""
app.py
------
Classe principale dell'applicazione GUI per la compressione DCT2.

DctCompressionApp gestisce:
  - costruzione e layout dei widget Tkinter
  - gestione degli eventi utente (selezione file, avvio compressione)
  - aggiornamento delle anteprime immagine (originale / compressa)
  - visualizzazione di 4 grafici interattivi con zoom/pan linkati a coppie:
      * istogramma originale  ↔  istogramma compressa
      * mappa DCT originale   ↔  mappa DCT troncata (coefficienti azzerati)
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from image_utils import load_grayscale_bmp, numpy_array_to_pil_image
from dct_compression import compress_image
from dct_analysis import build_dct_frequency_map
from constants import (
    WINDOW_TITLE, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
    LABEL_SELECT_IMAGE, LABEL_ORIGINAL, LABEL_COMPRESSED,
    BUTTON_SELECT_TEXT, BUTTON_COMPRESS_TEXT,
    PARAM_F_LABEL, PARAM_D_LABEL,
    PARAM_F_MIN, PARAM_F_MAX, PARAM_D_MIN,
    FILE_TYPES,
)
from widgets import ZoomableImageCanvas, LinkedAxesGroup, make_chart_panel
from gui import validate_compression_parameters


class DctCompressionApp:
    """
    Finestra principale dell'applicazione di compressione DCT2.
    """

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title(WINDOW_TITLE)
        self._root.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self._selected_image_path: str | None             = None
        self._original_grayscale_array: np.ndarray | None = None

        # Risorse grafici matplotlib (frame + canvas)
        self._chart_resources: list[tuple[tk.Widget, "FigureCanvasTkAgg"]] = []
        # Gruppi di assi linkati (ricreati a ogni compressione)
        self._linked_axes_groups: list[LinkedAxesGroup] = []
        # Frame contenitore dei 4 pannelli grafici
        self._charts_outer_frame: ttk.Frame | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Costruzione dell'interfaccia
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Crea la struttura principale con scrollbar verticale."""
        self._main_canvas = tk.Canvas(self._root, highlightthickness=0)
        self._scrollbar   = ttk.Scrollbar(
            self._root, orient=tk.VERTICAL, command=self._main_canvas.yview
        )
        self._main_canvas.configure(yscrollcommand=self._scrollbar.set)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner_frame = ttk.Frame(self._main_canvas)
        self._inner_window_id = self._main_canvas.create_window(
            (0, 0), window=self._inner_frame, anchor=tk.NW
        )

        self._inner_frame.bind("<Configure>", self._on_inner_frame_configure)
        self._main_canvas.bind("<Configure>", self._on_canvas_configure)
        self._main_canvas.bind("<MouseWheel>", self._on_main_scroll_windows)
        self._main_canvas.bind("<Button-4>",   self._on_main_scroll_up_linux)
        self._main_canvas.bind("<Button-5>",   self._on_main_scroll_down_linux)

        self._build_control_panel()
        self._build_image_preview_area()

    def _on_inner_frame_configure(self, _event: tk.Event) -> None:
        self._main_canvas.configure(scrollregion=self._main_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._main_canvas.itemconfig(self._inner_window_id, width=event.width)

    def _on_main_scroll_windows(self, event: tk.Event) -> None:
        self._main_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _on_main_scroll_up_linux(self, _event: tk.Event) -> None:
        self._main_canvas.yview_scroll(-1, "units")

    def _on_main_scroll_down_linux(self, _event: tk.Event) -> None:
        self._main_canvas.yview_scroll(1, "units")

    def _build_control_panel(self) -> None:
        """Crea la barra in cima con selezione file, parametri F/d e bottone Comprimi."""
        control_frame = ttk.Frame(self._inner_frame, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Riga selezione file
        file_row = ttk.Frame(control_frame)
        file_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(
            file_row, text=BUTTON_SELECT_TEXT,
            command=self._on_select_image_clicked,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._file_path_label = ttk.Label(
            file_row, text=LABEL_SELECT_IMAGE, foreground="gray",
        )
        self._file_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Riga parametri
        params_row = ttk.Frame(control_frame)
        params_row.pack(fill=tk.X)
        self._block_size_var  = tk.IntVar(value=8)
        self._threshold_d_var = tk.IntVar(value=0)

        ttk.Label(params_row, text=PARAM_F_LABEL).pack(side=tk.LEFT)
        ttk.Spinbox(
            params_row, from_=PARAM_F_MIN, to=PARAM_F_MAX,
            textvariable=self._block_size_var, width=6,
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(params_row, text=PARAM_D_LABEL).pack(side=tk.LEFT)
        ttk.Spinbox(
            params_row, from_=PARAM_D_MIN, to=9999,
            textvariable=self._threshold_d_var, width=6,
        ).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Button(
            params_row, text=BUTTON_COMPRESS_TEXT,
            command=self._on_compress_clicked,
        ).pack(side=tk.LEFT)

        ttk.Label(
            control_frame,
            text="\U0001f50d Rotella mouse: zoom  \u2022  \U0001f5b1 Trascina: pan  "
                 "\u2022  \u2b1b Doppio clic: reimposta vista",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(4, 0))

    def _build_image_preview_area(self) -> None:
        """Crea i due canvas affiancati per le anteprime originale / compressa."""
        preview_frame = ttk.Frame(self._inner_frame)
        preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        original_panel = ttk.LabelFrame(preview_frame, text=LABEL_ORIGINAL, padding=5)
        original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._original_canvas = ZoomableImageCanvas(
            original_panel, background="#2b2b2b", cursor="fleur", width=400, height=300,
        )
        self._original_canvas.pack(fill=tk.BOTH, expand=True)

        compressed_panel = ttk.LabelFrame(preview_frame, text=LABEL_COMPRESSED, padding=5)
        compressed_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._compressed_canvas = ZoomableImageCanvas(
            compressed_panel, background="#2b2b2b", cursor="fleur", width=400, height=300,
        )
        self._compressed_canvas.pack(fill=tk.BOTH, expand=True)

        self._original_canvas.sync_with(self._compressed_canvas)

    # ------------------------------------------------------------------
    # Grafici di analisi
    # ------------------------------------------------------------------

    def _remove_charts(self) -> None:
        """Distrugge tutti i widget e le figure matplotlib dei grafici precedenti."""
        self._linked_axes_groups.clear()
        for _frame, mpl_canvas in self._chart_resources:
            mpl_canvas.get_tk_widget().destroy()
            plt.close(mpl_canvas.figure)
        self._chart_resources.clear()
        if self._charts_outer_frame is not None:
            self._charts_outer_frame.destroy()
            self._charts_outer_frame = None

    def _show_charts(
        self,
        original: np.ndarray,
        compressed: np.ndarray,
        block_size: int,
        threshold_d: int,
    ) -> None:
        """
        Crea 4 pannelli grafici interattivi sotto alle anteprime immagine.

        Griglia 2×2:
          [0,0] Istogramma originale    [0,1] Istogramma compressa
          [1,0] Frequenze DCT originali [1,1] Frequenze DCT troncate
        """
        self._remove_charts()

        freq_full, freq_trunc = build_dct_frequency_map(original, block_size, threshold_d)
        vmax_full = freq_full.max() if freq_full.max() > 0 else 1.0
        kept  = int(np.sum(freq_trunc > 0))
        total = block_size * block_size
        pct   = 100.0 * kept / total if total > 0 else 0.0

        # ---- Contenitore esterno ----------------------------------------
        self._charts_outer_frame = ttk.LabelFrame(
            self._inner_frame,
            text=(
                "Analisi \u2014 istogrammi e frequenze DCT  "
                "[ \U0001f517 zoom/pan linkati per coppia ]  "
                "(toolbar: \U0001f50d zoom rett. \u00b7 \u270b pan \u00b7 \U0001f3e0 reset \u00b7 \U0001f4be salva)"
            ),
            padding=8,
        )
        self._charts_outer_frame.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10)
        )
        for col in range(2):
            self._charts_outer_frame.columnconfigure(col, weight=1)
        for row in range(2):
            self._charts_outer_frame.rowconfigure(row, weight=1)

        # ---- Funzioni di disegno dei 4 grafici --------------------------

        def draw_hist_original(fig, ax):
            ax.hist(original.ravel(), bins=256, range=(0, 255),
                    color="#4C72B0", alpha=0.85, edgecolor="none")
            ax.set_title("Istogramma \u2013 Immagine originale",  fontsize=10, fontweight="bold")
            ax.set_xlabel("Livello di grigio", fontsize=8)
            ax.set_ylabel("Conteggio pixel",   fontsize=8)
            ax.set_xlim(0, 255)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        def draw_hist_compressed(fig, ax):
            ax.hist(compressed.ravel(), bins=256, range=(0, 255),
                    color="#DD8452", alpha=0.85, edgecolor="none")
            ax.set_title("Istogramma \u2013 Immagine compressa", fontsize=10, fontweight="bold")
            ax.set_xlabel("Livello di grigio", fontsize=8)
            ax.set_ylabel("Conteggio pixel",   fontsize=8)
            ax.set_xlim(0, 255)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        def draw_dct_full(fig, ax):
            im = ax.imshow(
                np.log1p(freq_full), cmap="inferno", aspect="auto",
                interpolation="nearest", vmin=0,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + |coeff.|)")
            ax.set_title(
                f"Frequenze DCT originali\n(media |coeff.|, F={block_size})",
                fontsize=9, fontweight="bold",
            )
            ax.set_xlabel("Frequenza orizzontale (j)", fontsize=8)
            ax.set_ylabel("Frequenza verticale (i)",   fontsize=8)
            ax.tick_params(labelsize=7)
            # Linea diagonale di taglio
            if 0 < threshold_d <= 2 * block_size - 2:
                d, F = threshold_d, block_size
                x0 = min(d, F - 1);  y0 = max(d - (F - 1), 0)
                x1 = max(d - (F - 1), 0); y1 = min(d, F - 1)
                ax.plot(
                    [x0 - 0.5, x1 - 0.5], [y0 - 0.5, y1 - 0.5],
                    color="cyan", linewidth=1.5, linestyle="--",
                    label=f"soglia d={d}",
                )
                ax.legend(fontsize=7, loc="lower right")

        def draw_dct_trunc(fig, ax):
            im = ax.imshow(
                np.log1p(freq_trunc), cmap="inferno", aspect="auto",
                interpolation="nearest", vmin=0, vmax=np.log1p(vmax_full),
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + |coeff.|)")
            ax.set_title(
                f"Frequenze DCT troncate (d={threshold_d})\n"
                f"Coefficienti mantenuti: {kept}/{total} ({pct:.1f}%)",
                fontsize=9, fontweight="bold",
            )
            ax.set_xlabel("Frequenza orizzontale (j)", fontsize=8)
            ax.set_ylabel("Frequenza verticale (i)",   fontsize=8)
            ax.tick_params(labelsize=7)

        chart_specs = [
            ("Istogramma originale  \U0001f517",    draw_hist_original,   0, 0),
            ("Istogramma compressa  \U0001f517",    draw_hist_compressed, 0, 1),
            ("Frequenze DCT originali  \U0001f517", draw_dct_full,        1, 0),
            ("Frequenze DCT troncate  \U0001f517",  draw_dct_trunc,       1, 1),
        ]

        # ---- Creazione pannelli -----------------------------------------
        collected_axes: list[plt.Axes] = []
        for title, draw_fn, row, col in chart_specs:
            panel, mpl_canvas, ax = make_chart_panel(
                self._charts_outer_frame,
                title=title,
                draw_fn=draw_fn,
                fig_width=5.5,
                fig_height=3.6,
            )
            panel.grid(row=row, column=col, sticky=tk.NSEW, padx=6, pady=6)
            self._chart_resources.append((panel, mpl_canvas))
            collected_axes.append(ax)

        # ---- Collegamento zoom/pan a coppie -----------------------------
        # [0] istogramma originale  [1] istogramma compressa
        # [2] mappa DCT originale   [3] mappa DCT troncata
        self._linked_axes_groups = [
            LinkedAxesGroup([collected_axes[0], collected_axes[1]], sync_x=True, sync_y=True),
            LinkedAxesGroup([collected_axes[2], collected_axes[3]], sync_x=True, sync_y=True),
        ]

        # ---- Aggiorna scroll e salta ai grafici -------------------------
        self._inner_frame.update_idletasks()
        self._main_canvas.configure(scrollregion=self._main_canvas.bbox("all"))
        self._main_canvas.yview_moveto(1.0)

    # ------------------------------------------------------------------
    # Gestori degli eventi utente
    # ------------------------------------------------------------------

    def _on_select_image_clicked(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Seleziona un'immagine BMP in toni di grigio",
            filetypes=FILE_TYPES,
        )
        if not file_path:
            return
        try:
            self._original_grayscale_array = load_grayscale_bmp(file_path)
        except Exception as error:
            messagebox.showerror("Errore caricamento", str(error))
            return

        self._selected_image_path = file_path
        self._file_path_label.config(text=file_path, foreground="black")

        pil_img = numpy_array_to_pil_image(self._original_grayscale_array)
        self._original_canvas.set_image(pil_img)
        self._compressed_canvas.clear()
        self._original_canvas._reset_fit()
        self._remove_charts()

    def _on_compress_clicked(self) -> None:
        if self._original_grayscale_array is None:
            messagebox.showwarning("Nessuna immagine", "Seleziona prima un'immagine BMP.")
            return

        try:
            block_size  = self._block_size_var.get()
            threshold_d = self._threshold_d_var.get()
        except tk.TclError:
            messagebox.showerror("Parametri non validi", "F e d devono essere numeri interi.")
            return

        error_message = validate_compression_parameters(block_size, threshold_d)
        if error_message:
            messagebox.showerror("Parametri non validi", error_message)
            return

        image_height, image_width = self._original_grayscale_array.shape
        if image_height < block_size or image_width < block_size:
            messagebox.showerror(
                "Immagine troppo piccola",
                f"L'immagine ({image_width}\u00d7{image_height}) "
                f"\u00e8 pi\u00f9 piccola del blocco F={block_size}.",
            )
            return

        try:
            compressed_array = compress_image(
                grayscale_image=self._original_grayscale_array,
                block_size=block_size,
                threshold_d=threshold_d,
            )
        except Exception as error:
            messagebox.showerror("Errore compressione", str(error))
            return

        self._original_canvas._reset_fit()
        self._compressed_canvas.set_image(numpy_array_to_pil_image(compressed_array))
        self._show_charts(self._original_grayscale_array, compressed_array, block_size, threshold_d)
