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

I grafici sono renderizzati come canvas Tkinter nativi (ZoomableChartCanvas):
matplotlib viene usato solo come motore di disegno in memoria (backend Agg),
senza embedding di FigureCanvasTkAgg né NavigationToolbar2Tk.

Compatibilità macOS
-------------------
Tutte le operazioni CPU-intensive (compress_image, build_dct_frequency_map,
render matplotlib) girano in un thread secondario via _run_in_thread().
Il risultato viene restituito al main thread tramite root.after(0, callback),
evitando freeze e spinning beachball. Nessun widget Tkinter viene mai
toccato dal thread secondario.
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib.pyplot as plt

from image_utils import load_grayscale_bmp, numpy_array_to_pil_image
from dct_compression import compress_image
from dct_analysis import build_dct_frequency_map
from constants import (
    WINDOW_TITLE,
    LABEL_SELECT_IMAGE, LABEL_ORIGINAL, LABEL_COMPRESSED,
    BUTTON_SELECT_TEXT, BUTTON_COMPRESS_TEXT,
    PARAM_F_LABEL, PARAM_D_LABEL,
    PARAM_F_MIN, PARAM_F_MAX, PARAM_D_MIN,
    FILE_TYPES,
)
from widgets import ZoomableImageCanvas, LinkedChartGroup, make_chart_panel
from gui import validate_compression_parameters


class DctCompressionApp:
    """
    Finestra principale dell'applicazione di compressione DCT2.
    Il minsize e la geometria iniziale sono gestiti da gui.py al momento
    della creazione della root window, adattandosi allo schermo disponibile.
    """

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title(WINDOW_TITLE)

        self._selected_image_path: str | None             = None
        self._original_grayscale_array: np.ndarray | None = None

        # Canvas dei grafici (ZoomableChartCanvas)
        self._chart_canvases: list = []
        # Gruppi di canvas linkati (ricreati a ogni compressione)
        self._linked_chart_groups: list[LinkedChartGroup] = []
        # Frame contenitore dei 4 pannelli grafici
        self._charts_outer_frame: ttk.Frame | None = None

        # Flag per evitare compressioni concorrenti
        self._compression_running: bool = False

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

        # Scroll della finestra principale: Windows/macOS e Linux
        self._main_canvas.bind("<MouseWheel>", self._on_main_scroll_mousewheel)
        self._main_canvas.bind("<Button-4>",   self._on_main_scroll_up_linux)
        self._main_canvas.bind("<Button-5>",   self._on_main_scroll_down_linux)

        self._build_control_panel()
        self._build_image_preview_area()

    def _on_inner_frame_configure(self, _event: tk.Event) -> None:
        self._main_canvas.configure(scrollregion=self._main_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._main_canvas.itemconfig(self._inner_window_id, width=event.width)

    def _on_main_scroll_mousewheel(self, event: tk.Event) -> None:
        """Scroll verticale: Windows delta multiplo di 120, macOS delta in unità."""
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

        self._compress_button = ttk.Button(
            params_row, text=BUTTON_COMPRESS_TEXT,
            command=self._on_compress_clicked,
        )
        self._compress_button.pack(side=tk.LEFT)

        # Label di stato (mostrata durante la compressione)
        self._status_label = ttk.Label(params_row, text="", foreground="gray")
        self._status_label.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(
            control_frame,
            text="\U0001f50d Rotella mouse: zoom  \u2022  \U0001f5b1 Trascina: pan  "
                 "\u2022  \u2b1b Doppio clic: reimposta vista",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(4, 0))

    # Dimensione massima (in pixel) del lato più lungo del canvas di preview
    _PREVIEW_MAX_PX: int = 520
    # Dimensione di default quando non è ancora caricata alcuna immagine
    _PREVIEW_DEFAULT_PX: int = 400

    def _build_image_preview_area(self) -> None:
        """Crea i due canvas affiancati per le anteprime originale / compressa."""
        self._preview_frame = ttk.Frame(self._inner_frame)
        self._preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self._original_panel = ttk.LabelFrame(self._preview_frame, text=LABEL_ORIGINAL, padding=5)
        self._original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self._original_canvas = ZoomableImageCanvas(
            self._original_panel, background="#2b2b2b", cursor="fleur",
            width=self._PREVIEW_DEFAULT_PX, height=self._PREVIEW_DEFAULT_PX,
        )
        self._original_canvas.pack(fill=tk.BOTH, expand=True)

        self._compressed_panel = ttk.LabelFrame(self._preview_frame, text=LABEL_COMPRESSED, padding=5)
        self._compressed_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self._compressed_canvas = ZoomableImageCanvas(
            self._compressed_panel, background="#2b2b2b", cursor="fleur",
            width=self._PREVIEW_DEFAULT_PX, height=self._PREVIEW_DEFAULT_PX,
        )
        self._compressed_canvas.pack(fill=tk.BOTH, expand=True)

        self._original_canvas.sync_with(self._compressed_canvas)

        # Propaga lo scroll al canvas principale anche quando il mouse è sui canvas immagine
        for canvas in (self._original_canvas, self._compressed_canvas):  # type: ignore[assignment]
            canvas.bind("<MouseWheel>", self._on_child_scroll_mousewheel, add=True)
            canvas.bind("<Button-4>",   self._on_main_scroll_up_linux,    add=True)
            canvas.bind("<Button-5>",   self._on_main_scroll_down_linux,  add=True)

    # ------------------------------------------------------------------
    # Ridimensionamento canvas di preview in base all'aspect ratio dell'immagine
    # ------------------------------------------------------------------

    def _resize_preview_canvases(self, img_w: int, img_h: int) -> None:
        """
        Ridimensiona i due canvas di preview in modo che rispettino
        l'aspect ratio dell'immagine, limitando il lato più lungo a
        _PREVIEW_MAX_PX pixel.
        """
        max_px = self._PREVIEW_MAX_PX
        ratio  = img_w / img_h if img_h > 0 else 1.0
        if ratio >= 1.0:
            # Immagine più larga che alta (o quadrata)
            canvas_w = max_px
            canvas_h = max(1, int(round(max_px / ratio)))
        else:
            # Immagine più alta che larga
            canvas_h = max_px
            canvas_w = max(1, int(round(max_px * ratio)))

        for canvas in (self._original_canvas, self._compressed_canvas):
            canvas.config(width=canvas_w, height=canvas_h)

    # ------------------------------------------------------------------
    # Scroll propagation da widget figli al canvas principale
    # ------------------------------------------------------------------

    def _on_child_scroll_mousewheel(self, event: tk.Event) -> None:
        """
        Propaga lo scroll verticale al canvas principale quando il mouse è su un
        widget figlio (canvas immagine o grafico). Su macOS Tkinter non propaga
        automaticamente gli eventi di scroll verso i widget padre.
        """
        self._main_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _register_child_scroll(self, widget: "ZoomableChartCanvas") -> None:
        """
        Registra i binding di scroll sul widget Tk interno del canvas matplotlib
        in modo che lo scroll raggiunga il canvas principale.
        Il binding usa add=True per non sovrascrivere gli handler di zoom.
        """
        inner = widget._tk_widget
        inner.bind("<MouseWheel>", self._on_child_scroll_mousewheel, add=True)
        inner.bind("<Button-4>",   self._on_main_scroll_up_linux,    add=True)
        inner.bind("<Button-5>",   self._on_main_scroll_down_linux,  add=True)

    # ------------------------------------------------------------------
    # Threading: esecuzione asincrona senza freeze del main thread
    # ------------------------------------------------------------------

    def _run_in_thread(self, worker, on_done, on_error=None) -> None:
        """
        Esegue `worker()` in un thread secondario (daemon).
        Al termine chiama `on_done(result)` nel main thread tramite after(0).
        In caso di eccezione chiama `on_error(exc)` nel main thread.

        Nessun widget Tkinter viene toccato dal thread secondario.
        """
        def _thread_body():
            try:
                result = worker()
                self._root.after(0, lambda: on_done(result))
            except Exception as exc:
                if on_error is not None:
                    self._root.after(0, lambda: on_error(exc))
                else:
                    self._root.after(0, lambda: messagebox.showerror(
                        "Errore interno", str(exc)
                    ))

        t = threading.Thread(target=_thread_body, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Stato UI durante la compressione
    # ------------------------------------------------------------------

    def _set_busy(self, message: str) -> None:
        """Disabilita il bottone Comprimi e mostra un messaggio di stato."""
        self._compression_running = True
        self._compress_button.config(state=tk.DISABLED)
        self._status_label.config(text=message)
        self._root.config(cursor="watch")

    def _set_idle(self) -> None:
        """Riabilita il bottone Comprimi e rimuove il messaggio di stato."""
        self._compression_running = False
        self._compress_button.config(state=tk.NORMAL)
        self._status_label.config(text="")
        self._root.config(cursor="")

    # ------------------------------------------------------------------
    # Grafici di analisi
    # ------------------------------------------------------------------

    def _remove_charts(self) -> None:
        """Distrugge tutti i widget dei grafici precedenti."""
        self._linked_chart_groups.clear()
        self._chart_canvases.clear()
        if self._charts_outer_frame is not None:
            self._charts_outer_frame.destroy()
            self._charts_outer_frame = None

    def _show_charts(
        self,
        original:    np.ndarray,
        compressed:  np.ndarray,
        block_size:  int,
        threshold_d: int,
        freq_full:   np.ndarray,
        freq_trunc:  np.ndarray,
    ) -> None:
        """
        Crea 4 pannelli grafici interattivi sotto alle anteprime immagine.

        Griglia 2×2:
          [0,0] Istogramma originale    [0,1] Istogramma compressa
          [1,0] Frequenze DCT originali [1,1] Frequenze DCT troncate

        freq_full e freq_trunc sono già calcolate (nel thread) e passate qui.
        """
        self._remove_charts()

        vmax_full = freq_full.max() if freq_full.max() > 0 else 1.0
        kept  = int(np.sum(freq_trunc > 0))
        total = block_size * block_size
        pct   = 100.0 * kept / total if total > 0 else 0.0

        # ---- Contenitore esterno ----------------------------------------
        self._charts_outer_frame = ttk.LabelFrame(
            self._inner_frame,
            text=(
                "Analisi \u2014 istogrammi e frequenze DCT  "
                "[ \U0001f517 zoom/pan linkati per coppia ]"
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
        collected_canvases = []
        for title, draw_fn, row, col in chart_specs:
            panel, chart_canvas = make_chart_panel(
                self._charts_outer_frame,
                title=title,
                draw_fn=draw_fn,
                fig_width=5.5,
                fig_height=3.6,
            )
            panel.grid(row=row, column=col, sticky=tk.NSEW, padx=6, pady=6)
            self._chart_canvases.append(chart_canvas)
            collected_canvases.append(chart_canvas)
            # Propaga lo scroll al canvas principale anche da dentro i grafici
            self._register_child_scroll(chart_canvas)

        # ---- Collegamento zoom/pan a coppie -----------------------------
        self._linked_chart_groups = [
            LinkedChartGroup([collected_canvases[0], collected_canvases[1]]),
            LinkedChartGroup([collected_canvases[2], collected_canvases[3]]),
        ]

        # ---- Aggiorna la scrollregion senza spostare la vista -------------------------
        self._inner_frame.update_idletasks()
        self._main_canvas.configure(scrollregion=self._main_canvas.bbox("all"))

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
        # Ridimensiona i canvas in base all'aspect ratio dell'immagine caricata
        img_h, img_w = self._original_grayscale_array.shape
        self._resize_preview_canvases(img_w, img_h)
        self._original_canvas.set_image(pil_img)
        self._compressed_canvas.clear()
        self._original_canvas._reset_fit()
        self._remove_charts()

    def _on_compress_clicked(self) -> None:
        """
        Avvia la compressione in un thread secondario per non bloccare il main thread.
        Il flusso è:
          1. Validazione parametri (main thread)
          2. compress_image + build_dct_frequency_map (thread secondario)
          3. Aggiornamento UI con risultati (main thread via after(0, ...))
        """
        if self._compression_running:
            return  # compressione già in corso

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

        # Cattura le variabili necessarie al thread (evita race condition su self)
        source_array = self._original_grayscale_array
        self._set_busy("Compressione in corso\u2026")

        def _worker():
            """Eseguito nel thread secondario: solo numpy/scipy, niente Tkinter."""
            compressed = compress_image(
                grayscale_image=source_array,
                block_size=block_size,
                threshold_d=threshold_d,
            )
            freq_full, freq_trunc = build_dct_frequency_map(
                source_array, block_size, threshold_d
            )
            return compressed, freq_full, freq_trunc

        def _on_done(result):
            """Chiamato nel main thread al termine del calcolo."""
            compressed_array, freq_full, freq_trunc = result
            self._set_idle()
            self._original_canvas._reset_fit()
            self._compressed_canvas.set_image(numpy_array_to_pil_image(compressed_array))
            self._show_charts(
                source_array, compressed_array,
                block_size, threshold_d,
                freq_full, freq_trunc,
            )

        def _on_error(exc):
            """Chiamato nel main thread in caso di eccezione nel worker."""
            self._set_idle()
            messagebox.showerror("Errore compressione", str(exc))

        self._run_in_thread(_worker, _on_done, _on_error)
