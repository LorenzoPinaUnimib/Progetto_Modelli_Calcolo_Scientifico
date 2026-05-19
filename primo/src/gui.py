import sys
import os
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Palette & stile
# ---------------------------------------------------------------------------
BG        = "#0f1117"
PANEL     = "#181c27"
BORDER    = "#2a2f42"
ACCENT    = "#4f8ef7"
ACCENT2   = "#a78bfa"
SUCCESS   = "#34d399"
WARNING   = "#fbbf24"
DANGER    = "#f87171"
TEXT      = "#e8eaf0"
SUBTEXT   = "#7c84a0"

FONT_MONO  = ("Consolas", 10)
FONT_LABEL = ("Segoe UI", 10)
FONT_TITLE = ("Segoe UI Semibold", 13)
FONT_SMALL = ("Segoe UI", 9)

METHOD_COLORS = {
    "Jacobi":             "#4f8ef7",
    "Gauss-Seidel":       "#a78bfa",
    "Gradiente":          "#34d399",
    "Gradiente Coniugato":"#fbbf24",
}

# Funzioni di supporto
def _add_project_root_to_path():
    """Aggiunge la cartella contenente gui.py al sys.path."""
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_solvers():
    """Importa i solver dal progetto; ritorna un dict name→solve_fn."""
    _add_project_root_to_path()
    try:

        from utils.matrix_io import load_mtx
        from solvers import jacobi, gauss_seidel, gradient, cg

        return load_mtx, {
            "Jacobi":              jacobi.solve,
            "Gauss-Seidel":        gauss_seidel.solve,
            "Gradiente":           gradient.solve,
            "Gradiente Coniugato": cg.solve,
        }
    except ImportError as e:
        raise ImportError(
            f"Impossibile importare i moduli del progetto: {e}\n"
            "Assicurati di avere la cartella utils/ e solvers/ nella stessa directory di gui.py."
        )

def _compute_relative_error(x_true, x_comp):
    if x_comp is None:
        return float("nan")
    num = np.linalg.norm(x_true - x_comp)
    den = np.linalg.norm(x_true)
    return num / den if den else float("nan")

# Thread di esecuzione
class SolverThread(threading.Thread):
    """Esegue tutti i solver in background e notifica la GUI al termine."""

    # Inizializzazione
    def __init__(self, mtx_path, tol, on_progress, on_done, on_error):
        super().__init__(daemon=True)
        self.mtx_path    = mtx_path
        self.tol         = tol
        self.on_progress = on_progress
        self.on_done     = on_done
        self.on_error    = on_error

    # Esecuzione
    def run(self):
        import tracemalloc

        try:
            # Carico i vari risolutori
            load_mtx, methods = _load_solvers()

            # Carico matrice e faccio un analisi preliminare
            self.on_progress("Caricamento matrice…")
            A = load_mtx(self.mtx_path)
            n = A.shape[0]
            x_true = np.ones(n)
            b = A @ x_true
            self.on_progress(f"Matrice caricata: {n}×{n} | {A.nnz} elementi non-zero")

            # Esecuzione dei risolutori
            results = {}
            for name, solver in methods.items():
                self.on_progress(f"Esecuzione {name}…")
                tracemalloc.start()
                try:
                    x_sol, iters, elapsed = solver(A, b, self.tol)
                except Exception as exc:
                    tracemalloc.stop()
                    results[name] = {
                        "err": float("nan"), "iters": 0,
                        "time": 0, "peak_mem_mb": 0,
                        "failed": True, "msg": str(exc),
                    }
                    continue

                # Calcolo memoria utilizzata
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Calcolo errori
                err = _compute_relative_error(x_true, x_sol)
                results[name] = {
                    "err":          err,
                    "iters":        iters,
                    "time":         elapsed,
                    "peak_mem_mb":  peak_mem / 1e6,
                    "failed":       x_sol is None,
                }

            self.on_done(results)

        except Exception as exc:
            self.on_error(traceback.format_exc())

# Finestra principale
class App(tk.Tk):
    # Inizializzazione e applicazione stili
    def __init__(self):
        super().__init__()
        self.title("Primo assignment - analisi comparativa")
        self.geometry("1280x800")
        self.minsize(900, 620)
        self.configure(bg=BG)
        self._apply_style()

        self._mtx_path  = tk.StringVar(value="")
        self._tol_var   = tk.StringVar(value="1e-4")
        self._results   = None

        self._build_ui()
        self._configure_dnd()

    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",
                     background=BG, foreground=TEXT,
                     fieldbackground=PANEL, bordercolor=BORDER,
                     troughcolor=PANEL, insertcolor=TEXT,
                     selectbackground=ACCENT, selectforeground=BG,
                     font=FONT_LABEL)
        s.configure("TFrame",  background=BG)
        s.configure("Panel.TFrame", background=PANEL, relief="flat")
        s.configure("TLabel",  background=BG,    foreground=TEXT)
        s.configure("Sub.TLabel", background=PANEL, foreground=SUBTEXT, font=FONT_SMALL)
        s.configure("TEntry",  fieldbackground=PANEL, foreground=TEXT,
                     insertcolor=TEXT, relief="flat", padding=4)
        s.configure("Accent.TButton",
                     background=ACCENT, foreground=BG,
                     font=("Segoe UI Semibold", 10),
                     relief="flat", padding=(14, 6))
        s.map("Accent.TButton",
              background=[("active", "#6ea6ff"), ("disabled", BORDER)],
              foreground=[("disabled", SUBTEXT)])
        s.configure("Run.TButton",
                     background=SUCCESS, foreground=BG,
                     font=("Segoe UI Semibold", 11),
                     relief="flat", padding=(20, 8))
        s.map("Run.TButton",
              background=[("active", "#52e8b0"), ("disabled", BORDER)],
              foreground=[("disabled", SUBTEXT)])
        s.configure("TProgressbar", troughcolor=PANEL,
                     background=ACCENT, thickness=4)
        s.configure("Treeview",
                     background=PANEL, foreground=TEXT,
                     fieldbackground=PANEL, rowheight=28,
                     font=FONT_LABEL)
        s.configure("Treeview.Heading",
                     background=BORDER, foreground=SUBTEXT,
                     font=("Segoe UI Semibold", 9), relief="flat")
        s.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", BG)])
        s.configure("TNotebook",        background=BG, bordercolor=BORDER)
        s.configure("TNotebook.Tab",
                     background=PANEL, foreground=SUBTEXT,
                     padding=(14, 6), font=FONT_LABEL)
        s.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", TEXT)])

    # Costruzione della finestra effettiva
    def _build_ui(self):
        # Colonna sinistra (controlli)
        left = ttk.Frame(self, style="Panel.TFrame", width=260)
        left.pack(side="left", fill="y", padx=(12, 0), pady=12)
        left.pack_propagate(False)
        self._build_left_panel(left)

        # Colonna destra (output)
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        self._build_right_panel(right)

    # Pannello sinistro
    def _build_left_panel(self, parent):
        pad = dict(padx=16, pady=6)

        # Titolo
        tk.Label(parent, text="Primo assignment", font=("Segoe UI Semibold", 14),
                 fg=TEXT, bg=PANEL).pack(anchor="w", padx=16, pady=(18, 2))
        tk.Label(parent, text="Analisi comparativa metodi", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", padx=16, pady=(0, 14))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=16, pady=4)

        # Selezione file
        tk.Label(parent, text="File matrice (.mtx)", font=FONT_LABEL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", **pad)

        # Drop-zone
        self._drop_frame = tk.Frame(parent, bg=BORDER, relief="flat",
                                     cursor="hand2", height=72)
        self._drop_frame.pack(fill="x", padx=16, pady=(0, 4))
        self._drop_frame.pack_propagate(False)

        self._drop_label = tk.Label(
            self._drop_frame,
            text="↓  In attesa di file .mtx",
            font=FONT_SMALL, fg=SUBTEXT, bg=BORDER,
            wraplength=200, justify="center",
        )
        self._drop_label.pack(expand=True)
        self._drop_frame.bind("<Button-1>", lambda _: self._browse_file())
        self._drop_label.bind("<Button-1>", lambda _: self._browse_file())

        # Nome file selezionato
        self._file_label = tk.Label(parent, text="Nessun file selezionato",
                                     font=FONT_SMALL, fg=SUBTEXT, bg=PANEL,
                                     wraplength=220, justify="left")
        self._file_label.pack(anchor="w", padx=16, pady=(0, 8))

        ttk.Button(parent, text="Sfoglia…", style="Accent.TButton",
                   command=self._browse_file).pack(fill="x", padx=16, pady=(0, 10))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=16, pady=4)

        # Tolleranza
        tk.Label(parent, text="Tolleranza", font=FONT_LABEL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", **pad)

        tol_frame = ttk.Frame(parent, style="Panel.TFrame")
        tol_frame.pack(fill="x", padx=16, pady=(0, 4))

        self._tol_entry = ttk.Entry(tol_frame, textvariable=self._tol_var, width=14)
        self._tol_entry.pack(side="left", fill="x", expand=True)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=16, pady=10)

        # Pulsante esecuzione
        self._run_btn = ttk.Button(parent, text="▶  Esegui Solver",
                                    style="Run.TButton",
                                    command=self._run_solvers)
        self._run_btn.pack(fill="x", padx=16, pady=(0, 8))

        # Progress bar
        self._progress = ttk.Progressbar(parent, mode="indeterminate")
        self._progress.pack(fill="x", padx=16, pady=(0, 4))

        # Stato
        self._status_var = tk.StringVar(value="In attesa…")
        tk.Label(parent, textvariable=self._status_var,
                 font=FONT_SMALL, fg=SUBTEXT, bg=PANEL,
                 wraplength=220, justify="left").pack(anchor="w", padx=16)

        # Spaziatore inferiore
        ttk.Frame(parent, style="Panel.TFrame").pack(expand=True, fill="both")

        # Footer
        tk.Label(parent, text="Progetto a cura di Rancati Simone,\n Pina Lorenzo e Piovanelli Michele",
                 font=FONT_SMALL, fg=BORDER, bg=PANEL).pack(pady=(0, 12))

    # Pannello destro
    def _build_right_panel(self, parent):
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill="both", expand=True)

        # Tab 1: Dashboard grafici
        self._tab_charts = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_charts, text="  Dashboard  ")
        self._build_charts_tab(self._tab_charts)

        # Tab 2: Tabella numerica
        self._tab_table = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_table, text="  Tabella risultati  ")
        self._build_table_tab(self._tab_table)

        # Tab 3: Log
        self._tab_log = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_log, text="  Log  ")
        self._build_log_tab(self._tab_log)

    # Tab grafici
    def _build_charts_tab(self, parent):
        self._fig = plt.Figure(figsize=(10, 7), facecolor=BG)
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty_charts()

    def _draw_empty_charts(self):
        self._fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self._fig,
                               hspace=0.45, wspace=0.35,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.09)
        axes = [self._fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
        titles = ["Tempo di esecuzione (s)", "Numero di iterazioni",
                  "Errore relativo (scala log)", "Memoria di picco (MB)"]
        for ax, t in zip(axes, titles):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=SUBTEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.set_title(t, color=SUBTEXT, fontsize=9, pad=8)
            ax.text(0.5, 0.5, "Nessun dato", ha="center", va="center",
                    transform=ax.transAxes, color=BORDER, fontsize=11)
        self._canvas.draw()

    def _draw_charts(self, results):
        self._fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self._fig,
                               hspace=0.50, wspace=0.38,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.11)

        names  = list(results.keys())
        colors = [METHOD_COLORS.get(n, ACCENT) for n in names]
        times  = [r["time"]         for r in results.values()]
        iters  = [r["iters"]        for r in results.values()]
        errors = [r["err"]          for r in results.values()]
        mems   = [r["peak_mem_mb"]  for r in results.values()]

        def _style_ax(ax, title):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=SUBTEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.set_title(title, color=TEXT, fontsize=9, pad=8)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=20, ha="right",
                                fontsize=8, color=SUBTEXT)

        def _bar(ax, values, title, ylabel, fmt="{:.4f}"):
            bars = ax.bar(range(len(names)), values, color=colors,
                          width=0.55, zorder=3, edgecolor="none")
            ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=8)
            ax.yaxis.label.set_color(SUBTEXT)
            ax.tick_params(axis="y", colors=SUBTEXT)
            ax.grid(axis="y", color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
            _style_ax(ax, title)
            for bar, val in zip(bars, values):
                if not (np.isnan(val) or val == 0):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            fmt.format(val),
                            ha="center", va="bottom",
                            color=TEXT, fontsize=7)

        # 1. Tempo
        ax0 = self._fig.add_subplot(gs[0, 0])
        _bar(ax0, times, "Tempo di esecuzione (s)", "Secondi", "{:.3f}s")

        # 2. Iterazioni
        ax1 = self._fig.add_subplot(gs[0, 1])
        _bar(ax1, iters, "Numero di iterazioni", "Iterazioni", "{:.0f}")

        # 3. Errore relativo — log scale
        ax2 = self._fig.add_subplot(gs[1, 0])
        valid_errors = [max(e, 1e-16) if not np.isnan(e) else 1e-16 for e in errors]
        bars3 = ax2.bar(range(len(names)), valid_errors, color=colors,
                        width=0.55, zorder=3, edgecolor="none")
        ax2.set_yscale("log")
        ax2.set_ylabel("Errore relativo", color=SUBTEXT, fontsize=8)
        ax2.yaxis.label.set_color(SUBTEXT)
        ax2.tick_params(axis="y", colors=SUBTEXT)
        ax2.grid(axis="y", color=BORDER, linestyle="--", linewidth=0.5, zorder=0)
        _style_ax(ax2, "Errore relativo (scala log)")
        for bar, val in zip(bars3, errors):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height(),
                         f"{val:.1e}",
                         ha="center", va="bottom",
                         color=TEXT, fontsize=7)

        # 4. Memoria
        ax3 = self._fig.add_subplot(gs[1, 1])
        _bar(ax3, mems, "Memoria di picco (MB)", "MB", "{:.2f}")

        self._canvas.draw()

    # Tab tabella
    def _build_table_tab(self, parent):
        cols = ("Metodo", "Iterazioni", "Tempo (s)", "Errore Rel.", "Memoria (MB)", "Stato")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings",
                                   selectmode="browse")
        widths = (160, 90, 90, 110, 110, 80)
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True, padx=8, pady=8)

        self._tree.tag_configure("ok",     foreground=SUCCESS)
        self._tree.tag_configure("failed", foreground=DANGER)

    def _populate_table(self, results):
        for row in self._tree.get_children():
            self._tree.delete(row)
        for name, r in results.items():
            tag = "failed" if r.get("failed") else "ok"
            self._tree.insert("", "end", values=(
                name,
                r["iters"],
                f"{r['time']:.4f}",
                f"{r['err']:.2e}" if not np.isnan(r["err"]) else "—",
                f"{r['peak_mem_mb']:.3f}",
                "✗ Fallito" if r.get("failed") else "✓ OK",
            ), tags=(tag,))

    # Tab log
    def _build_log_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        self._log_text = tk.Text(frame, bg=PANEL, fg=TEXT,
                                  font=FONT_MONO, relief="flat",
                                  insertbackground=TEXT,
                                  wrap="word", state="disabled")
        vsb = ttk.Scrollbar(frame, orient="vertical",
                             command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True)

        # Redirect dal log CLI a log GUI
        sys.stdout = _TextRedirector(self._log_text, self)

    def _log(self, msg):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # Drag-and-drop - dà errori strani, rimarrà senza
    def _configure_dnd(self):
        try:
            self._drop_frame.dnd_bind('<<Drop>>', self._on_dnd_drop)
            self._drop_label.configure(text="↓  In attesa di file .mtx")
        except (ImportError, AttributeError):
            # tkinterdnd2 non disponibile: solo click per browse
            pass

    def _on_dnd_drop(self, event):
        path = event.data.strip().strip("{}")  # Windows può aggiungere {}
        if path.lower().endswith(".mtx"):
            self._set_file(path)
        else:
            messagebox.showerror("Formato non valido",
                                  "Seleziona un file .mtx valido.")

    # Azioni
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Seleziona matrice .mtx",
            filetypes=[("Matrix Market", "*.mtx"), ("Tutti i file", "*.*")],
        )
        if path:
            self._set_file(path)

    def _set_file(self, path):
        self._mtx_path.set(path)
        fname = os.path.basename(path)
        self._file_label.configure(text=fname, fg=TEXT)
        self._drop_label.configure(text=f"✓  {fname}", fg=SUCCESS)
        self._drop_frame.configure(bg="#1a2e1e")
        self._status_var.set("File pronto. Clicca Esegui.")

    def _run_solvers(self):
        path = self._mtx_path.get()
        if not path:
            messagebox.showwarning("Nessun file", "Seleziona prima un file .mtx.")
            return
        try:
            tol = float(self._tol_var.get())
        except ValueError:
            messagebox.showerror("Tolleranza non valida",
                                  "Inserisci un numero valido (es. 1e-6).")
            return

        # Blocca UI
        self._run_btn.configure(state="disabled")
        self._progress.start(12)
        self._status_var.set("Esecuzione in corso…")
        self._draw_empty_charts()
        self._notebook.select(self._tab_log)

        def on_progress(msg):
            self.after(0, lambda: self._status_var.set(msg))
            self.after(0, lambda: self._log(msg))

        def on_done(results):
            self.after(0, lambda: self._finish(results))

        def on_error(tb):
            self.after(0, lambda: self._handle_error(tb))

        SolverThread(path, tol, on_progress, on_done, on_error).start()

    def _finish(self, results):
        self._progress.stop()
        self._run_btn.configure(state="normal")
        self._results = results

        n_ok = sum(1 for r in results.values() if not r.get("failed"))
        self._status_var.set(f"Completato — {n_ok}/{len(results)} solver OK")

        self._draw_charts(results)
        self._populate_table(results)
        self._notebook.select(self._tab_charts)

        self._log("\n── Riepilogo ─────────────────────────────────────────")
        self._log(f"{'Metodo':<22} {'Iter':>6} {'Tempo':>9} {'Errore Rel.':>12} {'Mem (MB)':>10}")
        self._log("─" * 65)
        for name, r in results.items():
            stato = "FAIL" if r.get("failed") else "OK"
            err_s = f"{r['err']:.2e}" if not np.isnan(r["err"]) else "   —"
            self._log(
                f"{name:<22} {r['iters']:>6} {r['time']:>9.4f} {err_s:>12} "
                f"{r['peak_mem_mb']:>10.3f}  [{stato}]"
            )

    def _handle_error(self, tb):
        self._progress.stop()
        self._run_btn.configure(state="normal")
        self._status_var.set("Errore durante l'esecuzione.")
        self._log("\n[ERRORE]\n" + tb)
        messagebox.showerror("Errore", "Si è verificato un errore.\nControlla il tab Log.")

# Redirect stdout → widget Text
class _TextRedirector:
    def __init__(self, widget, app):
        self._widget = widget
        self._app    = app

    def write(self, s):
        if s:
            self._app.after(0, lambda msg=s: self._append(msg))

    def _append(self, msg):
        self._widget.configure(state="normal")
        self._widget.insert("end", msg)
        self._widget.see("end")
        self._widget.configure(state="disabled")

    def flush(self):
        pass

# Entrypoint
if __name__ == "__main__":
    app = App()
    app.mainloop()