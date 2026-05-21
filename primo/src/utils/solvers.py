import threading
import traceback

import numpy as np

def _load_solvers():
    """Importa i solver dal progetto; ritorna un dict name→solve_fn."""

    from utils.matrix_io import load_mtx
    from solvers import jacobi, gauss_seidel, gradient, cg

    return load_mtx, {
        "Jacobi":              jacobi.solve,
        "Gauss-Seidel":        gauss_seidel.solve,
        "Gradiente":           gradient.solve,
        "Gradiente Coniugato": cg.solve,
    }

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
        from utils.metrics import _compute_relative_error

        try:
            # Carico i vari risolutori
            load_mtx, methods = _load_solvers()

            # Carico matrice e faccio un analisi preliminare
            self.on_progress("Caricamento matrice…")
            A = load_mtx(self.mtx_path)
            n = A.shape[0]
            x_true = np.ones(n)
            b = A @ x_true
            self.on_progress(f"Matrice caricata: {n}x{n} | {A.nnz} elementi non-zero ({A.nnz / (n * n):.2f}%)")

            # Esecuzione dei risolutori
            results = {}
            for name, solver in methods.items():
                self.on_progress(f"Esecuzione {name}...")
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
                    "x_sol":        x_sol,
                    "err":          err,
                    "iters":        iters,
                    "time":         elapsed,
                    "peak_mem_mb":  peak_mem / 1e6,
                    "failed":       x_sol is None or iters >= 50000,
                }

            self.on_done(results)

        except Exception as exc:
            self.on_error(traceback.format_exc())
