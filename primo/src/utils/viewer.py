import matplotlib.pyplot as plt
import tracemalloc
import numpy as np
from utils.metrics import compute_relative_error
from solvers import jacobi, gauss_seidel, gradient, cg

def run_all_solvers(A, b, x_true, tol):
    """Esegue tutti i solutori monitorando tempo e memoria."""
    
    # Trasformazione di funzioni in elementi di array, per semplicità
    methods = {
        "Jacobi": jacobi.solve,
        "Gauss-Seidel": gauss_seidel.solve,
        "Gradiente": gradient.solve,
        "Gradiente Coniugato": cg.solve
    }
    
    results = {}
    
    # Esecuzione effettiva dei solver
    for name, solver in methods.items():
        print(f"Esecuzione {name} in corso...")
        
        # Monitoraggio memoria
        tracemalloc.start()
        
        # Esecuzione
        x_sol, iters, elapsed = solver(A, b, tol)
        
        # Stop monitoraggio
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calcolo errore
        err = compute_relative_error(x_true, x_sol)
        
        # Array degli output
        results[name] = {
            "err": err, 
            "iters": iters, 
            "time": elapsed,
            "peak_mem_mb": peak_mem / 10**6 # Converti in MB
        }
        
    return results