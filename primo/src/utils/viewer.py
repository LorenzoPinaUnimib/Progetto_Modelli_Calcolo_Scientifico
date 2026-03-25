import matplotlib.pyplot as plt
import tracemalloc
import numpy as np
from utils.metrics import compute_relative_error
from solvers import jacobi, gauss_seidel, gradient, cg

def run_all_solvers(A, b, x_true, tol):
    """Esegue tutti i solutori monitorando tempo e memoria."""
    methods = {
        "Jacobi": jacobi.solve,
        "Gauss-Seidel": gauss_seidel.solve,
        "Gradiente": gradient.solve,
        "Gradiente Coniugato": cg.solve
    }
    
    results = {}
    
    for name, solver in methods.items():
        print(f"Esecuzione {name} in corso...")
        
        # Monitoraggio memoria
        tracemalloc.start()
        
        # Esecuzione
        x_sol, iters, elapsed = solver(A, b, tol)
        
        # Stop monitoraggio
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calcolo errore
        err = compute_relative_error(x_true, x_sol)
        
        results[name] = {
            "err": err, 
            "iters": iters, 
            "time": elapsed,
            "peak_mem_mb": peak_mem / 10**6 # Converti in MB
        }
        
    return results

def show_dashboard(results, title_suffix):
    """Genera una dashboard con 3 grafici: Tempo, Iterazioni, Memoria e Errore."""
    names = list(results.keys())
    times = [r["time"] for r in results.values()]
    iters = [r["iters"] for r in results.values()]
    errors = [r["err"] for r in results.values()]
    mems = [r["peak_mem_mb"] for r in results.values()]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Analisi Solutori - {title_suffix}", fontsize=16)

    # 1. Grafico Tempo e Iterazioni (Doppio asse)
    ax1 = axs[0]
    ax1.bar(names, times, color='skyblue', label='Tempo (s)')
    ax1.set_ylabel('Tempo (Secondi)')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(names, iters, color='red', marker='o', linestyle='-', linewidth=2, label='Iterazioni')
    ax2.set_ylabel('Numero di Iterazioni')
    ax1.set_title('Tempo vs Iterazioni')

    # 2. Grafico Errore Relativo (Scala Logaritmica)
    axs[1].bar(names, errors, color='lightgreen')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Errore Relativo (Scala Log)')
    axs[1].set_title('Precisione (Errore Relativo)')
    axs[1].tick_params(axis='x', rotation=45)
    
    # Aggiunge una linea orizzontale per indicare la convergenza ideale
    axs[1].axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='Tol 10^-10')
    axs[1].legend()

    # 3. Grafico Memoria di Picco
    axs[2].bar(names, mems, color='salmon')
    axs[2].set_ylabel('Memoria di picco (MB)')
    axs[2].set_title('Consumo di Memoria')
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()