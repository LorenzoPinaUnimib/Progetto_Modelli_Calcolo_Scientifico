import sys
import numpy as np
from utils.matrix_io import load_mtx
from utils.viewer import run_all_solvers, show_dashboard

def main():
    if len(sys.argv) < 3:
        print("Uso: python main.py <percorso_matrice.mtx> <tolleranza>")
        sys.exit(1)
        
    mtx_path = sys.argv[1]
    tol = float(sys.argv[2])
    
    # Step 1-2: Preparazione sistema [cite: 33, 34, 35, 36]
    A = load_mtx(mtx_path)
    n = A.shape[0]
    x_true = np.ones(n)
    b = A @ x_true
    
    print(f"\n--- Analisi {mtx_path.split('/')[-1]} | Tolleranza: {tol} ---")
    
    # Step 3-4: Esecuzione e raccolta metriche [cite: 37, 38]
    results = run_all_solvers(A, b, x_true, tol)
    
    # Stampa in console
    print("\nRisultati Tabellari:")
    print(f"{'Metodo':<20} | {'Iterazioni':<10} | {'Tempo (s)':<10} | {'Errore Rel.':<12}")
    print("-" * 60)
    for m, res in results.items():
        print(f"{m:<20} | {res['iters']:<10} | {res['time']:<10.4f} | {res['err']:.2e}")
        
    # Visualizzazione Grafici
    show_dashboard(results, f"Matrice: {mtx_path.split('/')[-1]} | Tol: {tol}")

if __name__ == "__main__":
    main()