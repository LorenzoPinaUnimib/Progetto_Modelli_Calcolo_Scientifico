import numpy as np
import time

def solve(A, b, tol, max_iter=20000):
    """Metodo di Jacobi vettorizzato per matrici sparse."""
    n = A.shape[0]
    x = np.zeros(n)
    
    # Estrazione della diagonale
    D = A.diagonal()
    norm_b = np.linalg.norm(b)
    
    start_t = time.perf_counter()
    for k in range(max_iter):
        r = b - A @ x
        
        # Controllo convergenza
        if np.linalg.norm(r) / norm_b < tol:
            return x, k, time.perf_counter() - start_t
            
        # Aggiornamento
        x = x + r / D
        
    return x, max_iter, time.perf_counter() - start_t