import numpy as np
import time

def solve(A, b, tol, max_iter=20000):
    """Metodo del Gradiente (Steepest Descent)."""
    n = A.shape[0]
    x = np.zeros(n)
    r = b - A @ x
    norm_b = np.linalg.norm(b)
    
    start_t = time.perf_counter()
    for k in range(max_iter):
        if np.linalg.norm(r) / norm_b < tol:
            return x, k, time.perf_counter() - start_t
            
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x = x + alpha * r
        r = r - alpha * Ar # Aggiornamento efficiente del residuo
        
    return x, max_iter, time.perf_counter() - start_t