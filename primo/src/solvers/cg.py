import numpy as np
import time

def solve(A, b, tol, max_iter=20000):
    """
    Risolve Ax = b usando il metodo del Gradiente Coniugato[cite: 9].
    """
    n = A.shape[0]
    x = np.zeros(n)  # Partenza da vettore nullo [cite: 19]
    r = b - A @ x
    p = r.copy()
    nb = np.linalg.norm(b)
    
    iters = 0
    start_t = time.perf_counter()
    
    while iters < max_iter:
        # Controllo convergenza residuo relativo [cite: 20]
        if np.linalg.norm(r) / nb < tol:
            return x, iters, time.perf_counter() - start_t
        
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
        iters += 1
        
    return x, iters, time.perf_counter() - start_t # Ritorna anche se non converge [cite: 24]