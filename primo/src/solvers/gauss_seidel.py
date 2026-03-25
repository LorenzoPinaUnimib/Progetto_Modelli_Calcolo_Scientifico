import numpy as np
import time

def solve(A, b, tol, max_iter=20000):
    """Metodo di Gauss-Seidel sfruttando la struttura CSR."""
    n = A.shape[0]
    x = np.zeros(n)
    
    norm_b = np.linalg.norm(b)
    D = A.diagonal()
    
    # Puntatori diretti alla struttura sparsa per massima velocità
    indptr = A.indptr
    indices = A.indices
    data = A.data
    
    start_t = time.perf_counter()
    for k in range(max_iter):
        # Calcolo del residuo per il controllo di convergenza
        # Nota: calcolarlo ad ogni iterazione è costoso, ma richiesto dal criterio
        r = b - A @ x
        if np.linalg.norm(r) / norm_b < tol:
            return x, k, time.perf_counter() - start_t
            
        # Ciclo di aggiornamento componente per componente
        for i in range(n):
            start_idx = indptr[i]
            end_idx = indptr[i+1]
            
            # Prodotto scalare della i-esima riga di A con x
            ax_i = np.dot(data[start_idx:end_idx], x[indices[start_idx:end_idx]])
            
            # Aggiornamento: x_i = x_i + (b_i - A_i * x) / A_ii
            x[i] = x[i] + (b[i] - ax_i) / D[i]
            
    return x, max_iter, time.perf_counter() - start_t