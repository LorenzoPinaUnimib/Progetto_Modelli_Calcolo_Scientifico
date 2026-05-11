import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular
import time

def solve(A, b, tol, nmax=20000):
    M, N = A.shape
    
    #Verifica matrice quadrata
    if M != N:
        print("Matrix A is not a square matrix")
        return None, 0, 0

    # Creazione matrici e variabili usate durante la risoluzione
    # Formato csr richiesto per l'efficienza
    P = sp.tril(A, format='csr')
    N = A - P
    
    x = np.zeros(M)
    nit = 0

    # Calcolo errore
    err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)

    # Salvataggio tempo di inizio
    start_time = time.perf_counter()
    
    while err >= tol and nit < nmax:
        if nit % 1000 == 0:
            print("Gauss-Seidel: iterazione", nit)
        
        # rhs = (b - B*xold)
        rhs = b - A @ x

        # Risoluzione del sistema triangolare inferiore
        y = spsolve_triangular(P, rhs, lower=True)

        # Calcolo nuova x
        x = x + y

        # Calcolo errore
        err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)
        
        nit += 1
        
    # Calcolo tempo trascorso
    elapsed_time = time.perf_counter() - start_time
    
    return x, nit, elapsed_time