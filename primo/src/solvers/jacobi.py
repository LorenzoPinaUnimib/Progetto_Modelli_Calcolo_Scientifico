import numpy as np
import scipy.sparse as sp
import time

def solve(A, b, tol, nmax=20000):
    M, N = A.shape
    
    #Verifica matrice quadrata
    if M != N:
        print("Matrix A is not a square matrix")
        return None, 0, 0
        
    D_diag = A.diagonal()

    if np.any(D_diag == 0):
        print("At least a diagonal entry is zero. The method automatically fails")
        return None, 0, 0

    # Creazione matrici e variabili usate durante la risoluzione
    # D = sp.diags(D_diag)
    D_inv = sp.diags(1.0 / D_diag)
    # B = D - A
    x = np.zeros(M)
    nit = 0

    # Calcolo errore
    r = b - A @ x
    err = np.linalg.norm(r, np.inf) / np.linalg.norm(b, np.inf)

    # Salvataggio tempo di inizio
    start_time = time.perf_counter()
    
    while err >= tol and nit < nmax:
        if nit % 1000 == 0:
            print("Jacobi: iterazione", nit)

        # xnew = inv(D) * (B*xold + b) in formato array:
        # Calcolo nuova x
        x = x + D_inv @ r

        # Calcolo errore
        r = b - A @ x
        err = np.linalg.norm(r, np.inf) / np.linalg.norm(b, np.inf)

        nit += 1
        
    # Calcolo tempo trascorso
    elapsed_time = time.perf_counter() - start_time
    
    return x, nit, elapsed_time