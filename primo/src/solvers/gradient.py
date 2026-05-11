import numpy as np
import time

def solve(A, b, tol, nmax=20000):
    M, N = A.shape
    
    #Verifica matrice quadrata
    if M != N:
        print("Matrix A is not a square matrix")
        return None, 0, 0
        
    # Nota: la verifica degli autovalori (eig) su matrici sparse giganti bloccherebbe il PC.
    # Evitiamo di inserire il calcolo esplicito di eig(A) qui.

    # Creazione variabili usate durante la risoluzione
    nit = 0
    err = 1.0
    x = np.zeros(M)

    # Calcolo errore
    err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)

    # Salvataggio tempo di inizio
    start_time = time.perf_counter()
    
    while nit < nmax and err >= tol:
        if nit % 1000 == 0:
            print("Gradiente: iterazione", nit)

        # Calcolo residuo
        residual = b - A @ x
        
        # Calcolo dimensione passo
        A_res = A @ residual
        step = np.dot(residual, residual) / np.dot(residual, A_res)
        
        # Calcolo nuova x
        x = x + step * residual
        
        # Calcolo errore
        err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)

        nit += 1
        
    # Calcolo tempo trascorso
    elapsed_time = time.perf_counter() - start_time
    
    return x, nit, elapsed_time