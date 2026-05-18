import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular
import time

def solve(A, b, tol, nmax=20000):
    """
    Metodo di Jacobi
    INPUT  : A=matrice del sistema, b=termine noto,
             tol=tolleranza, nmax=massimo numero di iterazioni
    OUTPUT : x=soluzione, nit=numero di iterazioni, time=tempo trascorso
    """

    M, N = A.shape
    
    # Verifica che la matrice sia quadrata
    if M != N:
        print("La matrice passata in argomento non è quadrata, Gauß-Seidel non è applicabile")
        return None, 0, 0

    # Estrazione della matrice triangolare inferiore
    P = sp.tril(A, format='csr')
    # Matrice superiore, inutilizzata (?)
    N = A - P
    
    # Creazione del vettore della soluzione, composto da zeri, come da consegna
    x = np.zeros(M)

    # Contatore per il numero di iterazioni
    nit = 0

    # Calcolo errore
    err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)

    # Salvataggio tempo di inizio
    start_time = time.perf_counter()
    
    # Iterazione del metodo
    while err >= tol and nit < nmax:
        if nit % 1000 == 0:
            print("Gauss-Seidel: iterazione", nit)
        
        # Calcolo del residuo
        rhs = b - A @ x

        # Risoluzione del sistema triangolare inferiore
        y = spsolve_triangular(P, rhs, lower=True)

        # Aggiornamento della soluzione
        x = x + y

        # Calcolo errore
        err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)
        
        nit += 1
        
    # Calcolo tempo trascorso
    elapsed_time = time.perf_counter() - start_time
    
    return x, nit, elapsed_time