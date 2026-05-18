import numpy as np
import scipy.sparse as sp
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
        print("La matrice passata in argomento non è quadrata, Jacobi non è applicabile")
        return None, 0, 0
        
    D_diag = A.diagonal()

    # Controllo sulla presenza di 0 sulla diagonale
    if np.any(D_diag == 0):
        print("Almeno un enemento sulla diagonale è zero, Jacobi fallisce")
        return None, 0, 0

    # Creazione dell'inversa della diagonale
    D_inv = sp.diags(1.0 / D_diag)

    # Creazione del vettore della soluzione, composto da zeri, come da consegna
    x = np.zeros(M)

    # Contatore per il numero di iterazioni
    nit = 0

    # Calcolo errore
    r = b - A @ x
    err = np.linalg.norm(r, np.inf) / np.linalg.norm(b, np.inf)

    # Salvataggio tempo di inizio
    start_time = time.perf_counter()
    
    # Iterazione del metodo
    while err >= tol and nit < nmax:
        if nit % 1000 == 0:
            print("Jacobi: iterazione", nit)

        # Aggiornamento vettore delle soluzioni
        x = x + D_inv @ r

        # Calcolo errore
        r = b - A @ x
        err = np.linalg.norm(r, np.inf) / np.linalg.norm(b, np.inf)

        nit += 1
        
    # Calcolo tempo trascorso
    elapsed_time = time.perf_counter() - start_time
    
    return x, nit, elapsed_time