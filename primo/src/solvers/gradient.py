import numpy as np
import time

def solve(A, b, tol, nmax=20000):
    """
    Metodo del gradiente
    INPUT  : A=matrice del sistema, b=termine noto, 
             tol=tolleranza, nmax=massimo numero di iterazioni
    OUTPUT : x=soluzione, nit=numero di iterazioni, time=tempo trascorso
    """

    M, N = A.shape
    
    # Verifica che la matrice sia quadrata
    if M != N:
        print("La matrice passata in argomento non è quadrata, gradiente non è applicabile")
        return None, 0, 0
        
    # Nota: la verifica degli autovalori (eig) su matrici sparse giganti bloccherebbe il PC.
    # Evitiamo di inserire il calcolo esplicito di eig(A) qui.

    # Creazione del vettore della soluzione, composto da zeri, come da consegna
    x = np.zeros(M)

    # Contatore per il numero di iterazioni
    nit = 0

    # Calcolo errore
    err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)

    # Salvataggio tempo di inizio
    start_time = time.perf_counter()
    
    # Iterazione del metodo
    while nit < nmax and err >= tol:
        if nit % 1000 == 0:
            print("Gradiente: iterazione", nit)

        # Calcolo residuo
        rhs = b - A @ x
        
        # Calcolo della dimensione del passo
        A_res = A @ rhs
        step = np.dot(rhs, rhs) / np.dot(rhs, A_res)
        
        # Aggiornamento della soluzione
        x = x + step * rhs
        
        # Calcolo errore
        err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)

        nit += 1
        
    # Calcolo tempo trascorso
    elapsed_time = time.perf_counter() - start_time
    
    return x, nit, elapsed_time