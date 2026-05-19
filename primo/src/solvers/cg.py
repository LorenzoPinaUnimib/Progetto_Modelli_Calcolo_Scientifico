import numpy as np
import time
import scipy.sparse.linalg as spla

def solve(A, b, tol, nmax=20000):
    """
    Metodo del Gradiente Coniugato
    INPUT  : A=matrice del sistema, b=termine noto,
             tol=tolleranza, nmax=massimo numero di iterazioni
    OUTPUT : xk=soluzione, nit=numero di iterazioni, elapsed_time=tempo trascorso
    """
    
    M, N = A.shape
    
    # Verifica che la matrice sia quadrata
    if M != N:
        print("La matrice passata in argomento non è quadrata, gradiente coniugato non è applicabile")
        return None, 0, 0

    # Verifica simmetria della matrice
    if (A - A.T).nnz != 0:
        print("La matrice non è simmetrica, il metodo del gradiente coniugato non è applicabile.")
        return None, 0, 0

    # Verifica che sia definita positiva, calcolando solo autovalore più piccolo per ottimizzazione
    try:
        min_eigenval = spla.eigsh(A, k=1, which='SM', return_eigenvectors=False)
        if min_eigenval[0] <= 1e-10:  # Tolleranza per lo zero numerico
            print("La matrice non è definita positiva, il metodo del gradiente coniugato non è applicabile.")
            return None, 0, 0
        else: 
            print(f"Autovalore minimo per metodo gradiente coniugato: {min_eigenval[0]}")
    except (ValueError, RuntimeError):
        # Cattura eventuali problemi di convergenza del solutore ARPACK su matrici singolari
        print("Impossibile determinare se la matrice è definita positiva (mancata convergenza degli autovalori).")
        return None, 0, 0

    # Creazione del vettore della soluzione, composto da zeri, come da consegna
    x = np.zeros(M)
    
    # Contatore per il numero di iterazioni
    nit = 0

    # Calcolo residuo iniziale
    r = b - A @ x

    # Inizializzazione della direzione di ricerca
    d = r.copy()
    
    # Errore iniziale
    err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)
    
    # Salvataggio tempo di inizio
    start_time = time.perf_counter()
    
    # Iterazione del metodo
    while nit < nmax and err >= tol:
        if nit % 1000 == 0:
            print("Gradiente Coniugato: iterazione", nit)

        # Calcolo del passo ottimale
        y = A @ d
        step = np.dot(d, r) / np.dot(d, y)
        
        # Aggiornamento della soluzione
        x = x + step * d

        # Aggiornamento del residuo
        r = b - A @ x
        
        # Calcolo della nuova direzione di ricerca
        beta = np.dot(d, A @ r) / np.dot(d, y)
        d = r - beta * d

        # Aggiornamento residuo e errore per il prossimo ciclo
        err = np.linalg.norm(A @ x - b, np.inf) / np.linalg.norm(b, np.inf)
        
        nit += 1
    
    # Calcolo tempo trascorso
    elapsed_time = time.perf_counter() - start_time
    
    return x, nit, elapsed_time