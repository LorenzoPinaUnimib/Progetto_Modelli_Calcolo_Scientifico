import numpy as np
import time
import scipy.sparse.linalg as spla

def solve(A, b, tol, nmax=20000):
    """
    Metodo del Gradiente Coniugato
    INPUT  : A=matrice del sistema, b=termine noto,
             tol=tolleranza, nmax=massimo numero di iterazioni
    OUTPUT : xk=soluzione, nit=numero di iterazioni, time=tempo trascorso
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

    # Creazione dei vettori della soluzione, composto da zeri, come da consegna
    xold = np.zeros(M)
    xnew = np.zeros(M)
    
    # Contatore per il numero di iterazioni
    nit = 0

    # Calcolo residuo iniziale
    r = b - A @ xold

    # Inizializzazione della direzione di ricerca
    p = r.copy()
    
    # Errore iniziale, con check per evitare divisioni per zero se x è nullo
    err = np.linalg.norm(r) / (np.linalg.norm(xold) if np.linalg.norm(xold) != 0 else 1)
    
    # Salvataggio tempo di inizio
    start_time = time.perf_counter()
    
    # Iterazione del metodo
    while nit < nmax and err >= tol:
        if nit % 1000 == 0:
            print("Gradiente Coniugato: iterazione", nit)

        # Calcolo del passo ottimale
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        
        # Aggiornamento della soluzione
        xnew = xold + alpha * p

        # Aggiornamento del residuo
        r_new = r - alpha * Ap
        
        # Calcolo della nuova direzione di ricerca
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        
        # Aggiornamento residuo e errore per il prossimo ciclo
        r = r_new
        err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(xnew)
        
        xold = xnew
        nit += 1
    
    # Calcolo tempo trascorso
    elapsed_time = time.perf_counter() - start_time
    xk = xold
    
    return xk, nit, elapsed_time