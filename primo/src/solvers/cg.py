import numpy as np
import time

def solve(A, b, tol, nmax=20000):
    """
    Metodo del Gradiente Coniugato
    INPUT  : A=matrice del sistema, b=termine noto, x0=guess iniziale, 
             tol=tolleranza, nmax=massimo numero di iterazioni
    OUTPUT : xk=soluzione, nit=numero di iterazioni, time=tempo trascorso, err=errore finale
    """
    
    # Controllo delle proprietà della matrice A (come nei file .m)
    M, N = A.shape
    
    if M != N:
        print('Matrix A is not a square matrix')
        return None, 0, 0, 1

    nit = 0
    xold = np.zeros(M)
    
    # Calcolo residuo iniziale
    r = b - A @ xold
    p = r.copy()
    
    # Errore iniziale (usando la logica del tuo metodo_gradiente.m)
    # err = norm(b - A*x)/norm(x)
    # Per evitare divisioni per zero se x è nullo, usiamo una protezione
    err = np.linalg.norm(r) / (np.linalg.norm(xold) if np.linalg.norm(xold) != 0 else 1)
    
    start_time = time.perf_counter()
    
    while nit < nmax and err > tol:
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        
        xnew = xold + alpha * p
        r_new = r - alpha * Ap
        
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        
        # Aggiornamento residuo e errore per il prossimo ciclo
        r = r_new
        err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(xnew)
        
        xold = xnew
        nit += 1
    
    elapsed_time = time.perf_counter() - start_time
    xk = xold
    
    return xk, nit, elapsed_time