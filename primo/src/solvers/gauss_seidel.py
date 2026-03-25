import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular
import time

def solve(A, b, tol, nmax=20000):
    M, N = A.shape
    
    if M != N:
        print("Matrix A is not a square matrix")
        return None, 0, 0, 1

    # extract needed matrices
    # Formato csr richiesto per l'efficienza
    L = sp.tril(A, format='csr') 
    B = A - L
    
    xold = np.zeros(M)
    xnew = xold + 1.0
    nit = 0

    start_time = time.perf_counter()
    
    while np.linalg.norm(xnew - xold, np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        
        # rhs = (b - B*xold)
        rhs = b - B @ xold
        # Risoluzione del sistema triangolare inferiore L * xnew = rhs
        xnew = spsolve_triangular(L, rhs, lower=True)
        
        nit += 1
        
    elapsed_time = time.perf_counter() - start_time
    err = np.linalg.norm(xnew - xold, np.inf) / np.linalg.norm(xnew, np.inf)
    
    return xnew, nit, elapsed_time