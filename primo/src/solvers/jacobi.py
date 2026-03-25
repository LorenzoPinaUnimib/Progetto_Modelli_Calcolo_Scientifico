import numpy as np
import scipy.sparse as sp
import time

def solve(A, b, tol, nmax=20000):
    M, N = A.shape
    
    if M != N:
        print("Matrix A is not a square matrix")
        return None, 0, 0, 1
        
    D_diag = A.diagonal()
    if np.any(D_diag == 0):
        print("At least a diagonal entry is non-zero. The method automatically fails")
        return None, 0, 0, 1

    # extract needed matrices
    D = sp.diags(D_diag)
    B = D - A
    xold = np.zeros(M)
    xnew = xold + 1.0
    nit = 0

    start_time = time.perf_counter()
    
    # norm(..., inf) equivale a np.linalg.norm(..., np.inf)
    while np.linalg.norm(xnew - xold, np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        # xnew = inv(D) * (B*xold + b) in formato array:
        xnew = (B @ xold + b) / D_diag
        nit += 1
        
    elapsed_time = time.perf_counter() - start_time
    err = np.linalg.norm(xnew - xold, np.inf) / np.linalg.norm(xnew, np.inf)
    
    return xnew, nit, elapsed_time