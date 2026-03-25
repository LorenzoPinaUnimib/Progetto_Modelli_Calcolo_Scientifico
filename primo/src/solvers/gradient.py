import numpy as np
import time

def solve(A, b, tol, nmax=20000):
    M, N = A.shape
    
    if M != N:
        print("Matrix A is not a square matrix")
        return None, 0, 0, 1
        
    # Nota: la verifica degli autovalori (eig) su matrici sparse giganti bloccherebbe il PC.
    # Evitiamo di inserire il calcolo esplicito di eig(A) qui.

    nit = 0
    err = 1.0
    xold = np.zeros(M)

    start_time = time.perf_counter()
    
    while nit < nmax and err > tol:
        residual = b - A @ xold
        
        # step = (residual'*residual)/(residual'*A*residual)
        A_res = A @ residual
        step = np.dot(residual, residual) / np.dot(residual, A_res)
        
        xnew = xold + step * residual
        
        # err = norm(b - A*xnew)/norm(xnew)
        err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(xnew)
        xold = xnew
        nit += 1
        
    elapsed_time = time.perf_counter() - start_time
    
    return xnew, nit, elapsed_time