import math
import numpy as np
import matplotlib.pyplot as plt

def funzione(x):
    #return x
    #return x**3
    return np.where(x < 0.5, 1.0, 0.0)

def campiona_funzione(N):
    return [funzione((2 * i + 1) / (2 * N)) for i in range(N)]

def calcola_D(N):
    D = []

    D.append([1 / math.sqrt(N) * math.cos(0) for j in range(N)])

    for i in range(1, N):
        D.append([math.sqrt(2 / N) * math.cos(i * math.pi * (2 * j + 1) / (2 * N)) for j in range(N)])

    return D

def calcola_c(D, funzione):
    return D @ np.transpose(funzione)

def tronca(c, n):
    for i in range(n, len(c)):
        c[i] = 0

    return c

def IDCT1(c):
    return np.transpose(calcola_D(len(c))) @ c

def DCT1(f, n, m):
    c = calcola_c(calcola_D(n), campiona_funzione(n))

    return tronca(c, m)

N = 1000
M = 100

x = (2 * np.arange(N) + 1) / (2 * N)
f_samples = campiona_funzione(N)
c = DCT1(funzione, N, M)
f_rec = IDCT1(c)

plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(x, f_samples, label='original', linewidth=1)
plt.plot(x, f_rec, label='reconstructed (m={})'.format(M), linewidth=1)
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original vs Reconstructed')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(x, f_samples, color='C0', linewidth=1)
plt.title('Original')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(x, f_rec, color='C1', linewidth=1)
plt.title('Reconstructed (m={})'.format(M))
plt.xlabel('x')
plt.ylabel('f_rec(x)')
plt.grid(True)

plt.tight_layout()
plt.show()