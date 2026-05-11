import math
import numpy as np
import matplotlib.pyplot as plt

def funzione(x):
    #return x
    #return x**3
    return np.sign(x - 0.5)

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

def IDCT1(c, D):
    return np.transpose(D) @ c

def DCT1(f ,D):
    c = calcola_c(D, f)

    return c

N = 100
M = int(N * 0.2)

x = (2 * np.arange(N) + 1) / (2 * N)
f_samples = campiona_funzione(N)
D = calcola_D(N)

c = DCT1(f_samples, D)
tronca(c, M)
f_rec = IDCT1(c, D)

plt.figure(figsize=(10,8))

plt.subplot(4,1,1)
plt.plot(x, f_samples, label='Originale (N = {})'.format(N), linewidth=1)
plt.plot(x, f_rec, label='Ricostruita (M = {})'.format(M), linewidth=1)
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Originale vs Ricostruita')
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(x, f_samples, color='C0', linewidth=1)
plt.title('Originale (N = {})'.format(N))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(x, f_rec, color='C1', linewidth=1)
plt.title('Ricostruita (M = {})'.format(M))
plt.xlabel('x')
plt.ylabel('f_rec(x)')
plt.grid(True)

plt.subplot(4,1,4)
m_idx = np.arange(len(c))
plt.stem(x, c, basefmt=" ")
plt.title('Coefficienti DCT1')
plt.xlabel('Indice')
plt.ylabel('c[k]')
plt.grid(True)

plt.tight_layout()
plt.show()