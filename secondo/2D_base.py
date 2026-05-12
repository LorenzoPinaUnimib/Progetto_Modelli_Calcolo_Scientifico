import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def funzione(x, y):
    return x**3 + y**3
    #return np.where(x < 0.5, 1.0, 0.0)
    #return x + y

def campiona_funzione(N):
    return [[funzione(((2 * i + 1) / (2 * N)), ((2 * j + 1) / (2 * N))) for i in range(N)] for j in range(N)]

def calcola_D(N):
    D = []

    D.append([1 / math.sqrt(N) * math.cos(0) for j in range(N)])

    for i in range(1, N):
        D.append([math.sqrt(2 / N) * math.cos(i * math.pi * (2 * j + 1) / (2 * N)) for j in range(N)])

    return D

def calcola_c(D, funzione):
    return D @ np.transpose(funzione)

def tronca(c, n):
    for i in range(len(c)):
        for j in range(len(c[i])):
            if i > n or j > n:
                c[i][j] = 0

def split(matrice, dimensione):
    split = np.array([])

    for i in range(len(matrice) // dimensione):
        for j in range(len(matrice[i]) // dimensione):
            split = np.append(split, matrice[i * dimensione : i * (dimensione + 1)][j * dimensione : j * (dimensione + 1)])

    return split

def IDCT1(c):
    return np.transpose(calcola_D(len(c))) @ c

def DCT1(f, n):
    c = calcola_c(calcola_D(n), f)

    return c

def DCT2(f, n, m):
    c = [DCT1(col, n) for col in [DCT1(row, n) for row in f]]

    return c

def IDCT2(c):
    f = [IDCT1(row) for row in [IDCT1(col) for col in c]]

    return f

N = 10
M = int(N * 0.4)

x = (2 * np.arange(N) + 1) / (2 * N)
f_samples = np.array(campiona_funzione(N))

#print(split(f_samples, 5))
print(split_matrix_into_blocks(f_samples, 5))

c = DCT2(f_samples, N, M)
tronca(c, M)
f_rec = IDCT2(c)

# Preparazione griglia per il plotting
X, Y = np.meshgrid(x, x)
F_orig = np.array(f_samples)
F_rec = np.array(f_rec)

# Plot 3D affiancati
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, F_orig, cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.9)
ax1.set_title('f_samples (originale)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, F_rec, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.9)
ax2.set_title('f_rec (ricostruita)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f')

plt.tight_layout()
plt.show()