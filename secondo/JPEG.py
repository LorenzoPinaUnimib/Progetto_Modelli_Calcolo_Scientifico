import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def apri_immagine():
    img = mpimg.imread(os.path.join(__file__[:-8], "dati/bridge.bmp"))

    return img

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

def split(immagine, dimensione):
    righe = len(immagine) // dimensione
    colonne = len(immagine[0]) // dimensione

    blocchi = np.zeros((righe * colonne, dimensione, dimensione))

    for i in range(righe):
        for j in range(colonne):
            blocchi[i * colonne + j, :, :] = immagine[i * dimensione : (i + 1) * dimensione, j * dimensione : (j + 1) * dimensione]

    return blocchi, righe, colonne

def desplit(blocchi, righe, colonne):
    dim_blocco = len(blocchi[0])

    immagine = np.zeros((righe * dim_blocco, colonne * dim_blocco))

    for i in range(righe):
        for j in range(colonne):
            immagine[i * dim_blocco : (i + 1) * dim_blocco, j * dim_blocco : (j + 1) * dim_blocco] = blocchi[i * colonne + j]

    return immagine

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

img = apri_immagine()

blocchi, righe, colonne = split(img, 16)

test = desplit(blocchi, righe, colonne)

imgplot = plt.imshow(test)
plt.show()