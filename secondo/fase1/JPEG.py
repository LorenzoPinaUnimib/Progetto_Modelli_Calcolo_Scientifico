import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def apri_immagine():
    img = mpimg.imread(os.path.join(__file__[:-8], "dati/bridge.bmp"))

    return img

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

def calcola_D(N):
    D = []

    D.append([1 / math.sqrt(N) * math.cos(0) for j in range(N)])

    for i in range(1, N):
        D.append([math.sqrt(2 / N) * math.cos(i * math.pi * (2 * j + 1) / (2 * N)) for j in range(N)])

    return D

def tronca(c, n):
    c[:, n :, n :] = 0

def DCT1(f, D):
    return D @ f

def DCT2(blocchi, D):
    # c = [DCT1(col, D) for col in np.transpose([DCT1(row, D) for row in blocchi])]
    # c = [[DCT1(col, D) for col in ([DCT1(row, D) for row in blocco])] for blocco in blocchi]

    c = np.ndarray([])
    b = []

    for blocco in blocchi:
        tmp = np.array([DCT1(row, D) for row in blocco])
        cols_dct = np.array([DCT1(col, D) for col in tmp.T])

        b.append((np.asarray(cols_dct.T)))

    c = np.stack(b, axis = 0)
    return c

def IDCT1(c, D_tr):
    return D_tr @ c

def IDCT2(c, D_tr):
    # f = [IDCT1(col, D_tr) for col in np.transpose([IDCT1(row, D_tr) for row in c])]

    f = np.ndarray([])
    b = []

    for blocco in c:
        tmp = np.array([IDCT1(row, D_tr) for row in blocco])
        cols_dct = np.array([IDCT1(col, D_tr) for col in tmp.T])

        b.append((np.asarray(cols_dct.T)))

    f = np.stack(b, axis = 0)
    return f

img = apri_immagine()

blocchi, righe, colonne = split(img, 16)
print("Post split")

D = calcola_D(len(blocchi[0]))
print("Post D")

c = DCT2(blocchi, D)
print("Post DCT2")
tronca(c, 5)
print("Post troncamento")

blocchi = IDCT2(c, np.transpose(D))
print("Post IDCT2")

img_rec = desplit(blocchi, righe, colonne)
print("Post desplit")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray' if img.ndim == 2 else None)
axes[0].set_title('Originale')
axes[0].axis('off')

axes[1].imshow(img_rec, cmap='gray' if img_rec.ndim == 2 else None)
axes[1].set_title('Ricostruita')
axes[1].axis('off')

plt.tight_layout()
plt.show()