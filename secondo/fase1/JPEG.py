import math
import numpy as np
import os
from PIL import Image
import sys
from PyQt6.QtWidgets import QApplication
from helper.ImageComparisonWindow import ImageComparisonWindow

def apri_immagine(name):
    # Apro l'immagine
    img = np.array(Image.open(os.path.join(__file__[:-14], "dati/", name)).convert("L"), dtype=np.uint8)

    return img

def split(immagine, dimensione):
    # Stabilisco il numero di blocchi sulle righe e colonne in modo da poter ricostruire correttamente immagini non quadrate 
    righe = len(immagine) // dimensione
    colonne = len(immagine[0]) // dimensione

    # Definisco un array relativo ai blocchi che creeremo
    blocchi = np.zeros((righe * colonne, dimensione, dimensione))

    for i in range(righe):
        for j in range(colonne):
            # Spezzo l'immagine in blocchi e la salvo
            blocchi[i * colonne + j] = immagine[i * dimensione : (i + 1) * dimensione, j * dimensione : (j + 1) * dimensione]

    # Arrotondo i valori e li sposto di -128 (il range andrà da -128 a 127)
    blocchi = np.round(blocchi) - 128
    blocchi[blocchi < -128] = -128
    blocchi[blocchi > 127] = 127

    return blocchi, righe, colonne

def desplit(blocchi, righe, colonne):
    # Calcolo la dimensinoe di una riga di un blocco
    dim_blocco = len(blocchi[0])

    # Arrotondo i valori e li sposto di +128 (il range andrà da 0 a 255)
    blocchi = np.round(blocchi) + 128
    blocchi[blocchi < 0] = 0
    blocchi[blocchi > 255] = 255

    # Definisco un array relativo all'immagine che creeremo
    immagine = np.zeros((righe * dim_blocco, colonne * dim_blocco))

    for i in range(righe):
        for j in range(colonne):
            # Ricompongo l'immagine dai blocchi
            immagine[i * dim_blocco : (i + 1) * dim_blocco, j * dim_blocco : (j + 1) * dim_blocco] = blocchi[i * colonne + j]

    return immagine

def calcola_D(N):
    # Effettuo il calcolo della matrica D in base a quello visto a lezione per 1D
    D = []

    D.append([1 / math.sqrt(N) * math.cos(0) for j in range(N)])

    for i in range(1, N):
        D.append([math.sqrt(2 / N) * math.cos(i * math.pi * (2 * j + 1) / (2 * N)) for j in range(N)])

    return D

def tronca(c, n, triang = False):
    if (triang):
        N = c.shape[1]
        k, l = np.indices((N, N))
        mask = (k + l) < n # True = mantieni, False = azzera
        c[:, ~mask] = 0
    else:
        c[:, n :, :] = 0
        c[:, :, n :] = 0

def DCT1(f, D):
    return D @ f

def DCT2(blocchi, D):
    c = np.ndarray([])
    b = []

    # Eseguo la DCT1 sulle righe e poi sulle colonne di ogni blocco
    for blocco in blocchi:
        tmp = np.array([DCT1(row, D) for row in blocco])
        tmp = np.array([DCT1(col, D) for col in tmp.T])

        b.append((np.asarray(tmp.T)))

    c = np.stack(b, axis = 0)
    return c

def IDCT1(c, D_tr):
    return D_tr @ c

def IDCT2(c, D_tr):
    f = np.ndarray([])
    b = []

    # Eseguo la IDCT1 sulle righe e poi sulle colonne di ogni blocco
    for blocco in c:
        tmp = np.array([IDCT1(row, D_tr) for row in blocco])
        tmp = np.array([IDCT1(col, D_tr) for col in tmp.T])

        b.append((np.asarray(tmp.T)))

    f = np.stack(b, axis = 0)
    return f

def JPEG(img, N, M, grafico = True, triangolare = False):
    D = calcola_D(N)

    blocchi, righe, colonne = split(img, N)

    if (len(blocchi) == 0):
        print("Il numero di blocchi creato è 0, riprovare con N più piccolo")
        return

    c = DCT2(blocchi, D)
    tronca(c, M, triangolare)

    blocchi = IDCT2(c, np.transpose(D))

    img_rec = desplit(blocchi, righe, colonne)

    if grafico:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        window = ImageComparisonWindow(
            img, 
            img_rec, 
            title1="Originale",
            title2=f"Ricostruita (troncando da {N} a {M})"
        )

        window.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    img = apri_immagine("bridge.bmp")
    JPEG(img, 16, 16, True, False)