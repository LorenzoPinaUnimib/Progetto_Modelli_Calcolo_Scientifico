import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image

def apri_immagine(name):
    # Apro l'immagine
    img = np.array(Image.open(os.path.join(__file__[:-8], "dati/", name)).convert("L"), dtype=np.uint8)

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

def tronca(c, n):
    c[:, n :, n :] = 0

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

# Funzione creata con IA per sincronizzare gli zoom e pan sulle due immagini
def sync_axes(axs):
    """
    Sincronizza zoom e pan per le axes passate (lista o array).
    Quando si zooma/pana su una axis, le altre adottano gli stessi xlim/ylim.
    """
    # stato condiviso per evitare loop di callback
    state = {'updating': False}

    def on_xlim_changed(event_ax):
        if state['updating']:
            return
        state['updating'] = True
        try:
            new_xlim = event_ax.get_xlim()
            new_ylim = event_ax.get_ylim()
            for ax in axs:
                if ax is not event_ax:
                    ax.set_xlim(new_xlim)
                    ax.set_ylim(new_ylim)
            event_ax.figure.canvas.draw_idle()
        finally:
            state['updating'] = False

    # Connetti le callback di change limits per ogni axis
    for ax in axs:
        ax.callbacks.connect('xlim_changed', lambda evt_ax, ax=ax: on_xlim_changed(ax))
        ax.callbacks.connect('ylim_changed', lambda evt_ax, ax=ax: on_xlim_changed(ax))

    # Facoltativo: sincronizza anche lo zoom con la rotella del mouse (migliora interattività)
    def on_scroll(event):
        # event.inaxes è l'axis sotto il cursore
        ax = event.inaxes
        if ax is None:
            return
        base_scale = 1.1
        # scelta del fattore di zoom
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        new_xlim = (xdata - (1 - relx) * new_width, xdata + relx * new_width)
        new_ylim = (ydata - (1 - rely) * new_height, ydata + rely * new_height)

        # applica e propaga
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        on_xlim_changed(ax)

    fig = axs[0].figure
    fig.canvas.mpl_connect('scroll_event', on_scroll)

def JPEG(img, N, M, graph = True):
    D = calcola_D(N)

    blocchi, righe, colonne = split(img, N)

    if (len(blocchi) == 0):
        print("Il numero di blocchi creato è 0, riprovare con N più piccolo")
        return

    c = DCT2(blocchi, D)
    tronca(c, M)

    blocchi = IDCT2(c, np.transpose(D))

    img_rec = desplit(blocchi, righe, colonne)

    if graph:
        _, axes = plt.subplots(1, 2, figsize=(12, 6.5))
        axes[0].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[0].set_title('Originale')
        axes[0].axis('off')

        axes[1].imshow(img_rec, cmap='gray' if img_rec.ndim == 2 else None)
        axes[1].set_title('Ricostruita (troncando da {} a {})'.format(N, M))
        axes[1].axis('off')

        sync_axes(axes)

        plt.tight_layout()
        plt.show()

# img = apri_immagine("bridge.bmp")
# JPEG(img, 16, 1)