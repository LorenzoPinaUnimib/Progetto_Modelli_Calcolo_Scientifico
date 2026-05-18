import math
import numpy as np
import os
from PIL import Image
import cv2
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

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

# Funzione creata con IA per sincronizzare gli zoom e pan sulle due immagini
class SyncedImageLabel(QLabel):
    """Label per immagini con pan/zoom"""
    def __init__(self, img_array):
        super().__init__()
        self.original_img = img_array.astype(np.uint8)
        self.scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.dragging = False
        self.first_resize = True  # Aggiungi questo flag
        
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_display()
    
    def update_display(self):
        """Aggiorna il display con scale e pan attuali"""
        h, w = self.original_img.shape
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        # Ridimensiona
        if self.scale != 1.0:
            resized = cv2.resize(self.original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = self.original_img.copy()
        
        # Applica pan
        display_img = self._apply_pan(resized, new_w, new_h)
        
        # Converti a QPixmap
        h_disp, w_disp = display_img.shape
        bytes_per_line = w_disp
        q_img = QImage(display_img.data, w_disp, h_disp, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.setPixmap(pixmap)
    
    def _apply_pan(self, img, img_w, img_h):
        """Applica pan all'immagine"""
        label_w = self.width()
        label_h = self.height()
        
        # Crea canvas
        canvas = np.full((label_h, label_w), 128, dtype=np.uint8)
        
        # Calcola dove mettere l'immagine
        x_start = int(label_w / 2 - img_w / 2 + self.pan_x)
        y_start = int(label_h / 2 - img_h / 2 + self.pan_y)
        
        # Clipping
        src_x_start = max(0, -x_start)
        src_y_start = max(0, -y_start)
        dst_x_start = max(0, x_start)
        dst_y_start = max(0, y_start)
        
        src_x_end = min(img_w, label_w - x_start)
        src_y_end = min(img_h, label_h - y_start)
        
        if src_x_end > src_x_start and src_y_end > src_y_start:
            canvas[dst_y_start:dst_y_start + src_y_end - src_y_start,
                   dst_x_start:dst_x_start + src_x_end - src_x_start] = \
                img[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return canvas
    
    def mousePressEvent(self, event):
        self.dragging = True
        self.last_mouse_x = event.position().x()
        self.last_mouse_y = event.position().y()
    
    def mouseReleaseEvent(self, event):
        self.dragging = False
    
    def mouseMoveEvent(self, event):
        if self.dragging:
            dx = event.position().x() - self.last_mouse_x
            dy = event.position().y() - self.last_mouse_y
            
            self.pan_x += dx
            self.pan_y += dy
            
            self.last_mouse_x = event.position().x()
            self.last_mouse_y = event.position().y()
            
            self.update_display()
    
    def wheelEvent(self, event):
        """Zoom con rotella del mouse, centrato sul puntatore"""
        zoom_factor = 1.1
        
        # Posizione del mouse rispetto alla label
        mouse_x = event.position().x()
        mouse_y = event.position().y()
        
        # Centro della label
        label_w = self.width()
        label_h = self.height()
        center_x = label_w / 2
        center_y = label_h / 2
        
        # Distanza del mouse dal centro
        offset_x = mouse_x - center_x
        offset_y = mouse_y - center_y
        
        # Scala precedente
        old_scale = self.scale
        
        # Calcola la nuova scala
        if event.angleDelta().y() > 0:
            self.scale *= zoom_factor
        else:
            self.scale /= zoom_factor
        
        self.scale = max(0.1, min(5.0, self.scale))
        
        # Aggiusta il pan mantenendo il punto sotto il mouse fisso
        scale_ratio = self.scale / old_scale
        self.pan_x = offset_x - (offset_x - self.pan_x) * scale_ratio
        self.pan_y = offset_y - (offset_y - self.pan_y) * scale_ratio
        
        self.update_display()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        
        # Al primo resize, calcola lo zoom per adattare l'immagine intera
        if self.first_resize:
            h, w = self.original_img.shape
            label_w = self.width()
            label_h = self.height()
            
            if label_w > 0 and label_h > 0:
                scale_w = label_w / w
                scale_h = label_h / h
                self.scale = min(scale_w, scale_h)
                self.first_resize = False
        
        self.update_display()
    
    def sync_view(self, other_label):
        """Sincronizza scale e pan con un'altra label"""
        other_label.scale = self.scale
        other_label.pan_x = self.pan_x
        other_label.pan_y = self.pan_y
        other_label.update_display()

class ImageComparisonWindow(QMainWindow):
    def __init__(self, img1, img2, title1="Originale", title2="Ricostruita"):
        super().__init__()
        
        self.setWindowTitle("JPEG Compression Viewer")
        self.setGeometry(100, 100, 1400, 700)
        
        # Crea widget centrale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Colonna sinistra (titolo + immagine 1)
        left_layout = QVBoxLayout()
        title_label_1 = QLabel(title1)
        title_label_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label1 = SyncedImageLabel(img1)
        left_layout.addWidget(title_label_1)
        left_layout.addWidget(self.label1)
        left_layout.setStretchFactor(self.label1, 1)
        
        # Colonna destra (titolo + immagine 2)
        right_layout = QVBoxLayout()
        title_label_2 = QLabel(title2)
        title_label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label2 = SyncedImageLabel(img2)
        right_layout.addWidget(title_label_2)
        right_layout.addWidget(self.label2)
        right_layout.setStretchFactor(self.label2, 1)
        
        # Aggiungi le due colonne al layout principale
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        
        # Sincronizzazione manuale tra i due
        original_move_1 = self.label1.mouseMoveEvent
        original_wheel_1 = self.label1.wheelEvent
        original_move_2 = self.label2.mouseMoveEvent
        original_wheel_2 = self.label2.wheelEvent
        
        def move_1(event):
            original_move_1(event)
            self.label1.sync_view(self.label2)
        
        def move_2(event):
            original_move_2(event)
            self.label2.sync_view(self.label1)
        
        def wheel_1(event):
            original_wheel_1(event)
            self.label1.sync_view(self.label2)
        
        def wheel_2(event):
            original_wheel_2(event)
            self.label2.sync_view(self.label1)
        
        self.label1.mouseMoveEvent = move_1
        self.label2.mouseMoveEvent = move_2
        self.label1.wheelEvent = wheel_1
        self.label2.wheelEvent = wheel_2

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
            title2=f"Ricostruita (N={N}, M={M})"
        )
        window.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    img = apri_immagine("bridge.bmp")
    JPEG(img, 16, 5, True, False)