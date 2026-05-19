# Funzione creata con IA per sincronizzare gli zoom e pan sulle due immagini
import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

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