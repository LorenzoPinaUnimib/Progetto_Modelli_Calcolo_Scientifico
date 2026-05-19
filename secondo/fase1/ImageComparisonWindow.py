# Funzione creata con IA per sincronizzare gli zoom e pan sulle due immagini
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from SyncedImageLabel import SyncedImageLabel

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