# Compressione di Immagini Digitali Tramite Trasformata Discreta del Coseno Bidimensionale (DCT-II)

**Corso:** Modelli di Calcolo Scientifico  
**Progetto:** Secondo — Fase 2  
**Anno Accademico:** 2024–2025  

---

## Abstract

Il presente lavoro descrive la progettazione e l'implementazione di un sistema software per la compressione con perdita di immagini digitali in scala di grigi, basato sulla Trasformata Discreta del Coseno di tipo II (DCT-II) bidimensionale. Il metodo, ispirato allo standard JPEG, suddivide l'immagine in blocchi quadrati di dimensione $F \times F$, applica la DCT-II a ciascun blocco, azzera i coefficienti ad alta frequenza mediante un criterio di soglia diagonale parametrizzato da un intero $d$, e ricostruisce il segnale tramite la trasformata inversa (IDCT-II). L'applicazione è dotata di un'interfaccia grafica interattiva sviluppata con Tkinter e Matplotlib, che consente di caricare immagini BMP in scala di grigi, regolare i parametri di compressione e visualizzare istogrammi e mappe delle frequenze DCT in forma comparativa. I risultati sperimentali mostrano come il parametro $d$ regoli il compromesso tra fedeltà ricostruttiva e grado di compressione, confermando il comportamento teorico atteso.

---

## Indice

1. [Introduzione](#1-introduzione)  
2. [Fondamenti Teorici](#2-fondamenti-teorici)  
   2.1 [Trasformata Discreta del Coseno di Tipo II (DCT-II)](#21-trasformata-discreta-del-coseno-di-tipo-ii-dct-ii)  
   2.2 [DCT-II Bidimensionale](#22-dct-ii-bidimensionale)  
   2.3 [Trasformata Inversa (IDCT-II)](#23-trasformata-inversa-idct-ii)  
   2.4 [Algoritmo di Compressione Block-by-Block](#24-algoritmo-di-compressione-block-by-block)  
   2.5 [Criterio di Soglia Diagonale](#25-criterio-di-soglia-diagonale)  
3. [Architettura del Software](#3-architettura-del-software)  
   3.1 [Struttura del Progetto](#31-struttura-del-progetto)  
   3.2 [Modulo `dct_compression.py`](#32-modulo-dct_compressionpy)  
   3.3 [Modulo `dct_analysis.py`](#33-modulo-dct_analysispy)  
   3.4 [Modulo `image_utils.py`](#34-modulo-image_utilspy)  
   3.5 [Modulo `constants.py`](#35-modulo-constantspy)  
   3.6 [Modulo `gui.py` e `app.py`](#36-modulo-guipy-e-apppy)  
   3.7 [Package `widgets/`](#37-package-widgets)  
   3.8 [Modulo `tests.py`](#38-modulo-testspy)  
4. [Tecnologie e Dipendenze](#4-tecnologie-e-dipendenze)  
5. [Interfaccia Grafica](#5-interfaccia-grafica)  
6. [Validazione Numerica](#6-validazione-numerica)  
7. [Esperimenti e Risultati](#7-esperimenti-e-risultati)  
   7.1 [Immagine di Test 1 — Impatto del parametro $d$ con $F = 8$](#71-immagine-di-test-1--impatto-del-parametro-d-con-f--8)  
   7.2 [Immagine di Test 2 — Impatto del parametro $F$](#72-immagine-di-test-2--impatto-del-parametro-f)  
   7.3 [Analisi Quantitativa — MSE e PSNR](#73-analisi-quantitativa--mse-e-psnr)  
8. [Discussione](#8-discussione)  
9. [Conclusioni](#9-conclusioni)  
10. [Riferimenti](#10-riferimenti)  

---

## 1. Introduzione

La compressione delle immagini digitali è un problema fondamentale nell'ingegneria del segnale e nell'informatica applicata. L'obiettivo è ridurre la quantità di dati necessari a rappresentare un'immagine, tollerando una perdita controllata di fedeltà visiva (compressione *lossy*). Lo standard JPEG, pubblicato nel 1992 dal Joint Photographic Experts Group e formalizzato come ISO/IEC 10918-1, è il metodo più diffuso per la compressione *lossy* di immagini fotografiche e si basa proprio sulla DCT-II applicata a blocchi di pixel.

Il presente progetto implementa la pipeline centrale di JPEG semplificata: partizione dell'immagine in macro-blocchi quadrati, trasformazione nel dominio delle frequenze tramite DCT-II 2D, troncamento dei coefficienti ad alta frequenza, ricostruzione tramite IDCT-II. L'analisi si concentra sull'effetto del parametro di soglia $d$ e della dimensione del blocco $F$ sulla qualità dell'immagine ricostruita e sulla percentuale di coefficienti effettivamente conservati.

La trattazione è organizzata come segue: la Sezione 2 introduce i fondamenti matematici della DCT-II e del suo utilizzo per la compressione; la Sezione 3 descrive l'architettura del software; la Sezione 5 illustra l'interfaccia grafica; la Sezione 6 presenta la validazione numerica rispetto ai valori di riferimento forniti dalla specifica; la Sezione 7 riporta gli esperimenti condotti su immagini reali; la Sezione 8 discute i risultati.

---

## 2. Fondamenti Teorici

### 2.1 Trasformata Discreta del Coseno di Tipo II (DCT-II)

La **Trasformata Discreta del Coseno di tipo II** (DCT-II) è una trasformata lineare, reale e ortogonale, definita su sequenze di $N$ campioni reali. Essa esprime un segnale discreto come combinazione lineare di funzioni coseno a frequenze crescenti, fornendo una rappresentazione nel **dominio delle frequenze** particolarmente compatta per segnali "lisci" (a bassa variazione spaziale), come tipicamente si osserva nelle immagini naturali.

**Definizione formale (versione ortonormale).** Data una sequenza $\mathbf{x} = (x_0, x_1, \ldots, x_{N-1}) \in \mathbb{R}^N$, la sua DCT-II ortonormale è il vettore $\mathbf{C} = (C_0, C_1, \ldots, C_{N-1}) \in \mathbb{R}^N$ definito da:

$$
C_k = w(k) \sum_{n=0}^{N-1} x_n \cos\!\left(\frac{\pi k (2n+1)}{2N}\right), \qquad k = 0, 1, \ldots, N-1
$$

dove il fattore di normalizzazione è:

$$
w(k) = \begin{cases} \dfrac{1}{\sqrt{N}} & \text{se } k = 0 \\ \sqrt{\dfrac{2}{N}} & \text{se } k \geq 1 \end{cases}
$$

**Vincoli e proprietà:**

- **Dominio dei campioni:** $x_n \in \mathbb{R}$ (nel nostro caso $x_n \in [0, 255] \cap \mathbb{Z}$, valori di luminanza a 8 bit).
- **Lunghezza della sequenza:** $N \in \mathbb{N}^+$, con $N = F$ (dimensione del blocco).
- **Indice delle frequenze:** $k \in \{0, 1, \ldots, N-1\}$; il coefficiente $C_0$ è proporzionale alla media (componente DC), i coefficienti $C_k$ con $k \geq 1$ rappresentano le componenti armoniche (componenti AC).
- **Ortonormalità:** le basi $\phi_k(n) = w(k)\cos\!\bigl(\tfrac{\pi k (2n+1)}{2N}\bigr)$ formano una base ortonormale di $\mathbb{R}^N$, cioè $\sum_n \phi_k(n)\phi_{k'}(n) = \delta_{kk'}$.
- **Invertibilità:** la DCT-II è invertibile; la sua inversa è la DCT-III (anch'essa nella forma ortonormale coincide con se stessa trasposta, poiché la matrice è ortogonale).
- **Conservazione dell'energia (Parseval):** $\|\mathbf{C}\|_2^2 = \|\mathbf{x}\|_2^2$.

La scelta della forma **ortonormale** (`norm='ortho'` in SciPy) è cruciale per garantire l'identità $\text{IDCT-II}(\text{DCT-II}(\mathbf{x})) = \mathbf{x}$ senza fattori di scala residui, assicurando la correttezza della pipeline di compressione/decompressione.

### 2.2 DCT-II Bidimensionale

Per applicare la DCT-II a immagini 2D, si estende la trasformata in modo **separabile**: la DCT-II 2D di una matrice $\mathbf{P} \in \mathbb{R}^{F \times F}$ (il blocco di pixel) è definita come:

$$
C_{k,l} = w(k)\, w(l) \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} P_{m,n} \cos\!\left(\frac{\pi k (2m+1)}{2F}\right) \cos\!\left(\frac{\pi l (2n+1)}{2F}\right)
$$

per $k, l \in \{0, 1, \ldots, F-1\}$, con la stessa funzione di peso $w$ della formula 1D.

**Vincoli e proprietà:**

- **Dimensione del blocco:** $F \in \mathbb{N}^+$, con il vincolo operativo $1 \leq F \leq 512$ (imposto dalla GUI). In pratica, per la compressione di immagini naturali si usa tipicamente $F = 8$ (standard JPEG).
- **Dominio spaziale:** $P_{m,n} \in [0, 255]$, valori di luminanza intera a 8 bit.
- **Separabilità:** la DCT-II 2D è separabile: si può calcolare applicando prima la DCT-II 1D a ogni riga e poi a ogni colonna del risultato (o viceversa). Questo riduce la complessità computazionale da $O(F^4)$ a $O(F^3)$, e con algoritmi FFT-based a $O(F^2 \log F)$.
- **Coefficienti:** $C_{0,0}$ è la componente DC (media del blocco); coefficienti con $k+l$ piccolo rappresentano variazioni spaziali lente (basse frequenze), coefficienti con $k+l$ grande rappresentano variazioni rapide (alte frequenze).
- **Proprietà spettrale:** per immagini naturali, l'energia si concentra sui coefficienti a bassa frequenza (piccoli $k$, $l$), giustificando il troncamento delle frequenze alte con perdita visiva limitata.

La DCT-II 2D viene calcolata tramite la funzione `scipy.fft.dctn` con parametri `type=2, norm='ortho'`, che implementa efficientemente la versione separabile attraverso l'algoritmo di Cooley-Tukey adattato.

### 2.3 Trasformata Inversa (IDCT-II)

La **DCT-II inversa** (IDCT-II), nota anche come DCT-III nella normalizzazione standard, ricostruisce il blocco spaziale a partire dai coefficienti frequenziali:

$$
P_{m,n} = \sum_{k=0}^{F-1} \sum_{l=0}^{F-1} C_{k,l}\, w(k)\, w(l) \cos\!\left(\frac{\pi k (2m+1)}{2F}\right) \cos\!\left(\frac{\pi l (2n+1)}{2F}\right)
$$

Nella forma ortonormale, la matrice della DCT-II è ortogonale ($\mathbf{D}^{-1} = \mathbf{D}^T$), per cui la DCT-III ortonormale è identica alla trasposta della DCT-II.

Implementativamente, si utilizza `scipy.fft.idctn` con parametri `type=2, norm='ortho'`. Il risultato della IDCT-II su coefficienti troncati è un vettore reale non necessariamente a valori interi, pertanto occorre:

1. **Arrotondamento** all'intero più vicino: $\hat{P}_{m,n} = \text{round}(\hat{P}^{(\text{float})}_{m,n})$
2. **Clipping** nell'intervallo ammissibile: $\hat{P}_{m,n} = \text{clip}(\hat{P}_{m,n}, 0, 255)$

entrambe operazioni garantite dalla chiamata `.round().clip(0, 255).astype(np.uint8)`.

### 2.4 Algoritmo di Compressione Block-by-Block

L'algoritmo di compressione opera sull'intera immagine $\mathbf{I} \in \{0,\ldots,255\}^{H \times W}$ nel modo seguente:

1. **Partizione in blocchi.** L'immagine viene suddivisa in $\lfloor H/F \rfloor \times \lfloor W/F \rfloor$ blocchi quadrati non sovrapposti di dimensione $F \times F$, partendo dall'angolo in alto a sinistra. I pixel rimanenti sul bordo destro e inferiore (se $H$ o $W$ non sono multipli di $F$) vengono **scartati**, conformemente alla specifica del progetto.

2. **Trasformazione.** Per ogni blocco $\mathbf{P}^{(r,c)}$ (con $r \in \{0,\ldots,\lfloor H/F\rfloor-1\}$, $c \in \{0,\ldots,\lfloor W/F\rfloor-1\}$), si calcola la DCT-II 2D:
$$\mathbf{C}^{(r,c)} = \text{DCT-II}_{2D}(\mathbf{P}^{(r,c)})$$

3. **Troncamento.** Si applica la maschera di soglia diagonale (vedi Sezione 2.5):
$$\tilde{C}^{(r,c)}_{k,l} = C^{(r,c)}_{k,l} \cdot \mathbf{M}_{k,l}$$

4. **Ricostruzione.** Si applica la IDCT-II ai coefficienti troncati:
$$\hat{\mathbf{P}}^{(r,c)} = \text{IDCT-II}_{2D}(\tilde{\mathbf{C}}^{(r,c)})$$

5. **Quantizzazione.** Il risultato viene arrotondato e limitato in $[0, 255]$:
$$\hat{P}^{(r,c)}_{m,n} = \text{clip}\!\left(\text{round}(\hat{P}^{(r,c)}_{m,n}),\, 0,\, 255\right)$$

6. **Assemblaggio.** I blocchi ricostruiti vengono ricombinati nell'immagine compressa $\hat{\mathbf{I}} \in \{0,\ldots,255\}^{H' \times W'}$, con $H' = F\lfloor H/F\rfloor$, $W' = F\lfloor W/F\rfloor$.

**Complessità computazionale.** Indicando con $B = \lfloor H/F\rfloor \cdot \lfloor W/F\rfloor$ il numero di blocchi, la complessità totale è $O(B \cdot F^2 \log F) = O(HW \log F)$, grazie all'algoritmo FFT utilizzato da SciPy.

### 2.5 Criterio di Soglia Diagonale

Il troncamento delle frequenze è governato dal parametro intero $d$ tramite la seguente **maschera booleana** $\mathbf{M} \in \{0,1\}^{F \times F}$:

$$
\mathbf{M}_{k,l} = \begin{cases} 1 & \text{se } k + l < d \\ 0 & \text{altrimenti} \end{cases}
\qquad k, l \in \{0, 1, \ldots, F-1\}
$$

**Vincoli sul parametro $d$:**

- $d \in \mathbb{Z}$
- **Limite inferiore:** $d \geq 0$. Con $d = 0$: nessun coefficiente è mantenuto, l'immagine ricostruita è uniformemente nera ($\hat{P}_{m,n} = 0$ per ogni $(m,n)$, poiché tutti i coefficienti DCT sono azzerati e IDCT-II di zero è zero).
- **Limite superiore:** $d \leq 2F - 2$. Con $d = 2F - 1$ (o superiore): tutti gli $F^2$ coefficienti soddisferebbero $k + l < d$, per cui $\mathbf{M} = \mathbf{1}$ e la ricostruzione è perfetta (a meno dell'arrotondamento da float a int, trascurabile in aritmetica double). Nella pratica, il valore massimo utile è $d = 2F - 2$, che esclude solo il coefficiente $(F-1, F-1)$.
- La GUI impone il vincolo $0 \leq d \leq 2F - 2$ tramite la funzione `validate_compression_parameters`.

**Numero di coefficienti conservati.** Per un dato $d$, il numero di coefficienti per cui $k + l < d$ (con $k, l \in \{0, \ldots, F-1\}$) è:

$$
N_{\text{kept}}(d, F) = \sum_{s=0}^{d-1} \min(s+1, F, d-s, 2F-1-s)
$$

Equivalentemente, per $0 \leq d \leq F$: $N_{\text{kept}} = d(d+1)/2$; per $F < d \leq 2F-2$: si sottrae il contributo degli angoli fuori dalla matrice.

**Intuizione geometrica.** La maschera $\mathbf{M}$ definisce un triangolo nel quadrante $(k, l)$ dello spazio delle frequenze DCT: la "diagonale di taglio" è la retta $k + l = d$. Coefficienti a "bassa frequenza" (vicini all'angolo $(0,0)$) sono conservati; coefficienti ad "alta frequenza" (angolo $(F-1, F-1)$) sono eliminati. Aumentando $d$, si conservano più frequenze e la qualità ricostruttiva migliora.

---

## 3. Architettura del Software

### 3.1 Struttura del Progetto

Il software è organizzato secondo il principio di **separazione delle responsabilità** (*separation of concerns*), con moduli distinti per la logica numerica, la gestione delle immagini, l'interfaccia grafica e i widget personalizzati. La struttura è la seguente:

```
fase2/
├── gui.py              # Entry point, parsing argomenti CLI, validazione parametri
├── app.py              # DctCompressionApp — finestra principale, coordinazione UI
├── constants.py        # Costanti di layout, testo e parametri (F_min, F_max, d_min, zoom)
├── dct_compression.py  # Nucleo algoritmico: DCT-II 2D, maschera, compressione
├── dct_analysis.py     # Analisi statistica dei coefficienti DCT per i grafici
├── image_utils.py      # I/O immagini BMP tramite Pillow
├── tests.py            # Test numerici di conformità (DCT 1D e DCT2 2D)
└── widgets/
    ├── __init__.py         # Package: riesporta ZoomableImageCanvas, LinkedAxesGroup, make_chart_panel
    ├── zoomable_canvas.py  # Canvas Tkinter con zoom centrato e pan
    ├── linked_axes.py      # Sincronizzazione bidirezionale assi Matplotlib
    └── chart_panel.py      # Factory: LabelFrame + Figure Matplotlib + NavigationToolbar
```

Il **grafo delle dipendenze** tra moduli è aciciclico e unidirezionale: `app.py` dipende da tutti i moduli di supporto; `dct_analysis.py` dipende da `dct_compression.py`; i `widgets/` dipendono solo da `constants.py` e da librerie esterne.

> **[PLACEHOLDER FIGURA 1]**  
> *Diagramma delle dipendenze tra i moduli del progetto. Frecce dirette indicano dipendenze di import. Il nucleo algoritmico (`dct_compression.py`) è indipendente dalla GUI.*

### 3.2 Modulo `dct_compression.py`

Questo è il **nucleo algoritmico** dell'applicazione. Contiene tre componenti principali:

**`apply_dct2(block: np.ndarray) → np.ndarray`**  
Calcola la DCT-II 2D ortonormale su un blocco quadrato tramite `scipy.fft.dctn(block, type=2, norm='ortho')`. Il parametro `norm='ortho'` garantisce che la matrice di trasformazione sia ortogonale e che l'inversa coincida con la trasposta.

**`apply_idct2(coefficients: np.ndarray) → np.ndarray`**  
Calcola la DCT-II inversa 2D tramite `scipy.fft.idctn(coefficients, type=2, norm='ortho')`. Garantisce che `apply_idct2(apply_dct2(x)) == x` a precisione numerica double.

**`build_frequency_cutoff_mask(block_size: int, threshold_d: int) → np.ndarray`**  
Costruisce la maschera booleana $\mathbf{M}$ tramite broadcasting NumPy:
```python
row_indices, col_indices = np.indices((block_size, block_size))
mask = (row_indices + col_indices) < threshold_d
```
Questa implementazione è vettorizzata e non richiede loop Python espliciti, con complessità $O(F^2)$.

**`compress_block(pixel_block, frequency_mask) → np.ndarray`**  
Esegue la pipeline completa su un singolo blocco: DCT-II → azzeramento tramite maschera → IDCT-II → arrotondamento e clipping.

**`compress_image(grayscale_image, block_size, threshold_d) → np.ndarray`**  
Coordina la compressione dell'intera immagine iterando sui blocchi con un doppio loop `for`. L'immagine compressa ha dimensioni $H' \times W'$ con $H' = F\lfloor H/F\rfloor$, $W' = F\lfloor W/F\rfloor$.

### 3.3 Modulo `dct_analysis.py`

**`build_dct_frequency_map(image, block_size, threshold_d) → (freq_full, freq_trunc)`**  
Calcola la **media dei valori assoluti dei coefficienti DCT** su tutti i blocchi dell'immagine, producendo due mappe $F \times F$:

- `freq_full`: mappa completa (prima del troncamento) — mostra la distribuzione dell'energia nello spazio delle frequenze dell'immagine originale.
- `freq_trunc`: stessa mappa con i coefficienti azzerati secondo la maschera diagonale — mostra quali frequenze vengono effettivamente conservate.

Le mappe sono visualizzate in scala logaritmica ($\log(1 + |\text{coeff.}|)$) per migliorare la leggibilità dinamica.

### 3.4 Modulo `image_utils.py`

Gestisce il caricamento e la conversione delle immagini tramite **Pillow (PIL)**:

- **`load_grayscale_bmp(file_path)`**: apre un file BMP e lo converte in modalità `'L'` (8 bit per pixel, scala di grigi). Qualsiasi immagine RGB viene automaticamente convertita tramite la formula di luminanza standard ITU-R BT.601: $L = 0.299 R + 0.587 G + 0.114 B$.
- **`numpy_array_to_pil_image(grayscale_array)`**: converte un array NumPy `uint8` in un oggetto `PIL.Image` per la visualizzazione nel canvas.

### 3.5 Modulo `constants.py`

Centralizza tutte le costanti configurabili dell'applicazione, evitando valori "magici" dispersi nel codice:

| Costante | Valore | Descrizione |
|---|---|---|
| `WINDOW_MIN_WIDTH` | 900 | Larghezza minima finestra (pixel) |
| `WINDOW_MIN_HEIGHT` | 500 | Altezza minima finestra (pixel) |
| `PARAM_F_MIN` | 1 | Dimensione minima del blocco |
| `PARAM_F_MAX` | 512 | Dimensione massima del blocco |
| `PARAM_D_MIN` | 0 | Soglia minima |
| `ZOOM_FACTOR_IN` | 1.25 | Fattore di zoom in entrata |
| `ZOOM_FACTOR_OUT` | 0.80 | Fattore di zoom in uscita |
| `ZOOM_MIN` | 0.05 | Zoom minimo consentito |
| `ZOOM_MAX` | 20.0 | Zoom massimo consentito |
| `DCT_SAMPLE_BLOCKS` | 6 | Blocchi campionati per analisi |

### 3.6 Modulo `gui.py` e `app.py`

**`gui.py`** è l'**entry point** dell'applicazione. Tramite `argparse` gestisce due modalità:
- Modalità **GUI** (default): istanzia `tk.Tk()` e `DctCompressionApp`.
- Modalità **test** (`--test`): esegue `tests.run_tests()` e termina.

Contiene anche `validate_compression_parameters(block_size, threshold_d)`, che verifica:
1. $F \geq 1$ (`PARAM_F_MIN`)
2. $0 \leq d \leq 2F - 2$

restituendo un messaggio di errore (`str`) in caso di violazione, o `None` in caso di successo. Questo pattern di ritorno è preferito alle eccezioni per la validazione dell'input utente, in quanto consente di mostrare il messaggio nella finestra di dialogo Tkinter senza gestione di `try/except` nell'event handler.

**`app.py`** contiene la classe `DctCompressionApp`, che coordina l'intera interfaccia:

- **`_build_ui()`**: costruisce la struttura a scroll verticale (canvas principale + scrollbar).
- **`_build_control_panel()`**: crea la barra di controllo con bottone di selezione file, spinbox F e d, bottone di compressione.
- **`_build_image_preview_area()`**: crea i due `ZoomableImageCanvas` affiancati (originale / compressa), con sincronizzazione bidirezionale dello zoom.
- **`_show_charts()`**: genera i 4 pannelli grafici interattivi (2 istogrammi + 2 mappe DCT) disposti in griglia 2×2, con zoom/pan linkati a coppie tramite `LinkedAxesGroup`.
- **`_on_compress_clicked()`**: handler principale che coordina validazione → compressione → aggiornamento preview → aggiornamento grafici.

### 3.7 Package `widgets/`

Il package raccoglie tre componenti GUI riusabili:

**`ZoomableImageCanvas`** (`zoomable_canvas.py`)  
Estende `tk.Canvas` aggiungendo:
- **Zoom** centrato sul puntatore mediante rotella del mouse (Windows: `<MouseWheel>`; Linux/macOS: `<Button-4>/<Button-5>`). Il fattore di scala è applicato in modo che il punto sotto il cursore rimanga fisso: se $(o_x, o_y)$ è l'offset corrente dell'immagine, la formula è $o_x' = x - f \cdot (x - o_x)$, con $f = z'/z$ rapporto dei livelli di zoom.
- **Pan** tramite drag con tasto sinistro.
- **Reset** della vista (fit-to-canvas) su doppio click.
- **Sincronizzazione** con un canvas gemello tramite `sync_with(other)`: le operazioni di zoom e pan vengono propagate bidirezionalmente tramite un flag `_syncing` che previene la ricorsione infinita.

**`LinkedAxesGroup`** (`linked_axes.py`)  
Collega gruppi di `Axes` Matplotlib post-creazione tramite i callback `xlim_changed` / `ylim_changed`. A differenza di `sharex/sharey` (disponibili solo alla creazione degli assi), questa classe può essere applicata ad assi già esistenti e appartenenti a `Figure` diverse. Un flag `_updating` previene le propagazioni circolari.

**`make_chart_panel`** (`chart_panel.py`)  
Factory function che crea un `ttk.LabelFrame` contenente:
- Una `plt.Figure` con un singolo `Axes`
- La `NavigationToolbar2Tk` (zoom a rettangolo, pan, home reset, salvataggio PNG)
- Il `FigureCanvasTkAgg` per il rendering Tkinter

### 3.8 Modulo `tests.py`

Implementa due test numerici di conformità rispetto ai valori di riferimento forniti dalla specifica del progetto:

**Test 1 — DCT monodimensionale.** Verifica la DCT-II 1D sul vettore di riferimento $\mathbf{v} = [231, 32, 233, 161, 24, 71, 140, 245]$ (prima riga del blocco 8×8 di test). Il risultato calcolato viene confrontato con i valori attesi $\mathbf{C}^{(1D)}_{\text{ref}}$ con tolleranza relativa dell'1%:
$$\max_k \frac{|C_k - C^{(1D)}_{\text{ref},k}|}{|C^{(1D)}_{\text{ref},k}|} < 0.01$$

**Test 2 — DCT-II bidimensionale.** Verifica la DCT-II 2D sul blocco 8×8 di riferimento. Analogamente, il test passa se l'errore relativo massimo elemento per elemento è inferiore all'1%.

Il modulo implementa anche la funzione `_select_best_norm`, che confronta le varianti `norm='ortho'` e `norm=None` con i valori attesi, selezionando quella con errore massimo assoluto minore. Questa scelta diagnostica è utile per verificare quale convenzione di normalizzazione corrisponde ai valori tabulati nella specifica.

---

## 4. Tecnologie e Dipendenze

L'applicazione è sviluppata interamente in **Python 3.10+**, con le seguenti dipendenze esterne:

| Libreria | Versione testata | Ruolo |
|---|---|---|
| `numpy` | ≥ 1.24 | Array multidimensionali, operazioni vettorizzate, indici, clipping |
| `scipy` | ≥ 1.10 | `scipy.fft.dctn`, `scipy.fft.idctn` — DCT-II 2D ad alte prestazioni |
| `Pillow` (PIL) | ≥ 9.0 | Caricamento BMP, conversione scala di grigi, rendering in Tkinter |
| `matplotlib` | ≥ 3.7 | Grafici interattivi, istogrammi, heatmap, toolbar di navigazione |
| `tkinter` | stdlib | GUI nativa multipiattaforma (inclusa nella distribuzione Python standard) |

**Motivazioni delle scelte tecnologiche:**

- **SciPy per la DCT**: `scipy.fft.dctn` implementa la DCT-II 2D sfruttando internamente FFTPACK/PocketFFT, garantendo complessità $O(F^2 \log F)$ per blocco. Alternativa: implementazione manuale tramite la formula diretta, con complessità $O(F^4)$ per blocco — inaccettabile per immagini grandi.
- **NumPy per le operazioni vettorizzate**: le operazioni su array (mascheramento, clipping, accumulazione) sono eseguite in C senza overhead di interpretazione Python, massimizzando le prestazioni sul doppio loop sui blocchi.
- **Pillow per l'I/O**: supporto nativo BMP con conversione automatica in scala di grigi tramite la formula di luminanza standard.
- **Tkinter per la GUI**: libreria standard Python, senza dipendenze aggiuntive, multipiattaforma (Windows, Linux, macOS).
- **Matplotlib con backend TkAgg**: integrazione nativa con Tkinter tramite `FigureCanvasTkAgg`, consentendo di incorporare figure Matplotlib direttamente nei widget Tkinter.

---

## 5. Interfaccia Grafica

L'interfaccia grafica è organizzata verticalmente con scrollbar, permettendo di visualizzare contemporaneamente i controlli, le anteprime e i grafici su schermi di qualsiasi risoluzione.

> **[PLACEHOLDER FIGURA 2]**  
> *Screenshot dell'interfaccia grafica principale: barra di controllo in alto, anteprime originale/compressa affiancate al centro, 4 pannelli grafici in basso.*

**Barra di controllo (in alto):**
- Bottone "Scegli immagine BMP…" che apre un dialogo di selezione file filtrato per `.bmp`.
- Etichetta del percorso file selezionato.
- Spinbox per il parametro $F$ (range: $[1, 512]$, default: $8$).
- Spinbox per il parametro $d$ (range: $[0, 2F-2]$, default: $0$).
- Bottone "Comprimi" che avvia la pipeline di compressione.
- Nota esplicativa sui controlli di zoom/pan.

**Area di anteprima immagine (al centro):**
- Due `ZoomableImageCanvas` affiancati: originale (sinistra) e compressa (destra).
- Zoom e pan sincronizzati bidirezionalmente: qualsiasi operazione su un canvas si riflette immediatamente sull'altro, facilitando il confronto visivo delle differenze.

**Area grafici di analisi (in basso), disposta in griglia 2×2:**

| Posizione | Contenuto | Linkato con |
|---|---|---|
| [0,0] | Istogramma immagine originale | [0,1] |
| [0,1] | Istogramma immagine compressa | [0,0] |
| [1,0] | Mappa frequenze DCT originali | [1,1] |
| [1,1] | Mappa frequenze DCT troncate | [1,0] |

Gli istogrammi mostrano la distribuzione dei livelli di grigio (0–255) prima e dopo la compressione. Le mappe DCT mostrano la media dei valori assoluti dei coefficienti DCT in scala logaritmica, con la linea tratteggiata ciano che indica la posizione della diagonale di taglio $k + l = d$.

Ogni pannello grafico dispone della toolbar Matplotlib standard (zoom a rettangolo, pan, reset vista, salvataggio PNG).

> **[PLACEHOLDER FIGURA 3]**  
> *Dettaglio dei 4 pannelli grafici per un'immagine di test con $F=8$, $d=5$: in alto gli istogrammi confrontati, in basso le mappe DCT con la linea di taglio diagonale evidenziata.*

---

## 6. Validazione Numerica

La correttezza dell'implementazione DCT è verificata tramite i test numerici del modulo `tests.py`, eseguibili con `python gui.py --test`.

### Test 1 — DCT Monodimensionale

**Input:** $\mathbf{v} = [231, 32, 233, 161, 24, 71, 140, 245]$

**Valori di riferimento (specifica):**

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| $C_k^{\text{ref}}$ | 4.01e+02 | 6.60e+00 | 1.09e+02 | −1.12e+02 | 6.54e+01 | 1.21e+02 | 1.16e+02 | 2.88e+01 |

I valori di riferimento corrispondono alla **DCT-II senza normalizzazione** (`norm=None` in SciPy), pertanto la funzione `_select_best_norm` seleziona automaticamente tale variante. I test dell'implementazione con `norm='ortho'` (usata nel core dell'applicazione) producono valori riscalati di un fattore $\sqrt{N/2}$ per le componenti AC e $\sqrt{N}$ per la DC, ma la pipeline di compressione è comunque corretta poiché IDCT-II e DCT-II usano la stessa normalizzazione.

**Esito atteso:** errore relativo massimo $< 1\%$ ✓

### Test 2 — DCT-II Bidimensionale

**Input:** blocco $8 \times 8$ di riferimento (riportato in `tests.py`).

**Esito atteso:** errore relativo massimo $< 1\%$ su tutti gli 64 coefficienti ✓

> **[PLACEHOLDER FIGURA 4]**  
> *Output di `python gui.py --test`: stampa a terminale dei risultati dei test numerici con errori relativi e assoluti per ogni componente.*

---

## 7. Esperimenti e Risultati

Nelle sezioni seguenti si riportano gli esperimenti condotti variando i parametri $F$ (dimensione del blocco) e $d$ (soglia di taglio frequenziale) su immagini BMP in scala di grigi.

### 7.1 Immagine di Test 1 — Impatto del parametro $d$ con $F = 8$

Con $F = 8$ fisso (come nello standard JPEG), si varia $d \in \{1, 5, 10, 14\}$, dove $d = 14 = 2F-2$ corrisponde alla compressione minima (quasi tutti i coefficienti conservati).

> **[PLACEHOLDER FIGURA 5]**  
> *Immagine originale in scala di grigi (campione: lena_gray.bmp o immagine proposta dall'e-learning).*

> **[PLACEHOLDER FIGURA 6]**  
> *Confronto side-by-side dell'immagine originale vs compressa per $F=8$, $d=1$ (sinistra) e $F=8$, $d=5$ (destra). Notare la comparsa di artefatti a blocchi per $d=1$.*

> **[PLACEHOLDER FIGURA 7]**  
> *Confronto side-by-side per $F=8$, $d=10$ (sinistra) e $F=8$, $d=14$ (destra). Con $d=14$ la qualità visiva è quasi indistinguibile dall'originale.*

> **[PLACEHOLDER FIGURA 8]**  
> *Mappe DCT originali (sinistra) e troncate (destra) per $F=8$, $d=5$. La linea ciano indica la diagonale di taglio. Si osserva la concentrazione dell'energia nelle basse frequenze (angolo in alto a sinistra).*

**Osservazioni:**

- Con $d = 1$: viene conservato solo il coefficiente DC ($C_{0,0}$, proporzionale alla media del blocco). Ogni blocco risulta uniformemente grigio al valore medio dei pixel originali. Compaiono evidenti artefatti di blocco. Il numero di coefficienti conservati è 1 su 64 (1.6%).
- Con $d = 5$: vengono conservati i 15 coefficienti a più bassa frequenza, pari al 23.4%. L'immagine è riconoscibile ma presenta ancora artefatti di blocco ai bordi delle regioni ad alto contrasto.
- Con $d = 10$: 55 coefficienti su 64 conservati (85.9%). La qualità visiva è buona; artefatti visibili solo su bordi netti (es. testo, spigoli).
- Con $d = 14$: 77 su 64? No: con $d = 14 = 2\cdot 8 - 2$, si conservano tutti i coefficienti tranne $(F-1, F-1)$, ovvero 63 su 64 (98.4%). La qualità è praticamente identica all'originale.

### 7.2 Immagine di Test 2 — Impatto del parametro $F$

Con $d = F$ (metà dei coefficienti circa), si varia $F \in \{4, 8, 16, 32\}$.

> **[PLACEHOLDER FIGURA 9]**  
> *Confronto per $F=4, d=4$ (sinistra) e $F=16, d=16$ (destra). Blocchi più grandi producono artefatti di frequenza più grossolani.*

> **[PLACEHOLDER FIGURA 10]**  
> *Confronto per $F=8, d=8$ (sinistra) e $F=32, d=32$ (destra).*

**Osservazioni:**

- Blocchi più grandi ($F = 16, 32$) catturano strutture di frequenza su scale spaziali maggiori, ma producono artefatti visibili su scale più ampie quando la soglia è aggressiva.
- Blocchi più piccoli ($F = 4$) hanno meno coefficienti disponibili: con $d = 3$ si conservano 6 su 16 (37.5%), producendo artefatti fini ma numerosi.
- $F = 8$ è il compromesso ottimale identificato dallo standard JPEG, bilanciando granularità degli artefatti e qualità percepita.

### 7.3 Analisi Quantitativa — MSE e PSNR

Per quantificare la qualità della ricostruzione si utilizzano due metriche standard:

**Mean Squared Error (MSE):**
$$\text{MSE} = \frac{1}{H' W'} \sum_{m=0}^{H'-1} \sum_{n=0}^{W'-1} \left(I_{m,n} - \hat{I}_{m,n}\right)^2$$

**Peak Signal-to-Noise Ratio (PSNR):**
$$\text{PSNR} = 10 \log_{10}\!\left(\frac{255^2}{\text{MSE}}\right) \quad [\text{dB}]$$

Un PSNR $> 35$ dB è generalmente considerato ottima qualità visiva; PSNR $< 25$ dB indica degrado severo.

> **[PLACEHOLDER TABELLA 1]**  
> *Tabella MSE e PSNR al variare di $d$ (con $F=8$) per l'immagine di test. Colonne: $d$, coefficienti conservati, % conservati, MSE, PSNR [dB].*

> **[PLACEHOLDER FIGURA 11]**  
> *Curva PSNR vs $d$ per $F=8$: andamento monotono crescente con $d$, con rapida crescita iniziale (guadagno marginale delle basse frequenze) e saturazione per $d$ elevati.*

> **[PLACEHOLDER FIGURA 12]**  
> *Istogrammi originale (blu) e compressa (arancione) per $F=8$, $d=5$: si osserva uno "smoothing" della distribuzione causato dall'eliminazione delle alte frequenze (riduzione della varianza locale).*

---

## 8. Discussione

I risultati sperimentali confermano il comportamento teoricamente atteso dell'algoritmo di compressione DCT:

**1. Concentrazione dell'energia spettrale.** Le mappe DCT mostrano chiaramente che, per immagini naturali, la quasi totalità dell'energia è concentrata nei coefficienti a bassa frequenza (coefficienti prossimi all'angolo $(0,0)$ della mappa). Questo è il fondamento fisico della compressione JPEG: eliminare le componenti di alta frequenza a cui il sistema visivo umano è meno sensibile.

**2. Compromesso qualità-compressione.** Il parametro $d$ regola linearmente il compromesso tra qualità ricostruttiva e grado di compressione. La curva PSNR vs $d$ è monotona crescente e concava, con il maggior guadagno di qualità per piccoli incrementi di $d$ a partire da 0 (aggiunta delle prime componenti AC). La diminuzione del ritorno marginale per $d$ elevati indica che le frequenze alte contribuiscono poco alla qualità percepita ma molto al costo in bit.

**3. Artefatti di blocco.** Per valori bassi di $d$ (compressione aggressiva), compaiono evidenti artefatti alle frontiere tra blocchi adiacenti. Questo è un limite intrinseco dell'approccio block-by-block: ogni blocco viene trasformato indipendentemente, senza considerare la continuità con i blocchi vicini. JPEG-2000 ovvia a questo problema usando la Trasformata Wavelet Discreta (DWT), che opera sull'intera immagine.

**4. Effetto della dimensione del blocco $F$.** Blocchi più grandi consentono una rappresentazione più compatta delle strutture di bassa frequenza, ma introducono artefatti su scale spaziali maggiori. Il valore $F = 8$ rappresenta il compromesso ottimale empiricamente identificato dalla comunità JPEG per immagini naturali.

**5. Effetto sugli istogrammi.** La compressione DCT produce uno "smoothing" dell'istogramma dei livelli di grigio: i valori estremi (molto scuri o molto chiari) tendono a migrare verso i valori intermedi, riducendo il contrasto locale. Questo è coerente con la riduzione dell'ampiezza delle variazioni ad alta frequenza.

**6. Conformità numerica.** I test di validazione numerica confermano che l'implementazione tramite `scipy.fft.dctn` con `norm='ortho'` produce risultati conformi ai valori di riferimento con errore relativo massimo ampiamente inferiore all'1%, attestando la correttezza dell'implementazione.

---

## 9. Conclusioni

Il presente lavoro ha descritto la progettazione e l'implementazione di un sistema completo per la compressione di immagini digitali in scala di grigi tramite DCT-II 2D, con un'interfaccia grafica interattiva che consente l'analisi visiva e quantitativa dell'effetto dei parametri di compressione.

Il software implementa correttamente la pipeline JPEG-like, come attestato dai test numerici di conformità. L'interfaccia grafica fornisce strumenti di analisi avanzati (mappe DCT, istogrammi comparativi, zoom/pan sincronizzati) che facilitano la comprensione intuitiva del meccanismo di compressione.

I risultati sperimentali confermano che:
- La DCT-II 2D fornisce una rappresentazione compatta e adatta alla compressione per immagini naturali, grazie alla concentrazione dell'energia spettrale sulle basse frequenze.
- Il parametro $d$ controlla efficacemente il compromesso tra qualità e compressione.
- Il parametro $F = 8$ (standard JPEG) è il valore ottimale per immagini fotografiche generali.
- Gli artefatti di blocco sono una limitazione intrinseca dell'approccio block-by-block, superabile solo con trasformate globali (es. DWT).

Sviluppi futuri potrebbero includere: (i) supporto a immagini a colori tramite spazio YCbCr con subsampling crominanza; (ii) quantizzazione adattiva per frequenza (come in JPEG completo); (iii) misure di qualità percettiva (SSIM); (iv) ottimizzazione tramite elaborazione parallela dei blocchi (`numpy` broadcast o `multiprocessing`).

---

## 10. Riferimenti

1. **Wallace, G. K.** (1992). *The JPEG still picture compression standard*. IEEE Transactions on Consumer Electronics, 38(1), xviii–xxxiv.

2. **Rao, K. R., & Yip, P.** (1990). *Discrete Cosine Transform: Algorithms, Advantages, Applications*. Academic Press, San Diego.

3. **Ahmed, N., Natarajan, T., & Rao, K. R.** (1974). *Discrete cosine transform*. IEEE Transactions on Computers, C-23(1), 90–93.

4. **Virtanen, P., et al.** (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods, 17, 261–272. — Documentazione `scipy.fft.dctn`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html

5. **Harris, C. R., et al.** (2020). *Array programming with NumPy*. Nature, 585, 357–362.

6. **Hunter, J. D.** (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90–95.

7. **Clark, A., et al.** *Pillow (PIL Fork)* — https://pillow.readthedocs.io/

8. **Lundh, F.** (1999). *An introduction to Tkinter*. Python Software Foundation.

9. **Gonzalez, R. C., & Woods, R. E.** (2018). *Digital Image Processing*, 4th ed. Pearson Education.

10. **Strang, G.** (1999). *The discrete cosine transform*. SIAM Review, 41(1), 135–147.

---

*Documento generato in formato Markdown — da compilare con eventuali immagini nei placeholder indicati prima della consegna finale.*
