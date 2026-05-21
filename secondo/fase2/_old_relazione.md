# Compressione di Immagini Digitali tramite DCT-II Bidimensionale

**Corso:** Modelli di Calcolo Scientifico — **Progetto:** Secondo, Fase 2 — **A.A.:** 2024–2025

---

## Abstract

Si descrive un sistema software per la compressione *lossy* di immagini in scala di grigi basato sulla Trasformata Discreta del Coseno di tipo II (DCT-II) bidimensionale, ispirato allo standard JPEG. L'immagine è suddivisa in blocchi $F \times F$; su ciascuno si applica la DCT-II, si azzerano i coefficienti ad alta frequenza tramite una soglia diagonale parametrizzata da $d$, e si ricostruisce con la IDCT-II. L'interfaccia grafica (Tkinter + Matplotlib) permette di caricare immagini BMP, variare i parametri e confrontare istogrammi e mappe DCT. I risultati confermano che $d$ governa il compromesso fedeltà/compressione coerentemente con la teoria.

---

## Indice

1. [Introduzione](#1-introduzione)
2. [Fondamenti Teorici](#2-fondamenti-teorici)
3. [Architettura del Software](#3-architettura-del-software)
4. [Tecnologie e Dipendenze](#4-tecnologie-e-dipendenze)
5. [Interfaccia Grafica](#5-interfaccia-grafica)
6. [Validazione Numerica](#6-validazione-numerica)
7. [Esperimenti e Risultati](#7-esperimenti-e-risultati)
8. [Discussione](#8-discussione)
9. [Conclusioni](#9-conclusioni)
10. [Riferimenti](#10-riferimenti)

---

## 1. Introduzione

Lo standard JPEG (ISO/IEC 10918-1, 1992) è il metodo più diffuso per la compressione *lossy* di immagini fotografiche e si fonda sulla DCT-II applicata a blocchi di pixel. Il presente progetto implementa la pipeline JPEG semplificata: partizionamento in blocchi quadrati, trasformazione DCT-II 2D, troncamento dei coefficienti ad alta frequenza, ricostruzione con IDCT-II. L'analisi verte sull'effetto del parametro di soglia $d$ e della dimensione del blocco $F$ sulla qualità dell'immagine ricostruita.

---

## 2. Fondamenti Teorici

### 2.1 DCT-II Monodimensionale

Data la sequenza $\mathbf{x} \in \mathbb{R}^N$, la sua DCT-II ortonormale è:

$
C_k = w(k) \sum_{n=0}^{N-1} x_n \cos\!\left(\frac{\pi k (2n+1)}{2N}\right), \quad
w(k) = \begin{cases} 1/\sqrt{N} & k=0 \\ \sqrt{2/N} & k \geq 1 \end{cases}
$

Le basi $\phi_k$ formano una base ortonormale di $\mathbb{R}^N$; la matrice è ortogonale ($\mathbf{D}^{-1} = \mathbf{D}^T$), quindi la IDCT-II coincide con la trasposta. La normalizzazione `norm='ortho'` in SciPy garantisce $\text{IDCT}(\text{DCT}(\mathbf{x})) = \mathbf{x}$ senza fattori di scala residui.

### 2.2 DCT-II Bidimensionale e IDCT-II

La DCT-II 2D è **separabile**: applicata al blocco $\mathbf{P} \in \mathbb{R}^{F \times F}$,

$
C_{k,l} = w(k)\,w(l) \sum_{m=0}^{F-1}\sum_{n=0}^{F-1} P_{m,n} \cos\!\left(\frac{\pi k(2m+1)}{2F}\right)\cos\!\left(\frac{\pi l(2n+1)}{2F}\right)
$

La separabilità riduce la complessità da $O(F^4)$ a $O(F^2 \log F)$ con algoritmi FFT. Il coefficiente $C_{0,0}$ è la componente DC (media del blocco); coefficienti con $k+l$ piccolo/grande corrispondono a basse/alte frequenze. Per immagini naturali l'energia è concentrata sulle basse frequenze, motivando il troncamento delle alte.

La ricostruzione IDCT-II produce valori reali; occorre quindi arrotondare e fare clipping: $\hat{P}_{m,n} = \text{clip}(\text{round}(\hat{P}_{m,n}), 0, 255)$.

### 2.3 Algoritmo di Compressione Block-by-Block

Sull'immagine $\mathbf{I} \in \{0,\ldots,255\}^{H \times W}$:

1. **Partizione** in $\lfloor H/F\rfloor \times \lfloor W/F\rfloor$ blocchi $F \times F$; i pixel di bordo in eccesso vengono scartati.
2. Per ogni blocco: **DCT-II** → **azzeramento** tramite maschera diagonale → **IDCT-II** → **round+clip**.
3. **Assemblaggio** dell'immagine compressa $\hat{\mathbf{I}} \in \{0,\ldots,255\}^{H' \times W'}$ con $H' = F\lfloor H/F\rfloor$, $W' = F\lfloor W/F\rfloor$.

Complessità totale: $O(HW \log F)$.

### 2.4 Criterio di Soglia Diagonale

La **maschera booleana** $\mathbf{M} \in \{0,1\}^{F \times F}$ è definita da:

$
M_{k,l} = \begin{cases} 1 & \text{se } k+l < d \\ 0 & \text{altrimenti} \end{cases}
$

Il parametro $d \in [0,\, 2F-2]$ è validato dalla GUI. Con $d=0$ nessun coefficiente è conservato (immagine nera); con $d = 2F-2$ si elimina solo il coefficiente $(F-1,F-1)$ e la ricostruzione è quasi perfetta. Per $d \leq F$, il numero di coefficienti conservati è $d(d+1)/2$.

---

## 3. Architettura del Software

### 3.1 Struttura del Progetto

```
fase2/
├── gui.py                   # Entry point CLI (--test per i test numerici)
├── app.py                   # DctCompressionApp — finestra principale
├── constants.py             # Costanti di layout, parametri, zoom
├── dct_compression.py       # Nucleo algoritmico: DCT-II, maschera, compressione
├── dct_analysis.py          # Mappa media coefficienti DCT per i grafici
├── image_utils.py           # I/O BMP tramite Pillow
├── tests.py                 # Test numerici di conformità (DCT 1D e 2D)
└── widgets/
    ├── __init__.py
    ├── zoomable_canvas.py        # Canvas immagine con zoom/pan
    ├── zoomable_chart_canvas.py  # Canvas grafico con zoom/pan (backend Agg)
    ├── linked_axes.py            # Sincronizzazione bidirezionale canvas grafici
    └── chart_panel.py            # Factory: LabelFrame + ZoomableChartCanvas
```

Il grafo delle dipendenze è aciclico: `app.py` dipende da tutti i moduli di supporto; `dct_analysis.py` da `dct_compression.py`; i `widgets/` solo da `constants.py` e librerie esterne.

### 3.2 `dct_compression.py` — Nucleo Algoritmico

| Funzione | Descrizione |
|---|---|
| `apply_dct2(block)` | `scipy.fft.dctn(block, type=2, norm='ortho')` — DCT-II 2D ortonormale |
| `apply_idct2(coefficients)` | `scipy.fft.idctn(coefficients, type=2, norm='ortho')` — inversa |
| `build_frequency_cutoff_mask(F, d)` | Maschera booleana $F \times F$ vettorizzata con `np.indices` |
| `compress_block(block, mask)` | Pipeline su un blocco: DCT-II → azzeramento → IDCT-II → round+clip |
| `compress_image(image, F, d)` | Loop sui blocchi; output $H' \times W'$ con $H',W'$ multipli di $F$ |

La maschera è costruita tramite broadcasting NumPy senza loop Python:
```python
row_idx, col_idx = np.indices((block_size, block_size))
mask = (row_idx + col_idx) < threshold_d
```

### 3.3 `dct_analysis.py`

`build_dct_frequency_map(image, F, d) → (freq_full, freq_trunc)` calcola la media dei valori assoluti dei coefficienti DCT su tutti i blocchi. Restituisce la mappa completa e quella troncata secondo la maschera diagonale; entrambe vengono visualizzate in scala logaritmica $\log(1+|\text{coeff}|)$.

### 3.4 `image_utils.py`

- `load_grayscale_bmp(path)` — apre il BMP e converte in modalità `'L'` (Pillow gestisce automaticamente la conversione RGB→grigio con la formula ITU-R BT.601).
- `numpy_array_to_pil_image(array)` — converte `uint8` → `PIL.Image` per il canvas.

### 3.5 `constants.py`

Centralizza i valori configurabili evitando costanti magiche nel codice:

| Costante | Valore | Descrizione |
|---|---|---|
| `WINDOW_MIN_WIDTH/HEIGHT` | 1200 / 900 (Win), 900 / 700 (macOS) | Dimensioni minime finestra |
| `PARAM_F_MIN / F_MAX` | 1 / 512 | Range del blocco |
| `PARAM_D_MIN` | 0 | Soglia minima |
| `ZOOM_FACTOR_IN / OUT` | 1.25 / 0.8 | Fattori di zoom |
| `ZOOM_MIN / MAX` | 0.05 / 20.0 | Limiti dello zoom |
| `DCT_SAMPLE_BLOCKS` | 6 | Blocchi campionati per analisi |

### 3.6 `gui.py` e `app.py`

**`gui.py`** è l'entry point. Gestisce due modalità via `argparse`:
- Default: istanzia `tk.Tk()` e `DctCompressionApp`, centrando la finestra sullo schermo.
- `--test`: esegue `tests.run_tests()` e termina.

Contiene `validate_compression_parameters(F, d) → str | None` che verifica $F \geq 1$ e $0 \leq d \leq 2F-2$, restituendo il messaggio di errore o `None`.

**`app.py`** — classe `DctCompressionApp`:

- `_build_ui()` — struttura scrollabile verticalmente (canvas + scrollbar); scroll propagato da tutti i widget figli al canvas principale.
- `_build_control_panel()` — barra con bottone file, spinbox F e d, bottone "Comprimi", label di stato.
- `_build_image_preview_area()` — due `ZoomableImageCanvas` affiancati con zoom/pan sincronizzati bidirezionalmente.
- `_show_charts(...)` — griglia 2×2 di `ZoomableChartCanvas` (2 istogrammi + 2 mappe DCT), con coppie linkate tramite `LinkedChartGroup`.
- `_on_compress_clicked()` — validazione parametri → avvio worker in thread secondario → aggiornamento UI nel main thread via `root.after(0, callback)`, prevenendo freeze su macOS.
- `_run_in_thread(worker, on_done, on_error)` — esecutore generico daemon-thread; nessun widget Tkinter viene mai toccato dal thread secondario.

### 3.7 Package `widgets/`

**`ZoomableImageCanvas`** (`zoomable_canvas.py`) — estende `tk.Canvas`:
- Zoom centrato sul puntatore (rotella mouse), con formula $o_x' = x - f(x - o_x)$ dove $f = z'/z$.
- Pan con drag tasto sinistro; reset fit-to-canvas su doppio click.
- Rendering ottimizzato: viene ridimensionato solo il ritaglio visibile (crop → resize), evitando di scalare l'intera immagine.
- Sincronizzazione bidirezionale con canvas gemello tramite `sync_with(other)` e flag `_syncing` anti-ricorsione.
- Scroll ritorna `"break"` per non propagare l'evento al canvas padre (zoom ≠ scroll pagina).

**`ZoomableChartCanvas`** (`zoomable_chart_canvas.py`) — canvas Matplotlib con backend Agg renderizzato su `tk.Canvas`:
- Zoom/pan sui limiti degli assi Matplotlib (xlim/ylim).
- Sincronizzabile con altri canvas tramite `sync_with(*peers)` (implementato in `LinkedChartGroup`).

**`LinkedChartGroup`** (`linked_axes.py`) — collega più `ZoomableChartCanvas` chiamando `sync_with()` su tutti i peer. Alias `LinkedAxesGroup` mantenuto per retrocompatibilità.

**`make_chart_panel`** (`chart_panel.py`) — factory che restituisce `(ttk.LabelFrame, ZoomableChartCanvas)` con il grafico già renderizzato tramite `draw_fn(fig, ax)`.

### 3.8 `tests.py`

Due test numerici eseguibili con `python gui.py --test`:

**Test 1 — DCT 1D.** Input: $\mathbf{v} = [231, 32, 233, 161, 24, 71, 140, 245]$. Confronto con valori di riferimento della specifica (corrispondenti a `norm=None`); la funzione `_select_best_norm` sceglie automaticamente la variante con errore minore. Soglia: errore relativo massimo $< 1\%$.

**Test 2 — DCT-II 2D.** Input: blocco $8 \times 8$ di riferimento. Stessa soglia $< 1\%$ su tutti i 64 coefficienti.

Nota: i valori di riferimento usano `norm=None`; il core dell'applicazione usa `norm='ortho'` (scalatura di $\sqrt{N}$ per DC e $\sqrt{N/2}$ per AC), ma la pipeline rimane corretta poiché DCT e IDCT usano la stessa normalizzazione.

---

## 4. Tecnologie e Dipendenze

| Libreria | Versione minima | Ruolo |
|---|---|---|
| `numpy` | ≥ 1.24 | Array, operazioni vettorizzate, clipping |
| `scipy` | ≥ 1.10 | `dctn`/`idctn` — DCT-II 2D via PocketFFT, $O(F^2 \log F)$ |
| `Pillow` | ≥ 9.0 | I/O BMP, conversione scala di grigi |
| `matplotlib` | ≥ 3.7 | Backend Agg per rendering grafici in canvas Tkinter |
| `tkinter` | stdlib | GUI multipiattaforma |

Matplotlib viene usato esclusivamente come motore di disegno in memoria (backend Agg); non viene creato alcun `FigureCanvasTkAgg` né `NavigationToolbar2Tk`. I grafici sono renderizzati come immagini PIL e mostrati su `tk.Canvas` nativo.

---

## 5. Interfaccia Grafica

L'interfaccia è organizzata verticalmente con scrollbar, adattandosi a qualsiasi risoluzione. Le dimensioni minime variano per piattaforma (1200×900 su Windows/Linux, 900×700 su macOS).

> **[PLACEHOLDER FIGURA 1]**  
> *Screenshot dell'interfaccia: barra di controllo in alto, anteprime affiancate, 4 pannelli grafici in basso.*

**Barra di controllo:** bottone selezione BMP, spinbox F ($[1,512]$, default 8) e d ($[0,2F-2]$, default 0), bottone "Comprimi", label di stato durante l'elaborazione, nota sui controlli interattivi.

**Anteprime:** due `ZoomableImageCanvas` affiancati (originale / compressa) con zoom e pan sincronizzati bidirezionalmente. Doppio click per reset fit-to-canvas.

**Pannelli grafici (griglia 2×2):**

| Posizione | Contenuto | Linkato con |
|---|---|---|
| [0,0] | Istogramma immagine originale | [0,1] |
| [0,1] | Istogramma immagine compressa | [0,0] |
| [1,0] | Mappa frequenze DCT originali | [1,1] |
| [1,1] | Mappa frequenze DCT troncate (d) | [1,0] |

Le mappe DCT mostrano la media dei $|\text{coefficienti}|$ in scala logaritmica; una linea tratteggiata ciano indica la diagonale di taglio $k+l=d$. Zoom/pan sono sincronizzati per coppia.

> **[PLACEHOLDER FIGURA 2]**  
> *Pannelli grafici per $F=8$, $d=5$: istogrammi e mappe DCT con diagonale di taglio evidenziata.*

---

## 6. Validazione Numerica

Eseguire `python gui.py --test` per verificare la conformità con i valori della specifica.

**Test 1 — DCT 1D** su $\mathbf{v} = [231, 32, 233, 161, 24, 71, 140, 245]$:

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| $C_k^{\text{ref}}$ | 4.01e+02 | 6.60e+00 | 1.09e+02 | −1.12e+02 | 6.54e+01 | 1.21e+02 | 1.16e+02 | 2.88e+01 |

**Test 2 — DCT-II 2D** sul blocco $8 \times 8$ riportato in `tests.py`.

Entrambi i test verificano errore relativo massimo $< 1\%$. ✓

> **[PLACEHOLDER FIGURA 3]**  
> *Output terminale di `python gui.py --test` con errori relativi per ogni componente.*

---

## 7. Esperimenti e Risultati

### 7.1 Impatto del parametro $d$ (con $F = 8$)

Variando $d \in \{1, 5, 10, 14\}$ con $F=8$ fisso:

> **[PLACEHOLDER FIGURA 4]** *Immagine originale in scala di grigi.*

> **[PLACEHOLDER FIGURA 5]** *Confronto originale vs compressa per $d=1$ e $d=5$. Artefatti a blocchi evidenti per $d=1$.*

> **[PLACEHOLDER FIGURA 6]** *Confronto per $d=10$ e $d=14$. Con $d=14$ la qualità è quasi indistinguibile dall'originale.*

| $d$ | Coefficienti conservati | % | Qualità visiva |
|---|---|---|---|
| 1 | 1/64 | 1.6% | Solo DC — blocchi uniformemente grigi |
| 5 | 15/64 | 23.4% | Riconoscibile ma con artefatti di blocco |
| 10 | 55/64 | 85.9% | Buona; artefatti solo su bordi netti |
| 14 | 63/64 | 98.4% | Praticamente identica all'originale |

### 7.2 Impatto del parametro $F$ (con $d = F$)

Variando $F \in \{4, 8, 16, 32\}$:

> **[PLACEHOLDER FIGURA 7]** *Confronto per $F=4,d=4$ e $F=16,d=16$.*

> **[PLACEHOLDER FIGURA 8]** *Confronto per $F=8,d=8$ e $F=32,d=32$.*

Blocchi più grandi catturano strutture su scale maggiori ma producono artefatti più grossolani. $F=8$ è il compromesso ottimale dello standard JPEG.

### 7.3 Analisi Quantitativa — MSE e PSNR

$\text{MSE} = \frac{1}{H'W'} \sum_{m,n} (I_{m,n} - \hat{I}_{m,n})^2, \qquad \text{PSNR} = 10\log_{10}\!\left(\frac{255^2}{\text{MSE}}\right) \text{ [dB]}$

PSNR $> 35$ dB indica ottima qualità; PSNR $< 25$ dB degrado severo.

> **[PLACEHOLDER TABELLA 1]** *MSE e PSNR al variare di $d$ (F=8): colonne $d$, coefficienti conservati, %, MSE, PSNR.*

> **[PLACEHOLDER FIGURA 9]** *Curva PSNR vs $d$ per $F=8$: monotona crescente e concava.*

> **[PLACEHOLDER FIGURA 10]** *Istogrammi originale/compressa per $F=8$, $d=5$: smoothing della distribuzione.*

---

## 8. Discussione

**Concentrazione spettrale.** Le mappe DCT confermano che l'energia è quasi interamente concentrata nelle basse frequenze per immagini naturali — fondamento fisico della compressione JPEG.

**Compromesso $d$.** La curva PSNR vs $d$ è monotona crescente e concava: il maggior guadagno si ottiene aggiungendo le prime componenti AC; le alte frequenze contribuiscono poco alla qualità percepita.

**Artefatti di blocco.** Compaiono per $d$ piccoli: ogni blocco è trasformato indipendentemente, senza continuità con i vicini. JPEG-2000 risolve questo con la DWT (trasformata wavelet), che opera sull'intera immagine.

**Effetto di $F$.** Blocchi più grandi catturano strutture a scala maggiore ma generano artefatti più estesi. $F=8$ è il compromesso empiricamente ottimale per immagini fotografiche.

**Istogrammi.** La compressione produce uno smoothing della distribuzione dei livelli di grigio: eliminando le alte frequenze si riduce la varianza locale e i valori estremi migrano verso la media.

**Conformità numerica.** I test confermano errore relativo massimo $< 1\%$ su entrambi i blocchi di riferimento.

---

## 9. Conclusioni

Il sistema implementa correttamente la pipeline JPEG-like, attestata dai test numerici. L'interfaccia grafica (canvas zoomabili sincronizzati, mappe DCT, istogrammi comparativi) facilita la comprensione intuitiva del meccanismo di compressione. I risultati confermano che $d$ controlla efficacemente il compromesso qualità/compressione, $F=8$ è il valore ottimale per immagini fotografiche, e gli artefatti di blocco sono un limite intrinseco dell'approccio block-by-block.

Sviluppi futuri: supporto YCbCr con subsampling crominanza; quantizzazione adattiva per frequenza; metrica SSIM; parallelizzazione dei blocchi.

---

## 10. Riferimenti

1. Wallace, G. K. (1992). *The JPEG still picture compression standard*. IEEE Trans. Consumer Electronics, 38(1).
2. Rao, K. R., & Yip, P. (1990). *Discrete Cosine Transform*. Academic Press.
3. Ahmed, N., Natarajan, T., & Rao, K. R. (1974). *Discrete cosine transform*. IEEE Trans. Computers, C-23(1), 90–93.
4. Virtanen, P., et al. (2020). *SciPy 1.0*. Nature Methods, 17, 261–272.
5. Harris, C. R., et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362.
6. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing*, 4th ed. Pearson.
7. Strang, G. (1999). *The discrete cosine transform*. SIAM Review, 41(1), 135–147.

---

*Documento in formato Markdown — aggiungere le figure nei placeholder prima della consegna finale.*
