# Compressione di Immagini Digitali tramite Trasformata Discreta del Coseno (DCT-II)

**Corso:** Modelli di Calcolo Scientifico  
**Progetto:** Secondo Assignment — Fase 1 e Fase 2  
**Anno Accademico:** 2025–2026  
**URL repo GitHub:** [Progetto_Modelli_Calcolo_Scientifico](https://github.com/LorenzoPinaUnimib/Progetto_Modelli_Calcolo_Scientifico) 

---

## Abstract

Il presente lavoro descrive la progettazione e l'implementazione di un sistema software per la compressione con perdita (_lossy_) di immagini digitali in scala di grigi, basato sulla Trasformata Discreta del Coseno di tipo II (DCT-II) bidimensionale, ispirato allo standard JPEG. Il progetto è strutturato in due fasi distinte ma architetturalmente integrate: la **Fase 1** implementa la DCT-II manualmente, analizza le prestazioni rispetto a SciPy e visualizza i risultati di compressione tramite una GUI Tkinter; la **Fase 2** estende il sistema con un'interfaccia grafica completa che include anteprime interattive, istogrammi e mappe delle frequenze DCT con zoom e pan sincronizzati. Le due fasi condividono lo stesso package di widget UI (`widgets/`) e lo stesso modulo di costanti (`constants.py`), collocati nella root del progetto.

---

## Indice

1. [Introduzione](#1-introduzione)
2. [Fase 1 — Implementazione Manuale e Analisi delle Prestazioni](#2-fase-1--implementazione-manuale-e-analisi-delle-prestazioni)
    - 2.1 [DCT-II 2D (JPEG-like) e visualizzazione](#21-dct-ii-2d-jpeg-like-e-visualizzazione)
    - 2.2 [Analisi delle Prestazioni (test.py)](#22-analisi-delle-prestazioni-testpy)
    - 2.3 [GUI Fase 1 — ZoomableImageCanvas condiviso](#23-gui-fase-1--zoomableimagecanvas-condiviso)
3. [Fase 2 — Sistema Completo con Analisi DCT](#3-fase-2--sistema-completo-con-analisi-dct)
    - 3.1 [Architettura del Software](#31-architettura-del-software)
    - 3.2 [Modulo `dct_compression.py`](#32-modulo-dct_compressionpy)
    - 3.3 [Modulo `dct_analysis.py`](#33-modulo-dct_analysispy)
    - 3.4 [Modulo `image_utils.py`](#34-modulo-image_utilspy)
    - 3.5 [Modulo `gui.py` e `app.py`](#35-modulo-guipy-e-apppy)
    - 3.6 [Validazione Numerica (`tests.py`)](#36-validazione-numerica-testspy)
4. [Architettura Condivisa](#4-architettura-condivisa)
    - 4.1 [Struttura del Progetto](#41-struttura-del-progetto)
    - 4.2 [Package `widgets/`](#42-package-widgets)
    - 4.3 [Modulo `constants.py`](#43-modulo-constantspy)
5. [Interfaccia Grafica](#5-interfaccia-grafica)
6. [Tecnologie e Dipendenze](#6-tecnologie-e-dipendenze)
7. [Conclusioni](#7-conclusioni)

---

## 1. Introduzione

La compressione delle immagini digitali è un problema fondamentale nell'ingegneria del segnale e nell'informatica applicata. L'obiettivo è ridurre la quantità di dati necessari a rappresentare un'immagine, tollerando una perdita controllata di fedeltà visiva. Lo standard JPEG (ISO/IEC 10918-1, 1994) è il metodo più diffuso per la compressione _lossy_ di immagini fotografiche e si basa sulla DCT-II applicata a blocchi di pixel.

Il presente progetto è articolato in due fasi:

- **Fase 1**: implementazione della DCT-II _from scratch_ (senza SciPy), sviluppo dell'algoritmo di compressione JPEG-like, analisi comparativa delle prestazioni rispetto all'implementazione SciPy e visualizzazione dei risultati tramite una GUI Tkinter con zoom/pan sincronizzati.
- **Fase 2**: estensione del sistema con un'applicazione completa che, partendo da un'immagine BMP, permette di variare i parametri di compressione $F$ e $d$ e di visualizzare istogrammi e mappe delle frequenze DCT con grafici interattivi.

Le due fasi condividono il package di widget UI (`widgets/`) e il modulo di costanti (`constants.py`), eliminando la duplicazione del codice e garantendo uniformità dell'esperienza utente.

---

## 2. Fase 1 — Implementazione Manuale e Analisi delle Prestazioni

### 2.1 DCT-II 2D (JPEG-like) e visualizzazione

Il file `JPEG.py` implementa la pipeline completa di compressione 2D. La DCT-II è implementata manualmente tramite la stessa `calcola_D(N)`, applicata per separabilità: DCT-II 1D su ogni riga, poi su ogni colonna del risultato trasposto.

```python
def DCT2(blocchi, D):
    b = []
    for blocco in blocchi:
        tmp = np.array([DCT1(row, D) for row in blocco])
        tmp = np.array([DCT1(col, D) for col in tmp.T])
        b.append(np.asarray(tmp.T))
    return np.stack(b, axis=0)
```

La funzione `JPEG(img, N, M, grafico, triangolare)` accetta come parametri:

- `N` — dimensione del blocco $F$
- `M` — numero di coefficienti da mantenere (lungo ciascuna dimensione), equivalente a $d = M$ nella notazione della Fase 2 con `triangolare=False`, oppure soglia sulla diagonale $k+l < M$ con `triangolare=True`
- `grafico` — se True, apre la finestra di confronto Tkinter
- `triangolare` — se True, usa la maschera diagonale; se False, azzera le righe/colonne oltre la $M$-esima

La GUI di visualizzazione è descritta nella Sezione 3.4.

### 2.2 Analisi delle Prestazioni (test.py)

Il file `test.py` confronta sistematicamente le prestazioni della DCT-II 2D implementata manualmente (`DCT2` in `JPEG.py`) rispetto a `scipy.fftpack.dctn`, variando la dimensione del blocco $N$ su un'immagine casuale $1 \times N \times N$.

Per ogni $N$ vengono misurati:

- Il tempo di esecuzione delle due implementazioni (`time.perf_counter`)
- L'errore assoluto medio tra i coefficienti: $\varepsilon = \frac{1}{N^2}\sum_{k,l}|C^{\text{manuale}}_{k,l} - C^{\text{SciPy}}_{k,l}|$

I risultati vengono confrontati con le curve teoriche di complessità:

- $O(N^3)$ per l'implementazione manuale (applicazione diretta della DCT-II per separabilità senza FFT)
- $O(N^2 \log N)$ per SciPy

> ![](/secondo/immagini/test.png)
> _Output di `test.py`: grafico sinistra — tempi di esecuzione JPEG vs SciPy con curve teoriche $O(N^3)$ e $O(N^2 \log N) $ (scala logaritmica); grafico destra — errore assoluto medio tra le due implementazioni._

**Osservazioni:**

- L'implementazione manuale cresce cubicamente: raddoppiando $N$, il tempo scala di circa $8\times$.
- SciPy cresce quasi quadraticamente (con il fattore $\log N$ trascurabile per questi valori di $N$).
- L'errore assoluto medio rimane nell'ordine di $10^{-12}$ - $10^{-13}$, confermando la correttezza numerica dell'implementazione manuale rispetto alla versione ottimizzata.

### 2.3 GUI Fase 1 — ZoomableImageCanvas condiviso

La GUI della Fase 1 (invocata da `JPEG.py` quando `grafico=True`) riusa direttamente il `ZoomableImageCanvas` del package `widgets/` condiviso con la Fase 2. La finestra Tkinter mostra le due immagini (originale e compressa) affiancate con zoom e pan sincronizzati, senza alcuna dipendenza da PyQt6 o dal vecchio package `helper/`.

```
secondo/
└── fase1/
    └── JPEG.py   →  from widgets import ZoomableImageCanvas
```

La funzione `_show_comparison_window` crea la finestra Tkinter con:

- Due `ZoomableImageCanvas` in griglia, sincronizzati con `canvas_orig.sync_with(canvas_rec)`
- Titoli parametrizzabili (`title1`, `title2`)
- Label con istruzioni per zoom/pan (rotella, drag, doppio clic)

Il metodo `sync_with` accetta un numero arbitrario di canvas (`*others`), costruendo una rete di sincronizzazione N-vie identica a quella di `ZoomableChartCanvas`. Il doppio clic su uno qualsiasi dei canvas ripristina la vista fit-to-canvas su tutti i peer collegati.

---

## 3. Fase 2 — Sistema Completo con Analisi DCT

### 3.1 Architettura del Software

La Fase 2 è un'applicazione Tkinter completa per la compressione interattiva di immagini BMP. I moduli interni sono:

```
fase2/
├── gui.py              # Entry point: argparse, validazione, avvio finestra
├── app.py              # DctCompressionApp — finestra principale
├── dct_compression.py  # Nucleo algoritmico: DCT-II, maschera, compressione
├── dct_analysis.py     # Mappa media coefficienti DCT per i grafici
├── image_utils.py      # I/O BMP tramite Pillow
└── tests.py            # Test numerici di conformità (DCT 1D e 2D)
```

I moduli condivisi (`widgets/`, `constants.py`) sono nella root del progetto. Il grafo delle dipendenze è aciclico: `app.py` dipende da tutti i moduli di supporto; `dct_analysis.py` da `dct_compression.py`; i `widgets/` solo da `constants.py` e librerie esterne.

### 3.2 Modulo `dct_compression.py`

Contiene il nucleo algoritmico della Fase 2, basato su SciPy (`scipy.fft.dctn`/`idctn` con `norm='ortho'`):

| Funzione                            | Descrizione                                                  |
| ----------------------------------- | ------------------------------------------------------------ |
| `apply_dct2(block)`                 | DCT-II 2D ortonormale via `scipy.fft.dctn`                   |
| `apply_idct2(coefficients)`         | IDCT-II 2D via `scipy.fft.idctn`                             |
| `build_frequency_cutoff_mask(F, d)` | Maschera booleana $F \times F$ vettorizzata con `np.indices` |
| `compress_block(block, mask)`       | DCT-II → azzeramento → IDCT-II → round + clip                |
| `compress_image(image, F, d)`       | Iterazione sui blocchi; output $H' \times W'$                |

La maschera è costruita senza loop Python tramite broadcasting NumPy:

```python
row_idx, col_idx = np.indices((block_size, block_size))
mask = (row_idx + col_idx) < threshold_d
```

### 3.3 Modulo `dct_analysis.py`

`build_dct_frequency_map(image, F, d) → (freq_full, freq_trunc)` calcola la media dei valori assoluti dei coefficienti DCT su tutti i blocchi, restituendo due mappe $F \times F$:

- `freq_full`: distribuzione dell'energia prima del troncamento
- `freq_trunc`: stessa mappa con coefficienti azzerati secondo la maschera diagonale

Entrambe vengono visualizzate in scala logaritmica $\log(1 + |\text{coeff}|)$ per migliorare la leggibilità del range dinamico.

### 3.4 Modulo `image_utils.py`

- `load_grayscale_bmp(path)` — apre il file BMP e converte in modalità `'L'` (8 bit per pixel). Pillow gestisce automaticamente la conversione RGB → grigio
- `numpy_array_to_pil_image(array)` — converte un array NumPy `uint8` in `PIL.Image` per la visualizzazione nel canvas.

### 3.5 Modulo `gui.py` e `app.py`

**`gui.py`** è l'entry point. Gestisce due modalità via `argparse`:

- Default: istanzia `tk.Tk()` e `DctCompressionApp`, centrando la finestra sullo schermo.
- `--test`: esegue `tests.run_tests()` e termina.

Aggiunge automaticamente la root del progetto al `sys.path` per permettere l'import di `widgets/` e `constants.py` anche invocando direttamente `python fase2/gui.py`.

`validate_compression_parameters(F, d) → str | None` verifica $F \geq 1$ e $0 \leq d \leq 2F-2$, restituendo il messaggio di errore o `None`.

**`app.py`** — classe `DctCompressionApp`:

- `_build_ui()` — struttura scrollabile verticalmente; lo scroll è propagato da tutti i widget figli al canvas principale.
- `_build_control_panel()` — barra con bottone file, spinbox F e d, bottone "Comprimi", label di stato.
- `_build_image_preview_area()` — due `ZoomableImageCanvas` affiancati con zoom/pan sincronizzati.
- `_show_charts(...)` — griglia 2×2 di `ZoomableChartCanvas` (2 istogrammi + 2 mappe DCT), con coppie linkate tramite `LinkedChartGroup`.
- `_on_compress_clicked()` — validazione → worker in thread secondario → aggiornamento UI nel main thread via `root.after(0, callback)`.
- `_run_in_thread(worker, on_done, on_error)` — esecutore daemon-thread generico; nessun widget Tkinter viene mai toccato dal thread secondario.

### 3.6 Validazione Numerica (`tests.py`)

Due test numerici eseguibili con `python fase2/run.py --test`:

**Test 1 — DCT 1D.** Input: $\mathbf{v} = [231, 32, 233, 161, 24, 71, 140, 245]$.

| $k$                | 0        | 1        | 2        | 3         | 4        | 5        | 6        | 7        |
| ------------------ | -------- | -------- | -------- | --------- | -------- | -------- | -------- | -------- |
| $C_k^{\text{ref}}$ | 4.01e+02 | 6.60e+00 | 1.09e+02 | −1.12e+02 | 6.54e+01 | 1.21e+02 | 1.16e+02 | 2.88e+01 |

I valori di riferimento corrispondono alla DCT-II senza normalizzazione (`norm=None`). La funzione `_select_best_norm` seleziona automaticamente la variante con errore minore. Soglia: errore relativo massimo $< 1\%$.

**Test 2 — DCT-II 2D.** Input: blocco $8 \times 8$ di riferimento (riportato in `tests.py`). Stessa soglia $< 1\%$ su tutti i 64 coefficienti.

> ![](/secondo/immagini/output_tests.png)
> _Output terminale di `python fase2/gui.py --test`: errori relativi per ogni componente del vettore e del blocco 8×8._

---

## 4. Architettura Condivisa

### 4.1 Struttura del Progetto

```
secondo/
├── constants.py            # Costanti condivise (zoom, finestre, parametri F e d)
├── widgets/                # Package UI condiviso tra fase1 e fase2
│   ├── __init__.py
│   ├── zoomable_canvas.py        # Canvas immagine con zoom/pan (sync N-vie)
│   ├── zoomable_chart_canvas.py  # Canvas grafico con zoom/pan (backend TkAgg)
│   ├── linked_axes.py            # Sincronizzazione N-vie canvas grafici
│   └── chart_panel.py            # Factory: LabelFrame + ZoomableChartCanvas
├── dati/                   # Immagini BMP di test
├── fase1/
│   ├── JPEG.py             # Compressione 2D manuale + GUI Tkinter (widget condivisi)
│   └── test.py             # Benchmark prestazioni vs SciPy
└── fase2/
    ├── gui.py              # Entry point fase2
    ├── run.py              # Launcher fase2
    ├── app.py              # Finestra principale
    ├── dct_compression.py
    ├── dct_analysis.py
    ├── image_utils.py
    └── tests.py
```

`JPEG.py` inserisce direttamente la root `secondo/` nel `sys.path` all'avvio, rendendo `widgets/` e `constants.py` accessibili senza launcher esterno. La Fase 2 usa lo stesso meccanismo in `gui.py`.

### 4.2 Package `widgets/`

**`ZoomableImageCanvas`** (`zoomable_canvas.py`) — estende `tk.Canvas`:

- **Zoom** centrato sul puntatore con rotella mouse (cross-platform: Windows `<MouseWheel>`, Linux `<Button-4>/<Button-5>`, macOS `<MouseWheel>`). Formula: $o_x' = x - f(x - o_x)$ con $f = z'/z$.
- **Pan** con drag tasto sinistro.
- **Reset** fit-to-canvas su doppio click — propagato a tutti i canvas sincronizzati.
- **Rendering ottimizzato**: solo il ritaglio visibile viene ridimensionato (crop → resize con `Image.NEAREST`), evitando di scalare l'intera immagine in memoria.
- **Sincronizzazione N-vie** tramite `sync_with(*others)`: gestisce una lista di peer (`_synced_canvases`) con flag `_syncing` anti-ricorsione, identica all'architettura di `ZoomableChartCanvas`. Zoom, pan e doppio clic si propagano a tutti i canvas collegati.
- Lo scroll restituisce `"break"` per non propagare l'evento al canvas padre (zoom ≠ scroll pagina).

**`ZoomableChartCanvas`** (`zoomable_chart_canvas.py`) — frame Tkinter con figura Matplotlib embedded via `FigureCanvasTkAgg`:

- Zoom/pan aggiornano i limiti degli assi Matplotlib (xlim/ylim) direttamente, senza toolbar.
- Sincronizzabile con altri canvas tramite `sync_with(*peers)` (propagazione tramite `_propagate_lims` / `_apply_lims`).

**`LinkedChartGroup`** (`linked_axes.py`) — collega più `ZoomableChartCanvas` chiamando `sync_with(*peers)` su ciascun canvas del gruppo, costruendo una rete completamente connessa. L'alias `LinkedAxesGroup` è mantenuto per retrocompatibilità.

**`make_chart_panel`** (`chart_panel.py`) — factory che restituisce `(ttk.LabelFrame, ZoomableChartCanvas)` con il grafico già renderizzato tramite `draw_fn(fig, ax)`.

### 4.3 Modulo `constants.py`

Centralizza tutti i valori configurabili, evitando costanti magiche nel codice:

| Costante                  | Valore                          | Descrizione                    |
| ------------------------- | ------------------------------- | ------------------------------ |
| `WINDOW_MIN_WIDTH/HEIGHT` | 1200/900 (Win), 900/700 (macOS) | Dimensioni minime finestra     |
| `PARAM_F_MIN / F_MAX`     | 1 / 512                         | Range del blocco $F$           |
| `PARAM_D_MIN`             | 0                               | Soglia minima $d$              |
| `ZOOM_FACTOR_IN / OUT`    | 1.25 / 0.80                     | Fattori di zoom                |
| `ZOOM_MIN / MAX`          | 0.05 / 20.0                     | Limiti dello zoom              |
| `DCT_SAMPLE_BLOCKS`       | 6                               | Blocchi campionati per analisi |

---

## 5. Interfaccia Grafica

### Fase 1

La finestra di confronto della Fase 1 è una finestra Tkinter con layout a griglia 2×2:

- Riga 0: titoli "Originale" e "Compressa" centrati
- Riga 1: due `ZoomableImageCanvas` affiancati con zoom/pan sincronizzati
- Riga 2: label con istruzioni interattive

La finestra si ridimensiona liberamente; le immagini si adattano fit-to-canvas all'avvio e si sincronizzano automaticamente durante zoom e pan.

> ![](/secondo/immagini/JPEG.png)
> _Finestra Fase 1: immagine originale (sinistra) e compressa (destra) con zoom sincronizzato. Esempio con N=8, M=3._

### Fase 2

L'interfaccia della Fase 2 è organizzata verticalmente con scrollbar, adattandosi a qualsiasi risoluzione.

> ![](/secondo/immagini/gui.png)
> _Screenshot completo della GUI Fase 2: barra di controllo in alto, anteprime affiancate al centro, 4 pannelli grafici in basso._

**Barra di controllo:** bottone selezione BMP, spinbox F ($[1,512]$, default 8) e d ($[0,2F-2]$, default 0), bottone "Comprimi", label di stato durante l'elaborazione.

**Anteprime:** due `ZoomableImageCanvas` affiancati (originale / compressa) con zoom e pan sincronizzati N-vie. Doppio click su uno qualsiasi per reset fit-to-canvas su entrambi.

**Pannelli grafici — griglia 2×2:**

| Posizione | Contenuto                     | Linkato con |
| --------- | ----------------------------- | ----------- |
| [0,0]     | Istogramma immagine originale | [0,1]       |
| [0,1]     | Istogramma immagine compressa | [0,0]       |
| [1,0]     | Mappa frequenze DCT originali | [1,1]       |
| [1,1]     | Mappa frequenze DCT troncate  | [1,0]       |

Le mappe DCT mostrano la media dei $|\text{coefficienti}|$ in scala logaritmica; una linea tratteggiata ciano indica la diagonale di taglio $k+l=d$. Zoom/pan sono sincronizzati per coppia tramite `LinkedChartGroup`.

> ![](/secondo/immagini/istogrammi.png)
> _Pannelli grafici per $F=8$, $d=5$: istogrammi comparativi (in alto) e mappe DCT con diagonale di taglio (in basso)._

---

## 6. Tecnologie e Dipendenze

| Libreria     | Versione minima | Ruolo                                                              |
| ------------ | --------------- | ------------------------------------------------------------------ |
| `numpy`      | ≥ 1.24          | Array multidimensionali, operazioni vettorizzate, clipping         |
| `scipy`      | ≥ 1.10          | `dctn`/`idctn` — DCT-II 2D via PocketFFT, $O(F^2 \log F)$          |
| `Pillow`     | ≥ 9.0           | I/O BMP, conversione scala di grigi                                |
| `matplotlib` | ≥ 3.7           | Backend TkAgg per grafici interattivi, Agg per rendering in canvas |
| `tkinter`    | stdlib          | GUI nativa multipiattaforma                                        |

**Motivazioni delle scelte tecnologiche:**

- **SciPy per la DCT (Fase 2)**: `dctn` garantisce $O(F^2 \log F)$ per blocco, contro $O(F^3)$ dell'implementazione manuale.
- **NumPy vettorizzato**: le operazioni su array (mascheramento, clipping, accumulazione) vengono eseguite in C senza overhead di interpretazione Python.
- **Tkinter**: libreria standard Python, multipiattaforma, senza dipendenze aggiuntive. La scelta di una sola libreria GUI per entrambe le fasi semplifica l'installazione e la manutenzione.
- **matplotlib con backend TkAgg**: integrazione nativa con Tkinter tramite `FigureCanvasTkAgg`, che permette di incorporare figure Matplotlib direttamente nei widget Tkinter senza finestre separate.

---

## 7. Conclusioni

Il progetto ha prodotto un sistema completo per la compressione di immagini tramite DCT-II, articolato in due fasi architetturalmente integrate:

- La **Fase 1** dimostra l'implementazione manuale della DCT-II 2D, la sua correttezza numerica rispetto a SciPy, e il costo computazionale dell'approccio diretto rispetto all'algoritmo FFT-based.
- La **Fase 2** estende il sistema con un'applicazione GUI completa che permette di esplorare interattivamente l'effetto dei parametri di compressione $F$ e $d$ tramite anteprime, istogrammi e mappe DCT con zoom/pan sincronizzati.

Le due fasi condividono lo stesso package di widget UI (`widgets/`) e lo stesso modulo di costanti, garantendo uniformità dell'esperienza utente e riducendo la duplicazione del codice.

I risultati confermano che la DCT-II è una trasformata efficace per la compressione di immagini naturali.

Sviluppi futuri: supporto a immagini a colori, quantizzazione adattiva per frequenza, parallelizzazione dei blocchi, interfaccia parametrica per `JPEG.py` (scelta F, M e modalità di troncamento da riga di comando).
