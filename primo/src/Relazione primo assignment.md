# Analisi Comparativa di Metodi Iterativi per Sistemi Lineari Sparsi

**Corso:** Modelli di Calcolo Scientifico  
**Progetto:** Primo Assignment - alternativo  
**Anno Accademico:** 2025–2026  
**URL repo GitHub:** [Progetto_Modelli_Calcolo_Scientifico](https://github.com/LorenzoPinaUnimib/Progetto_Modelli_Calcolo_Scientifico)  

**Progetto svolto da:**  
_Pina Lorenzo - 894396  
Piovanelli Michele - 894433  
Rancati Simone - 900052_  

---

## Utilizzo IA

Nel progetto è stata utilizzata l'IA generativa per la creazione delle interfacce grafiche e per leggere correzioni del codice.

---

## Indice

1. [Introduzione](#1-introduzione)
2. [Architettura del Software](#2-architettura-del-software)
   * 2.1 [`gui.py` — Entry Point](#21-guipy--entry-point)
   * 2.2 [`utils/app.py` — Finestra Principale](#22-utilsapppy--finestra-principale)
   * 2.3 [`utils/matrix_io.py` — Lettura delle Matrici](#23-utilsmatrix_iopy--lettura-delle-matrici)
   * 2.4 [`utils/metrics.py` — Calcolo dell'Errore](#24-utilsmetricspy--calcolo-dellerrore)
   * 2.5 [`utils/solvers.py` — Orchestrazione dei Solutori](#25-utilssolverspy--orchestrazione-dei-solutori)
   * 2.6 [`utils/styles.py` — Stile e Palette](#26-utilsstylespy--stile-e-palette)
   * 2.7 [`solvers/jacobi.py` — Metodo di Jacobi](#27-solversjacobipy--metodo-di-jacobi)
   * 2.8 [`solvers/gauss_seidel.py` — Metodo di Gauss-Seidel](#28-solversgauss_seidelpy--metodo-di-gauss-seidel)
   * 2.9 [`solvers/gradient.py` — Metodo del Gradiente](#29-solversgradientpy--metodo-del-gradiente)
   * 2.10 [`solvers/cg.py` — Metodo del Gradiente Coniugato](#210-solverscgpy--metodo-del-gradiente-coniugato)
3. [Tecnologie e Dipendenze](#3-tecnologie-e-dipendenze)
4. [Interfaccia Grafica](#4-interfaccia-grafica)
5. [Esperimenti e Risultati](#5-esperimenti-e-risultati)
   * 5.1 [spa1.mtx](#51-spa1mtx)
   * 5.2 [spa2.mtx](#52-spa2mtx)
   * 5.3 [vem1.mtx](#53-vem1mtx)
   * 5.4 [vem2.mtx](#54-vem2mtx)
   * 5.5 [Numero di condizionamento](#55-numero-di-condizionamento)
   * 5.6 [Analisi sotto tolleranze stringenti](#56-analisi-sotto-tolleranze-stringenti)
   * 5.7 [Note aggiuntive sull'utilizzo di memoria](#57-note-aggiuntive-sullutilizzo-di-memoria)
6. [Conclusioni](#6-conclusioni)
7. [Riferimenti](#7-riferimenti)

---

## 1. Introduzione

Il presente elaborato descrive la progettazione e lo sviluppo di una libreria per l'esecuzione di metodi di risoluzione di sistemi lineari di equazioni tramite risolutori iterativi. I metodi in questione sono il **metodo di Jacobi, di Gauß-Seidel, del gradiente e del gradiente coniugato**. Questa viene applicata limitatamente a matrici simmetriche e definite positive.
L'applicazione risultante dal lavoro di sviluppo eseguito è dotata di un interfaccia grafica sviluppata con Tkinter, la quale provvede la possibilità di caricare una matrice in formato .mtx e scegliere una tolleranza, anche in notazione scentifica. Una volta forniti gli input desiderati ed eseguito il programma, verranno forniti dei grafici che descrivono tempo, n° di iterazioni, errore e memoria utilizzati per ognuno dei quattro metodi sopra elencati.

---

## 2. Architettura del Software

Il progetto è strutturato secondo una separazione netta tra logica di calcolo, gestione dell'input/output e presentazione visiva. La directory principale si articola come segue:

```
src/
├── gui.py                  # Entry point dell'applicazione
├── solvers/
│   ├── cg.py               # Solutore per il metodo del Gradiente Coniugato
│   ├── gauss_seidel.py     # Solutore per Gauß-Seidel
│   ├── gradient.py         # Solutore per il metodo del Gradiente
│   └── jacobi.py           # Solutore per Jacobi
└── utils/
    ├── app.py              # Finestra principale (tk.Tk)
    ├── matrix_io.py        # Lettura matrici .mtx
    ├── metrics.py          # Calcolo errore relativo
    ├── solvers.py          # Thread di orchestrazione dei solutori
    └── styles.py           # Palette cromatica e costanti tipografiche
```

---

### 2.1 `gui.py` — Entry Point

Il file `gui.py` costituisce il punto di ingresso dell'applicazione. Il suo contenuto è volutamente minimale: si limita a impostare il backend grafico di Matplotlib su `TkAgg` e a importare la classe `App` dal sottomodulo `utils.app`.
Questa separazione tra entry point e logica applicativa segue il principio di singola responsabilità: `gui.py` non contiene alcuna logica di business, delegando interamente la costruzione dell'interfaccia e la gestione degli eventi al modulo dedicato.

---

### 2.2 `utils/app.py` — Finestra Principale

Il modulo `app.py` definisce la classe `App`, che racchiude tutta la logica dell'interfaccia grafica. Nel costruttore vengono inizializzate le variabili di stato, applicato il tema visivo tramite `ttk.Style`, e costruita la UI con `_build_ui()`.

Il metodo `_run_solvers()` è il cuore della logica di interazione: valida l'input, disabilita il pulsante durante l'esecuzione, delega il calcolo a un `SolverThread` e registra tre callback che aggiornano l'interfaccia in modo thread-safe tramite `self.after(0, ...)`. Al completamento, `_finish()` aggiorna grafici e tabella, mentre la classe interna `_TextRedirector` redirige lo `stdout` dell'intero processo verso il widget di log, catturando anche i messaggi stampati direttamente dai solutori.

---

### 2.3 `utils/matrix_io.py` — Lettura delle Matrici

Il modulo `matrix_io.py` espone un'unica funzione pubblica, `load_mtx()`, che accetta il percorso di un file nel formato Matrix Market (`.mtx`) e ne restituisce la matrice corrispondente in formato CSR (_Compressed Sparse Row_). La conversione è effettuata in due passaggi: `scipy.io.mmread()` legge il file e produce una matrice sparsa in formato COO, che viene poi convertita con `csr_matrix()`.

---

### 2.4 `utils/metrics.py` — Calcolo dell'Errore

Il modulo `metrics.py` contiene la funzione `_compute_relative_error()`, che calcola l'errore relativo tra la soluzione esatta `x_true` e quella calcolata `x_comp` secondo la formula $\epsilon = \|x^* - x^{(k)}\|_2 / \|x^*\|_2$. La funzione gestisce esplicitamente i casi degeneri: se `x_comp` è `None` (solutore fallito) restituisce `NaN`, e se la norma del denominatore è zero evita la divisione restituendo anch'essa `NaN`.

---

### 2.5 `utils/solvers.py` — Orchestrazione dei Solutori

Il modulo `solvers.py` definisce la classe `SolverThread`, che esegue l'intera pipeline di calcolo in background. Il costruttore riceve il percorso della matrice, la tolleranza e le tre callback di comunicazione con la GUI. Il metodo `run()` carica dinamicamente i quattro solutori tramite la funzione privata `_load_solvers()`, costruisce il vettore termine noto $b = A \cdot \mathbf{1}$ e itera sui metodi disponibili.

Per ciascun solutore, `run()` attiva `tracemalloc` prima della chiamata e ne legge il picco di memoria al termine, gestisce le eccezioni interne marcando il risultato come fallito senza interrompere l'esecuzione degli altri metodi, e accumula i risultati in un dizionario che viene infine passato alla callback `on_done`.

---

### 2.6 `utils/styles.py` — Stile e Palette

Il modulo `styles.py` centralizza tutte le costanti visive dell'applicazione: colori della palette (sfondo, pannelli, bordi, accenti, stati di successo/avviso/errore), famiglie e dimensioni dei font (`FONT_MONO`, `FONT_LABEL`, `FONT_TITLE`, `FONT_SMALL`) e il dizionario `METHOD_COLORS` che associa a ciascun solutore un colore identificativo univoco.

---

### 2.7 `solvers/jacobi.py` — Metodo di Jacobi

Il modulo implementa il metodo iterativo di Jacobi per la risoluzione di sistemi lineari sparsi. Dopo aver verificato che la matrice sia quadrata e che la diagonale principale non contenga elementi nulli, costruisce la matrice dell'inversa della diagonale come operatore sparso `D_inv = sp.diags(1.0 / D_diag)`. Il ciclo di iterazione aggiorna la soluzione con $x^{(k+1)} = x^{(k)} + D^{-1} r^{(k)}$ e ricalcola il residuo ad ogni passo fino al soddisfacimento del criterio di arresto o al raggiungimento del numero massimo di iterazioni.

Il metodo restituisce la tripla `(x, nit, elapsed_time)`, interfaccia comune a tutti i solutori del progetto. Il tempo è misurato con `time.perf_counter()` ad alta risoluzione, escludendo le operazioni di setup (costruzione di `D_inv`) dal conteggio. Un messaggio di avanzamento viene stampato sullo `stdout` ogni 1000 iterazioni, che la GUI cattura automaticamente tramite il meccanismo di redirezione implementato in `app.py`.

---

### 2.8 `solvers/gauss_seidel.py` — Metodo di Gauss-Seidel

Il modulo implementa il metodo di Gauss-Seidel sfruttando la struttura triangolare inferiore della matrice sparsa. Dopo i controlli di validità, estrae la parte triangolare inferiore $P = \text{tril}(A)$ tramite `sp.tril()` in formato CSR. Ad ogni iterazione calcola il residuo $r = b - Ax$ e risolve il sistema triangolare $Py = r$ con `spsolve_triangular()`, aggiornando poi la soluzione come $x \leftarrow x + y$.

---

### 2.9 `solvers/gradient.py` — Metodo del Gradiente

Il modulo implementa il metodo della discesa del gradiente per sistemi SPD. Prima di avviare l'iterazione esegue tre controlli: quadratezza della matrice, simmetria esatta e definita positività calcolando il minimo autovalore con `scipy.sparse.linalg.eigsh(..., which='SM')`.

Il ciclo iterativo calcola ad ogni passo il residuo $r = b - Ax$, il passo ottimale $\alpha = r^T r / (r^T A r)$ e aggiorna la soluzione. Il costo dominante per iterazione è la doppia moltiplicazione matrice-vettore, che per matrici sparse rimane comunque efficiente. La verifica della definita positività tramite ARPACK introduce un overhead iniziale significativo, giustificato dalla necessità di garantire la correttezza matematica dell'applicazione del metodo.

---

### 2.10 `solvers/cg.py` — Metodo del Gradiente Coniugato

Il modulo implementa il metodo del gradiente coniugato, condividendo con `gradient.py` gli stessi tre controlli preliminari. L'iterazione mantiene due vettori ausiliari — il residuo $r$ e la direzione di ricerca $d$ — aggiornati secondo lo schema classico: il passo ottimale è $\alpha_k = d^T r / (d^T A d)$, la soluzione è aggiornata come $x \leftarrow x + \alpha_k d$, il residuo come $r \leftarrow b - Ax$, e infine la nuova direzione è calcolata come $d \leftarrow r - \beta_k d$ con $\beta_k = (d^T A r) / (d^T A d)$.

---

## 3. Tecnologie e Dipendenze

Il progetto è implementato interamente in Python e fa uso delle seguenti librerie:

- **NumPy** (`numpy`) costituisce il substrato computazionale per tutte le operazioni vettoriali.
- **SciPy** (`scipy`) è impiegata per la lettura delle matrici nel formato Matrix Market (`scipy.io.mmread`), la conversione in formato CSR (`scipy.sparse.csr_matrix`), la risoluzione di sistemi triangolari (`scipy.sparse.linalg.spsolve_triangular`), la verifica della definita positività tramite calcolo del minimo autovalore (`scipy.sparse.linalg.eigsh`), e la rappresentazione efficiente delle matrici sparse.
- **Matplotlib** (`matplotlib`) con backend `TkAgg` gestisce la visualizzazione dei grafici comparativi direttamente all'interno della finestra Tkinter.
- **Tkinter** (`tkinter`), incluso nella libreria standard di Python, fornisce l'infrastruttura per l'interfaccia grafica. Il modulo `tracemalloc`, anch'esso parte della libreria standard, è utilizzato per la profilazione della memoria di picco durante l'esecuzione di ciascun solutore.

---

## 4. Interfaccia Grafica

L'interfaccia è composta da due colonne principali. Il pannello sinistro, di larghezza fissa pari a 260 pixel, raccoglie i controlli operativi: una drop-zone cliccabile per la selezione del file `.mtx`, un campo di testo per l'impostazione della tolleranza (valore predefinito `10e-4`), un pulsante di avvio, una barra di progresso indeterminata e un'area di stato testuale. Il pannello destro, espandibile, ospita un `ttk.Notebook` con tre schede:

La scheda **Dashboard** mostra quattro grafici a barre prodotti con Matplotlib su una griglia 2×2: tempo di esecuzione in secondi, numero di iterazioni, errore relativo in scala logaritmica e memoria di picco in megabyte. Ogni metodo è associato a un colore fisso definito nel dizionario `METHOD_COLORS` di `styles.py`, garantendo coerenza visiva tra grafico e tabella.

La scheda **Tabella risultati** presenta un `ttk.Treeview` con le colonne Metodo, Iterazioni, Tempo, Errore Relativo, Memoria e Stato. Le righe sono colorate in verde per i solutori convergenti e in rosso per quelli falliti.

La scheda **Log** redirige lo `stdout` dell'applicazione verso un widget `tk.Text` in sola lettura, mostrando i messaggi di avanzamento emessi da ciascun solutore durante l'iterazione (ogni 1000 cicli) e un riepilogo tabulare al termine.

![](/primo/immagini/gui.png)

Il tema grafico adotta una palette scura con sfondo `#0f1117`, pannelli `#181c27` e accento principale `#4f8ef7`, definiti come costanti in `styles.py`. Lo stile è applicato globalmente tramite `ttk.Style` con tema base `"clam"`.

---

## 5. Esperimenti e Risultati

Sono stati eseguiti più run sulle quattro matrici fornite di esempio, con le tolleranze indicate ($10^{-4}$,$10^{-6}$,$10^{-8}$,$10^{-10}$) ed una esecuzione aggiuntiva con tolleranza $10^{-20}$, per stressare il sistema.

---

### 5.1 spa1.mtx

Per quello che riguarda la matrice `spa1.mtx`, di dimensioni 1000x1000 ed un totale di 182434 elementi non zero (0,18%), possiamo notare un trend di difficoltà nell'esecuzione del metodo del Gradiente, particolarmente accentuata nel provare la tolleranza $10^{-10}$.

Contrariamente, il metodo del Gradiente coniugato fornisce risultati molto più precisi sotto condizioni più impegnative; Jacobi e Gauß-Seidel non presentano qualità di spicco, eccetto che per un'alta precisione di Jacobi in condizioni rilassate.

![](/primo/immagini/spa1.png)

---

### 5.2 spa2.mtx

Per la matrice `spa2.mtx`, di dimensioni 3000x3000 ed un totale di 1633298 elementi non-zero (0.18%), si può osservare un comportamento analogo a quello di `spa1.mtx` per quello che concerne i metodi del gradiente, mentre si nota un miglioramento netto nel metodo di Gauß-Seidel rispetto alla prima.

![](/primo/immagini/spa2.png)

---

### 5.3 vem1.mtx

Per la matrice `vem1.mtx`, di dimensioni 1681x1681 ed un totale di 13385 elementi non-zero (>0.01%), si ha che Jacobi e Gauß-Seidel sono messi in difficoltà, per quello che riguarda il tempo di esecuzione, suggerendo che sia più favorevole al metodo del gradiente rispetto a `spa1.mtx` e `spa2.mtx`

Inoltre, si nota che il gradiente coniugato sia il migliore di gran lunga per quello che riguarda l'errore, indipendentemente dalle condizioni di precisione richieste.

![](/primo/immagini/vem1.png)

---

### 5.4 vem2.mtx

Infine, per `vem2.mtx`, di dimensioni 2601x2601 ed un totale di 21225 elementi non-zero (>0.01%), possiamo notare che si ripete un comportamento analogo a `vem1.mtx`.

![](/primo/immagini/vem2.png)

---

### 5.5 Numero di condizionamento

Anche se non calcolato nell'applicazione in quanto oneroso, il numero di condizionamento correlato alle matrici, calcolato in maniera differita, permette di spiegare il comportamento relativo al metodo del gradiente per tutte le matrici analizzate:

| Matrice:                       | `spa1.mtx`     | `spa2.mtx`     | `vem1.mtx`    | `vem2.mtx`    |
| ------------------------------ | -------------- | -------------- | ------------- | ------------- |
| N° di condizionamento $\kappa$ | $\approx$ 2048 | $\approx$ 1411 | $\approx$ 324 | $\approx$ 507 |

Ricordando che il numero di condizionamento $\kappa$ è dato da $\frac {\lambda_{max}}{\lambda_{min}}$, possiamo osservare che tutte le matrici sono _mal condizionate_, in quanto $\kappa>>1$.

Sapendo inoltre che più alto il numero di condizionamento, più fanno fatica i vari metodi a convergere.
Inoltre, per il metodo del gradiente, sappiamo che più alto è il numero di condizionamento e più le linee di livello sono dissimili a dei cerchi; risulta quindi evidente che la convergenza proceda a zig-zag. Si nota che con `spa1.mtx` e `spa2.mtx` questo metodo vada così in crisi, mentre con `vem1.mtx` e `vem2.mtx` sia più in linea con gli altri metodi.

---

### 5.6 Analisi sotto tolleranze stringenti

Come menzionato prima, è stato eseguito un giro di test con tolleranza di $10^{-20}$, con i seguenti risultati:

![](/primo/immagini/lowtol.png)

Si può notare che con questa tolleranza, tutti i metodi arrivano al numero massimo di iterazioni senza convergere, il che permette di avere una stima più precisa della velocità di convergenza dei vari metodi.

Inoltre, come precedentemente notato, `spa1.mtx` e `spa2.mtx` presentano difficoltà nella convergenza per il metodo del gradiente, dovuta all'alto numero di condizionamento.

Si nota anche che il metodo di Gauß-Seidel è consistentemente più lento a pari numero di iterazioni rispetto agli altri tre metodi; questo è probabilmente dovuto alla presenza di `spsolve_triangular` all'interno della singola iterazione. In compenso, sembra avere in media una precisione maggiore degli altri metodi.

Contrariamente, il metodo di Jacobi è consistentemente meno preciso, con la sola eccezione del metodo del gradiente che è particolarmente mal posto in `spa1.mtx` e `spa2.mtx`.

---

### 5.7 Note aggiuntive sull'utilizzo di memoria

In aggiunta alle tre metriche richieste dalla consegna, è stata aggiunta anche l'analisi sull'utilizzo della memoria, che rimane sufficientemente consistente con la dimensione delle matrici analizzate:

![](/primo/immagini/memusage.png)

Inoltre, si può notare che Jacobi in particolare è estremamente efficiente sotto questo ambito, mentre gli altri tre metodi sono pressochè equivalenti.

Risulta quindi possibile utilizzare questa libreria anche per matrici molto più onerose in termini di dimensioni, ma considerato il significativo aumento di tempo di computazione su `spa2.mtx`, è improbabile che sia pratico in termini di tempo per matrici di dimensioni elevate.

---

## 6. Conclusioni

I risultati sperimentali confermano in modo netto la superiorità del metodo del Gradiente Coniugato come solutore general-purpose per sistemi lineari sparsi con matrice simmetrica e definita positiva. Su tutte e quattro le matrici testate, il gradiente coniugato converge in un numero di iterazioni drasticamente inferiore rispetto agli altri metodi e raggiunge gli errori relativi più bassi, indipendentemente dalla tolleranza richiesta. Questo comportamento è coerente con la teoria: la coniugazione delle direzioni di ricerca permette al metodo di non subire il degrado da zig-zag tipico del gradiente semplice, e il tasso di convergenza governato da $\sqrt{\kappa(A)}$​ anziché da $\kappa(A)$ si traduce in un vantaggio concreto e misurabile anche per le matrici più mal condizionate del benchmark.

Il metodo del Gradiente semplice si è rivelato il punto debole dell'insieme, in particolare su `spa1.mtx` e `spa2.mtx`, dove l'elevato numero di condizionamento ($\kappa \approx 2048$ e $\kappa \approx 1411$ rispettivamente) causa una convergenza estremamente lenta o addirittura il raggiungimento del limite massimo di iterazioni. Al contrario, su `vem1.mtx` e `vem2.mtx`, caratterizzate da $\kappa$ più contenuto, il metodo si comporta in modo più competitivo. Jacobi e Gauß-Seidel si collocano in una posizione intermedia: Jacobi eccelle per efficienza di memoria, risultando di gran lunga il metodo più parsimonioso, mentre Gauß-Seidel tende a fornire soluzioni mediamente più accurate, al costo però di un tempo per iterazione più elevato a causa dell'impiego di `spsolve_triangular`.

Dal punto di vista implementativo, il progetto ha dimostrato come sia possibile costruire una libreria didattica funzionale e misurabile attorno a questi quattro metodi classici, integrando profilazione di memoria, criteri di arresto uniformi e un'interfaccia grafica interattiva. I limiti principali riguardano la scalabilità: l'aumento di dimensione da `spa1.mtx` a `spa2.mtx` comporta già un incremento di tempo di calcolo significativo, rendendo l'utilizzo poco pratico su matrici di grandi dimensioni senza ulteriori ottimizzazioni — ad esempio l'adozione di precondizionatori per il gradiente coniugato, o una parallelizzazione delle operazioni di prodotto matrice-vettore. Queste rappresentano naturali direzioni di sviluppo futuro per estendere le capacità della libreria a problemi di scala reale.

---

## 7. Riferimenti

1. Appunti del corso: https://elearning.unimib.it/course/view.php?id=62129
2. SciPy Documentation — `scipy.sparse.linalg`. https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
3. Iterative methods for sparse linear systems 2nd edition (2003)
