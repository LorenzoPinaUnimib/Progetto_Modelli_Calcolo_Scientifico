# Analisi Comparativa di Metodi Iterativi per Sistemi Lineari Sparsi

**Corso:** Modelli di Calcolo Scientifico  
**Progetto:** Primo - alternativo  
**Anno Accademico:** 2025–2026  

---

## Abstract

Il presente elaborato descrive la progettazione e lo sviluppo di una libreria per l'esecuzione di metodi di risoluzione di sistemi lineari di equazioni tramite risolutori iterativi. I metodi in questione sono il **metodo di Jacobi, di Gauß-Seidel, del gradiente e del gradiente coniugato**. Questa viene applicata limitatamente a matrici simmetriche e definite positive.
L'applicazione risultante dal lavoro di sviluppo eseguito è dotata di un interfaccia grafica sviluppata con Tkinter, la quale provvede la possibilità di caricare una matrice in formato .mtx e scegliere una tolleranza, anche in notazione scentifica. Una volta forniti gli input desiderati ed eseguito il programma, verranno forniti dei grafici che descrivono n° di iterazioni, errore, tempo e memoria utilizzati per ognuno dei quattro metodi sopra elencati.

---

## 1. Introduzione

La risoluzione di sistemi lineari della forma $Ax = b$ rappresenta uno dei problemi fondamentali del calcolo scientifico. Quando la matrice $A$ è di grandi dimensioni ma strutturalmente sparsa — ovvero quando la maggior parte degli elementi è nulla — i metodi diretti come la fattorizzazione LU diventano proibitivi sia in termini di memoria sia di tempo computazionale. I metodi iterativi, al contrario, sfruttano direttamente la struttura sparsa della matrice, costruendo una successione di approssimazioni $\{x^{(k)}\}$ che converge alla soluzione esatta sotto opportune condizioni.

Il presente elaborato documenta la progettazione, l'implementazione e la valutazione comparativa di quattro metodi iterativi classici: **Jacobi**, **Gauss-Seidel**, **Gradiente** e **Gradiente Coniugato**. Il progetto include un'applicazione desktop con interfaccia grafica che consente di caricare matrici in formato Matrix Market (`.mtx`), eseguire i quattro solutori in parallelo e confrontarne i risultati tramite visualizzazioni interattive.

---

## 2. Fondamenti Teorici

### 2.1 Metodo di Jacobi

Il metodo di Jacobi è un metodo stazionario di tipo splitting. Dato il sistema $Ax = b$, si decompone $A = D + R$ dove $D$ è la matrice diagonale e $R = L + U$ raccoglie le parti triangolari strettamente inferiore e superiore. L'iterazione si definisce come:

$$x^{(k+1)} = D^{-1}(b - Rx^{(k)})$$

Equivalentemente, ad ogni passo si aggiorna $x^{(k+1)} = x^{(k)} + D^{-1}r^{(k)}$ dove $r^{(k)} = b - Ax^{(k)}$ è il residuo corrente. La convergenza è garantita quando la matrice di iterazione $B_J = -D^{-1}R$ ha raggio spettrale $\rho(B_J) < 1$, condizione soddisfatta, ad esempio, quando $A$ è a dominanza diagonale stretta. Il metodo non richiede che la matrice sia simmetrica o definita positiva, ma è generalmente più lento di Gauss-Seidel.

### 2.2 Metodo di Gauss-Seidel

Gauss-Seidel è una variante di Jacobi che utilizza immediatamente i valori aggiornati durante la stessa iterazione. Lo splitting utilizzato è $A = P + N$ dove $P = D + L$ è la parte triangolare inferiore (inclusa la diagonale). L'iterazione diventa:

$$Px^{(k+1)} = b - Ux^{(k)}$$

La risoluzione del sistema triangolare inferiore viene effettuata in modo efficiente con la tecnica di *forward substitution*. Rispetto a Jacobi, Gauss-Seidel converge tipicamente in meno iterazioni; per matrici simmetriche e definite positive, la convergenza è garantita. La matrice di iterazione associata è $B_{GS} = -(D+L)^{-1}U$.

### 2.3 Metodo del Gradiente

Il metodo del gradiente (o *steepest descent*) è un metodo di discesa applicabile a sistemi con matrice $A$ simmetrica e definita positiva (SPD). Il sistema $Ax = b$ è equivalente alla minimizzazione del funzionale quadratico $\phi(x) = \frac{1}{2}x^TAx - b^Tx$. Ad ogni iterazione si procede nella direzione del residuo (gradiente negativo della funzione costo):

$$x^{(k+1)} = x^{(k)} + \alpha_k r^{(k)}, \qquad \alpha_k = \frac{(r^{(k)})^T r^{(k)}}{(r^{(k)})^T A r^{(k)}}$$

Il passo ottimale $\alpha_k$ minimizza $\phi$ lungo la direzione corrente. La convergenza dipende dal numero di condizionamento $\kappa(A) = \lambda_{\max}/\lambda_{\min}$: per matrici mal condizionate la convergenza può essere molto lenta, poiché le direzioni di ricerca successive tendono a formare angoli piccoli causando un andamento a zig-zag.

### 2.4 Metodo del Gradiente Coniugato

Il gradiente coniugato (CG) supera il limite del metodo del gradiente semplice generando direzioni di ricerca $A$-coniugate, ovvero ortogonali rispetto al prodotto scalare indotto da $A$. L'aggiornamento della direzione introduce un termine correttivo $\beta_k$:

$$d^{(k+1)} = r^{(k+1)} - \beta_k d^{(k)}, \qquad \beta_k = \frac{(d^{(k)})^T A r^{(k+1)}}{(d^{(k)})^T A d^{(k)}}$$

In aritmetica esatta, il CG converge in al più $n$ iterazioni (dove $n$ è la dimensione del sistema), comportandosi come un metodo diretto. In pratica, per matrici SPD ben condizionate, la convergenza avviene in un numero di iterazioni molto inferiore a $n$. Il tasso di convergenza è governato da $\sqrt{\kappa(A)}$ anziché da $\kappa(A)$ come nel gradiente semplice, il che costituisce un vantaggio sostanziale.

### 2.5 Criteri di Arresto

Tutti i metodi implementati adottano un criterio di arresto basato sul residuo relativo in norma infinito:

$$\frac{\|Ax^{(k)} - b\|_\infty}{\|b\|_\infty} < \text{tol}$$

Il numero massimo di iterazioni è fissato a $n_{\max} = 20000$ per ciascun metodo.

---

## 3. Architettura del Software

Il progetto è strutturato secondo una separazione netta tra logica di calcolo, gestione dell'input/output e presentazione visiva. La directory principale si articola come segue:

```
project/
├── gui.py                  # Entry point dell'applicazione
├── utils/
│   ├── app.py              # Finestra principale (tk.Tk)
│   ├── solvers.py          # Thread di orchestrazione dei solutori
│   ├── matrix_io.py        # Lettura matrici .mtx
│   ├── metrics.py          # Calcolo errore relativo
│   └── styles.py           # Palette cromatica e costanti tipografiche
└── solvers/
    ├── jacobi.py
    ├── gauss_seidel.py
    ├── gradient.py
    └── cg.py
```

Il modulo `gui.py` funge da entry point e istanzia la classe `App`. Quest'ultima, definita in `utils/app.py`, eredita da `tk.Tk` e si occupa sia della costruzione dell'interfaccia sia della gestione degli eventi utente. La comunicazione tra la GUI e i solutori avviene attraverso la classe `SolverThread` (in `utils/solvers.py`), che esegue il calcolo in un thread demone separato per non bloccare il ciclo degli eventi di Tkinter. Al completamento, i risultati sono passati alla GUI tramite callback invocate con `self.after(0, ...)`, garantendo così la thread-safety nell'aggiornamento dei widget.

Ogni solutore espone un'interfaccia uniforme:

```python
def solve(A, b, tol, nmax=20000) -> (x, iters, elapsed_time)
```

Questa uniformità permette a `SolverThread` di iterare dinamicamente sul dizionario dei metodi disponibili senza accoppiamento specifico con nessuno di essi.

---

## 4. Tecnologie e Dipendenze

Il progetto è implementato interamente in Python e fa uso delle seguenti librerie:

**NumPy** (`numpy`) costituisce il substrato computazionale per tutte le operazioni vettoriali (norme, prodotti scalari, inizializzazione dei vettori). **SciPy** (`scipy`) è impiegata per la lettura delle matrici nel formato Matrix Market (`scipy.io.mmread`), la conversione in formato CSR (`scipy.sparse.csr_matrix`), la risoluzione di sistemi triangolari (`scipy.sparse.linalg.spsolve_triangular`), la verifica della definita positività tramite calcolo del minimo autovalore (`scipy.sparse.linalg.eigsh` con opzione `which='SM'`), e la rappresentazione efficiente delle matrici sparse.

**Matplotlib** (`matplotlib`) con backend `TkAgg` gestisce la visualizzazione dei grafici comparativi direttamente all'interno della finestra Tkinter. **Tkinter** (`tkinter`), incluso nella libreria standard di Python, fornisce l'infrastruttura per l'interfaccia grafica. Il modulo `tracemalloc`, anch'esso parte della libreria standard, è utilizzato per la profilazione della memoria di picco durante l'esecuzione di ciascun solutore.

---

## 5. Interfaccia Grafica

L'interfaccia è composta da due colonne principali. Il pannello sinistro, di larghezza fissa pari a 260 pixel, raccoglie i controlli operativi: una drop-zone cliccabile per la selezione del file `.mtx`, un campo di testo per l'impostazione della tolleranza (valore predefinito `1e-4`), un pulsante di avvio, una barra di progresso indeterminata e un'area di stato testuale. Il pannello destro, espandibile, ospita un `ttk.Notebook` con tre schede:

La scheda **Dashboard** mostra quattro grafici a barre prodotti con Matplotlib su una griglia 2×2: tempo di esecuzione in secondi, numero di iterazioni, errore relativo in scala logaritmica e memoria di picco in megabyte. Ogni metodo è associato a un colore fisso definito nel dizionario `METHOD_COLORS` di `styles.py`, garantendo coerenza visiva tra grafico e tabella.

La scheda **Tabella risultati** presenta un `ttk.Treeview` con le colonne Metodo, Iterazioni, Tempo, Errore Relativo, Memoria e Stato. Le righe sono colorate in verde per i solutori convergenti e in rosso per quelli falliti.

La scheda **Log** redirige lo `stdout` dell'applicazione verso un widget `tk.Text` in sola lettura, mostrando i messaggi di avanzamento emessi da ciascun solutore durante l'iterazione (ogni 1000 cicli) e un riepilogo tabulare al termine.

Il tema grafico adotta una palette scura con sfondo `#0f1117`, pannelli `#181c27` e accento principale `#4f8ef7`, definiti come costanti in `styles.py`. Lo stile è applicato globalmente tramite `ttk.Style` con tema base `"clam"`.

---

## 6. Validazione Numerica

La correttezza delle implementazioni è verificata costruendo sistemi lineali con soluzione nota. Per ogni matrice $A$ caricata, il vettore termine noto $b$ è calcolato come $b = A \cdot \mathbf{1}$, dove $\mathbf{1}$ è il vettore di tutti uno. La soluzione esatta è quindi $x^* = \mathbf{1}$. L'errore relativo è calcolato dalla funzione `_compute_relative_error` in `metrics.py` come:

$$\epsilon_{\text{rel}} = \frac{\|x^* - x^{(k)}\|_2}{\|x^*\|_2}$$

I metodi del gradiente e del gradiente coniugato eseguono verifiche preliminari sulla matrice prima di avviare l'iterazione. In particolare, verificano che $A$ sia quadrata, simmetrica (controllando che `(A - A.T).nnz == 0`) e definita positiva tramite il calcolo del minimo autovalore con ARPACK. Qualora uno di questi controlli fallisca, il solutore restituisce `None` con zero iterazioni e il risultato viene marcato come fallito nella GUI.

Il metodo di Jacobi e Gauss-Seidel verificano invece la presenza di elementi nulli sulla diagonale principale, condizione che renderebbe la divisione per gli elementi diagonali non definita.

---

## 7. Esperimenti e Risultati

I solutori sono stati progettati per essere valutati su matrici sparse SPD di varie dimensioni, tipicamente provenienti dalla raccolta SuiteSparse Matrix Collection o generate da discretizzazioni di equazioni differenziali alle derivate parziali. Di seguito si riporta uno schema rappresentativo del comportamento atteso, basato sulle proprietà teoriche dei metodi e sulla struttura del codice.

Per una matrice SPD ben condizionata di dimensione moderata (es. $n \approx 1000$, densità < 1%), con tolleranza `1e-4`:

| Metodo | Convergenza | Iterazioni attese | Note |
|---|---|---|---|
| Jacobi | Dipende da $\rho(B_J)$ | Alta (migliaia) | Lento ma robusto |
| Gauss-Seidel | Garantita (SPD) | Moderata | ~2× più veloce di Jacobi |
| Gradiente | Garantita (SPD) | Moderata–alta | Dipende da $\kappa(A)$ |
| Gradiente Coniugato | Garantita (SPD) | Bassa | Metodo ottimale per SPD |

Per matrici mal condizionate o non SPD, i metodi del gradiente e del gradiente coniugato vengono correttamente esclusi dalla computazione tramite il controllo sull'autovalore minimo.

La profilazione della memoria di picco tramite `tracemalloc` permette di quantificare l'overhead introdotto da ciascun metodo rispetto alla sola memorizzazione della matrice sparsa. In generale, il gradiente coniugato richiede la memorizzazione di due vettori aggiuntivi (residuo e direzione di ricerca), contro il singolo vettore residuo dei metodi stazionari.

---

## 8. Discussione

Il confronto tra i quattro metodi evidenzia un chiaro trade-off tra generalità e efficienza. Jacobi e Gauss-Seidel sono applicabili a una classe più ampia di matrici (non richiedono simmetria o definita positività, a differenza degli altri), ma mostrano tipicamente tassi di convergenza inferiori al gradiente coniugato. Quest'ultimo, quando applicabile, rappresenta lo stato dell'arte tra i metodi iterativi non precondizionati per sistemi SPD.

Un aspetto critico dell'implementazione del gradiente coniugato è la scelta della direzione di aggiornamento. La formula per $\beta_k$ adottata nel codice, $\beta_k = (d^{(k)T} A r^{(k+1)}) / (d^{(k)T} A d^{(k)})$, è equivalente alla formulazione di Fletcher-Reeves in aritmetica esatta, ma può accumulare errori numerici su matrici mal condizionate. In tali contesti, un riavvio periodico del metodo (resettando la direzione di ricerca al residuo) rappresenta una pratica comune per stabilizzare la convergenza.

Il controllo della definita positività tramite ARPACK introduce un costo computazionale non trascurabile (calcolo dell'autovalore minimo tramite metodo delle potenze inverse), giustificato tuttavia dalla necessità di evitare applicazioni non corrette del metodo. Su matrici di grandi dimensioni questo controllo potrebbe essere sostituito da euristiche meno costose (ad esempio, verifica della dominanza diagonale).

L'esecuzione in un thread demone separato garantisce la reattività dell'interfaccia durante il calcolo, ma richiede attenzione alla sincronizzazione: tutti gli aggiornamenti UI sono delegati al thread principale attraverso il meccanismo `after(0, ...)` di Tkinter, evitando così race conditions sui widget grafici.

---

## 9. Conclusioni

Il progetto realizza un ambiente integrato per la valutazione comparativa di metodi iterativi su sistemi lineari sparsi, combinando implementazioni numericamente corrette con una GUI professionale e misurazioni quantitative di tempo, iterazioni e memoria. L'architettura modulare, con interfacce uniformi tra i solutori e disaccoppiamento netto tra logica e presentazione, facilita l'aggiunta di nuovi metodi o metriche senza modificare il codice esistente.

Dall'analisi emerge che il Gradiente Coniugato è il metodo preferibile per sistemi SPD grazie al suo tasso di convergenza superlineare e al ridotto numero di iterazioni. Per sistemi non simmetrici o indefiniti, Gauss-Seidel rappresenta una scelta pragmatica, con una convergenza mediamente più rapida rispetto a Jacobi a parità di requisiti. In prospettiva futura, l'introduzione di precondizionatori (ILU, Jacobi a blocchi, AMG) potrebbe ridurre sensibilmente il numero di iterazioni per matrici mal condizionate, estendendo la praticabilità dei metodi di gradiente a problemi di scala industriale.

---

## 10. Riferimenti

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
2. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
3. Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
4. Shewchuk, J. R. (1994). *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*. Carnegie Mellon University, Technical Report CMU-CS-94-125.
5. SciPy Documentation — `scipy.sparse.linalg`. https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
6. Davis, T. A., & Hu, Y. (2011). The University of Florida Sparse Matrix Collection. *ACM Transactions on Mathematical Software*, 38(1), 1–25.
