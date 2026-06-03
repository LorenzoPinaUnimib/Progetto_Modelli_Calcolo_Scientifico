## Metodo di Jacobi

Il metodo di Jacobi è un metodo stazionario di tipo splitting. Dato il sistema $Ax = b$, si decompone $A = D + R$ dove $D$ è la matrice diagonale e $R = L + U$ raccoglie le parti triangolari strettamente inferiore e superiore. L'iterazione si definisce come:

$$x^{(k+1)} = D^{-1}(b - Rx^{(k)})$$

Equivalentemente, ad ogni passo si aggiorna $x^{(k+1)} = x^{(k)} + D^{-1}r^{(k)}$ dove $r^{(k)} = b - Ax^{(k)}$ è il residuo corrente. La convergenza è garantita quando la matrice di iterazione $B_J = -D^{-1}R$ ha raggio spettrale $\rho(B_J) < 1$, condizione soddisfatta, ad esempio, quando $A$ è a dominanza diagonale stretta. Il metodo non richiede che la matrice sia simmetrica o definita positiva, ma è generalmente più lento di Gauß-Seidel.

## Metodo di Gauß-Seidel

Gauß-Seidel è una variante di Jacobi che utilizza immediatamente i valori aggiornati durante la stessa iterazione. Lo splitting utilizzato è $A = P + N$ dove $P = D + L$ è la parte triangolare inferiore (inclusa la diagonale). L'iterazione diventa:

$$Px^{(k+1)} = b - Ux^{(k)}$$

La risoluzione del sistema triangolare inferiore viene effettuata in modo efficiente con la tecnica di _forward substitution_. Rispetto a Jacobi, Gauß-Seidel converge tipicamente in meno iterazioni; per matrici simmetriche e definite positive, la convergenza è garantita. La matrice di iterazione associata è $B_{GS} = -(D+L)^{-1}U$.

## Metodo del Gradiente

Il metodo del gradiente (o _steepest descent_) è un metodo di discesa applicabile a sistemi con matrice $A$ simmetrica e definita positiva (SPD). Il sistema $Ax = b$ è equivalente alla minimizzazione del funzionale quadratico $\phi(x) = \frac{1}{2}x^TAx - b^Tx$. Ad ogni iterazione si procede nella direzione del residuo (gradiente negativo della funzione costo):

$$x^{(k+1)} = x^{(k)} + \alpha_k r^{(k)}, \qquad \alpha_k = \frac{(r^{(k)})^T r^{(k)}}{(r^{(k)})^T A r^{(k)}}$$

Il passo ottimale $\alpha_k$ minimizza $\phi$ lungo la direzione corrente. La convergenza dipende dal numero di condizionamento $\kappa(A) = \lambda_{\max}/\lambda_{\min}$: per matrici mal condizionate la convergenza può essere molto lenta, poiché le direzioni di ricerca successive tendono a formare angoli piccoli causando un andamento a zig-zag.

## Metodo del Gradiente Coniugato

Il gradiente coniugato (CG) supera il limite del metodo del gradiente semplice generando direzioni di ricerca $A$-coniugate, ovvero ortogonali rispetto al prodotto scalare indotto da $A$. L'aggiornamento della direzione introduce un termine correttivo $\beta_k$:

$$d^{(k+1)} = r^{(k+1)} - \beta_k d^{(k)}, \qquad \beta_k = \frac{(d^{(k)})^T A r^{(k+1)}}{(d^{(k)})^T A d^{(k)}}$$

In aritmetica esatta, il CG converge in al più $n$ iterazioni (dove $n$ è la dimensione del sistema), comportandosi come un metodo diretto. In pratica, per matrici SPD ben condizionate, la convergenza avviene in un numero di iterazioni molto inferiore a $n$. Il tasso di convergenza è governato da $\sqrt{\kappa(A)}$ anziché da $\kappa(A)$ come nel gradiente semplice, il che costituisce un vantaggio sostanziale.

## Criteri di Arresto

Tutti i metodi implementati adottano un criterio di arresto basato sul residuo relativo in norma infinito:

$$\frac{\|Ax^{(k)} - b\|}{\|b\|} < \text{tol}$$

Il numero massimo di iterazioni è fissato a $n_{\max} = 50000$ per ciascun metodo; nel caso un metodo non riesca a convergere entro questo numero, verrà avvisato l'utente nel log e nella tabella dei risultati.
