import time
import matplotlib.pyplot as plt
from JPEG import calcola_D, DCT2
import numpy as np
from scipy.fftpack import dctn

# Costruisco liste utilizzate durante l'esecuzione
Ns = list(range(50, 701, 50))
timesJPEG = []
timesScipy = []
errors = []

for N in Ns:
    # Creo un vettore casuale di dimensione N * N
    img = np.random.randint(0, 256, size = (1, N, N))
    D = calcola_D(N)

    # Calcolo tempistiche funzione nostrana
    start = time.perf_counter()
    c1 = DCT2(img, D)
    end = time.perf_counter()

    timesJPEG.append(end - start)

    # Calcolo tempistiche funzione SciPy
    start = time.perf_counter()
    c2 = dctn(img, norm = "ortho")
    end = time.perf_counter()

    timesScipy.append(end - start)

    # Calcolo errore soluzione nostrana e soluzione SciPy
    error = np.mean(np.abs(c1 - c2))
    errors.append(error)

    # Stampo N per vedere il progresso nella console
    print(N)

# Costruisco la scala N^2 rispetto al punto iniziale di SciPy e N^3 rispetto al punto iniziale nostrano
N0 = Ns[0]
o_n2 = [timesScipy[0] * (N / N0)**2 * np.log(N) / np.log(N0) for N in Ns]
o_n3 = [timesJPEG[0] * (N / N0)**3 for N in Ns]

# Grafici
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(Ns, timesJPEG, marker='o', linestyle='-', label='JPEG', linewidth=2)
ax1.plot(Ns, timesScipy, marker='o', linestyle='-', label='SciPy', linewidth=2)
ax1.plot(Ns, o_n2, linestyle='--', label='O(N²logN)', linewidth=2)
ax1.plot(Ns, o_n3, linestyle='--', label='O(N³)', linewidth=2)
ax1.set_xlabel('N')
ax1.set_ylabel('Tempo in secondi')
ax1.set_title('Tempi di esecuzione')
ax1.set_yscale('log')
ax1.grid(True)
ax1.legend()

ax2.plot(Ns, errors, marker='s', linestyle='-', color='red', linewidth=2, markersize=6)
ax2.set_xlabel('N')
ax2.set_ylabel('Errore assoluto medio')
ax2.set_title('Errore tra JPEG e SciPy')
ax2.grid(True)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()
