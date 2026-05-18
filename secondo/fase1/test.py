import time
import matplotlib.pyplot as plt
from JPEG import calcola_D , DCT2
import numpy as np
from scipy.fftpack import dctn

Ns = list(range(50, 700, 50))
timesJPEG = []
timesScipy = []

for N in Ns:
    print(N)
    img = np.random.randint(0, 256, size=(1, N, N))

    start = time.perf_counter()
    D = calcola_D(N)
    DCT2(img, D)
    end = time.perf_counter()

    timesJPEG.append(end - start)

    start = time.perf_counter()
    dctn(img, norm = "ortho")
    end = time.perf_counter()

    timesScipy.append(end - start)

N0 = Ns[0]
o_n2 = [timesScipy[0] * (N / N0)**2 * np.log(N) / np.log(N0) for N in Ns]
o_n3 = [timesJPEG[0] * (N / N0)**3 for N in Ns]

plt.figure(figsize=(8, 8))
plt.plot(Ns, timesJPEG, marker='o', linestyle='-', label='JPEG', linewidth=2)
plt.plot(Ns, timesScipy, marker='o', linestyle='-', label='Scipy', linewidth=2)
plt.plot(Ns, o_n2, linestyle='--', label='O(N²logN)', linewidth=2)
plt.plot(Ns, o_n3, linestyle='--', label='O(N³)', linewidth=2)
plt.xlabel('N')
plt.ylabel('Tempo in secondi')
plt.title('JPEG vs N')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()