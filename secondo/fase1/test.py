import time
import matplotlib.pyplot as plt
from JPEG import JPEG, DCT2, calcola_D
import numpy as np

# img = apri_immagine("bridge.bmp")

Ns = list(range(50, 500, 50))
times = []

for N in Ns:
    print(N)
    img = np.random.randint(0, 256, size=(1, N, N))

    start = time.perf_counter()
    # JPEG(img, N, N, False)
    D = calcola_D(N)
    DCT2(img, D)
    end = time.perf_counter()

    times.append(end - start)

N0 = Ns[0]
t0 = times[0]
o_n2 = [t0 * (N / N0)**2 for N in Ns]
o_n3 = [t0 * (N / N0)**3 for N in Ns]

plt.figure(figsize=(8, 8))
plt.plot(Ns, times, marker='o', linestyle='-', label='JPEG', linewidth=2)
plt.plot(Ns, o_n2, linestyle='--', label='O(N²)', linewidth=2)
plt.plot(Ns, o_n3, linestyle='--', label='O(N³)', linewidth=2)
plt.xlabel('N')
plt.ylabel('Tempo in secondi')
plt.title('JPEG vs N')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
