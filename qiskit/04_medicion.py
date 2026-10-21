# %% [markdown]
# # 04 - Medicion: P(0) = |alpha|^2 y P(1) = |beta|^2

# %% Celda 1: importaciones
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator  # simulador local de circuitos
from qiskit.visualization import plot_histogram

# La consola de Windows (cp1252) no imprime el dibujo del circuito sin esto.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# %% Celda 2: preparar un qubit con P(1) = 0.25
# La rotacion Ry(theta) lleva |0> a cos(theta/2)|0> + sin(theta/2)|1>.
# Queremos sin^2(theta/2) = 0.25, es decir theta = 2*arcsin(sqrt(0.25)).
theta = 2 * np.arcsin(np.sqrt(0.25))

# QuantumCircuit(1, 1): 1 qubit y 1 bit clasico donde guardar la medicion.
qc = QuantumCircuit(1, 1)
qc.ry(theta, 0)     # prepara el estado con las amplitudes deseadas
qc.measure(0, 0)    # mide el qubit 0 y guarda el resultado en el bit clasico 0
print(qc.draw())    # dibujo del circuito en texto

# %% Celda 3: repetir la medicion (shots)
# Una sola medicion da un unico 0 o 1 y destruye la superposicion.
# Para estimar las probabilidades hay que repetir el experimento muchas
# veces: cada repeticion se llama "shot".
sim = AerSimulator()
for shots in [10, 100, 1000, 10000]:
    # run() lanza el circuito; result().get_counts() devuelve cuantas veces
    # salio cada resultado, por ejemplo {'0': 749, '1': 251}.
    counts = sim.run(qc, shots=shots).result().get_counts()
    print(shots, "shots ->", counts, f"| freq(1) = {counts.get('1', 0) / shots:.3f}")
# Ley de los grandes numeros: con mas shots, freq(1) se acerca a 0.25.

# %% Celda 4: histograma
# En Colab o Jupyter la figura se muestra sola; en terminal se puede
# guardar con plot_histogram(...).savefig("hist.png").
plot_histogram(sim.run(qc, shots=1000).result().get_counts())
