# %% [markdown]
# # 07 - Practica guiada: Bell, variantes y reto GHZ

# %% Celda 1: importaciones y simulador
import sys

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# La consola de Windows (cp1252) no imprime el dibujo del circuito sin esto.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sim = AerSimulator()  # un solo simulador reutilizado en todas las celdas

# %% Celda 2: circuito de Bell
qc = QuantumCircuit(2, 2)   # 2 qubits y 2 bits clasicos
qc.h(0)                     # superposicion en el qubit 0
qc.cx(0, 1)                 # CNOT: entrelaza el qubit 1 con el 0
qc.measure([0, 1], [0, 1])  # qubit i -> bit clasico i
print(qc.draw())
# Esperado: ~50% '00' y ~50% '11', y casi nunca '01' ni '10'.
print(sim.run(qc, shots=1000).result().get_counts())

# %% Celda 3: cambiar h(0) por x(0)
# X es el NOT: lleva |0> a |1> sin crear superposicion. El CNOT copia
# ese 1 al qubit 1, asi que el resultado es determinista: siempre '11'.
# Moraleja: el aleatorio de la celda 2 venia de la superposicion de H.
qx = QuantumCircuit(2, 2)
qx.x(0)
qx.cx(0, 1)
qx.measure([0, 1], [0, 1])
print(sim.run(qx, shots=1000).result().get_counts())

# %% Celda 4: reto - estado GHZ de 3 qubits
# Generaliza Bell: (|000> + |111>)/sqrt(2). Cada CNOT "propaga" el valor
# del qubit anterior al siguiente.
ghz = QuantumCircuit(3, 3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(1, 2)
ghz.measure([0, 1, 2], [0, 1, 2])
print(sim.run(ghz, shots=1000).result().get_counts())  # solo '000' y '111'

# %% Celda 5: reto - efecto del numero de shots
# Con pocos shots el histograma es muy irregular (ej. 8 contra 2);
# con muchos se acerca al 50/50 teorico. Es ruido estadistico, no fisico.
for shots in [10, 100, 10000]:
    print(shots, sim.run(qc, shots=shots).result().get_counts())
plot_histogram(sim.run(qc, shots=10000).result().get_counts())
