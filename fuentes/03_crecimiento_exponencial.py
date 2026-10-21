# %% [markdown]
# # 03 - Con n qubits hay 2^n amplitudes

# %% Celda 1: importaciones
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# %% Celda 2: memoria necesaria para simular n qubits
# Un estado de n qubits tiene 2^n amplitudes complejas.
# Cada complejo de doble precision (complex128) ocupa 16 bytes
# (8 para la parte real y 8 para la imaginaria).
for n in [1, 2, 10, 20, 30, 40, 50]:
    amplitudes = 2 ** n                 # numero de amplitudes
    gib = amplitudes * 16 / 2 ** 30     # bytes -> GiB (1 GiB = 2^30 bytes)
    print(f"n={n:>2}  amplitudes={amplitudes:>18,}  memoria ~ {gib:>14,.4f} GiB")
# Con 30 qubits ya son 16 GiB (una laptop buena); con 50, millones de GiB.
# Este crecimiento exponencial es la razon por la que simular es inviable
# a gran escala, y una de las motivaciones de la computacion cuantica.

# %% Celda 3: ver las 2^n amplitudes de un caso pequeno
n = 4
qc = QuantumCircuit(n)
# qc.h(range(n)) aplica Hadamard a cada uno de los n qubits.
# El resultado es la superposicion uniforme de las 2^n cadenas de bits.
qc.h(range(n))
sv = Statevector(qc)  # calcula el vector de estado del circuito
# Todas las amplitudes valen 1/sqrt(2^n) = 1/4 cuando n = 4.
print(f"{len(sv.data)} amplitudes, cada una = {sv.data[0].real:.4f}")
