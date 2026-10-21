# %% [markdown]
# # 02 - Las puertas cuanticas son matrices unitarias
# H = 1/sqrt(2) * [[1, 1], [1, -1]]

# %% Celda 1: importaciones
import numpy as np
from qiskit import QuantumCircuit  # un circuito es una lista de puertas
from qiskit.quantum_info import Operator, Statevector
# Operator convierte un circuito (o una puerta) en su matriz unitaria.

# %% Celda 2: la matriz de cada puerta
# Para cada puerta construimos un circuito de 1 qubit con solo esa puerta
# y le pedimos a Qiskit su matriz.
for nombre, puerta in [("H", "h"), ("X", "x"), ("Z", "z")]:
    qc = QuantumCircuit(1)
    # getattr(qc, "h")(0) es lo mismo que qc.h(0): aplica la puerta al qubit 0.
    getattr(qc, puerta)(0)
    # np.round limpia el ruido numerico (0.7071067811865476 -> 0.707).
    print(nombre, "=\n", np.round(Operator(qc).data, 3))
# H: crea superposicion. X: "NOT" cuantico (intercambia |0> y |1>).
# Z: deja |0> igual y cambia el signo de |1> (puerta de fase).

# %% Celda 3: aplicar una puerta = multiplicar matriz por vector
H = Operator.from_label("H").data  # matriz 2x2 de Hadamard
ket0 = np.array([1, 0])            # el vector |0>
# El operador @ es el producto matricial de numpy.
print("H @ |0> =", H @ ket0)       # (1/sqrt2, 1/sqrt2): superposicion igual

# Qiskit hace exactamente lo mismo con .evolve(): aplica el operador al estado.
evolucionado = Statevector.from_label("0").evolve(Operator.from_label("H"))
print("Statevector:", evolucionado.data)

# %% Celda 4: las puertas son unitarias
# Una matriz U es unitaria si U^dagger U = I, donde U^dagger es la
# transpuesta conjugada (.conj().T). Esto garantiza que la norma del
# vector se conserve: una puerta nunca rompe la normalizacion.
print("H^dagger H = I:", np.allclose(H.conj().T @ H, np.eye(2)))

# Como consecuencia, toda puerta es reversible: su inversa es U^dagger.
# En particular H es su propia inversa (H H = I), lo que se usa en 05a.
print("H H = I:", np.allclose(H @ H, np.eye(2)))
