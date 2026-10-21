# %% [markdown]
# # 06 - Grover: busqueda cuadratica O(sqrt(N)) frente a O(N)

# %% Celda 1: importaciones
import numpy as np
from qiskit import QuantumCircuit
# grover_operator arma el operador de Grover G = (difusion)(oraculo).
from qiskit.circuit.library import grover_operator
from qiskit.quantum_info import Statevector

# %% Celda 2: el oraculo
# Problema: entre N = 2^3 = 8 cadenas, encontrar la marcada "101".
# El oraculo es una puerta que cambia el signo (fase -1) SOLO de la
# cadena marcada, sin revelar cual es: U|x> = -|x> si x == "101", y |x> si no.
n, marcado = 3, "101"

# Qiskit numera el qubit 0 a la derecha de la cadena, por eso se invierte
# con reversed(). Buscamos los qubits donde la cadena marcada tiene un 0.
ceros = [i for i, b in enumerate(reversed(marcado)) if b == "0"]

oraculo = QuantumCircuit(n)
oraculo.x(ceros)                     # convierte "101" en "111" temporalmente
oraculo.h(n - 1)                     # H - mcx - H sobre el ultimo qubit
oraculo.mcx(list(range(n - 1)), n - 1)  # equivale a un CCZ: fase -1 si todo es 1
oraculo.h(n - 1)
oraculo.x(ceros)                     # deshace el primer paso

# %% Celda 3: aplicar Grover
G = grover_operator(oraculo)         # un paso = oraculo + difusion
# El numero optimo de iteraciones es ~ (pi/4)*sqrt(N). Aqui es 2.
iteraciones = int(np.floor(np.pi / 4 * np.sqrt(2 ** n)))

qc = QuantumCircuit(n)
qc.h(range(n))                       # superposicion uniforme de las 8 cadenas
for _ in range(iteraciones):
    qc.compose(G, inplace=True)      # agrega G al final del circuito
print("Iteraciones de Grover:", iteraciones)

# Cada iteracion amplifica la amplitud de la cadena marcada por interferencia.
probs = Statevector(qc).probabilities_dict()
print("P(", marcado, ") =", round(float(probs[marcado]), 3))  # ~0.945

# %% Celda 4: comparacion de consultas
# Clasico: en promedio hay que revisar N/2 elementos. Grover: ~(pi/4)*sqrt(N).
# La ventaja es cuadratica (no exponencial como en Shor).
for N in [8, 1_000, 1_000_000, 10**12]:
    print(f"N={N:>16,}  clásico ~ {N // 2:>16,}  Grover ~ {int(np.pi / 4 * np.sqrt(N)):>10,}")
