# %% [markdown]
# # 05a - Interferencia
# Las amplitudes se suman o se cancelan. Como H H = I, el |1> se cancela.

# %% Celda 1: importaciones
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# %% Celda 2: una H frente a dos H
# Una sola H: superposicion 50/50.
una = QuantumCircuit(1)
una.h(0)
p1 = {str(k): round(float(v), 4) for k, v in Statevector(una).probabilities_dict().items()}
print("1 H :", Statevector(una).data.round(3), p1)

# Dos H seguidas: la segunda H combina las dos amplitudes de la primera.
# La contribucion hacia |1> es (+1/2) + (-1/2) = 0: interferencia
# DESTRUCTIVA. La contribucion hacia |0> es (+1/2) + (+1/2) = 1:
# interferencia CONSTRUCTIVA. El qubit vuelve a |0> con certeza.
dos = QuantumCircuit(1)
dos.h(0)
dos.h(0)
p2 = {str(k): round(float(v), 4) for k, v in Statevector(dos).probabilities_dict().items()}
print("2 H :", Statevector(dos).data.round(3), p2)

# %% Celda 3: la fase decide hacia donde interfiere
# La puerta Z cambia el signo de la amplitud de |1> (fase de pi).
# Entre dos H, ese signo invierte la interferencia: H Z H = X,
# asi que el resultado pasa de |0> a |1> con certeza.
fase = QuantumCircuit(1, 1)
fase.h(0)
fase.z(0)
fase.h(0)
fase.measure(0, 0)
# Esperado: {'1': 1000}. Solo cambiamos una fase y cambio el resultado,
# aunque la Z "no hace nada" a las probabilidades por si sola.
print(AerSimulator().run(fase, shots=1000).result().get_counts())
