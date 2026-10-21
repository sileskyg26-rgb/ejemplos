# %% [markdown]
# # 08 - Era NISQ: el ruido produce los resultados 01 y 10

# %% Celda 1: importaciones
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
# NoiseModel agrupa los errores; depolarizing_error(p, k) es un error
# que, con probabilidad p, reemplaza el estado de k qubits por uno aleatorio.
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram

# %% Celda 2: circuito y modelo de ruido
# El mismo circuito de Bell de los ejemplos anteriores.
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

ruido = NoiseModel()
# Error de 1% en cada puerta H (puertas de 1 qubit: errores chicos).
ruido.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h"])
# Error de 5% en cada CNOT. En hardware real las puertas de 2 qubits
# son bastante peores que las de 1, por eso se les da mas error.
ruido.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])
# Los valores son ilustrativos, NO miden ningun equipo concreto.

# %% Celda 3: comparar ideal y ruidoso
ideal = AerSimulator().run(qc, shots=2000).result().get_counts()
# Pasar noise_model hace que el simulador inyecte los errores al ejecutar.
noisy = AerSimulator(noise_model=ruido).run(qc, shots=2000).result().get_counts()
print("Ideal  :", ideal)    # solo 00 y 11
print("Ruidoso:", noisy)    # aparecen 01 y 10: errores que rompen la correlacion
# Este es el panorama NISQ (Noisy Intermediate-Scale Quantum): pocos qubits
# y con errores, lo que limita la profundidad de los circuitos utiles.
plot_histogram([ideal, noisy], legend=["ideal", "ruidoso"])
