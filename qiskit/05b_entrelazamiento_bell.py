# %% [markdown]
# # 05b - Entrelazamiento: estado de Bell (|00> + |11>)/sqrt(2)

# %% Celda 1: importaciones
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
# partial_trace "olvida" un qubit y deja el estado del otro; entropy mide
# cuanta incertidumbre (mezcla) tiene ese estado reducido.
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.visualization import plot_histogram

# %% Celda 2: construir el estado de Bell
bell = QuantumCircuit(2)
bell.h(0)        # H sobre el qubit 0: (|0> + |1>)/sqrt(2) en ese qubit
bell.cx(0, 1)    # CNOT: invierte el qubit 1 solo si el qubit 0 vale 1
# Resultado: (|00> + |11>)/sqrt(2). Ojo: Qiskit escribe las cadenas con el
# qubit 0 a la DERECHA (orden little-endian), pero aqui es simetrico.
sv = Statevector(bell)
print("Estado:", sv.data.round(3))  # amplitudes de |00>, |01>, |10>, |11>

# %% Celda 3: prueba de que NO es un producto de dos qubits
# Si el estado fuera producto |a>|b>, cada qubit tendria un estado propio
# bien definido (entropia 0). Aqui, al descartar el qubit 1 el qubit 0
# queda totalmente mezclado: entropia = 1 bit, el maximo para un qubit.
# Eso es entrelazamiento: el todo esta bien definido, las partes no.
rho0 = partial_trace(sv, [1])   # matriz densidad del qubit 0
print("Entropía de entrelazamiento:", entropy(rho0, base=2))  # 1.0

# %% Celda 4: la correlacion al medir
m = bell.copy()
m.measure_all()   # agrega un registro clasico y mide todos los qubits
counts = AerSimulator().run(m, shots=1000).result().get_counts()
# Solo aparecen '00' y '11' (~50% cada uno). Nunca '01' ni '10':
# los resultados estan perfectamente correlacionados, aunque cada uno
# por separado es aleatorio. Esto NO permite enviar informacion
# (no-signaling): el resultado de uno no se puede elegir.
print(counts)
plot_histogram(counts)
