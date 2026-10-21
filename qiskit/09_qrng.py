# %% [markdown]
# # 09 - Aleatoriedad cuantica (QRNG), conecta con el Tic-Tac-Toe

# %% Celda 1: importaciones
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# %% Celda 2: generador de bits
def bits_cuanticos(n_bits: int) -> str:
    """Devuelve una cadena de n_bits bits generada midiendo n_bits qubits en
    superposicion. Con H, cada qubit da 0 o 1 con probabilidad 1/2."""
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))                       # cada qubit en (|0> + |1>)/sqrt(2)
    qc.measure(range(n_bits), range(n_bits))
    # memory=True guarda el resultado de CADA shot, no solo el conteo.
    # Con shots=1 obtenemos una unica cadena, p.ej. '11011110'.
    return AerSimulator().run(qc, shots=1, memory=True).result().get_memory()[0]

# %% Celda 3: usar los bits
bits = bits_cuanticos(8)
print("Bits:", bits, "-> entero:", int(bits, 2))  # int(..., 2) lee base 2

# Elegir una casilla libre del tablero 3x3 (indices 0-8).
# El modulo (%) mapea el entero a un indice valido de la lista.
# Nota: 16 valores % 5 casillas no es perfectamente uniforme (leve sesgo);
# para un uso serio se descartan los valores sobrantes y se repite.
libres = [0, 2, 3, 5, 8]
print("Jugada del bot:", libres[int(bits_cuanticos(4), 2) % len(libres)])

# IMPORTANTE: AerSimulator usa un generador pseudoaleatorio. Solo en
# hardware real la aleatoriedad es de origen fisico.
