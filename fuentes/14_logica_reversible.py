# %% [markdown]
# # 14 - Puertas universales y logica clasica reversible (Toffoli)

# %% Celda 1: importaciones
import itertools
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
# OneQubitEulerDecomposer encuentra los
# angulos de una puerta de un qubit.
from qiskit.quantum_info import Statevector, random_unitary
from qiskit.synthesis import OneQubitEulerDecomposer

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def Rz(t):
    return expm(-1j * t / 2 * sz)


def Ry(t):
    return expm(-1j * t / 2 * sy)


# %% Celda 2: cualquier puerta de un qubit es  U = e^{i a} Rz(b) Ry(g) Rz(d)
# Toda puerta de un qubit es una matriz unitaria 2x2. Siempre se puede escribir como
# una rotacion alrededor de z, luego una alrededor de y y otra alrededor de z, mas una
# fase global e^{i a}. Probamos con una unitaria al azar.
U = random_unitary(2, seed=5).data
# Qiskit calcula los angulos (theta = g, phi, lam) de Rz(phi) Ry(theta) Rz(lam).
theta, phi, lam, fase = OneQubitEulerDecomposer("ZYZ").angles_and_phase(U)
reconstruida = np.exp(1j * fase) * Rz(phi) @ Ry(theta) @ Rz(lam)
print("U se reconstruye con Rz Ry Rz:", np.allclose(U, reconstruida))
print(f"angulos: beta = {phi:.3f}, gamma = {theta:.3f}, delta = {lam:.3f}")

# %% Celda 3: U = e^{i a} A X B X C  con  A B C = I
# Esta descomposicion permite construir la version "controlada" de U (aplicar U solo
# si otro qubit vale 1) con puertas de uno y dos qubits. Se toma:
#   A = Rz(b) Ry(g/2),   B = Ry(-g/2) Rz(-(d+b)/2),   C = Rz((d-b)/2).
beta, gamma, delta = phi, theta, lam
A = Rz(beta) @ Ry(gamma / 2)
B = Ry(-gamma / 2) @ Rz(-(delta + beta) / 2)
C = Rz((delta - beta) / 2)
print("A B C = I      :", np.allclose(A @ B @ C, np.eye(2)))
print("e^{ia} A X B X C = U:", np.allclose(np.exp(1j * fase) * A @ sx @ B @ sx @ C, U))
# Si el qubit de control vale 0 se aplica A B C = I (no pasa nada); si vale 1, las X
# se activan y se aplica A X B X C = U.

# %% Celda 4: la puerta Toffoli (CCX)
# Tiene tres qubits: dos de control y uno objetivo. El objetivo se invierte solo si
# los dos controles valen 1. En Qiskit el orden de la cadena es 'q2 q1 q0': aqui
# q0 y q1 son los controles y q2 el objetivo.
toffoli = QuantumCircuit(3)
toffoli.ccx(0, 1, 2)
print("Tabla de verdad (entrada q2q1q0 -> salida q2q1q0):")
for bits in itertools.product("01", repeat=3):
    entrada = "".join(bits)
    salida = Statevector.from_label(entrada).evolve(toffoli).probabilities_dict()
    print(f"  {entrada} -> {max(salida, key=salida.get)}")
# Solo cambian 011 y 111: el objetivo (q2) se invierte cuando q0 = q1 = 1.

# %% Celda 5: computacion clasica con puertas cuanticas
# Toda puerta logica clasica se puede reescribir con puertas reversibles: X (NOT),
# CNOT y Toffoli. Las entradas son A y B (q0 y q1); la salida queda en C (q2).
def circuito_logico(nombre):
    qc = QuantumCircuit(3)
    if nombre == "AND":                # C empieza en 0; Toffoli lo invierte si A = B = 1
        qc.ccx(0, 1, 2)
    elif nombre == "OR":               # C empieza en 1; sale 0 solo si A = B = 0
        qc.x(2)
        qc.x([0, 1])
        qc.ccx(0, 1, 2)
        qc.x([0, 1])
    elif nombre == "NAND":             # AND y luego NOT a la salida
        qc.ccx(0, 1, 2)
        qc.x(2)
    elif nombre == "NOR":              # OR y luego NOT a la salida
        qc.compose(circuito_logico("OR"), inplace=True)
        qc.x(2)
    elif nombre == "XOR":              # C = A xor B: dos CNOT hacia C
        qc.cx(0, 2)
        qc.cx(1, 2)
    elif nombre == "XNOR":             # XOR y luego NOT a la salida
        qc.compose(circuito_logico("XOR"), inplace=True)
        qc.x(2)
    return qc


clasicas = {
    "AND": lambda a, b: a & b, "OR": lambda a, b: a | b,
    "NAND": lambda a, b: 1 - (a & b), "NOR": lambda a, b: 1 - (a | b),
    "XOR": lambda a, b: a ^ b, "XNOR": lambda a, b: 1 - (a ^ b),
}

# %% Celda 6: comprobar las tablas de verdad
# Para cada puerta, se pone A y B, se corre el circuito y se lee C. Para OR (y NOR)
# el qubit C debe empezar en 1 (lo hace el circuito mismo con la X inicial).
for nombre, f in clasicas.items():
    ok = True
    for a, b in itertools.product((0, 1), repeat=2):
        # Cadena q2 q1 q0 = "0" + str(b) + str(a), con C = q2 en 0.
        inicial = Statevector.from_label(f"0{b}{a}")
        probs = inicial.evolve(circuito_logico(nombre)).probabilities_dict()
        final = max(probs, key=probs.get)
        c = int(final[0])
        ok &= (c == f(a, b))
    print(f"{nombre:5s} coincide con la tabla de verdad clasica: {ok}")
