# %% [markdown]
# # 13 - Tomografia de estados: reconstruir un estado a partir de mediciones

# %% Celda 1: importaciones
import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": sx, "Y": sy, "Z": sz}
sim = AerSimulator(seed_simulator=123)   # semilla fija: las salidas se pueden reproducir


def fidelidad(rho, estado_puro):
    """Fidelidad de rho con un estado puro: <psi| rho |psi> (1 = identico)."""
    return float(np.real(estado_puro.data.conj() @ rho @ estado_puro.data))


# %% Celda 2: la idea
# Al terminar un calculo hay que "leer" el estado de los qubits. Medir lo
# destruye (colapso), pero la informacion queda escrita en las PROBABILIDADES.
# Las coordenadas del qubit en la esfera de Bloch son (a, b, c) = (<X>, <Y>, <Z>), y
#     P(0) = (1 + c)/2,   P(1) = (1 - c)/2.
# Repitiendo muchas veces la medicion se estima c. Con (a, b, c) se arma la
# matriz densidad:
#     rho = (I + a sx + b sy + c sz) / 2.
# Eso es la tomografia: medir los tres ejes y reconstruir rho.


# %% Celda 3: medir el promedio de un producto de Pauli
# Solo se puede medir en el eje z. Para medir x o y se gira el qubit antes de
# medir (en el laboratorio son pulsos pi/2) y asi ese eje queda en z.
def promedio(preparacion, paulis, shots):
    """Estima <paulis> (por ejemplo "X", "ZZ" o "XI") de un estado.

    `paulis` tiene una letra por qubit, escrita de izquierda a derecha como
    qubit 0, qubit 1, ... La letra "I" significa que ese qubit no se usa.
    """
    qc = preparacion.copy()
    for q, p in enumerate(paulis):
        if p == "X":
            qc.h(q)          # lleva x a z
        elif p == "Y":
            qc.sdg(q)        # sdg y luego h llevan y a z
            qc.h(q)
    qc.measure_all()
    cuentas = sim.run(qc, shots=shots).result().get_counts()
    total = 0
    for bits, n in cuentas.items():
        bits = bits[::-1]    # Qiskit escribe el qubit 0 a la derecha; lo ponemos a la izquierda
        # El resultado de la medicion vale +1 si hay un numero par de 1s entre
        # los qubits medidos, y -1 si es impar.
        unos = sum(1 for q, p in enumerate(paulis) if p != "I" and bits[q] == "1")
        total += (-1) ** unos * n
    return total / shots


# %% Celda 4: reconstruir un qubit
# Preparamos |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1> y lo medimos
# como si no lo conociéramos. Luego comparamos con el estado real.
theta, phi = np.pi / 3, np.pi / 4
prep = QuantumCircuit(1)
prep.ry(theta, 0)
prep.rz(phi, 0)
real = Statevector(prep)
print("Bloch real:", np.round([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 3))
for shots in (100, 1000, 10000):
    a, b, c = (promedio(prep, eje, shots) for eje in "XYZ")
    rho = (I2 + a * sx + b * sy + c * sz) / 2
    fid = fidelidad(rho, real)
    largo = np.sqrt(a**2 + b**2 + c**2)
    print(f"{shots:6d} shots -> (a, b, c) = ({a:+.3f}, {b:+.3f}, {c:+.3f}) | |(a,b,c)| = {largo:.3f} | fidelidad = {fid:.4f}")
# Un estado fisico tiene |(a,b,c)| <= 1. Con pocos shots el ruido estadistico
# puede hacerlo un poco mayor: entonces rho reconstruida no es un estado valido
# y la fidelidad puede salir un poco por encima de 1 (como con 100 shots).
# La fidelidad (1 = identico) mejora al aumentar los shots, como en el capitulo 04.

# %% Celda 5: varios qubits - hay que medir TODAS las correlaciones
# Para dos qubits, rho se expande en los 16 productos de Pauli:
#     rho = (1/4) * suma de  c_ij * (sigma_i (x) sigma_j),   con c_II = 1.
# Los c_ij se llaman coeficientes. No basta con hacer tomografia de cada qubit
# por separado: si estan entrelazados, parte de la informacion esta en las
# correlaciones entre ellos (terminos como XX, YY o ZZ).
def tomografia_2_qubits(preparacion, shots):
    rho = np.zeros((4, 4), dtype=complex)
    coefs = {}
    for p0, p1 in itertools.product("IXYZ", repeat=2):
        c = 1.0 if (p0, p1) == ("I", "I") else promedio(preparacion, p0 + p1, shots)
        coefs[p0 + p1] = c
        # np.kron(A, B): A actua en el qubit 1 (izquierda) y B en el qubit 0 (derecha).
        rho += c * np.kron(PAULI[p1], PAULI[p0]) / 4
    return rho, coefs


# %% Celda 6: tomografia de un estado de Bell
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
rho_bell, coefs = tomografia_2_qubits(bell, 5000)
print("Coeficientes distintos de 0 (el resto ~ 0):", {k: round(v, 2) for k, v in coefs.items() if abs(v) > 0.2})
print("Fidelidad con el Bell ideal:", round(fidelidad(rho_bell, Statevector(bell)), 4))

# %% Celda 7: por que no basta con medir cada qubit por separado
# Cada qubit del estado de Bell, visto solo, esta totalmente mezclado
# (coeficientes XI, YI, ZI, IX, IY, IZ ~ 0, es decir rho = I/2). Si juntamos
# esas dos tomografias en un producto obtenemos I/4, que NO es el estado real:
# lo que le falta son las correlaciones XX, YY y ZZ.
producto = np.kron(I2 / 2, I2 / 2)
print("Fidelidad del producto I/4 con el Bell real:", round(fidelidad(producto, Statevector(bell)), 4))
