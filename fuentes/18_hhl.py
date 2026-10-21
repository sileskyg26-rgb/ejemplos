# %% [markdown]
# # 18 - Algoritmo HHL: resolver un sistema lineal Ax = b con dos qubits

# %% Celda 1: importaciones
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# %% Celda 2: el problema clasico
# Dos grupos hacen vasijas: 2.5 x - 0.5 y = 1  y  -0.5 x + 2.5 y = 0.
# En forma matricial es  A x = b  con
A = np.array([[2.5, -0.5], [-0.5, 2.5]])
b = np.array([1.0, 0.0])
x_clasico = np.linalg.solve(A, b)
print("Solucion clasica x = A^-1 b =", x_clasico.round(4), "(5/12, 1/12)")

# %% Celda 3: la idea de HHL, con autovalores y autovectores
# A se escribe con sus autovalores l_i y autovectores |u_i>:  A = suma l_i |u_i><u_i|.
# El vector b se escribe en esa base:  |b> = suma beta_i |u_i>,  beta_i = <u_i|b>.
# Entonces la solucion es   |x> proporcional a  suma (beta_i / l_i) |u_i>:
# basta dividir cada coeficiente entre su autovalor.
autovalores, U = np.linalg.eigh(A)      # las columnas de U son los autovectores
b_norm = b / np.linalg.norm(b)
beta = U.T @ b_norm
x_por_autovalores = U @ (beta / autovalores)
print("Autovalores:", autovalores)
print("beta_i = <u_i|b>:", beta.round(4))
print("Solucion con autovalores (normalizada):", (x_por_autovalores / np.linalg.norm(x_por_autovalores)).round(4))
print("Solucion clasica       (normalizada):", (x_clasico / np.linalg.norm(x_clasico)).round(4))


# %% Celda 4: el circuito de HHL de dos qubits
# HHL tiene tres partes: estimacion de fase (para hallar los autovalores), una
# rotacion controlada (para dividir entre ellos) y la estimacion de fase inversa.
# Para una matriz 2x2 con autovalores conocidos, el circuito se simplifica a dos
# qubits (como en el PDF): q0 = qubit de codificacion, q1 = qubit auxiliar.
#   1. Codificar |b> en q0.
#   2. U^dagger: lleva los autovectores |u_i> a |0> y |1>.
#   3. Rotacion controlada: si q0 vale |i>, Ry(theta_i) sobre q1, con
#      theta_i = 2 arcsin(c / l_i). Asi la amplitud de |1> en q1 es c / l_i.
#   4. U: regresa a la base original.
# c es un numero ajustable con c <= l_min. Aqui se toma el mayor posible.
def hhl_2x2(A, b):
    autovalores, U = np.linalg.eigh(A)
    c = autovalores.min()
    qc = QuantumCircuit(2)
    qc.prepare_state(b / np.linalg.norm(b), [0])        # 1. |b> en q0
    qc.unitary(U.conj().T, [0], label="U^dagger")        # 2. cambio de base
    for k, l in enumerate(autovalores):                  # 3. rotaciones controladas
        theta = 2 * np.arcsin(c / l)
        if k == 0:
            qc.x(0)                                      # controla cuando q0 = |0>
        qc.cry(theta, 0, 1)
        if k == 0:
            qc.x(0)
    qc.unitary(U, [0], label="U")                        # 4. vuelta a la base original
    return qc, c


# En Colab o Jupyter el circuito se dibuja solo; en terminal no se muestra.
hhl_2x2(A, b)[0].draw()


# %% Celda 5: ejecutar y quedarse con el caso "auxiliar = 1"
# La solucion aparece en q0 solo cuando q1 se mide en |1>. Se obtiene con
# probabilidad P1 (depende de c). Las amplitudes de q0 dadas q1 = 1 son proporcionales a x.
def resolver(A, b):
    qc, c = hhl_2x2(A, b)
    amplitudes = Statevector(qc).data           # indices: q1 q0 = 00, 01, 10, 11
    cond = amplitudes[2:]                        # q1 = 1: q0 = 0, 1
    prob_exito = float(np.sum(abs(cond) ** 2))
    return np.real(cond / np.sqrt(prob_exito)), prob_exito


x_cuantico, p_exito = resolver(A, b)
# Una fase global (signo) no importa: se compara el valor absoluto.
x_ref = x_clasico / np.linalg.norm(x_clasico)
print("x cuantico (normalizado):", np.abs(x_cuantico).round(4), "| clasico:", np.abs(x_ref).round(4))
print("Probabilidad de que el auxiliar valga 1:", round(p_exito, 4))

# %% Celda 6: la matriz del experimento
# En el laboratorio se resuelve  A x = b  con
#     A = [[2.14645, -0.35355], [-0.35355, 2.85355]],  b = (0.70711, 0.70711).
# Tiene autovalores 2 y 3. Sus autovectores son (cos(pi/8), sin(pi/8)) y
# (-sin(pi/8), cos(pi/8)): la matriz U es una rotacion Ry(pi/4).
A2 = np.array([[2.14645, -0.35355], [-0.35355, 2.85355]])
b2 = np.array([0.70711, 0.70711])
autovalores2, U2 = np.linalg.eigh(A2)
Ry_pi_4 = np.array([[np.cos(np.pi / 8), -np.sin(np.pi / 8)], [np.sin(np.pi / 8), np.cos(np.pi / 8)]])
print("Autovalores de A:", autovalores2.round(4))
print("U coincide con Ry(pi/4) (salvo signos):", np.allclose(abs(U2), abs(Ry_pi_4), atol=1e-4))

x2, p2 = resolver(A2, b2)
x2_clasico = np.linalg.solve(A2, b2)
x2_clasico = x2_clasico / np.linalg.norm(x2_clasico)
print("x cuantico (normalizado):", np.abs(x2).round(4), "| clasico:", np.abs(x2_clasico).round(4))
print("Probabilidad de que el auxiliar valga 1:", round(p2, 4))

# %% Celda 7: probabilidades que se medirian
# En un equipo real se mide en la base computacional. Las probabilidades de
# |q1 q0> = |00>, |01>, |10>, |11> salen de las amplitudes del estado final.
qc2, _ = hhl_2x2(A2, b2)
probs = np.abs(Statevector(qc2).data) ** 2
for etiqueta, p in zip(("00", "01", "10", "11"), probs):
    print(f"P(|{etiqueta}>) = {p:.4f}")
