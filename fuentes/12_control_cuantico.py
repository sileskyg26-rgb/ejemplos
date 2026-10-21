# %% [markdown]
# # 12 - Control cuantico: Hamiltoniano, rotaciones y acoplamiento entre qubits

# %% Celda 1: importaciones
import numpy as np
from scipy.linalg import expm
# Statevector es el estado; partial_trace y entropy sirven para ver si hay entrelazamiento.
from qiskit.quantum_info import DensityMatrix, Statevector, entropy, partial_trace

# Matrices de Pauli. Trabajamos con hbar = 1 y frecuencias en Hz (por eso aparece 2*pi).
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# %% Celda 2: la ecuacion de Schrodinger da una operacion unitaria
# El Hamiltoniano H es la energia total del sistema y manda su evolucion:
#     i d|psi>/dt = H |psi>.
# Si H no cambia en el tiempo, la solucion es  |psi(t)> = U(t)|psi(0)>  con
#     U(t) = exp(-i H t).
# "Controlar" un qubit es cambiar H para obtener la U que queremos.
H = 2 * np.pi * 1000 * (0.8 * sx + 0.6 * sz) / 2   # un H cualquiera
U = expm(-1j * H * 0.3e-3)
print("U^dagger U = I (es unitaria):", np.allclose(U.conj().T @ U, np.eye(2)))

# Unitaria significa que la suma de probabilidades sigue valiendo 1: no se
# pierde ni se crea nada, y por eso la operacion se puede invertir.
psi = np.array([0.6, 0.8j])
print("Suma de probabilidades antes  :", round(float(np.sum(abs(psi) ** 2)), 6))
print("Suma de probabilidades despues:", round(float(np.sum(abs(U @ psi) ** 2)), 6))


# %% Celda 3: rotaciones de un qubit con pulsos
# Un pulso de duracion t, con frecuencia de Rabi w1 y fase phi, aplica
#     U = exp(-i * 2*pi*w1 * (cos(phi) sx + sin(phi) sy)/2 * t)
# o sea, una ROTACION de angulo  theta = 2*pi*w1*t  alrededor del eje
# cos(phi) x + sin(phi) y.  phi = 0 -> eje x;  phi = pi/2 -> eje y.
def rotacion(theta, phi):
    """Rotacion de angulo theta alrededor del eje cos(phi) x + sin(phi) y."""
    return expm(-1j * theta / 2 * (np.cos(phi) * sx + np.sin(phi) * sy))


def Rx(theta):
    return rotacion(theta, 0.0)


def Ry(theta):
    return rotacion(theta, np.pi / 2)


def Rz(theta):
    return expm(-1j * theta / 2 * sz)   # un pulso solo no la da directamente (ver Celda 4)


w1, t = 1000.0, 0.25e-3                      # pulso de 0.25 ms a 1000 Hz
theta = 2 * np.pi * w1 * t
print(f"theta = 2*pi*w1*t = {theta:.4f} rad (pi/2 = {np.pi / 2:.4f})")
print("El pulso coincide con Rx(pi/2):", np.allclose(expm(-1j * 2 * np.pi * w1 * sx / 2 * t), Rx(np.pi / 2)))

# %% Celda 4: la rotacion alrededor de z se arma combinando tres pulsos
# Un pulso solo puede girar alrededor de ejes del plano xy, no de z. Truco:
#   1. Girar pi/2 alrededor de y   (el eje x pasa a ocupar el lugar de z).
#   2. Girar theta alrededor de x  (equivale a girar alrededor de z).
#   3. Girar pi/2 alrededor de -y  (volver al sistema original).
# El orden importa, porque las rotaciones no conmutan.
theta_z = 0.9
# En un producto de matrices, el primer pulso aplicado queda a la DERECHA.
secuencia = Ry(-np.pi / 2) @ Rx(theta_z) @ Ry(np.pi / 2)
cambiada = Ry(np.pi / 2) @ Rx(theta_z) @ Ry(-np.pi / 2)
print("Ry(pi/2), Rx(th), Ry(-pi/2) == Rz(+th):", np.allclose(secuencia, Rz(theta_z)))
print("Ry(pi/2), Rx(th), Ry(-pi/2) == Rz(-th):", np.allclose(secuencia, Rz(-theta_z)))
print("Con el orden cambiado        == Rz(+th):", np.allclose(cambiada, Rz(theta_z)),
      "| == Rz(-th):", np.allclose(cambiada, Rz(-theta_z)))

# %% Celda 5: un pulso desafinado (fuera de resonancia)
# Si la frecuencia del pulso difiere de la del qubit en dw, aparece un giro
# extra alrededor de z:   H = 2*pi*(dw sz + w1 sx)/2.
# Con desintonia grande el pulso casi no logra voltear al qubit.
ket0 = np.array([1, 0], dtype=complex)
for dw in (0.0, 1000.0, 10000.0):
    Hd = 2 * np.pi * (dw * sz + w1 * sx) / 2
    p1 = abs((expm(-1j * Hd * 0.5e-3) @ ket0)[1]) ** 2   # pulso "pi" de 0.5 ms
    print(f"desintonia = {dw:7.0f} Hz -> P(1) tras un pulso pi = {p1:.3f}")

# %% Celda 6: dos qubits acoplados
# Cuando dos qubits interactuan con acoplamiento J, el Hamiltoniano es
#     H = 2*pi*J * (sz (x) sz)/4.
# Es una rotacion alrededor del eje sz(x)sz: el eje de giro de un qubit
# DEPENDE del estado del otro (+z si el otro es |0>, -z si es |1>).
J = 696.0                       # acoplamiento en Hz
zz = np.kron(sz, sz)


def U_acople(t):
    return expm(-1j * 2 * np.pi * J * zz / 4 * t)


# El qubit 0 empieza en |+> (sobre el ecuador). Segun el estado del otro qubit,
# acumula una fase relativa de signo contrario.
t = 1 / (2 * J)                 # con este tiempo el angulo de rotacion es pi/2
mas = np.array([1, 1], dtype=complex) / np.sqrt(2)
for otro, nombre in ((np.array([1, 0], dtype=complex), "|0>"), (np.array([0, 1], dtype=complex), "|1>")):
    # np.kron(otro, mas): el factor de la izquierda es el qubit 1 y el de la derecha el qubit 0.
    estado = Statevector(U_acople(t) @ np.kron(otro, mas))
    rho = partial_trace(DensityMatrix(estado), [1]).data   # qubit 0 solo
    x, y = np.real(np.trace(rho @ sx)), np.real(np.trace(rho @ sy))
    print(f"otro qubit en {nombre}: qubit 0 queda en (x, y) = ({x:+.2f}, {y:+.2f})")

# %% Celda 7: el acoplamiento entrelaza
# Desde |++>, tras t = 1/(2J) los dos qubits quedan entrelazados. Criterio:
# un estado NO entrelazado se puede escribir como producto de los estados de
# cada qubit (rho = rho_A (x) rho_B). Si no se puede, hay entrelazamiento.
inicial = np.kron(mas, mas)
for t in (0.0, 1 / (2 * J)):
    rho = DensityMatrix(Statevector(U_acople(t) @ inicial))
    rho_A, rho_B = partial_trace(rho, [1]), partial_trace(rho, [0])
    es_producto = np.allclose(rho.data, np.kron(rho_B.data, rho_A.data))
    print(f"t = {t * 1e3:.3f} ms: rho == rho_A (x) rho_B? {es_producto} | "
          f"entropia del qubit 0 = {float(entropy(rho_A, base=2)):.3f} bit")
