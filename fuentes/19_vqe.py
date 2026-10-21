# %% [markdown]
# # 19 - VQE: encontrar la energia del estado fundamental con un circuito parametrizado

# %% Celda 1: importaciones
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
import matplotlib.pyplot as plt

# %% Celda 2: el problema
# Se busca la energia mas baja de un Hamiltoniano (la energia del estado fundamental).
# Aqui: el modelo de Heisenberg de dos qubits,  H = XX + YY + ZZ.
H = SparsePauliOp(["XX", "YY", "ZZ"])
H_matriz = H.to_matrix()
# Como el sistema es pequeno, se puede diagonalizar para saber la respuesta exacta.
print("Energias exactas de H:", np.linalg.eigvalsh(H_matriz).round(4))
print("Energia del estado fundamental:", float(np.linalg.eigvalsh(H_matriz).min()))
# VQE busca ese minimo SIN diagonalizar, usando solo valores esperados.

# %% Celda 3: dos formas de calcular el valor esperado <H>
# (1) Con el vector de estado:   <H> = <psi| H |psi>.
# (2) Con la matriz densidad:    <H> = tr(H rho),   rho = |psi><psi|.
# Dan el mismo numero. Se prueba con un estado cualquiera.
psi = Statevector(np.array([1, 1j, -1, 0.5]) / np.linalg.norm([1, 1j, -1, 0.5]))
via_vector = np.real(psi.data.conj() @ H_matriz @ psi.data)
via_traza = np.real(np.trace(H_matriz @ DensityMatrix(psi).data))
print(f"<psi|H|psi> = {via_vector:.4f} | tr(H rho) = {via_traza:.4f}")


# %% Celda 4: el circuito parametrizado ("ansatz")
# El estado de prueba |psi(theta)> sale de aplicar un circuito con parametros
# theta = (theta1, theta2) al estado inicial |00>:
#     Ry(theta1) en q0, CNOT(q0 -> q1), Rx(theta2) en q0.
def ansatz(theta):
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.cx(0, 1)
    qc.rx(theta[1], 0)
    return qc


def energia(theta):
    """Valor esperado <H> del estado de prueba (via tr(H rho))."""
    rho = DensityMatrix(Statevector(ansatz(theta)))
    return float(np.real(np.trace(H_matriz @ rho.data)))


# En Colab o Jupyter el circuito se dibuja solo; en terminal no se muestra.
ansatz([0.3, 0.5]).draw()

# %% Celda 5: regla del desplazamiento de parametros (parameter-shift)
# Para saber hacia donde mover cada parametro hace falta el gradiente dE/dtheta.
# Con una puerta de rotacion exp(-i theta P / 2) se puede obtener exacto con
# solo dos evaluaciones del circuito:
#     dE/dtheta = ( E(theta + s) - E(theta - s) ) / 2,   con s = pi/2.
def gradiente(theta, s=np.pi / 2):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        mas, menos = theta.copy(), theta.copy()
        mas[i] += s
        menos[i] -= s
        g[i] = (energia(mas) - energia(menos)) / 2
    return g


# Se comprueba contra una derivada numerica con un paso muy pequeno.
theta0 = np.array([1.1, 2.3])
eps = 1e-6
numerico = np.array([(energia(theta0 + eps * e) - energia(theta0 - eps * e)) / (2 * eps) for e in np.eye(2)])
print("Gradiente (parameter-shift):", gradiente(theta0).round(5))
print("Gradiente (numerico)       :", numerico.round(5))

# %% Celda 6: el ciclo de VQE
# 1. Elegir parametros iniciales.
# 2. Calcular <H> (computador cuantico).
# 3. Calcular el gradiente con parameter-shift.
# 4. Actualizar:  theta <- theta - lr * gradiente  (computador clasico).
# 5. Repetir hasta que <H> converja.
theta = np.array([1.3, 2.0])      # parametros iniciales (radianes)
lr = 0.3                          # tasa de aprendizaje
historia = [energia(theta)]
print("iteracion | theta1  theta2 |  <H>")
for it in range(1, 41):
    theta = theta - lr * gradiente(theta)
    historia.append(energia(theta))
    if it <= 5 or it % 10 == 0:
        print(f"{it:9d} | {theta[0]:6.3f} {theta[1]:6.3f} | {historia[-1]:+.4f}")
print(f"Energia final = {historia[-1]:.4f} (exacta: -3)")

# %% Celda 7: grafica de convergencia
plt.plot(historia, "o-")
plt.axhline(-3, color="gray", linestyle="--", label="energía exacta (-3)")
plt.xlabel("iteración")
plt.ylabel("<H>")
plt.title("Convergencia de VQE")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
