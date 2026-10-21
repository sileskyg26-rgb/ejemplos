# %% [markdown]
# # 20 - QAOA: resolver el problema Max-Cut con un circuito variacional

# %% Celda 1: importaciones
import itertools
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, SparsePauliOp

# %% Celda 2: el problema Max-Cut
# Dado un grafo, se reparten los vertices en dos grupos (0 o 1) para que haya
# el MAYOR numero posible de aristas entre grupos distintos ("aristas cortadas").
# Ejemplo del PDF: dos lugares turisticos unidos por una arista; x_i dice en
# que dia se visita el lugar i. Funcion a maximizar:
#     L(x) = suma sobre aristas (u,v) de  x_u + x_v - 2 x_u x_v.
aristas = [(0, 1)]
n = 2


def corte(x):
    return sum(x[u] + x[v] - 2 * x[u] * x[v] for u, v in aristas)


for x in itertools.product((0, 1), repeat=n):
    print(f"x = {x} -> aristas cortadas = {corte(x)}")

# %% Celda 3: pasar el problema a un Hamiltoniano
# Se cambia x_i por (I - Z_i)/2 (vale 0 si el qubit esta en |0> y 1 si esta en |1>):
#     Hc = suma sobre aristas de (1 - Z_u Z_v)/2.
# Su valor esperado es el numero promedio de aristas cortadas. Maximizarlo
# resuelve el problema.
def hamiltoniano_costo():
    terminos, coefs = [], []
    for u, v in aristas:
        terminos += ["I" * n, "".join("Z" if i in (u, v) else "I" for i in range(n))[::-1]]
        coefs += [0.5, -0.5]
    return SparsePauliOp(terminos, coefs).simplify()


Hc = hamiltoniano_costo()
Hc_matriz = Hc.to_matrix()
print("Autovalores de Hc:", np.linalg.eigvalsh(Hc_matriz).round(3), "-> el maximo es", Hc_matriz.diagonal().real.max())


# %% Celda 4: el circuito QAOA
# Estado inicial |+>...|+> (todas las cadenas con la misma amplitud). Cada capa tiene:
#   - capa de costo   e^{-i gamma Hc}: para cada arista, una rotacion ZZ,
#   - capa de mezcla  e^{-i beta sum X_i}: una Rx(2 beta) en cada qubit.
# Aqui los parametros se usan como angulos de las puertas: a = -gamma en la rotacion
# ZZ (puerta Rzz(a) = CNOT, Rz(a), CNOT) y b = 2 beta en la Rx.
def circuito_qaoa(params, desplazar=None):
    """Circuito QAOA. `desplazar` = (parametro, numero_de_uso, delta) mueve el angulo
    de UNA sola puerta (se necesita para el gradiente, ver Celda 5)."""
    usos = {}

    def angulo(i):
        k = usos.get(i, 0)
        usos[i] = k + 1
        if desplazar is not None and desplazar[0] == i and desplazar[1] == k:
            return params[i] + desplazar[2]
        return params[i]

    p = len(params) // 2
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for capa in range(p):
        for u, v in aristas:
            qc.cx(u, v)
            qc.rz(angulo(2 * capa), v)
            qc.cx(u, v)
        for q in range(n):
            qc.rx(angulo(2 * capa + 1), q)
    return qc


# Comprobacion: la capa de costo con angulo a = -gamma es e^{-i gamma Hc}
# (salvo una fase global).
gamma = 0.7
capa_costo = QuantumCircuit(n)
capa_costo.cx(0, 1)
capa_costo.rz(-gamma, 1)
capa_costo.cx(0, 1)
exacta = expm(-1j * gamma * Hc_matriz)
M = Operator(capa_costo).data
fase = (exacta[0, 0] / M[0, 0])
print("La capa de costo coincide con e^{-i gamma Hc} (salvo fase):", np.allclose(exacta, fase * M))

# En Colab o Jupyter el circuito se dibuja solo; en terminal no se muestra.
circuito_qaoa([0.5, 0.3]).draw()


# %% Celda 5: el valor esperado <Hc> y su gradiente
def valor_esperado(params, desplazar=None):
    return float(np.real(Statevector(circuito_qaoa(params, desplazar)).expectation_value(Hc)))


def gradiente(params, s=np.pi / 2):
    """Regla del desplazamiento de parametros: dE/d(angulo) = (E(+s) - E(-s)) / 2.

    Un mismo parametro aparece en varias puertas (la mezcla usa el mismo beta en
    todos los qubits). La regla vale para UNA puerta a la vez, asi que se suma la
    contribucion de cada puerta donde aparece el parametro."""
    g = np.zeros_like(params)
    for i in range(len(params)):
        usos = len(aristas) if i % 2 == 0 else n
        for k in range(usos):
            g[i] += (valor_esperado(params, (i, k, s)) - valor_esperado(params, (i, k, -s))) / 2
    return g


# Se comprueba contra una derivada numerica.
p0 = np.array([0.4, 0.9])
eps = 1e-6
numerico = np.array([(valor_esperado(p0 + eps * e) - valor_esperado(p0 - eps * e)) / (2 * eps) for e in np.eye(2)])
print("Gradiente (parameter-shift):", gradiente(p0).round(5), "| numerico:", numerico.round(5))


# %% Celda 6: optimizar (maximizar <Hc>) con p = 1 capa
# Ascenso por gradiente:  params <- params + lr * gradiente.
def optimizar(p, lr=0.3, iteraciones=60, semilla=1):
    rng = np.random.default_rng(semilla)
    params = rng.uniform(0, 1, 2 * p)
    for it in range(iteraciones):
        params = params + lr * gradiente(params)
    return params, valor_esperado(params)


params, valor = optimizar(1)
print(f"p = 1: parametros = {params.round(3)} | <Hc> = {valor:.4f} (maximo posible = 1)")

# %% Celda 7: que cadena se mide con mas probabilidad
# Con los parametros optimos se miden casi siempre cadenas que cortan la arista.
probs = Statevector(circuito_qaoa(params)).probabilities_dict()
for cadena, prob in sorted(probs.items()):
    print(f"cadena {cadena}: probabilidad = {prob:.4f} | aristas cortadas = {corte([int(c) for c in cadena[::-1]])}")

# %% Celda 8: cambiar el numero de capas p
# El PDF pide repetir con otros p para ver si el resultado depende de p.
for p in (1, 2, 3):
    _, valor = optimizar(p)
    print(f"p = {p}: <Hc> = {valor:.4f}")
