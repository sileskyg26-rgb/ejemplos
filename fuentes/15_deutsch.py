# %% [markdown]
# # 15 - Algoritmo de Deutsch: una consulta en vez de dos

# %% Celda 1: importaciones
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

sim = AerSimulator()

# %% Celda 2: el problema (las cajas de pintura)
# Una funcion f(x) con x en {0, 1} es una "caja negra". Hay cuatro posibles:
#   constante: f(0) = f(1)  (siempre 0, o siempre 1)    -> caja de un solo color
#   balanceada: f(0) != f(1)  (identidad o negacion)    -> caja de dos colores
# Clasicamente hay que preguntar dos veces (f(0) y f(1)) para saberlo.
funciones = {
    "constante 0": (0, 0),
    "constante 1": (1, 1),
    "balanceada f(x)=x": (0, 1),
    "balanceada f(x)=1-x": (1, 0),
}
for nombre, (f0, f1) in funciones.items():
    tipo = "constante" if f0 == f1 else "balanceada"
    print(f"{nombre:22s}: f(0) = {f0}, f(1) = {f1} -> {tipo}")


# %% Celda 3: el oraculo Uf
# El oraculo es una puerta unitaria que actua asi:  Uf |x>|y> = |x>|y xor f(x)>.
# q0 es el registro de datos (x) y q1 el registro objetivo (y).
# Se construye aplicando una X al objetivo cuando f(x) = 1:
def oraculo(f0, f1):
    qc = QuantumCircuit(2)
    if (f0, f1) == (1, 1):
        qc.x(1)                         # f(x) = 1 siempre: se invierte siempre
    elif (f0, f1) == (0, 1):
        qc.cx(0, 1)                     # f(x) = x: se invierte si x = 1
    elif (f0, f1) == (1, 0):
        qc.cx(0, 1)                     # f(x) = 1 - x: se invierte si x = 0
        qc.x(1)
    return qc                           # (0, 0): no se hace nada


# %% Celda 4: el circuito de Deutsch
#   1. Datos en |0> y objetivo en |1>.
#   2. H en los dos: datos en (|0>+|1>)/sqrt2 y objetivo en (|0>-|1>)/sqrt2.
#   3. Una sola consulta al oraculo Uf.
#   4. H en el registro de datos y se mide.
# Si sale 0, f es constante; si sale 1, f es balanceada.
def deutsch(f0, f1):
    qc = QuantumCircuit(2, 1)
    qc.x(1)                    # objetivo en |1>
    qc.h([0, 1])
    qc.compose(oraculo(f0, f1), inplace=True)
    qc.h(0)
    qc.measure(0, 0)
    return qc


# En Colab o Jupyter el circuito se dibuja solo; en terminal no se muestra.
deutsch(0, 1).draw()

# %% Celda 5: ejecutar con las cuatro funciones
# Una sola consulta basta, y el resultado es siempre el mismo (no es aleatorio).
for nombre, (f0, f1) in funciones.items():
    cuentas = sim.run(deutsch(f0, f1), shots=1000).result().get_counts()
    dato = max(cuentas, key=cuentas.get)
    veredicto = "constante" if dato == "0" else "balanceada"
    print(f"{nombre:22s}: medicion = {cuentas} -> {veredicto}")

# %% Celda 6: por que funciona (cambio de signo por la fase)
# Con el objetivo en (|0>-|1>)/sqrt2, el oraculo solo cambia el signo de cada x:
#   Uf |x> (|0>-|1>)/sqrt2 = (-1)^f(x) |x> (|0>-|1>)/sqrt2.
# El registro de datos queda en ((-1)^f(0)|0> + (-1)^f(1)|1>)/sqrt2:
#   constante  -> (|0>+|1>)/sqrt2  (o su opuesto)  -> la H final lo lleva a |0>
#   balanceada -> (|0>-|1>)/sqrt2  (o su opuesto)  -> la H final lo lleva a |1>
for nombre, (f0, f1) in funciones.items():
    qc = QuantumCircuit(2)
    qc.x(1)
    qc.h([0, 1])
    qc.compose(oraculo(f0, f1), inplace=True)
    amplitudes = Statevector(qc).data
    # El objetivo (q1) vale (|0>-|1>)/sqrt2, asi que las amplitudes con q1 = 0 (indices
    # 0 y 1) son las de los datos multiplicadas por 1/sqrt2. Multiplicando por 2 queda
    # +-1, que son los signos (-1)^f(x) de la formula.
    a0, a1 = amplitudes[0] * 2, amplitudes[1] * 2
    print(f"{nombre:22s}: datos = ({a0.real:+.0f}|0> {a1.real:+.0f}|1>)/sqrt2")
