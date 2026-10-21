# %% [markdown]
# # 16 - Algoritmo de Bernstein-Vazirani: adivinar un numero oculto con una consulta

# %% Celda 1: importaciones
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, entropy, partial_trace

sim = AerSimulator()

# %% Celda 2: el problema
# Hay un numero oculto de n bits, a = a1 a2 ... an. Solo se puede hacer una pregunta:
# se elige un x de n bits y se recibe f(x) = a . x (mod 2), es decir, la suma (mod 2)
# de los bits donde a y x valen 1. La respuesta es un solo bit (0 o 1).
a_secreto = "1011"          # lo "desconoce" el algoritmo; a1 es el primer bit de la izquierda
n = len(a_secreto)
a = [int(b) for b in a_secreto]


def f(x):
    """La caja negra: devuelve a . x (mod 2)."""
    return sum(ai * xi for ai, xi in zip(a, x)) % 2


# %% Celda 3: solucion clasica, n consultas
# Se pregunta con x = 1000, 0100, 0010, 0001: la respuesta i-esima da el bit a_i.
encontrado = ""
for i in range(n):
    x = [1 if j == i else 0 for j in range(n)]
    encontrado += str(f(x))
    print(f"consulta {i + 1}: x = {''.join(map(str, x))} -> f(x) = {f(x)}")
print("Numero encontrado:", encontrado, f"({n} consultas)")

# %% Celda 4: el oraculo cuantico
# En la version de este ejemplo el oraculo actua asi:  Uf |x> = (-1)^(a . x) |x>.
# Se arma como un producto de puertas de UN solo qubit: una Z en cada qubit donde a_i = 1
# y nada donde a_i = 0. (Z cambia el signo de |1>.)
# Qiskit numera el qubit 0 a la DERECHA al escribir cadenas; aqui el qubit i-1 guarda a_i.
def oraculo(a_bits):
    qc = QuantumCircuit(len(a_bits))
    for i, ai in enumerate(a_bits):
        if ai == 1:
            qc.z(i)
    return qc


# %% Celda 5: el circuito de Bernstein-Vazirani
#   |0...0> -> H en todos -> Uf (una sola consulta) -> H en todos -> medir.
# La H final deshace la superposicion y deja en los qubits justo el numero a.
def bernstein_vazirani(a_bits):
    n = len(a_bits)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.compose(oraculo(a_bits), inplace=True)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


# En Colab o Jupyter el circuito se dibuja solo; en terminal no se muestra.
bernstein_vazirani(a).draw()

# %% Celda 6: ejecutar, con una sola consulta
cuentas = sim.run(bernstein_vazirani(a), shots=1000).result().get_counts()
# Qiskit escribe el qubit 0 a la derecha, asi que se invierte la cadena para leer a1...an.
medido = max(cuentas, key=cuentas.get)[::-1]
print("Mediciones:", cuentas)
print("Numero oculto:", a_secreto, "| medido:", medido, "| 1 consulta en vez de", n)

# %% Celda 7: sin entrelazamiento
# Todas las puertas son de un solo qubit y el estado inicial no esta entrelazado, asi
# que el estado nunca lo esta: la ventaja viene de la superposicion, no del
# entrelazamiento. Se comprueba con la entropia de un qubit (0 = no entrelazado).
sin_medir = QuantumCircuit(n)
sin_medir.h(range(n))
sin_medir.compose(oraculo(a), inplace=True)
sin_medir.h(range(n))
sv = Statevector(sin_medir)
for q in range(n):
    resto = [i for i in range(n) if i != q]
    print(f"entropia del qubit {q}: {float(entropy(partial_trace(sv, resto), base=2)):.3f}")
