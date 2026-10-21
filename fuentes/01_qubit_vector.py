# %% [markdown]
# # 01 - El qubit como vector
# |psi> = alpha|0> + beta|1>, con |alpha|^2 + |beta|^2 = 1

# %% Celda 1: importaciones
# numpy nos da numeros complejos y funciones matematicas (sqrt, exp, pi).
import numpy as np

# Statevector representa el estado de un sistema cuantico como un vector
# de amplitudes complejas (el "vector de estado").
from qiskit.quantum_info import Statevector

# %% Celda 2: los estados de la base computacional
# from_label("0") construye |0>, que en forma de vector columna es (1, 0).
# Estos dos vectores son la base canonica: cualquier qubit es una
# combinacion lineal de ellos.
print("|0> =", Statevector.from_label("0").data)  # [1+0j, 0+0j]
print("|1> =", Statevector.from_label("1").data)  # [0+0j, 1+0j]

# %% Celda 3: un qubit arbitrario
# Elegimos dos amplitudes complejas alpha y beta.
# alpha es real; beta tiene modulo sqrt(2/3) y una fase de pi/4 radianes.
# La fase e^{i*pi/4} NO cambia la probabilidad de medir, pero si cambia
# como interfiere el qubit con otros (ver el ejemplo 05a).
alpha = 1 / np.sqrt(3)
beta = np.sqrt(2 / 3) * np.exp(1j * np.pi / 4)

# Construimos el estado a partir de la lista de amplitudes [alpha, beta].
psi = Statevector([alpha, beta])

# .data devuelve el arreglo de numpy con las amplitudes.
print("Vector:", psi.data)

# Condicion de normalizacion: la suma de los modulos al cuadrado debe ser 1.
# np.abs de un complejo es su modulo; el 1.0000000000000002 que se imprime
# es error de redondeo de punto flotante, no un error del estado.
print("Norma |a|^2+|b|^2 =", np.sum(np.abs(psi.data) ** 2))

# %% Celda 4: de amplitudes a probabilidades (regla de Born)
# probabilities_dict() aplica la regla de Born: P(k) = |amplitud_k|^2.
# Esperamos P(0) = 1/3 y P(1) = 2/3. Convertimos a float para imprimir limpio.
probs = {str(k): round(float(v), 4) for k, v in psi.probabilities_dict().items()}
print("Probabilidades:", probs)

# %% Celda 5: un vector NO normalizado no es un estado valido
# (1, 1) tiene norma^2 = 2, asi que no representa un qubit fisico.
# is_valid() comprueba justamente que la norma sea 1.
print("(1,1) es valido?:", Statevector([1, 1]).is_valid())  # False
print("(1,1)/sqrt(2) es valido?:", Statevector([1, 1] / np.sqrt(2)).is_valid())  # True
