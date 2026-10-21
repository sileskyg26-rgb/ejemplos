# %% [markdown]
# # 10 - La esfera de Bloch: un qubit como un punto en una esfera

# %% Celda 1: importaciones
import numpy as np
# Statevector guarda el estado; Pauli permite calcular los promedios <X>, <Y>, <Z>.
from qiskit.quantum_info import Pauli, Statevector
from qiskit.visualization import plot_bloch_multivector


def vector_bloch(sv):
    """Devuelve (x, y, z) = (<X>, <Y>, <Z>): la posicion del qubit en la esfera."""
    return np.array([sv.expectation_value(Pauli(p)).real for p in "XYZ"])


# %% Celda 2: de dos amplitudes complejas a dos angulos
# Un qubit es  |psi> = r1 e^{i g1}|0> + r2 e^{i g2}|1>.  Dos ideas permiten
# simplificarlo:
#  1) Solo importa la fase RELATIVA (g2 - g1); la fase global no se puede observar.
#  2) Como r1^2 + r2^2 = 1, se pueden escribir r1 = cos(theta/2), r2 = sin(theta/2).
# Resultado:  |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>,  con phi = g2 - g1.
# Cada qubit queda determinado por dos angulos (theta, phi): un punto de la esfera.
theta, phi = np.pi / 3, np.pi / 4
sv = Statevector([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
print("Amplitudes:", sv.data.round(3))

# %% Celda 3: la fase global no se puede observar
# Multiplicar todo el estado por e^{i gamma} no cambia nada medible:
# el punto de la esfera es el mismo.
con_fase_global = Statevector(np.exp(1j * 1.234) * sv.data)
print("Sin fase global:", vector_bloch(sv).round(4))
print("Con fase global:", vector_bloch(con_fase_global).round(4))

# %% Celda 4: los estados especiales
# |0> es el polo norte (z = +1) y |1> el polo sur (z = -1).
# Las superposiciones con igual probabilidad de 0 y 1 caen en el ecuador.
estados = {
    "|0>": Statevector.from_label("0"),
    "|1>": Statevector.from_label("1"),
    "|+>": Statevector.from_label("+"),    # (|0> + |1>)/sqrt2
    "|->": Statevector.from_label("-"),    # (|0> - |1>)/sqrt2
    "|+i>": Statevector.from_label("r"),   # (|0> + i|1>)/sqrt2
}
for nombre, s in estados.items():
    x, y, z = vector_bloch(s)
    print(f"{nombre:5s} -> (x, y, z) = ({x:+.2f}, {y:+.2f}, {z:+.2f})")

# %% Celda 5: theta y phi son los angulos de la esfera
# theta es el angulo desde el eje z (decide las probabilidades de medir 0 o 1).
# phi es el giro alrededor del eje z (la fase relativa).
# El punto es (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)).
esperado = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
print("Calculado:", vector_bloch(sv).round(4))
print("Formula  :", np.round(esperado, 4))
print("P(0), P(1) =", sv.probabilities().round(4), " (dependen solo de theta)")

# %% Celda 6: cambiar phi no cambia las probabilidades
# Al girar alrededor del eje z cambian x e y, pero z (y por tanto P(0) y P(1))
# se queda igual. La fase relativa solo se nota despues, por interferencia.
for phi_nuevo in (0, np.pi / 2, np.pi):
    s = Statevector([np.cos(theta / 2), np.exp(1j * phi_nuevo) * np.sin(theta / 2)])
    x, y, z = vector_bloch(s)
    print(f"phi = {phi_nuevo:4.2f} -> (x, y, z) = ({x:+.3f}, {y:+.3f}, {z:+.3f}) | P(1) = {s.probabilities()[1]:.3f}")

# %% Celda 7: dibujar la esfera
# En Colab o Jupyter la figura se muestra sola.
plot_bloch_multivector(sv)
