# %% [markdown]
# # 11 - Oscilacion de Rabi: como un pulso de radiofrecuencia cambia al qubit

# %% Celda 1: importaciones
import numpy as np
# expm calcula la exponencial de una matriz: lo necesitamos para evolucionar el estado.
from scipy.linalg import expm
# curve_fit ajusta una curva a datos (como se hace en el laboratorio para hallar w1).
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Matrices de Pauli. Trabajamos con hbar = 1 y frecuencias en Hz (por eso el 2*pi).
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
ket0 = np.array([1, 0], dtype=complex)

# %% Celda 2: el modelo (en el sistema que gira junto con el pulso)
# Un qubit de espin esta en un campo magnetico fijo B0 (eje z) y recibe un
# campo de radiofrecuencia (RF). En el "sistema rotante" el efecto del pulso es
# un campo FIJO de intensidad w1 (frecuencia de Rabi) cuya direccion en el
# plano xy la decide la fase phi del pulso:  eje = cos(phi) x + sin(phi) y.
# Si ademas el pulso esta desafinado (desintonia dw = w0 - w_rf), el qubit gira
# tambien alrededor de z. El Hamiltoniano completo es:
#     H = 2*pi*dw * sz/2  +  2*pi*w1 * (cos(phi) sx + sin(phi) sy)/2
def hamiltoniano(w1, dw=0.0, phi=0.0):
    return 2 * np.pi * (dw * sz + w1 * (np.cos(phi) * sx + np.sin(phi) * sy)) / 2


def estado_tras_pulso(w1, t, dw=0.0, phi=0.0):
    """Estado despues de un pulso de duracion t, partiendo de |0>: U(t)|0>, con U = exp(-iHt)."""
    return expm(-1j * hamiltoniano(w1, dw, phi) * t) @ ket0


# %% Celda 3: con pulso resonante (dw = 0), el angulo de giro es 2*pi*w1*t
# El estado gira alrededor del eje del pulso un angulo  theta = 2*pi*w1*t.
# Partiendo de |0>, el angulo theta de la esfera de Bloch es justo ese,
# asi que  P(1) = sin^2(theta/2) = sin^2(pi*w1*t): oscila entre 0 y 1.
w1 = 1000.0   # frecuencia de Rabi: 1000 Hz (1 oscilacion completa por milisegundo)
for t_ms in (0.25, 0.5, 0.75, 1.0):
    t = t_ms * 1e-3
    p1 = abs(estado_tras_pulso(w1, t)[1]) ** 2
    print(f"t = {t_ms:4.2f} ms -> theta = {2 * np.pi * w1 * t:5.2f} rad | P(1) simulada = {p1:.3f} | sin^2(pi w1 t) = {np.sin(np.pi * w1 * t) ** 2:.3f}")
# t = 0.5 ms es un pulso "pi" (invierte el qubit, como la puerta X);
# t = 0.25 ms es un pulso "pi/2" (superposicion 50/50).

# %% Celda 4: la fase del pulso decide el eje de giro
# phi = 0 gira alrededor de x; phi = pi/2, alrededor de y. Un pulso pi/2
# deja al qubit en puntos distintos del ecuador segun el eje elegido.
t = 0.25e-3   # pulso pi/2
for phi, nombre in ((0.0, "eje x"), (np.pi / 2, "eje y")):
    est = estado_tras_pulso(w1, t, phi=phi)
    x, y, z = (np.real(est.conj() @ (s @ est)) for s in (sx, sy, sz))
    print(f"phi = {phi:4.2f} ({nombre}) -> (x, y, z) = ({x:+.2f}, {y:+.2f}, {z:+.2f})")

# %% Celda 5: simular el experimento del laboratorio
# En el laboratorio no se mide el estado de un solo golpe. Se hace asi:
#   1. Se prepara siempre |0>.
#   2. Se aplica un pulso de duracion t y se mide muchas veces (shots).
#   3. Se repite aumentando poco a poco la duracion del pulso (cada medicion
#      colapsa el estado, asi que se reinicia el experimento).
#   4. Se ajusta una funcion seno a los datos y de ahi sale w1.
rng = np.random.default_rng(7)          # semilla: el resultado es reproducible
tiempos = np.linspace(0, 2e-3, 41)       # anchos de pulso de 0 a 2 ms
shots = 500
# (min/max evita que un redondeo deje la probabilidad en 1.0000000000000002)
p1_medida = np.array([rng.binomial(shots, min(1.0, abs(estado_tras_pulso(w1, t)[1]) ** 2)) / shots for t in tiempos])


def modelo(t, f):
    return np.sin(np.pi * f * t) ** 2


(f_ajustada,), _ = curve_fit(modelo, tiempos, p1_medida, p0=[900.0])
print(f"w1 real = {w1:.0f} Hz | w1 ajustada con {shots} shots = {f_ajustada:.1f} Hz")

# %% Celda 6: si el pulso esta desafinado, casi no excita
# Con desintonia, el qubit gira alrededor de z mientras el pulso intenta
# girarlo: P(1) ya no llega a 1. Si la desintonia es muy grande, el pulso
# practicamente no tiene efecto.
t_fino = np.linspace(0, 2e-3, 401)
for dw in (0.0, 500.0, 5000.0):
    maximo = max(abs(estado_tras_pulso(w1, t, dw=dw)[1]) ** 2 for t in t_fino)
    print(f"desintonia = {dw:6.0f} Hz -> maximo de P(1) = {maximo:.3f}")

# %% Celda 7: grafica
plt.plot(tiempos * 1e3, p1_medida, "o", label=f"datos ({shots} shots)")
plt.plot(t_fino * 1e3, modelo(t_fino, w1), label="teoria sin^2(pi w1 t)")
plt.plot(t_fino * 1e3, [abs(estado_tras_pulso(w1, t, dw=500.0)[1]) ** 2 for t in t_fino],
         "--", label="desintonia 500 Hz")
plt.xlabel("ancho del pulso (ms)")
plt.ylabel("P(medir 1)")
plt.title("Oscilaciones de Rabi")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
