# %% [markdown]
# # 21 - Decoherencia y spin echo: perder y recuperar la fase de un qubit

# %% Celda 1: importaciones
import numpy as np

rng = np.random.default_rng(11)   # semilla fija: las salidas se pueden reproducir

# %% Celda 2: relajacion y estados mezclados
# Un qubit puro es un punto en la SUPERFICIE de la esfera de Bloch (vector de longitud 1).
# Con el tiempo el entorno lo "acorta": el vector entra en la esfera y el estado se
# vuelve MEZCLADO (se pierde informacion). Hay dos procesos (en este modelo, ademas, la
# poblacion termina volviendo al estado de equilibrio |0>, por eso el vector
# vuelve a alargarse al final):
#   - relajacion longitudinal (T1): cambia la poblacion (la componente z),
#   - relajacion transversal (T2): se pierde la fase (las componentes x e y).
# Modelo de Bloch con estos dos tiempos (T2 <= 2 T1). Los valores de T1 y T2 de
# este ejemplo son elegidos para ilustrar, no medidos.
T1, T2 = 100e-3, 40e-3        # segundos


def vector_bloch(t, r0=(1.0, 0.0, 0.0)):
    x0, y0, z0 = r0
    z_eq = 1.0                  # estado de equilibrio: |0>, z = +1
    return np.array([x0 * np.exp(-t / T2), y0 * np.exp(-t / T2), z_eq - (z_eq - z0) * np.exp(-t / T1)])


# Se prepara el qubit sobre el ecuador con un pulso Ry(pi/2): r0 = (1, 0, 0).
print("t (ms) |  x      y      z   | |r|   | pureza = (1 + |r|^2)/2")
for t_ms in (0, 10, 40, 100, 300):
    r = vector_bloch(t_ms * 1e-3)
    largo = np.linalg.norm(r)
    print(f"{t_ms:6d} | {r[0]:+.3f} {r[1]:+.3f} {r[2]:+.3f} | {largo:.3f} | {(1 + largo ** 2) / 2:.3f}")
# La componente transversal cae con exp(-t/T2) (mas rapido aqui); la longitudinal
# se recupera con 1 - exp(-t/T1).

# %% Celda 3: desfase en un conjunto de espines (inhomogeneidad)
# Una muestra tiene muchos espines. Si el campo magnetico no es uniforme, cada
# espin gira con una frecuencia ligeramente distinta f_i. Todos empiezan en la
# direccion x (fase 0), pero tras un tiempo t tienen fases 2*pi*f_i*t distintas, y la
# magnetizacion neta (el promedio) se acorta:  M(t) = promedio de e^{i 2 pi f_i t}.
N = 5000
sigma = 100.0                       # dispersion de frecuencias: 100 Hz
f = rng.normal(0.0, sigma, N)       # una frecuencia por espin
T2_estrella = 1 / (2 * np.pi * sigma)
print(f"T2* (escala del desfase) ~ {T2_estrella * 1e3:.2f} ms")
for t_ms in (0, 1, 2, 4, 8):
    t = t_ms * 1e-3
    M = abs(np.mean(np.exp(2j * np.pi * f * t)))
    teoria = np.exp(-((2 * np.pi * sigma * t) ** 2) / 2)
    print(f"t = {t_ms} ms: |M| simulado = {M:.3f} | teoria exp(-(2 pi sigma t)^2/2) = {teoria:.3f}")

# %% Celda 4: el spin echo recupera la senal
# Idea de Hahn (1950): a mitad del tiempo se aplica un pulso pi (giro de 180 grados
# alrededor de x). Eso invierte las fases acumuladas, theta -> -theta, y en la
# segunda mitad cada espin "desacumula" lo que acumulo:
#   fase final = -(w t/2) + w t/2 = 0   para todos los espines.
# Todos vuelven a apuntar igual y aparece el "eco". Pero el eco es menor que la
# senal inicial, porque la relajacion natural (T2) no se corrige.
print("\nt (ms) | sin eco |M| | con eco |M| | e^{-t/T2}")
for t_ms in (2, 4, 8, 16, 32):
    t = t_ms * 1e-3
    sin_eco = abs(np.mean(np.exp(2j * np.pi * f * t)))
    fases = -2 * np.pi * f * (t / 2) + 2 * np.pi * f * (t / 2)      # = 0 para todos
    con_eco = abs(np.mean(np.exp(1j * fases))) * np.exp(-t / T2)   # solo queda T2
    print(f"{t_ms:6d} | {sin_eco:11.3f} | {con_eco:11.3f} | {np.exp(-t / T2):.3f}")
# Con eco, la unica perdida es la relajacion natural: el eco decae con T2, y sin
# eco la senal decae mucho mas rapido (T2*). Por eso T2 >= T2*.

# %% Celda 5: la fase del eco depende del pulso pi
# Con un solo pulso pi, la fase del eco puede no coincidir con la fase inicial: depende
# del eje del pulso pi y de la direccion inicial. Se sigue la fase de UN espin
# con frecuencia w en el plano xy (vector de Bloch con rotaciones 3x3).
def giro_z(angulo):
    c, s = np.cos(angulo), np.sin(angulo)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


PI_X = np.diag([1.0, -1.0, -1.0])    # giro de 180 grados alrededor de x
PI_Y = np.diag([-1.0, 1.0, -1.0])    # giro de 180 grados alrededor de y


def fase_eco(fase_inicial, pulso, dos_pulsos, w=2 * np.pi * 123.0, t=4e-3):
    r = np.array([np.cos(fase_inicial), np.sin(fase_inicial), 0.0])
    r = giro_z(w * t / 2) @ r       # primera mitad: precesion
    r = pulso @ r                   # pulso pi
    r = giro_z(w * t / 2) @ r       # segunda mitad
    if dos_pulsos:
        r = pulso @ r               # segundo pulso pi para "corregir" la fase
    return round(float(np.degrees(np.arctan2(r[1], r[0])))) % 360


print("\nfase inicial | pulso | eco con 1 pulso | eco con 2 pulsos")
for fase0 in (0, 90):
    for nombre, pulso in (("pi_x", PI_X), ("pi_y", PI_Y)):
        print(f"{fase0:10d}° | {nombre} | {fase_eco(np.radians(fase0), pulso, False):13.0f}° | {fase_eco(np.radians(fase0), pulso, True):14.0f}°")
# Con 1 pulso la fase del eco cambia segun el caso; con el segundo pulso del mismo
# eje coincide siempre con la fase inicial: la secuencia "mantiene la fase".

# %% Celda 6: spin echo frente a CPMG
# Si el campo ademas fluctua LENTAMENTE en el tiempo (aqui con un tiempo de
# correlacion de 10 ms), un solo pulso pi en un intervalo largo ya no refocaliza todo:
# el ruido de la primera mitad ya es distinto al de la segunda. La secuencia
# CPMG-n (Carr, Purcell, Meiboom, Gill) pone n pulsos pi repartidos en el tiempo
# t, con la separacion t/n y el primero y el ultimo a t/(2n) de los extremos.
# Con pulsos mas frecuentes que los cambios del ruido, cada fluctuacion se cancela.
paso = 0.5e-3                  # paso de tiempo
t_total = 40e-3
pasos = int(round(t_total / paso))
tau_c = 10e-3                                   # tiempo de correlacion del ruido
sigma_ruido = 30.0                              # fluctuacion de 30 Hz
estatico = rng.normal(0.0, sigma, N)
# Ruido que cambia suavemente: cada paso se parece al anterior (proceso AR(1)).
rho = np.exp(-paso / tau_c)
ruido = np.zeros((N, pasos))
ruido[:, 0] = rng.normal(0.0, sigma_ruido, N)
for i in range(1, pasos):
    ruido[:, i] = rho * ruido[:, i - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, sigma_ruido, N)


def magnetizacion(n_pulsos):
    """Magnitud de la magnetizacion neta al final, con n_pulsos pulsos pi repartidos."""
    posiciones = {int(round(pasos * (2 * k - 1) / (2 * n_pulsos))) for k in range(1, n_pulsos + 1)} if n_pulsos else set()
    signo, fase = 1.0, np.zeros(N)
    for i in range(pasos):
        if i in posiciones:
            signo = -signo             # cada pulso pi invierte la fase acumulada
        fase += signo * 2 * np.pi * (estatico + ruido[:, i]) * paso
    return abs(np.mean(np.exp(1j * fase)))


print("\nSecuencia                | |M| al final")
print(f"sin pulsos               | {magnetizacion(0):.3f}")
print(f"spin echo (1 pulso)      | {magnetizacion(1):.3f}")
print(f"CPMG-8 (8 pulsos)        | {magnetizacion(8):.3f}")
