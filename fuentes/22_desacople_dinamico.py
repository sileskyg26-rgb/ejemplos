# %% [markdown]
# # 22 - Desacople dinamico: apagar el acoplamiento entre dos espines con pulsos pi

# %% Celda 1: importaciones
import numpy as np
from scipy.linalg import expm

# Matrices de Pauli e identidad. Trabajamos con hbar = 1 y frecuencias en Hz.
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# Dos espines: H (primer factor del producto) y P (segundo factor).
# Acoplamiento entre ellos (el mismo valor del capitulo 12).
J = 696.0
H_acople = 2 * np.pi * J * np.kron(sz, sz) / 4

# %% Celda 2: herramientas
def evolucion(t, J_activo=True):
    """Evolucion libre durante un tiempo t bajo el acoplamiento."""
    return expm(-1j * (H_acople if J_activo else 0 * H_acople) * t)


def pi_en(espin):
    """Pulso pi (giro de 180 grados alrededor de x) sobre el espin 'H' o 'P' (o 'HP' = ambos)."""
    xH = np.kron(sx, I2) if "H" in espin else np.eye(4)   # 'H' aparece en 'H' y en 'HP'
    xP = np.kron(I2, sx) if "P" in espin else np.eye(4)
    return -1j * (xH @ xP)        # el factor -i es la fase global de un giro de 180 grados


def bloch_de_P(estado):
    """Vector de Bloch (x, y, z) del espin P (se descarta el espin H: traza parcial)."""
    rho = np.outer(estado, estado.conj()).reshape(2, 2, 2, 2)
    rho_P = np.einsum("ijik->jk", rho)
    return np.array([np.real(np.trace(rho_P @ s)) for s in (sx, sy, sz)])


def fase_P(estado):
    x, y, _ = bloch_de_P(estado)
    return float(np.degrees(np.arctan2(y, x)))


mas = np.array([1, 1], dtype=complex) / np.sqrt(2)        # |+>: espin apuntando a +x
arriba, abajo = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)

# %% Celda 3: el efecto del acoplamiento (el espin H en |0> o en |1>)
# P empieza apuntando a +x. Segun H este en |0> o |1>, P gira en sentido contrario.
# La fase acumulada vale  +-pi*J*t  (con signo segun el estado de H).
print("t (ms) | H en |0> | H en |1> | sin acoplamiento")
for t_ms in (0.5, 1.0, 2.0):
    t = t_ms * 1e-3
    fases = []
    for h in (arriba, abajo):
        fases.append(fase_P(evolucion(t) @ np.kron(h, mas)))
    sin = fase_P(evolucion(t, J_activo=False) @ np.kron(arriba, mas))
    print(f"{t_ms:6.1f} | {fases[0]:+8.1f}° | {fases[1]:+8.1f}° | {sin:+.1f}°")

# %% Celda 4: H en superposicion: P pierde su fase (queda entrelazado)
# Con H en cos(a)|0> + sin(a)|1>, P gira en los dos sentidos "a la vez". Su vector
# de Bloch se acorta (queda mezclado): la fase de P se pierde.
print("\nProporcion |0> de H | t (ms) | longitud transversal de P")
for a in (0.0, np.pi / 8, np.pi / 4):          # a = pi/4 es la superposicion 50/50
    h = np.array([np.cos(a), np.sin(a)], dtype=complex)
    for t_ms in (0.5, 1.0):
        x, y, _ = bloch_de_P(evolucion(t_ms * 1e-3) @ np.kron(h, mas))
        print(f"{np.cos(a) ** 2:20.2f} | {t_ms:6.1f} | {np.hypot(x, y):.3f}")

# %% Celda 5: pulso pi en la mitad del tiempo (desacople)
# Un pulso pi sobre uno de los espines a mitad de la evolucion invierte el signo del
# acoplamiento en la segunda mitad, y la fase acumulada en la primera se cancela.
# Se compara con la misma secuencia SIN acoplamiento: si P termina igual, hubo desacople.
def secuencia(t, pulsos, J_activo=True):
    """Evolucion t/2, pulso(s) pi, evolucion t/2."""
    U = evolucion(t / 2, J_activo)
    for esp in pulsos:
        U = pi_en(esp) @ U
    return evolucion(t / 2, J_activo) @ U


print("\nt (ms) | pulso pi sobre | fase de P (con J) | fase de P (J = 0) | desacople")
for t_ms in (1.0, 2.0):
    t = t_ms * 1e-3
    estado0 = np.kron(arriba, mas)                 # H en |0>, P en |+>
    for nombre, pulsos in (("ninguno", []), ("H", ["H"]), ("P", ["P"]), ("H y P", ["HP"])):
        con = fase_P(secuencia(t, pulsos) @ estado0)
        sin = fase_P(secuencia(t, pulsos, J_activo=False) @ estado0)
        desacoplado = np.isclose((con - sin + 180) % 360 - 180, 0, atol=1e-6)
        print(f"{t_ms:6.1f} | {nombre:14s} | {con:+16.1f}° | {sin:+16.1f}° | {'si' if desacoplado else 'no'}")
