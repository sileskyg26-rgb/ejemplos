# %% [markdown]
# # 23 - Pulsos optimizados con GRAPE: preparar un estado de Bell con pulsos de forma libre

# %% Celda 1: importaciones
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# %% Celda 2: el problema de control
# Dos espines acoplados (J = 696 Hz, como en el capitulo 12). Se controlan con
# campos de radiofrecuencia en x e y sobre cada espin; la amplitud de cada campo
# (en Hz) puede cambiar en el tiempo. El Hamiltoniano es
#     H(t) = 2*pi*J*(sz (x) sz)/4  +  2*pi * suma_j  u_j(t) * sigma_j / 2,
# con cuatro controles: sx y sy en cada espin. Se busca la forma de u_j(t) que lleva
# |00> al estado de Bell (|00> + |11>)/sqrt2.
J = 696.0
H_deriva = 2 * np.pi * J * np.kron(sz, sz) / 4
OPERADORES = [np.kron(sx, I2), np.kron(sy, I2), np.kron(I2, sx), np.kron(I2, sy)]
H_controles = [2 * np.pi * op / 2 for op in OPERADORES]

T = 1.0e-3                       # duracion total del pulso: 1 ms
N = 20                           # numero de pasos de amplitud constante
dt = T / N
U_MAX = 5000.0                   # amplitud maxima de cada control: 5 kHz

inicial = np.array([1, 0, 0, 0], dtype=complex)                    # |00>
objetivo = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)      # Bell


def H_con_errores(desint=0.0, escala=1.0):
    """Deriva con un error de frecuencia (desint, en Hz) en ambos espines; los controles se escalan por 'escala'."""
    error = 2 * np.pi * desint * (np.kron(sz, I2) + np.kron(I2, sz)) / 2
    return H_deriva + error, escala


# %% Celda 3: GRAPE
# GRAPE (gradient ascent pulse engineering) trata el pulso como un problema de
# optimizacion: la forma del pulso se divide en N pasos y la amplitud de cada
# control en cada paso es un parametro. Se maximiza la fidelidad
#     F = |<objetivo| U(T) |inicial>|^2.
# El gradiente respecto de cada amplitud se obtiene propagando el estado hacia
# adelante y el objetivo hacia atras (aproximacion de primer orden en dt).
def propagadores(u, H0, escala):
    return [expm(-1j * (H0 + escala * sum(u[k, j] * H_controles[j] for j in range(4))) * dt) for k in range(N)]


def fidelidad_y_gradiente(u, H0, escala):
    U = propagadores(u, H0, escala)
    adelante = [inicial]
    for Uk in U:
        adelante.append(Uk @ adelante[-1])
    atras = [None] * (N + 1)
    atras[N] = objetivo
    for k in range(N - 1, -1, -1):
        atras[k] = U[k].conj().T @ atras[k + 1]
    c = np.vdot(objetivo, adelante[N])
    grad = np.zeros_like(u)
    for k in range(N):
        for j in range(4):
            dc = np.vdot(atras[k + 1], (-1j * dt * escala * H_controles[j]) @ adelante[k + 1])
            grad[k, j] = 2 * np.real(np.conj(c) * dc)
    return abs(c) ** 2, grad


def optimizar(errores, semilla=0, intentos=4):
    """Maximiza la fidelidad PROMEDIO sobre una lista de errores (desintonia, escala)."""
    mejor = None
    for s in range(intentos):
        rng = np.random.default_rng(semilla + s)
        u0 = rng.uniform(-1000, 1000, (N, 4))

        def costo(x):
            u = x.reshape(N, 4)
            F, G = 0.0, np.zeros_like(u)
            for desint, escala in errores:
                H0, esc = H_con_errores(desint, escala)
                f, g = fidelidad_y_gradiente(u, H0, esc)
                F += f / len(errores)
                G += g / len(errores)
            return -F, -G.ravel()

        res = minimize(costo, u0.ravel(), jac=True, method="L-BFGS-B",
                       bounds=[(-U_MAX, U_MAX)] * (N * 4), options={"maxiter": 400})
        if mejor is None or res.fun < mejor.fun:
            mejor = res
    return mejor.x.reshape(N, 4), -mejor.fun


# %% Celda 4: pulso optimizado para el caso ideal (sin errores)
u_ideal, F_ideal = optimizar([(0.0, 1.0)])
print(f"Fidelidad del pulso GRAPE (caso ideal): {F_ideal:.4f}")
print("Amplitud maxima usada:", round(float(np.abs(u_ideal).max())), "Hz (limite", int(U_MAX), "Hz)")

# %% Celda 5: ese pulso frente a errores
# En un equipo real la frecuencia del pulso puede estar algo desafinada y la
# amplitud puede no ser exacta. Se evalua el pulso con desintonias de +-10 Hz
# y errores de amplitud de +-5 %.
CASOS = [(d, e) for d in (-10.0, 0.0, 10.0) for e in (0.95, 1.0, 1.05)]


def evaluar(u):
    fid = [fidelidad_y_gradiente(u, *H_con_errores(d, e))[0] for d, e in CASOS]
    return np.array(fid)


f1 = evaluar(u_ideal)
print(f"Sin robustez: fidelidad promedio = {f1.mean():.4f}, minima = {f1.min():.4f}")

# %% Celda 6: pulso robusto
# Para hacer el pulso resistente, se optimiza el PROMEDIO de la fidelidad sobre
# todos los errores a la vez (9 combinaciones de desintonia y amplitud).
u_robusto, _ = optimizar(CASOS)
f2 = evaluar(u_robusto)
print(f"Con robustez: fidelidad promedio = {f2.mean():.4f}, minima = {f2.min():.4f}")
print(f"En el caso ideal (sin errores) el pulso robusto da {evaluar(u_robusto)[CASOS.index((0.0, 1.0))]:.4f}")

# %% Celda 7: errores mayores
# Con los errores del experimento (+-10 Hz, +-5 %) el pulso sin robustez ya se
# comporta bien, porque el pulso dura solo 1 ms. Con errores mas grandes la
# diferencia se nota: desintonias de +-300 Hz y errores de amplitud de +-20 %.
CASOS_GRANDES = [(d, e) for d in (-300.0, 0.0, 300.0) for e in (0.8, 1.0, 1.2)]


def evaluar_grandes(u):
    return np.array([fidelidad_y_gradiente(u, *H_con_errores(d, e))[0] for d, e in CASOS_GRANDES])


u_robusto_grande, _ = optimizar(CASOS_GRANDES)
for nombre, u in (("sin robustez", u_ideal), ("robusto (errores grandes)", u_robusto_grande)):
    f = evaluar_grandes(u)
    print(f"{nombre:26s}: fidelidad promedio = {f.mean():.4f}, minima = {f.min():.4f}")
