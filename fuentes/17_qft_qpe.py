# %% [markdown]
# # 17 - Transformada de Fourier cuantica (QFT) y estimacion de fase (QPE)

# %% Celda 1: importaciones
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Statevector

sim = AerSimulator()

# %% Celda 2: la QFT como matriz
# La QFT lleva cada estado de la base |j> al estado
#     |j>  ->  (1/sqrt(N)) * suma_k  e^{2 pi i j k / N} |k>,     N = 2^n.
# Es la version cuantica de la transformada de Fourier discreta (DFT) y es una
# matriz unitaria con entradas  U[k, j] = e^{2 pi i j k / N} / sqrt(N).
def matriz_qft(n):
    N = 2 ** n
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    return np.exp(2j * np.pi * j * k / N) / np.sqrt(N)


U = matriz_qft(3)
print("La QFT es unitaria (U^dagger U = I):", np.allclose(U.conj().T @ U, np.eye(8)))


# %% Celda 3: el circuito de la QFT
# Para cada qubit, de arriba hacia abajo:
#   1. Una Hadamard.
#   2. Fases controladas CR_k por cada qubit siguiente, que pasan la informacion
#      de los demas bits a la fase de |1>. CR_k aplica la fase e^{2 pi i / 2^k}
#      (CR_2 es la puerta S controlada).
# Al final una serie de SWAP invierte el orden de los qubits.
# Qiskit numera el qubit 0 a la derecha, asi que el bit mas significativo es el
# qubit n-1 y se empieza por el.
def circuito_qft(n):
    qc = QuantumCircuit(n)
    for objetivo in reversed(range(n)):
        qc.h(objetivo)
        for control in range(objetivo - 1, -1, -1):
            k = objetivo - control + 1
            qc.cp(2 * np.pi / 2 ** k, control, objetivo)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


for n in (2, 3, 4):
    coincide = np.allclose(Operator(circuito_qft(n)).data, matriz_qft(n))
    puertas = sum(circuito_qft(n).count_ops().values())
    print(f"n = {n}: el circuito coincide con la matriz de la QFT: {coincide} ({puertas} puertas)")

# En Colab o Jupyter el circuito se dibuja solo; en terminal no se muestra.
circuito_qft(3).draw()

# %% Celda 4: la QFT de dos qubits sobre cada estado de la base
# Cada entrada |j> da cuatro amplitudes iguales en modulo (1/2) y lo que cambia es la
# fase: e^{2 pi i j k / 4} para cada k, es decir, multiplos de 90 grados.
qft2 = circuito_qft(2)
print("Fases (en grados) de las amplitudes de |k> para k = 00, 01, 10, 11:")
for j in range(4):
    entrada = format(j, "02b")
    amplitudes = Statevector.from_label(entrada).evolve(qft2).data
    # Los indices de Statevector crecen con el valor binario de la cadena (q1 q0).
    fases = [round(float(np.degrees(np.angle(c))) % 360) for c in amplitudes]
    print(f"  |{entrada}> -> {fases}")

# %% Celda 5: estimacion de fase (QPE)
# Sea U una puerta unitaria y |psi> un autovector con U|psi> = e^{2 pi i phi}|psi>.
# QPE estima phi con m bits: phi = 0.phi1 phi2 ... phim (en binario).
#   1. Primer registro (m qubits) en |0>, segundo registro en el autovector |psi>.
#   2. Hadamard en cada qubit del primer registro.
#   3. Puertas U^(2^k) controladas por el qubit k del primer registro.
#   4. QFT inversa sobre el primer registro.
#   5. Se mide el primer registro: sale el numero binario phi1 ... phim.
# La QFT pasa informacion de la base a la fase; la QPE hace lo contrario.
def qpe(phi, m):
    qc = QuantumCircuit(m + 1, m)
    qc.x(m)                      # el qubit m es el segundo registro: autovector |1> de la puerta fase
    qc.h(range(m))
    for k in range(m):
        # La puerta de fase P(2 pi phi) tiene autovalor e^{2 pi i phi} en |1>.
        # Controlada por el qubit k y elevada a 2^k: fase 2 pi phi 2^k.
        qc.cp(2 * np.pi * phi * 2 ** k, k, m)
    qc.compose(circuito_qft(m).inverse(), qubits=range(m), inplace=True)
    qc.measure(range(m), range(m))
    return qc


def estimar(phi, m, shots=1000):
    cuentas = sim.run(qpe(phi, m), shots=shots).result().get_counts()
    mejor = max(cuentas, key=cuentas.get)
    return mejor, int(mejor, 2) / 2 ** m, cuentas[mejor] / shots


# %% Celda 6: fases que se pueden escribir exactamente con m bits
# La puerta S tiene autovector |1> con autovalor i = e^{2 pi i (1/4)}, o sea phi = 1/4.
# En binario 1/4 = 0.01, y con m = 2 la QPE deberia dar 01.
# La puerta T tiene phi = 1/8 = 0.001: hacen falta m = 3.
for nombre, phi, m in (("S", 1 / 4, 2), ("T", 1 / 8, 3), ("Z", 1 / 2, 2)):
    bits, estimada, prob = estimar(phi, m)
    print(f"puerta {nombre}: phi = {phi:.3f} | QPE con m = {m}: {bits} -> {estimada:.3f} (probabilidad {prob:.2f})")

# %% Celda 7: fases que NO caben exactamente
# phi = 1/3 no se puede escribir con pocos bits. La QPE da el valor mas cercano con
# alta probabilidad, y mas bits dan mas precision.
phi = 1 / 3
for m in (2, 4, 6):
    bits, estimada, prob = estimar(phi, m, shots=2000)
    print(f"m = {m}: resultado mas probable {bits} -> {estimada:.4f} | error = {abs(estimada - phi):.4f} | probabilidad {prob:.2f}")
