"""Simulador de un qubit en NumPy puro (sin Qiskit).

Un qubit es un vector de 2 amplitudes complejas, una puerta es una matriz
unitaria 2x2 y medir aplica la regla de Born.
"""
import numpy as np

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    """Rotacion real alrededor del eje Y."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


class Qubit:
    def __init__(self, alfa: complex = 1, beta: complex = 0):
        self.estado = np.array([alfa, beta], dtype=complex)
        if not np.isclose(np.linalg.norm(self.estado), 1.0):
            raise ValueError("El estado debe estar normalizado: |alfa|^2 + |beta|^2 = 1")

    def aplicar(self, puerta: np.ndarray) -> "Qubit":
        """Aplica una puerta (matriz unitaria) y devuelve el propio qubit."""
        if not np.allclose(puerta.conj().T @ puerta, np.eye(2)):
            raise ValueError("La puerta no es unitaria")
        self.estado = puerta @ self.estado
        return self

    def probabilidades(self) -> np.ndarray:
        """Regla de Born: P(0), P(1)."""
        return np.abs(self.estado) ** 2

    def medir(self, shots: int = 1, rng: np.random.Generator | None = None) -> dict[str, int]:
        """Mide `shots` veces; devuelve conteos {'0': n0, '1': n1}."""
        rng = rng or np.random.default_rng()
        unos = int(rng.binomial(shots, self.probabilidades()[1]))
        return {"0": shots - unos, "1": unos}
