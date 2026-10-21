import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "desde-cero"))
from qubit import H, X, Qubit  # noqa: E402


def test_hadamard_da_50_50():
    assert np.allclose(Qubit().aplicar(H).probabilidades(), [0.5, 0.5])


def test_interferencia_hh_vuelve_a_cero():
    assert np.allclose(Qubit().aplicar(H).aplicar(H).probabilidades(), [1, 0])


def test_estado_no_normalizado_falla():
    with pytest.raises(ValueError):
        Qubit(1, 1)


def test_puerta_no_unitaria_falla():
    with pytest.raises(ValueError):
        Qubit().aplicar(np.array([[1, 1], [0, 1]], dtype=complex))


def test_medir_cuenta_todos_los_shots():
    conteos = Qubit().aplicar(X).medir(500)
    assert conteos == {"0": 0, "1": 500}
