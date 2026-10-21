"""Ejecuta cada notebook de notebooks/ de principio a fin y verifica que no falle."""
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_se_ejecuta(ruta):
    nb = nbformat.read(ruta, as_version=4)
    # La primera celda es `!pip install ...` (para Colab); aqui ya esta instalado.
    nb.cells = [c for c in nb.cells if not c.source.lstrip().startswith("!pip")]
    NotebookClient(nb, timeout=300, kernel_name="python3").execute()


def test_hay_notebooks():
    assert NOTEBOOKS
