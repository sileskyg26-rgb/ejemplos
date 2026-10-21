"""Convierte cada ejemplo .py (celdas separadas por '# %%') en un notebook .ipynb para Colab."""
import re
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).parent
DESTINO = AQUI.parent / "notebooks"
INSTALAR = "!pip install -q qiskit==2.5.0 qiskit-aer==0.17.2 matplotlib pylatexenc"


def a_notebook(ruta: Path) -> nbf.NotebookNode:
    celdas = [nbf.v4.new_code_cell(INSTALAR)]
    for bloque in re.split(r"^# %%[ \t]*", ruta.read_text(encoding="utf-8"), flags=re.M):
        if not bloque.strip():
            continue
        if bloque.startswith("[markdown]"):
            lineas = bloque.removeprefix("[markdown]").strip().splitlines()
            texto = "\n".join(l.removeprefix("# ").removeprefix("#") for l in lineas)
            celdas.append(nbf.v4.new_markdown_cell(texto))
        else:
            # La primera linea es el titulo de la celda: lo dejamos como comentario.
            titulo, _, codigo = bloque.partition("\n")
            encabezado = f"# {titulo.strip()}\n" if titulo.strip() else ""
            celdas.append(nbf.v4.new_code_cell(encabezado + codigo.strip()))
    nb = nbf.v4.new_notebook()
    nb.cells = celdas
    return nb


for py in sorted(AQUI.glob("[0-9]*.py")):
    destino = DESTINO / py.with_suffix(".ipynb").name
    nbf.write(a_notebook(py), destino)
    print("generado", destino.name)
