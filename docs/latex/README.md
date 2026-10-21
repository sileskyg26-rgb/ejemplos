# Documentos LaTeX de los ejemplos de Qiskit

Un `.tex` por concepto, más un capítulo de prerrequisitos (`00_prerrequisitos.tex`)
y un documento maestro (`main.tex`) que los reúne. Cada capítulo incluye el código
tal cual desde `../../fuentes/NN_*.py` (una sola fuente de verdad) y la salida real desde
`../../fuentes/salidas/`. Los PDF ya compilados están en `../pdf/`.

## Compilar

Requiere **XeLaTeX** o **LuaLaTeX** (el preámbulo usa Unicode y `quantikz`).

- Overleaf: subir esta carpeta **y** la carpeta `fuentes/` (los `.py` y `salidas/`)
  (conservando la estructura), elegir compilador XeLaTeX y compilar `main.tex`.
- Local con Tectonic: `tectonic -X compile main.tex` (desde esta carpeta).
- Local con TeX Live/MiKTeX: `xelatex main.tex` (dos pasadas para el índice).

Cada capítulo también compila solo, p. ej. `xelatex 05b_entrelazamiento_bell.tex`.

## Mantener sincronizado

Si se edita un `.py`, regenerar su salida (en `../../fuentes/salidas/`) y recompilar. Las
salidas con conteos aleatorios cambian en cada ejecución; el texto de cada
capítulo explica qué parte es reproducible.
