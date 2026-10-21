"""Ejemplos con el simulador desde cero: superposicion e interferencia."""
from qubit import H, Qubit, Z

q = Qubit().aplicar(H)
print("H|0>        P(0), P(1) =", q.probabilidades().round(3))
print("medir 1000  ->", q.medir(1000))

q = Qubit().aplicar(H).aplicar(H)
print("HH|0>       P(0), P(1) =", q.probabilidades().round(3), "(interferencia destructiva en |1>)")

q = Qubit().aplicar(H).aplicar(Z).aplicar(H)
print("HZH|0>      P(0), P(1) =", q.probabilidades().round(3), "(solo cambio una fase)")
