# Computación cuántica desde cero

Una introducción práctica a la computación cuántica con Qiskit: cada concepto tiene una explicación teórica, un notebook que se ejecuta en Google Colab sin instalar nada y un documento LaTeX/PDF con el código comentado, las salidas y ejercicios.

**A quién va:** estudiantes con álgebra lineal básica (vectores, matrices, números complejos) y algo de Python. Si algo de la notación no se entiende, empieza por el capítulo de [prerrequisitos y notación](docs/pdf/00_prerrequisitos.pdf).

**Orden de lectura:** de arriba hacia abajo, del 1 al 13. Cada sección tiene la teoría, el botón de Colab y el PDF completo. También hay un [PDF con todo reunido](docs/pdf/main.pdf).

## Estructura del repositorio

```
├── README.md          # esta guía
├── requirements.txt   # versiones fijas (qiskit, qiskit-aer, ...)
├── notebooks/         # un notebook por concepto (se abren en Colab)
├── desde-cero/        # simulador de un qubit en NumPy puro
├── fuentes/           # los .py de los ejemplos, sus salidas y el generador de notebooks
├── docs/              # documentación: latex/ (fuentes .tex) y pdf/ (compilados)
├── tests/             # ejecuta cada notebook y verifica que no falle
└── .github/workflows/probar-notebooks.yml
```

## Ejecutar en local

```
pip install -r requirements.txt
pytest                         # ejecuta todos los notebooks y los tests
python desde-cero/ejemplo.py   # el simulador de un qubit
python fuentes/build_notebooks.py   # regenera notebooks/ desde fuentes/*.py
```

**Índice**

1. [El qubit como vector](#1-el-qubit-como-vector)
2. [Las puertas son matrices unitarias](#2-las-puertas-son-matrices-unitarias)
3. [Crecimiento exponencial](#3-crecimiento-exponencial)
4. [Medición y regla de Born](#4-medición-y-regla-de-born)
5. [Interferencia](#5a-interferencia) y [entrelazamiento](#5b-entrelazamiento-el-estado-de-bell)
6. [Algoritmo de Grover](#6-algoritmo-de-grover)
7. [Práctica: Bell y GHZ](#7-práctica-bell-variantes-y-ghz)
8. [Ruido y era NISQ](#8-ruido-y-era-nisq)
9. [Generador de números aleatorios cuántico](#9-generador-de-números-aleatorios-cuántico-qrng)
10. [La esfera de Bloch](#10-la-esfera-de-bloch)
11. [Oscilación de Rabi](#11-oscilación-de-rabi)
12. [Control cuántico](#12-control-cuántico)
13. [Tomografía de estados](#13-tomografía-de-estados)

> **Nota sobre el orden de bits:** Qiskit escribe las cadenas con el qubit 0 a la derecha. `'10'` significa $q_1=1,\ q_0=0$.

---

## 1. El qubit como vector

Un bit clásico vale 0 o 1. Un qubit es un vector de dos amplitudes complejas:

$$\lvert\psi\rangle=\alpha\lvert0\rangle+\beta\lvert1\rangle,\qquad |\alpha|^2+|\beta|^2=1.$$

Por la **regla de Born**, al medir se obtiene 0 con probabilidad $|\alpha|^2$ y 1 con probabilidad $|\beta|^2$. La normalización garantiza que las probabilidades sumen 1.

- **Fase global:** $e^{i\gamma}\lvert\psi\rangle$ y $\lvert\psi\rangle$ son indistinguibles.
- **Fase relativa:** la fase de $\beta$ respecto a $\alpha$ no cambia las probabilidades de esta medición, pero sí cómo interfiere el qubit después (ver [interferencia](#5a-interferencia)).

En el ejemplo: $\alpha=1/\sqrt3$ y $\beta=(1+i)/\sqrt3$, así que $P(0)=1/3$ y $P(1)=2/3$. La suma sale `1.0000000000000002` por redondeo de punto flotante.

**Error común:** olvidar normalizar. Qiskit no lo hace solo.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/01_qubit_vector.ipynb) [📄 Documento completo (PDF)](docs/pdf/01_qubit_vector.pdf)

---

## 2. Las puertas son matrices unitarias

Una puerta de un qubit es una matriz unitaria $U$ ($U^\dagger U=I$) y el estado evoluciona como $\lvert\psi'\rangle=U\lvert\psi\rangle$:

$$H=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.$$

**Por qué unitarias:** conservan la norma ($\|U\lvert\psi\rangle\|=\|\lvert\psi\rangle\|$), así que las probabilidades siguen sumando 1. Además son invertibles con $U^{-1}=U^\dagger$: la evolución cuántica es **reversible**. Para $H$ se cumple $H^\dagger=H$ y $H^2=I$.

El ejemplo calcula $H\lvert0\rangle=(0.707,\,0.707)$ a mano con `@` y con `.evolve()`, y verifica numéricamente la unitariedad con `np.allclose` (nunca `==` con decimales).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/02_puertas_matrices.ipynb) [📄 Documento completo (PDF)](docs/pdf/02_puertas_matrices.pdf)

---

## 3. Crecimiento exponencial

El espacio de $n$ qubits tiene dimensión $2^n$: un estado general necesita $2^n$ amplitudes complejas. Con `complex128` (16 bytes cada una):

$$\text{memoria}=16\cdot2^n\ \text{bytes}.$$

- $n=30$: 16 GiB.
- $n=50$: 16 PiB, unas $1.13\times10^{15}$ amplitudes.

Cada qubit extra **duplica** la memoria. Por eso simular es difícil, y es parte de la motivación del hardware cuántico.

**Matiz:** esto aplica a la simulación de circuitos *generales* con vector de estado. Circuitos con estructura (por ejemplo, solo puertas Clifford) se simulan eficientemente. Y un computador cuántico no "guarda" $2^n$ números accesibles: al medir solo salen $n$ bits.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/03_crecimiento_exponencial.ipynb) [📄 Documento completo (PDF)](docs/pdf/03_crecimiento_exponencial.pdf)

---

## 4. Medición y regla de Born

Para $\lvert\psi\rangle=\alpha\lvert0\rangle+\beta\lvert1\rangle$, medir da 0 con probabilidad $|\alpha|^2$ y 1 con $|\beta|^2$. Una sola medición no revela $\alpha,\beta$: hay que repetir el experimento $N$ veces (cada repetición es un **shot**). Tras medir, el estado **colapsa** al resultado observado.

La frecuencia observada tiene error típico

$$\sigma=\sqrt{\frac{p(1-p)}{N}},$$

que para $p=0.25$ baja de $0.137$ ($N=10$) a $0.0043$ ($N=10000$): la incertidumbre estadística decrece como $1/\sqrt N$.

Para preparar $P(1)=0.25$ se usa $R_y(\theta)\lvert0\rangle=\cos\frac\theta2\lvert0\rangle+\sin\frac\theta2\lvert1\rangle$ con $\theta=\pi/3$.

Los conteos exactos cambian en cada ejecución; lo que se reproduce es la tendencia.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/04_medicion.ipynb) [📄 Documento completo (PDF)](docs/pdf/04_medicion.pdf)

---

## 5a. Interferencia

Las amplitudes se **suman** (con su signo o fase) *antes* de elevarse al cuadrado, y por eso pueden cancelarse. Aplicando $H$ dos veces:

$$HH\lvert0\rangle=\tfrac1{\sqrt2}\Big[\tfrac1{\sqrt2}(\lvert0\rangle+\lvert1\rangle)+\tfrac1{\sqrt2}(\lvert0\rangle-\lvert1\rangle)\Big]=\lvert0\rangle.$$

La amplitud de $\lvert1\rangle$ es $\frac12-\frac12=0$ (interferencia **destructiva**) y la de $\lvert0\rangle$ es $\frac12+\frac12=1$ (**constructiva**). Un modelo probabilístico clásico (50/50 y luego 50/50) daría otra vez 50/50, porque las probabilidades nunca se cancelan.

Si se inserta $Z$ entre las dos $H$, el resultado pasa a ser siempre $\lvert1\rangle$ (en general $HZH=X$). Solo se cambió una fase.

Este es el principio de los algoritmos cuánticos: diseñar las fases para que las respuestas incorrectas se cancelen y la correcta se refuerce.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/05a_interferencia.ipynb) [📄 Documento completo (PDF)](docs/pdf/05a_interferencia.pdf)

---

## 5b. Entrelazamiento: el estado de Bell

Con $H$ en $q_0$ y un CNOT ($q_0\to q_1$) se obtiene

$$\lvert\Phi^+\rangle=\tfrac1{\sqrt2}(\lvert00\rangle+\lvert11\rangle).$$

**No es un estado producto.** Si fuera $(a\lvert0\rangle+b\lvert1\rangle)\otimes(c\lvert0\rangle+d\lvert1\rangle)$ habría que tener $ad=0$ y $bc=0$ pero $ac=bd=\frac1{\sqrt2}\neq0$: imposible.

**Cómo cuantificarlo:** el estado del qubit 0 solo (traza parcial) es $\rho_0=I/2$, con entropía de von Neumann $S=1$ bit. El conjunto es un estado puro bien definido, pero cada parte por separado es máximamente incierta.

**Correlación sin comunicación:** al medir solo salen `00` o `11`, cada uno con probabilidad $\frac12$. Cada resultado individual es aleatorio, así que no se puede enviar un mensaje con esto (*no-signaling*).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/05b_entrelazamiento_bell.ipynb) [📄 Documento completo (PDF)](docs/pdf/05b_entrelazamiento_bell.pdf)

---

## 6. Algoritmo de Grover

Busca un elemento marcado entre $N=2^n$ sin estructura. Se parte de la superposición uniforme $\lvert s\rangle=\frac1{\sqrt N}\sum_x\lvert x\rangle$ y cada iteración aplica $G=D\,O$:

- **Oráculo** $O$: cambia el signo de la solución, $O\lvert x\rangle=-\lvert x\rangle$ si $x$ es el marcado.
- **Difusión** $D=2\lvert s\rangle\langle s\rvert-I$: reflexión respecto al promedio de las amplitudes.

Geométricamente $G$ gira el estado un ángulo $2\theta$ con $\sin\theta=1/\sqrt N$. Tras $k$ iteraciones:

$$P_k=\sin^2\!\big((2k+1)\theta\big),\qquad k_{\text{opt}}\approx\frac\pi4\sqrt N.$$

Para $N=8$: $k=2$ y $P_2\approx0.945$.

**La ventaja es cuadrática, no exponencial.** Clásicamente hacen falta $\sim N/2$ consultas; Grover usa $\sim\frac\pi4\sqrt N$ (óptimo para este problema). Con $N=10^{12}$: unas $7.9\times10^5$ consultas contra $5\times10^{11}$. Es distinto de Shor, cuya ventaja sobre lo mejor conocido clásicamente sí es exponencial.

Pasarse de iteraciones empeora el resultado, porque $P_k$ es periódica en $k$.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/06_grover_vs_clasico.ipynb) [📄 Documento completo (PDF)](docs/pdf/06_grover_vs_clasico.pdf)

---

## 7. Práctica: Bell, variantes y GHZ

Práctica guiada para predecir resultados antes de ejecutarlos.

- **Bell:** $H$ + CNOT da `00` y `11` con ~50 % cada uno.
- **Variante con $X$ en vez de $H$:** $X\lvert0\rangle=\lvert1\rangle$ no crea superposición, el CNOT copia ese 1 y el resultado es siempre `11`. Aísla el papel de $H$: es la fuente de la aleatoriedad.
- **GHZ (3 qubits):** $H$ y dos CNOT en cadena dan $\frac1{\sqrt2}(\lvert000\rangle+\lvert111\rangle)$; solo salen `000` y `111`.
- **Pocos shots:** con 10 shots las proporciones pueden alejarse mucho de 50/50 (por ejemplo, 8 contra 2). Es ruido *estadístico*, no físico.

Pregunta clave: ¿por qué nunca sale `01` ni `10`? Porque el CNOT fuerza que $q_1$ copie el valor de $q_0$ en la base computacional.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/07_bell_practica.ipynb) [📄 Documento completo (PDF)](docs/pdf/07_bell_practica.pdf)

---

## 8. Ruido y era NISQ

**NISQ** (*Noisy Intermediate-Scale Quantum*, Preskill 2018): dispositivos de decenas a pocos cientos de qubits, con errores y sin corrección de errores a escala. La corrección de errores codifica un qubit *lógico* en muchos qubits *físicos*, y ese costo es hoy una barrera.

El **canal depolarizante** sobre $k$ qubits es

$$\rho\mapsto(1-\lambda)\rho+\lambda\,\frac{I}{2^k}:$$

con probabilidad $\lambda$ el estado se reemplaza por el estado máximamente mezclado.

El ejemplo aplica 1 % de error tras cada $H$ y 5 % tras cada CNOT (las puertas de dos qubits suelen tener más error, como en hardware real). **Son valores ilustrativos, no de ningún dispositivo.** En una corrida aparecieron 42 de 2000 shots (2.1 %) con paridad incorrecta (`01` o `10`), imposibles en el circuito ideal.

Un equipo real tiene además errores de lectura y decoherencia que este modelo no incluye.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/08_ruido_nisq.ipynb) [📄 Documento completo (PDF)](docs/pdf/08_ruido_nisq.pdf)

---

## 9. Generador de números aleatorios cuántico (QRNG)

$H\lvert0\rangle$ da 0 o 1 con probabilidad $\frac12$. Con $n$ qubits en paralelo se obtienen $n$ bits independientes, es decir, un entero uniforme en $\{0,\dots,2^n-1\}$. El ejemplo lo usa para elegir la jugada de un bot de Tic-Tac-Toe.

**Sesgo por módulo.** Elegir entre $m$ opciones con $x \bmod m$ solo es uniforme si $m$ divide a $2^n$. Con 4 bits (16 valores) y $m=5$ casillas, $16=3\cdot5+1$: la casilla 0 sale con probabilidad $4/16$ y las demás con $3/16$. La solución es el **muestreo por rechazo**: descartar los valores $\ge15$ y repetir.

**Advertencia importante.** `AerSimulator` calcula las probabilidades y luego muestrea con un generador pseudoaleatorio, así que **en simulación los bits no son aleatoriedad física**. Solo en hardware real la fuente es física (e incluso ahí el ruido puede sesgar y se recomienda post-procesar).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/09_qrng.ipynb) [📄 Documento completo (PDF)](docs/pdf/09_qrng.pdf)

---

## 10. La esfera de Bloch

Los coeficientes de un qubit son complejos: $\lvert\psi\rangle=r_1e^{i\gamma_1}\lvert0\rangle+r_2e^{i\gamma_2}\lvert1\rangle$. Dos ideas lo simplifican:

1. **Solo importa la fase relativa.** La fase global no se puede observar (multiplicar todo el estado por $e^{i\gamma}$ no cambia nada medible), así que se queda solo $\varphi=\gamma_2-\gamma_1$.
2. **Normalización.** Como $r_1^2+r_2^2=1$, se escribe $r_1=\cos\frac\theta2$ y $r_2=\sin\frac\theta2$.

Resultado:

$$\lvert\psi\rangle=\cos\tfrac\theta2\,\lvert0\rangle+e^{i\varphi}\sin\tfrac\theta2\,\lvert1\rangle.$$

Cada qubit queda determinado por **dos ángulos**, es decir, es un punto de una esfera de radio 1: la **esfera de Bloch** (Felix Bloch, 1946). El polo norte es $\lvert0\rangle$, el polo sur es $\lvert1\rangle$, $\theta$ se mide desde el eje $z$ y $\varphi$ gira alrededor de él. Las coordenadas del punto son $(\langle X\rangle,\langle Y\rangle,\langle Z\rangle)=(\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta)$.

Las probabilidades dependen solo de $\theta$. Cambiar $\varphi$ mueve el punto alrededor de $z$ sin cambiar $P(0)$ ni $P(1)$: la fase relativa solo se nota después, por interferencia.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/10_esfera_bloch.ipynb) [📄 Documento completo (PDF)](docs/pdf/10_esfera_bloch.pdf)

---

## 11. Oscilación de Rabi

Cuando la frecuencia de un campo de radiofrecuencia (RF) coincide con la del espín (**resonancia**), el espín absorbe energía y cambia de nivel. Es la base del control de qubits en estos sistemas.

Si se mira desde un sistema que gira junto con el campo, el campo de RF se vuelve **estático**, con una dirección en el plano $xy$ que depende solo de la fase del pulso. En resonancia, el espín gira alrededor de ese eje a la **frecuencia de Rabi** $\omega_1$, y el ángulo de giro es

$$\theta=2\pi\,\omega_1\,t.$$

Partiendo de $\lvert0\rangle$, la probabilidad de medir 1 es $P(1)=\sin^2(\pi\omega_1 t)$: **oscila** entre 0 y 1. Un pulso con $\theta=\pi$ voltea el qubit (como la puerta $X$) y uno con $\theta=\pi/2$ lo deja en una superposición 50/50.

**Cómo se mide $\omega_1$:** medir colapsa el estado, así que se prepara siempre $\lvert0\rangle$, se aplica un pulso de duración $t$, se mide muchas veces y se repite **aumentando poco a poco $t$**. Se ajusta un seno a los datos y de ahí sale $\omega_1$ (el código lo simula).

**Desintonia:** si el pulso no es resonante, el espín gira además alrededor de $z$. Con una desintonia grande ese giro cancela el efecto del pulso y el qubit casi no se excita.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/11_oscilacion_rabi.ipynb) [📄 Documento completo (PDF)](docs/pdf/11_oscilacion_rabi.pdf)

---

## 12. Control cuántico

Toda evolución física es **unitaria** ($U^\dagger U=I$): conserva la suma de probabilidades, así que no se pierde información. La evolución la manda el **Hamiltoniano** $H$ (la energía total) mediante la ecuación de Schrödinger $i\hbar\,\frac{d}{dt}\lvert\psi\rangle=H\lvert\psi\rangle$. Con $\hbar=1$ y $H$ constante:

$$U(t)=e^{-iHt}.$$

**Controlar** un computador cuántico es modificar $H$ para obtener la $U$ que se quiere. $H$ se separa en una parte interna (energía de cada qubit y acoplamiento entre ellos) y una externa (el campo de RF).

- **Un qubit:** un pulso de fase $\varphi$ y duración $t$ es una **rotación** de ángulo $\theta=2\pi\omega_1t$ alrededor del eje $\cos\varphi\,\hat x+\sin\varphi\,\hat y$.
- **Rotación alrededor de $z$:** un pulso no la da directamente, pero se arma con tres: $R_y(\pi/2)$, luego $R_x(\theta)$, luego $R_{-y}(\pi/2)$. El **orden importa**.
- **Pulso desafinado:** gira a la vez alrededor de $z$ y del eje del pulso; con desintonia grande no excita.
- **Dos qubits acoplados:** $H_0=2\pi J\,\sigma_z\otimes\sigma_z/4$. El eje de giro de un qubit **depende del estado del otro** ($+z$ si el otro es $\lvert0\rangle$, $-z$ si es $\lvert1\rangle$). Con el otro qubit en superposición, los dos quedan **entrelazados**.

Criterio: sin entrelazamiento, $\rho=\rho_A\otimes\rho_B$; si $\rho$ no se puede escribir así, hay entrelazamiento.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/12_control_cuantico.ipynb) [📄 Documento completo (PDF)](docs/pdf/12_control_cuantico.pdf)

---

## 13. Tomografía de estados

Al terminar un cálculo hay que **leer** el estado de los qubits. La tomografía lo hace por completo. Medir colapsa el estado, pero la información queda en las probabilidades.

**Un qubit.** Las coordenadas en la esfera de Bloch son $(a,b,c)=(\langle\sigma_x\rangle,\langle\sigma_y\rangle,\langle\sigma_z\rangle)$ y $P(0)=\frac{1+c}2$, $P(1)=\frac{1-c}2$. Repitiendo la medición se estima cada una, y con ellas:

$$\rho=\tfrac12\big(I+a\,\sigma_x+b\,\sigma_y+c\,\sigma_z\big).$$

Si solo se puede medir en $z$, para medir $x$ o $y$ se aplica antes un pulso $\pi/2$ que gira ese eje hasta $z$.

**Varios qubits.** $\rho$ se expande en productos de matrices de Pauli. Para dos qubits son 16 términos (uno es la normalización, hay que medir 15):

$$\rho=\tfrac14\sum_{i,j}c_{ij}\,\sigma_i\otimes\sigma_j.$$

Términos como $\sigma_x\otimes I$ son el momento angular de un qubit; términos como $\sigma_z\otimes\sigma_z$ son la **correlación** entre ambos. Con entrelazamiento no basta medir cada qubit por separado, porque parte de la información está en las correlaciones.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sileskyg26-rgb/ejemplos/blob/main/notebooks/13_tomografia.ipynb) [📄 Documento completo (PDF)](docs/pdf/13_tomografia.pdf)
