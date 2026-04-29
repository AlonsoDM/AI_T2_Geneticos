# Tarea 2: Algoritmos Genéticos – CartPole-v1

Control de un carro con palo balanceante (Gymnasium / Farama AI) mediante un algoritmo genético que evoluciona los pesos de una red neuronal pequeña.

---

## Estructura del proyecto

```
AI_T2_Geneticos/
├── individual.py      # Cromosoma: red neuronal feedforward + método act()
├── fitness.py         # Evaluación de fitness sobre el entorno CartPole-v1
├── population.py      # Creación y estadísticas de la población
├── operators.py       # Selección, cruce y mutación
├── main.py            # Clase GeneticAlgorithm + función run_experiment()
├── experiments.py     # 3 configuraciones experimentales + gráficas
├── env.py             # Wrapper opcional de CartPole-v1
└── README.md          # Este archivo
```

---

## Instalación y ejecución

### 1. Crear y activar el entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 2. Instalar dependencias

```bash
pip install gymnasium numpy matplotlib
```

### 3. Ejecutar los experimentos completos (3 configuraciones, 50 generaciones c/u)

```bash
python experiments.py
```

Esto genera tres archivos PNG en el directorio de trabajo:

| Archivo | Contenido |
|---------|-----------|
| `plot_individual_configs.png` | Media ± std y máximo por generación, un subplot por configuración |
| `plot_comparison.png` | Comparación directa de las 3 configuraciones (max y media) |
| `plot_bar_summary.png` | Mejor fitness y media final por configuración |

### 4. Ejecutar una sola configuración personalizada

```python
from main import run_experiment

result = run_experiment({
    "population_size": 30,
    "n_generations":   50,
    "mutation_rate":   0.10,
    "crossover_type":  "single_point",   # "single_point" | "uniform" | "arithmetic"
    "selection_type":  "tournament",     # "tournament" | "roulette"
    "seed":            42,
})
print(f"Mejor fitness: {result['best_fitness']:.1f}")
```

### 5. Pruebas rápidas de módulos individuales

```bash
python individual.py   # verifica la red neuronal y act()
python fitness.py      # evalúa individuos aleatorios en CartPole
python main.py         # smoke-test: 5 generaciones con 30 individuos
```

---

## Diseño del algoritmo genético

### Representación del cromosoma

Cada individuo es una red neuronal feedforward con una capa oculta:

```
Entrada (4) ──→ ReLU ──→ Oculta (4) ──→ Lineal ──→ Salida (2)
```

Las 4 entradas corresponden al estado de CartPole:
`[posición_carro, velocidad_carro, ángulo_palo, velocidad_angular_palo]`

Las 2 salidas representan la preferencia por cada acción (0 = izquierda, 1 = derecha). Se toma `argmax` de la salida, sin softmax.

El cromosoma es un vector numpy de 30 reales: W1 (16) + b1 (4) + W2 (8) + b2 (2).

### Función de fitness

```
fitness(individuo) = promedio de pasos sobrevividos en N episodios
```

CartPole entrega +1 de recompensa por cada paso que el palo se mantiene vertical. El máximo teórico es 500 (límite de la versión v1). Se promedian 5 episodios con estados iniciales aleatorios distintos para reducir la varianza de la estimación.

### Operadores

| Operador | Opciones implementadas |
|----------|----------------------|
| Selección | Torneo (k configurable), Ruleta proporcional al fitness |
| Cruce | Un punto, Uniforme (p=0.5), Aritmético (α aleatorio) |
| Mutación | Gaussiana: cada gen perturba con probabilidad `mutation_rate` sumando N(0, σ) |
| Elitismo | Los top-k individuos pasan intactos a la siguiente generación |

---

## Configuraciones experimentales

| | Config 1 | Config 2 | Config 3 |
|---|---|---|---|
| **Estrategia** | Conservadora | Balanceada | Agresiva |
| `population_size` | 30 | 50 | 60 |
| `n_generations` | 50 | 50 | 50 |
| `mutation_rate` | 0.05 | 0.15 | 0.30 |
| `crossover_type` | `single_point` | `uniform` | `arithmetic` |
| `selection_type` | `tournament` | `tournament` | `tournament` |
| `tournament_k` | 3 | 5 | 5 |
| `elitism` | 2 | 2 | 3 |
| `crossover_rate` | 0.80 | 0.85 | 0.90 |

---

## Análisis crítico

### Contexto y motivación

El problema CartPole-v1 es un entorno de control clásico en el que un agente debe aplicar fuerzas laterales a un carro para mantener un palo vertical en equilibrio. El espacio de observación es continuo (4D) y el espacio de acción es discreto binario. Aunque existen soluciones de gradiente (PPO, DQN) altamente optimizadas para este problema, los algoritmos genéticos ofrecen una perspectiva complementaria: evolucionan directamente los parámetros de la política sin calcular gradientes, lo que los hace robustos frente a funciones de fitness no diferenciables y espacios de búsqueda multimodales.

### Diseño de la representación

La elección de codificar los pesos de una red neuronal como cromosoma real (en lugar de binario o simbólico) es natural para este dominio: los pesos son continuos y su espacio de búsqueda es el de los números reales. Con `hidden_size=4` y la arquitectura 4→4→2, el cromosoma tiene solo 30 genes, un espacio de búsqueda suficientemente compacto para ser explorado por un AG pequeño en pocas generaciones. Arquitecturas más grandes (más neuronas ocultas) aumentarían la capacidad de la red pero también la dimensionalidad del espacio de búsqueda, requiriendo poblaciones más grandes y más generaciones.

### Comparación de configuraciones

**Config 1 (Conservadora, Pop=30, mut=0.05, cruce en un punto):** La baja tasa de mutación limita la perturbación genética por generación, lo que permite conservar bloques de genes ya buenos. El cruce en un punto es el más sencillo: intercambia segmentos contiguos del cromosoma, preservando cierta coherencia local entre genes. Con solo 30 individuos, la presión de selección es alta y la convergencia tiende a ser rápida, aunque con mayor riesgo de converger a soluciones subóptimas si el mejor individuo inicial domina la población prematuramente.

**Config 2 (Balanceada, Pop=50, mut=0.15, cruce uniforme):** El cruce uniforme mezcla genes de forma independiente por posición, generando mayor diversidad genética entre los hijos que el cruce en un punto. La tasa de mutación moderada (0.15) permite explorar el espacio sin destruir soluciones parcialmente buenas. Con 50 individuos, la población es suficientemente grande para mantener diversidad durante más generaciones. Esta configuración representa el punto de equilibrio entre exploración (descubrir nuevas regiones del espacio de búsqueda) y explotación (refinar soluciones existentes).

**Config 3 (Agresiva, Pop=60, mut=0.30, cruce aritmético):** El cruce aritmético produce hijos que son combinaciones lineales de los padres, lo que genera una exploración más suave dentro del convex hull del espacio genético: no introduce valores extremos, sino interpolaciones. Sin embargo, la alta tasa de mutación (0.30) introduce mucho ruido, lo que puede ser beneficioso para escapar de mínimos locales o perjudicial al perturbar genes ya bien calibrados. Con 60 individuos y elitismo=3, la población mantiene diversidad incluso cuando la mutación es alta.

### Resultados observados y justificación

En general, las tres configuraciones logran encontrar políticas con fitness cercano a 500 (el máximo posible) dentro de las 50 generaciones, lo cual es consistente con la relativa simplicidad del problema CartPole cuando se usa una red neuronal pequeña. La diferencia principal entre configuraciones radica en la **velocidad de convergencia** y en la **estabilidad de la media poblacional**:

- La Config 1 converge rápidamente en términos del individuo máximo, pero su media poblacional puede quedarse baja si pocos individuos dominan la selección (pérdida de diversidad).
- La Config 2 tiende a elevar la media poblacional de forma más gradual y sostenida, ya que el cruce uniforme distribuye mejor los genes exitosos entre más individuos.
- La Config 3 muestra mayor variabilidad en la media por el alto ruido de mutación, pero el individuo élite está protegido por el elitismo, preservando la mejor solución encontrada.

### Elección del operador de selección

Se usó torneo (tournament selection) como mecanismo primario porque es paramétrico en la presión de selección (ajustable por `k`), eficiente en tiempo y no requiere normalizar el fitness. La alternativa de ruleta proporcional tiene el problema conocido de super-individuos: cuando un individuo tiene un fitness muy superior al resto, domina las selecciones y reduce la diversidad drásticamente.

### Limitaciones y trabajo futuro

1. **Sin gradientes adaptativos:** La tasa de mutación es fija por configuración. Estrategias como la auto-adaptación de parámetros (ES – Evolution Strategies, CMA-ES) ajustan σ dinámicamente según el progreso de la población.
2. **Fitness con varianza:** Promediar 5 episodios reduce pero no elimina el ruido estocástico del estado inicial. Aumentar a 10–15 episodios mejoraría la precisión de la evaluación, a costa de tiempo de cómputo.
3. **Arquitectura fija:** Explorar neuroevolution (NEAT) permitiría co-evolucionar la topología de la red junto con sus pesos.
4. **Reproducibilidad vs. exploración:** Fijar la semilla garantiza resultados reproducibles para comparación, pero en la práctica conviene promediar sobre múltiples semillas para obtener intervalos de confianza en los resultados.

### Conclusión

Los tres enfoques demuestran que los algoritmos genéticos son capaces de resolver CartPole-v1 de forma efectiva y sin acceso a gradientes. La configuración balanceada (Config 2) ofrece el mejor compromiso entre velocidad de convergencia, diversidad poblacional y robustez. Las variantes conservadora y agresiva ilustran los extremos del trade-off exploración/explotación: poca mutación converge rápido pero puede estancarse; mucha mutación explora ampliamente pero puede degradar soluciones buenas. El diseño del algoritmo genético —representación, operadores y parámetros— debe adaptarse a la naturaleza del problema, y CartPole demuestra ser un banco de pruebas ideal para experimentar con estas decisiones.
