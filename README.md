# Tarea 2: Algorítmos Genéticos

## Estructura de la Tarea

```
  AI_T2_Geneticos/
  │
  ├── environment.py     # wrapper de CartPole, evalúa un individuo
  ├── individual.py      # cromosoma (red neuronal pequeña) + método act()
  ├── population.py      # inicialización y manejo de la población
  ├── operators.py       # selección, cruce, mutación
  ├── fitness.py         # función de fitness
  ├── ga.py              # bucle principal del AG
  └── experiments.py     # corre las 3 configuraciones y grafica
```

## Entorno Virtual e instalación de gymnasium

```bash
  # 1. Crear el entorno virtual (solo una vez)
  python3 -m venv venv
  
  # 2. Activarlo
  # En macOS / Linux:
  source venv/bin/activate
  
  # 3. Instalar las dependencias (con el entorno activo)
  pip install gymnasium numpy matplotlib
  
  # 4. Verificar que estás dentro del entorno
  which python  
  
  # 5. Cuando termines, desactivar
  deactivate
```

## Diseño por módulo

### `indiviual.py`

Define el cromosoma y la política de control para CartPole-v1.

Arquitectura de la red neuronal:
- Entrada (4) → Capa oculta (hidden_size) → Salida (2)

- Las 4 entradas son las observaciones de CartPole:
  `[posición_carro, velocidad_carro, ángulo_palo, velocidad_angular_palo]`

- Las 2 salidas representan la preferencia por cada acción:
  ```
    0 = empujar izquierda
    1 = empujar derecha
  ```

- Se toma la acción con mayor valor (argmax), sin softmax, porque solo necesitamos la decisión, no una probabilidad.

- Estructura del cromosoma (vector plano de reales):
  ```
    Total de pesos = (4 × hidden_size) + hidden_size + (hidden_size × 2) + 2 = hidden_size × (4 + 1 + 2 + 1)  [con biases]

    Con hidden_size=4:
        W1: 4×4 = 16 pesos
        b1: 4        biases
        W2: 4×2 = 8  pesos
        b2: 2        biases
        ─────────────────
        Total: 30 valores reales
  ```

Un individuo representa una política completa de control para CartPole.

Su "genoma" es un vector numpy de números reales que codifica todos los
pesos y biases de una red neuronal pequeña (feedforward, 1 capa oculta).

#### Atributos

hidden_size : int
- Número de neuronas en la capa oculta.
- Más neuronas = más expresividad, pero también más genes que evolucionar.

genes : np.ndarray
- Vector 1D con todos los pesos/biases de la red. Es el cromosoma.
- Si se pasa, usa esos genes directamente (útil en cruce/mutación).
- Si no se pasa, inicializa con pesos aleatorios pequeños.

fitness : float
  Aptitud evaluada. None hasta que se llama a evaluate().
    
#### Métodos
- `_count_genes(hidden_size)`: Calcula el total de pesos+biases para una arquitectura dada.
- `_unpack_weights()`: Extrae los pesos del vector de genes y los reordena en matrices.
```
  Retorna:
  
  W1 : np.ndarray, shape (INPUT_SIZE, hidden_size)
  b1 : np.ndarray, shape (hidden_size,)
  W2 : np.ndarray, shape (hidden_size, OUTPUT_SIZE)
  b2 : np.ndarray, shape (OUTPUT_SIZE,)
```
- `act(observation)`: Dado el estado actual del entorno, decide qué acción tomar. El proceso:
  1. Propagación hacia adelante (forward pass) de la red.
  2. Capa oculta: ReLU(observation @ W1 + b1)
      ReLU introduce no-linealidad: sin ella, la red sería
      equivalente a una simple regresión lineal. Es decir, elimina valores negativos e introduce no linealidad.
  3. Capa salida: activación lineal (los logits se comparan
      directamente).
  4. Se elige la acción con el mayor logit (números sin normalizar).
```
  Parámetros:
  observation : np.ndarray, shape (4,)
      Vector [pos_carro, vel_carro, ángulo_palo, vel_angular_palo]
  
  Retorna:
  int : 0 (izquierda) o 1 (derecha)
```  

- `clone()`: Crea una copia independiente de este individuo (sin fitness).
- `n_genes()`: Retorna el número total de genes (pesos) del cromosoma.



### `fitness.py`

Define y evalúa la función de fitness para el problema CartPole-v1.

**¿Por qué esta función de fitness?**
CartPole ya entrega +1 de recompensa por cada paso que el palo no cae.
Entonces la recompensa acumulada de un episodio = número de pasos sobrevividos.

    fitness = promedio de pasos sobrevividos en N episodios

Decisiones de diseño importantes:
    
    1. Promediar varios episodios:
       El estado inicial de CartPole es ALEATORIO (uniforme en [-0.05, 0.05]).
       Si evaluamos con un solo episodio, un individuo mediocre puede tener
       "suerte" con un estado fácil y parecer mejor de lo que es.
       Con N=5 episodios se reduce mucho esa varianza sin costar demasiado tiempo.

    2. NO penalizar el ángulo o la posición manualmente:
       Ya está implícito: si el palo cae, el episodio termina y el individuo
       recibe menos pasos. Agregar penalizaciones extras complica la función
       sin necesidad.

    3. Normalización opcional [0, 1]:
       Dividir entre 500 (el máximo posible en v1) facilita comparar
       configuraciones con distintos max_steps o versiones del entorno.

#### Funciones

- `evaluate_fitness(individual, env, n_episodes, normalize, seed)`: 
    Evalúa qué tan buena es la política de un individuo en CartPole-v1.

    El individuo controla el carro durante N episodios independientes.
    El fitness es el número promedio de pasos que logra sobrevivir.

    Parámetros:
    - individual : Individual
      El cromosoma a evaluar. Se usa individual.act() para tomar decisiones.
    - env : gym.Env, opcional
      Entorno ya creado. Si es None, se crea uno internamente y se cierra
      al terminar. Pasar el entorno desde afuera es más eficiente cuando
      se evalúa una población completa.
    - n_episodes : int
      Número de episodios a promediar. Más episodios = fitness más estable, pero más lento con 5 es suficiente.
    - normalize : bool
      Si True, divide el resultado entre DEFAULT_MAX_STEPS (500) para
      obtener un valor en [0, 1]. Útil para comparar experimentos.
    - seed : int, opcional
      Semilla para reproducibilidad. Para comparar configuraciones.

    Retorna:
    - float
      Fitness del individuo. Sin normalizar: valores en [1, 500].
      Normalizado: valores en [0.002, 1.0].

- `evaluate_population(population: list, n_episodes: int = DEFAULT_N_EPISODES, normalize: bool = False, seed: int = None)`: Evalúa todos los individuos de una población y retorna sus fitness. Reutiliza un único entorno para toda la población, lo cual es más eficiente que crear/destruir el entorno por cada individuo.

    Parámetros:
    - population : Lista de individuos a evaluar.
    - n_episodes : Episodios por individuo.
    - normalize : Si True, normaliza entre 0 y 1.
    - seed : Semilla base. Cada individuo usa seed + su_índice para diversidad, pero reproducibilidad.

    Retorna:
    - list: Lista de fitness en el mismo orden que population. Los valores también quedan guardados en individual.fitness.

- `population_stats(fitness_values)`: Calcula estadísticas básicas del fitness de una generación. Para graficar la evolución del AG.

    Retorna:
    - dict con claves: mean, max, min, std
