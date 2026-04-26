import numpy as np



INPUT_SIZE  = 4   # dimensión del espacio de observación de CartPole
OUTPUT_SIZE = 2   # Acciones posibles (izquierda / derecha)



class Individual:

    def __init__(self, hidden_size: int = 4, genes: np.ndarray = None):

        self.hidden_size = hidden_size
        self.fitness = None

        if genes is not None:
            self.genes = genes.copy()
        else:
            # Inicialización aleatoria con distribución normal pequeña.
            # (siempre elige la misma acción), lo cual frena la evolución.
            n_genes = self._count_genes(hidden_size)
            self.genes = np.random.randn(n_genes) * 0.5


    @staticmethod
    def _count_genes(hidden_size: int) -> int:
        w1 = INPUT_SIZE * hidden_size    # pesos entrada → oculta
        b1 = hidden_size                 # biases capa oculta
        w2 = hidden_size * OUTPUT_SIZE   # pesos oculta → salida
        b2 = OUTPUT_SIZE                 # biases capa salida
        return w1 + b1 + w2 + b2

    def _unpack_weights(self):
        h = self.hidden_size
        idx = 0

        # Extraer W1: INPUT_SIZE × hidden_size
        w1_size = INPUT_SIZE * h
        W1 = self.genes[idx: idx + w1_size].reshape(INPUT_SIZE, h)
        idx += w1_size

        # Extraer b1: hidden_size
        b1 = self.genes[idx: idx + h]
        idx += h

        # Extraer W2: hidden_size × OUTPUT_SIZE
        w2_size = h * OUTPUT_SIZE
        W2 = self.genes[idx: idx + w2_size].reshape(h, OUTPUT_SIZE)
        idx += w2_size

        # Extraer b2: OUTPUT_SIZE
        b2 = self.genes[idx: idx + OUTPUT_SIZE]

        return W1, b1, W2, b2

    def act(self, observation: np.ndarray) -> int:
        W1, b1, W2, b2 = self._unpack_weights()

        # Capa oculta con ReLU
        hidden = np.maximum(0, observation @ W1 + b1)

        # Capa de salida (logits)
        output = hidden @ W2 + b2

        # Acción = índice del logit más alto
        return int(np.argmax(output))

    def clone(self) -> "Individual":
        return Individual(hidden_size=self.hidden_size, genes=self.genes.copy())

    def n_genes(self) -> int:
        return len(self.genes)

    def __repr__(self) -> str:
        fitness_str = f"{self.fitness:.2f}" if self.fitness is not None else "N/A"
        return (f"Individual(hidden={self.hidden_size}, "
                f"genes={self.n_genes()}, fitness={fitness_str})")


def random_individual(hidden_size: int = 4) -> Individual:
    """Crea un individuo con genes aleatorios. Proximante para population.py"""
    return Individual(hidden_size=hidden_size)


"""
    Test para ver si funciona
"""
if __name__ == "__main__":
    print("=== Verificación de Individual ===\n")

    ind = Individual(hidden_size=4)
    print(f"Individuo creado: {ind}")
    print(f"Primeros 5 genes: {ind.genes[:5].round(3)}")

    # Simular una observación de CartPole
    obs = np.array([0.01, -0.02, 0.03, 0.04])
    accion = ind.act(obs)
    print(f"\nObservación: {obs}")
    print(f"Acción elegida: {accion} ({'derecha' if accion == 1 else 'izquierda'})")

    # Verificar clonación
    clon = ind.clone()
    clon.genes[0] = 999.0
    print(f"\nGen[0] original: {ind.genes[0]:.3f}  (debe ser distinto de 999)")
    print(f"Gen[0] clon:     {clon.genes[0]:.3f}  (debe ser 999)")
