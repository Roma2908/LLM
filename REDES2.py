#!/usr/bin/env python3
"""
Visualizador interactivo de una red neuronal con datos de clasificación y gráfico del loss.

Este script implementa:
  - Una red neuronal configurable entrenada con backpropagation.
  - Visualización en tiempo real de la estructura de la red (nodos y conexiones con colores y grosores según el peso).
  - Una nube de puntos que muestra el problema de clasificación (ej.: XOR) junto con la frontera de decisión.
  - Una gráfica que muestra la evolución del loss (error) en el tiempo.
  
Requisitos:
  - Python 3.x
  - numpy
  - matplotlib
  - Un backend interactivo para matplotlib (por ejemplo, TkAgg o Qt5Agg)
"""

# Forzamos el uso de un backend interactivo
import matplotlib
matplotlib.use("TkAgg")  # O, si prefieres, "Qt5Agg"
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ----------------- Funciones y clases de la red -----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, topology):
        """
        Inicializa la red neuronal.
        topology: lista de enteros que indica el número de neuronas por capa.
                  Ejemplo: [2, 3, 1] => 2 entradas, 3 neuronas en capa oculta y 1 salida.
        """
        self.topology = topology
        self.num_layers = len(topology)
        self.weights = []
        self.biases = []
        # Inicialización de pesos con una inicialización tipo He
        for i in range(self.num_layers - 1):
            w = np.random.randn(topology[i+1], topology[i]) * np.sqrt(2 / topology[i])
            b = np.zeros((topology[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        """
        Propagación hacia adelante.
        x: vector de entrada (columna)
        Retorna:
          - activations: lista con las activaciones de cada capa (incluyendo la entrada)
          - zs: lista con los valores pre-activación (Wx + b) de cada capa
        """
        a = x
        activations = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return activations, zs

    def predict(self, x):
        """Realiza la predicción para una entrada x."""
        a = x
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """
        Ejecuta la retropropagación para un ejemplo (x, y) y retorna
        las derivadas respecto a cada peso y bias.
        """
        activations, zs = self.forward(x)
        # Error en la salida (derivada del MSE junto con la derivada de la sigmoide)
        delta = (activations[-1] - y) * (activations[-1] * (1 - activations[-1]))
        nabla_w = [None] * len(self.weights)
        nabla_b = [None] * len(self.biases)
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        nabla_b[-1] = delta
        
        # Propagación hacia atrás en las capas anteriores
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = activations[-l] * (1 - activations[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            nabla_b[-l] = delta
        
        return nabla_w, nabla_b

    def update_mini_batch(self, x, y, learning_rate):
        """
        Actualiza los parámetros de la red usando un ejemplo (x, y)
        """
        nabla_w, nabla_b = self.backprop(x, y)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * nabla_w[i]
            self.biases[i]  -= learning_rate * nabla_b[i]

    def compute_loss(self, X, Y):
        """
        Calcula el error cuadrático medio (MSE) sobre el conjunto (X, Y)
        """
        loss = 0
        m = X.shape[0]
        for i in range(m):
            x = X[i].reshape(-1, 1)
            y = Y[i].reshape(-1, 1)
            a = self.predict(x)
            loss += np.sum((a - y) ** 2)
        return loss / m

# ---------------- Funciones para la visualización ----------------
def compute_positions(topology):
    """
    Calcula las posiciones (x, y) para cada nodo de la red.
    Los nodos de cada capa se distribuyen verticalmente y las capas horizontalmente.
    Retorna un diccionario con claves (capa, índice_nodo) y valores (x, y).
    """
    positions = {}
    num_layers = len(topology)
    for i, n in enumerate(topology):
        x = i / (num_layers - 1) if num_layers > 1 else 0.5
        for j in range(n):
            y = (j + 1) / (n + 1)
            positions[(i, j)] = (x, y)
    return positions

def draw_network(ax, nn, positions):
    """
    Dibuja la red neuronal en el eje 'ax'.
      - Cada conexión se dibuja con un grosor y color que dependen del peso.
      - Los nodos se dibujan como círculos.
    """
    # Se normaliza respecto al peso máximo
    max_weight = 0
    for w in nn.weights:
        max_weight = max(max_weight, np.max(np.abs(w)))
    if max_weight == 0:
        max_weight = 1

    cmap = plt.cm.bwr  # Colormap: azul para negativo, rojo para positivo
    for i, w in enumerate(nn.weights):
        for j in range(w.shape[1]):
            for k in range(w.shape[0]):
                weight = w[k, j]
                norm = (weight + max_weight) / (2 * max_weight)  # normalización a [0, 1]
                color = cmap(norm)
                linewidth = 0.5 + (abs(weight) / max_weight) * 4
                pos1 = positions[(i, j)]
                pos2 = positions[(i + 1, k)]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color=color, linewidth=linewidth)
    # Dibuja los nodos
    for (i, j), (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.03, fc='lightgray', ec='k', zorder=3)
        ax.add_artist(circle)

def draw_classification(ax, nn, X, Y):
    """
    Dibuja la nube de puntos y la frontera de decisión.
    Se asume que:
       - Los datos de entrada tienen dos dimensiones.
       - La salida es un valor entre 0 y 1 (clasificación binaria).
    """
    ax.clear()
    # Creamos una malla para evaluar la red en cada punto y dibujar la frontera de decisión
    x0 = np.linspace(-0.1, 1.1, 200)
    x1 = np.linspace(-0.1, 1.1, 200)
    xx, yy = np.meshgrid(x0, x1)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Se evalúa la red en cada punto de la malla
    preds = np.array([nn.predict(np.array(p).reshape(-1,1)) for p in grid_points])
    preds = preds.reshape(xx.shape)
    
    # Dibujamos un contourf: niveles <0.5 y >0.5 se colorean diferente
    ax.contourf(xx, yy, preds, levels=[-0.1, 0.5, 1.1], cmap='RdBu', alpha=0.3)
    
    # Dibujamos los puntos de entrenamiento
    for i, point in enumerate(X):
        label = Y[i][0]
        color = 'red' if label > 0.5 else 'blue'
        ax.scatter(point[0], point[1], c=color, edgecolors='k', s=100)
    ax.set_title("Clasificación y Frontera de Decisión")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

# ------------------- Función principal -------------------
def main():
    # ---------------- Configuración ----------------
    # Define la topología de la red. Ejemplo:
    #   [2, 3, 1] -> 2 entradas, 3 neuronas en la capa oculta, 1 salida.
    topology = [2, 3, 1]
    learning_rate = 0.5
    epochs = 2000
    update_interval = 50  # Se actualiza la visualización cada 50 iteraciones
    # -------------------------------------------------
    
    # Inicializa la red y calcula las posiciones para dibujarla.
    nn = NeuralNetwork(topology)
    positions = compute_positions(topology)
    
    # Definición del conjunto de entrenamiento:
    # Si la red es de 2 entradas y 1 salida se entrena en XOR, en otro caso se generan datos aleatorios.
    if topology[0] == 2 and topology[-1] == 1:
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
        Y = np.array([[0],
                      [1],
                      [1],
                      [0]])
    else:
        np.random.seed(42)
        num_samples = 100
        X = np.random.rand(num_samples, topology[0])
        Y = np.random.rand(num_samples, topology[-1])
    
    # Configuración de la figura con GridSpec para tres paneles:
    #  - Panel superior izquierdo: red neuronal.
    #  - Panel superior derecho: clasificación y frontera.
    #  - Panel inferior: gráfica del loss.
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
    ax_network = fig.add_subplot(gs[0, 0])
    ax_classification = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, :])
    
    plt.ion()  # Activa el modo interactivo
    
    loss_history = []
    epochs_list = []
    
    # Bucle de entrenamiento
    for epoch in range(epochs):
        # Selecciona un ejemplo aleatorio para entrenamiento estocástico
        idx = np.random.randint(0, X.shape[0])
        x_sample = X[idx].reshape(-1, 1)
        y_sample = Y[idx].reshape(-1, 1)
        nn.update_mini_batch(x_sample, y_sample, learning_rate)
        
        if epoch % update_interval == 0:
            # Actualiza la visualización de la red
            ax_network.clear()
            draw_network(ax_network, nn, positions)
            ax_network.set_title("Red Neuronal")
            ax_network.set_xlim(-0.1, 1.1)
            ax_network.set_ylim(0, 1)
            
            # Actualiza la visualización de la clasificación (solo si es el caso de 2 entradas y 1 salida)
            if topology[0] == 2 and topology[-1] == 1:
                draw_classification(ax_classification, nn, X, Y)
            
            # Calcula y guarda el loss
            loss = nn.compute_loss(X, Y)
            loss_history.append(loss)
            epochs_list.append(epoch)
            
            # Actualiza la gráfica del loss
            ax_loss.clear()
            ax_loss.plot(epochs_list, loss_history, 'b-', lw=2)
            ax_loss.set_title("Pérdida (Loss) en el Tiempo")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            
            # Actualiza el título global de la figura
            fig.suptitle(f"Epoch {epoch}   Loss: {loss:.4f}", fontsize=14)
            plt.draw()
            plt.pause(0.001)
    
    plt.ioff()  # Desactiva el modo interactivo
    plt.show()

if __name__ == "__main__":
    main()
