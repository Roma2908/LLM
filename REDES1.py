#!/usr/bin/env python3
"""
Visualizador interactivo de redes neuronales en Python.
"""

import matplotlib
matplotlib.use("TkAgg")  # Forzamos un backend interactivo
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, topology):
        self.topology = topology
        self.num_layers = len(topology)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            w = np.random.randn(topology[i+1], topology[i]) * np.sqrt(2 / topology[i])
            b = np.zeros((topology[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
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
        a = x
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        activations, zs = self.forward(x)
        delta = (activations[-1] - y) * (activations[-1] * (1 - activations[-1]))
        nabla_w = [None] * len(self.weights)
        nabla_b = [None] * len(self.biases)
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        nabla_b[-1] = delta
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = activations[-l] * (1 - activations[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            nabla_b[-l] = delta
        return nabla_w, nabla_b

    def update_mini_batch(self, x, y, learning_rate):
        nabla_w, nabla_b = self.backprop(x, y)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * nabla_w[i]
            self.biases[i]  -= learning_rate * nabla_b[i]

    def compute_loss(self, X, Y):
        loss = 0
        m = X.shape[0]
        for i in range(m):
            x = X[i].reshape(-1, 1)
            y = Y[i].reshape(-1, 1)
            a = self.predict(x)
            loss += np.sum((a - y) ** 2)
        return loss / m

def compute_positions(topology):
    positions = {}
    num_layers = len(topology)
    for i, n in enumerate(topology):
        x = i / (num_layers - 1) if num_layers > 1 else 0.5
        for j in range(n):
            y = (j + 1) / (n + 1)
            positions[(i, j)] = (x, y)
    return positions

def draw_network(ax, nn, positions):
    max_weight = 0
    for w in nn.weights:
        max_weight = max(max_weight, np.max(np.abs(w)))
    if max_weight == 0:
        max_weight = 1

    cmap = plt.cm.bwr
    for i, w in enumerate(nn.weights):
        for j in range(w.shape[1]):
            for k in range(w.shape[0]):
                weight = w[k, j]
                norm = (weight + max_weight) / (2 * max_weight)
                color = cmap(norm)
                linewidth = 0.5 + (abs(weight) / max_weight) * 4
                pos1 = positions[(i, j)]
                pos2 = positions[(i + 1, k)]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color=color, linewidth=linewidth)
    for (i, j), (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.03, fc='lightgray', ec='k', zorder=3)
        ax.add_artist(circle)

def main():
    topology = [2, 3, 1]
    learning_rate = 0.5
    epochs = 2000
    update_interval = 50

    nn = NeuralNetwork(topology)
    positions = compute_positions(topology)
    
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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.ion()  # Modo interactivo activado
    
    for epoch in range(epochs):
        idx = np.random.randint(0, X.shape[0])
        x_sample = X[idx].reshape(-1, 1)
        y_sample = Y[idx].reshape(-1, 1)
        nn.update_mini_batch(x_sample, y_sample, learning_rate)
        if epoch % update_interval == 0:
            ax.clear()
            loss = nn.compute_loss(X, Y)
            ax.set_title(f"Epoch {epoch}   Loss: {loss:.4f}")
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(0, 1)
            draw_network(ax, nn, positions)
            plt.draw()
            plt.pause(0.001)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
