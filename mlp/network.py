# -*- coding: utf-8 -*-
"""
Implementação do Multi-Layer Perceptron (MLP).

Esta classe implementa uma rede neural feedforward com número arbitrário
de camadas. Usa as funções de ativação, loss e otimizador definidos nos
outros módulos do pacote.
"""

import numpy as np
from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import cross_entropy_loss, cross_entropy_gradient
from mlp.optimizers import SGD


class MLP:
    """
    Multi-Layer Perceptron com número arbitrário de camadas.
    
    Arquitetura:
        - Camadas ocultas usam ReLU como ativação
        - Camada de saída usa Softmax (para classificação multiclasse)
        - Loss: Cross-Entropy
        - Otimizador: SGD
    
    Exemplo de uso:
        >>> mlp = MLP([784, 128, 64, 10])  # 2 camadas ocultas
        >>> mlp.forward(X_batch)            # forward pass
    
    Parâmetros:
        layer_sizes (list[int]): Número de neurônios em cada camada.
            O primeiro elemento é o tamanho da entrada (ex: 784 para MNIST),
            o último é o número de classes (ex: 10 para dígitos 0-9),
            e os intermediários são as camadas ocultas.
    """
    
    def __init__(self, layer_sizes):
        """
        Inicializa os pesos e biases da rede.
        
        Usamos He Initialization para os pesos:
            W ~ N(0, sqrt(2/n_entrada))
        
        Essa inicialização é projetada especificamente para a ReLU:
        - Se inicializarmos com valores muito pequenos, o sinal "encolhe" 
          a cada camada e a rede não aprende (vanishing gradients)
        - Se inicializarmos com valores muito grandes, o sinal "explode"
          e a loss diverge (exploding gradients)
        - He initialization mantém a variância do sinal aproximadamente 
          constante através das camadas, garantindo gradientes saudáveis
        
        Os biases são inicializados com zero, o que é a prática padrão.
        Diferente dos pesos, não há problema de simetria com biases zerados
        porque os pesos já são diferentes entre si.
        
        Parâmetros:
            layer_sizes (list[int]): Ex: [784, 128, 64, 10]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # número de camadas com pesos
        
        # Inicializa pesos e biases para cada camada
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            n_in = layer_sizes[i]      # neurônios na camada anterior
            n_out = layer_sizes[i + 1]  # neurônios na camada atual
            
            # He initialization: escala pelo número de neurônios de entrada
            w = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Armazena valores intermediários para uso no backpropagation
        # z = saída linear (antes da ativação): z = W @ x + b
        # a = saída ativada (depois da ativação): a = f(z)
        self.z_values = []  # saídas lineares de cada camada
        self.a_values = []  # ativações de cada camada
    
    def forward(self, X):
        """
        Forward pass — propaga a entrada pela rede, camada por camada.
        
        Para cada camada oculta:
            z = X @ W + b      (transformação linear)
            a = relu(z)        (ativação não-linear)
        
        Para a última camada (saída):
            z = X @ W + b
            a = softmax(z)     (converte em probabilidades)
        
        Armazena os valores intermediários (z e a) porque serão necessários
        no backpropagation para calcular os gradientes.
        
        Parâmetros:
            X (np.ndarray): Dados de entrada, shape (batch_size, n_features)
        
        Retorna:
            np.ndarray: Probabilidades previstas, shape (batch_size, num_classes)
        """
        self.z_values = []
        self.a_values = [X]  # a[0] é a própria entrada (útil no backprop)
        
        current_input = X
        
        for i in range(self.num_layers):
            # Transformação linear: z = input @ W + b
            z = current_input @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # Aplica a ativação: ReLU para camadas ocultas, Softmax para a saída
            if i < self.num_layers - 1:
                a = relu(z)
            else:
                a = softmax(z)
            
            self.a_values.append(a)
            current_input = a
        
        return self.a_values[-1]  # retorna a saída da última camada
    
    def predict(self, X):
        """
        Faz predições retornando a classe com maior probabilidade.
        
        Parâmetros:
            X (np.ndarray): Dados de entrada, shape (batch_size, n_features)
        
        Retorna:
            np.ndarray: Índices das classes preditas, shape (batch_size,)
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
