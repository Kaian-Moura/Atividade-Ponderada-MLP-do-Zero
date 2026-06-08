# -*- coding: utf-8 -*-
"""
Otimizadores para atualização dos pesos da rede neural.

O otimizador é responsável por usar os gradientes calculados no backpropagation
para atualizar os pesos e biases da rede, fazendo-a "aprender".
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent (SGD) — Descida de Gradiente Estocástica.
    
    A regra de atualização é simples:
        w = w - learning_rate * gradiente
    
    "Estocástico" porque não usamos o dataset inteiro para calcular o gradiente,
    mas sim mini-batches (subconjuntos aleatórios). Isso introduz ruído no
    gradiente, mas torna o treinamento muito mais rápido e pode até ajudar
    a escapar de mínimos locais.
    
    Parâmetros:
        learning_rate (float): Taxa de aprendizado. Controla o tamanho do passo
            na direção oposta ao gradiente. Valores típicos: 0.001 a 0.1.
            - Muito grande: a loss oscila e pode divergir
            - Muito pequeno: o treinamento fica lento demais
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, weights, biases, grad_weights, grad_biases):
        """
        Atualiza os pesos e biases usando a regra do SGD.
        
        Para cada camada i:
            W_i = W_i - lr * dL/dW_i
            b_i = b_i - lr * dL/db_i
        
        A atualização é feita in-place (modifica os arrays diretamente),
        o que é mais eficiente em memória.
        
        Parâmetros:
            weights (list[np.ndarray]): Lista de matrizes de pesos, uma por camada
            biases (list[np.ndarray]): Lista de vetores de bias, um por camada
            grad_weights (list[np.ndarray]): Gradientes dos pesos (dL/dW)
            grad_biases (list[np.ndarray]): Gradientes dos biases (dL/db)
        """
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * grad_weights[i]
            biases[i] -= self.learning_rate * grad_biases[i]
