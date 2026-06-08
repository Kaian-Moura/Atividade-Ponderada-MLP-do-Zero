# -*- coding: utf-8 -*-
"""
Funções de ativação e suas derivadas.

Cada função de ativação possui uma versão "forward" (usada no forward pass)
e uma versão "derivative" (usada no backpropagation para calcular gradientes).
"""

import numpy as np


def relu(z):
    """
    ReLU (Rectified Linear Unit).
    
    Retorna max(0, z) para cada elemento do array.
    É a ativação mais usada em camadas ocultas porque:
    - É simples e computacionalmente barata
    - Não sofre do problema de vanishing gradients (para valores positivos)
    - Introduz não-linearidade suficiente para a rede aprender padrões complexos
    
    Parâmetros:
        z (np.ndarray): Saída linear da camada (z = W @ x + b)
    
    Retorna:
        np.ndarray: Ativação aplicada elemento a elemento
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivada da ReLU.
    
    Retorna 1 onde z > 0 e 0 onde z <= 0.
    Na prática, a derivada em z=0 não é definida, mas convencionamos como 0.
    Isso funciona bem na prática porque a probabilidade de z ser exatamente 0
    é desprezível com valores float.
    
    Parâmetros:
        z (np.ndarray): Saída linear da camada (ANTES da ativação)
    
    Retorna:
        np.ndarray: Derivada da ReLU, mesmo shape que z
    """
    return (z > 0).astype(np.float64)


def softmax(z):
    """
    Softmax — converte logits em probabilidades.
    
    Para um vetor z, softmax(z_i) = exp(z_i) / sum(exp(z_j)) para todo j.
    O resultado é um vetor de probabilidades que soma 1.
    
    Usamos o truque de estabilidade numérica: subtraímos o valor máximo de z
    antes de calcular a exponencial. Isso evita overflow (exp de números muito
    grandes) sem alterar o resultado matemático, pois:
        softmax(z) = softmax(z - max(z))
    
    Parâmetros:
        z (np.ndarray): Logits da camada de saída, shape (batch_size, num_classes)
    
    Retorna:
        np.ndarray: Probabilidades, mesmo shape que z, cada linha soma 1
    """
    # Subtrai o máximo de cada amostra para estabilidade numérica
    # keepdims=True mantém a dimensão para broadcasting correto
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
