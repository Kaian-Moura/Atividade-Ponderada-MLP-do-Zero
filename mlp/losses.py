# -*- coding: utf-8 -*-
"""
Funções de loss (custo) e seus gradientes.

A loss mede o quão longe as predições da rede estão dos rótulos reais.
Durante o treinamento, queremos minimizar esse valor.
"""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-Entropy Loss para classificação multiclasse.
    
    Fórmula: L = -1/N * sum( sum( y_true * log(y_pred) ) )
    
    Onde:
    - y_pred são as probabilidades previstas (saída do softmax)
    - y_true são os rótulos em formato one-hot
    - N é o número de amostras no batch
    
    Usamos np.clip para evitar log(0), que resultaria em -inf.
    O clipping limita y_pred ao intervalo [1e-15, 1-1e-15], o que é
    pequeno o suficiente para não afetar o resultado numérico, mas
    grande o suficiente para evitar instabilidades.
    
    Parâmetros:
        y_pred (np.ndarray): Probabilidades previstas, shape (batch_size, num_classes)
        y_true (np.ndarray): Rótulos one-hot, shape (batch_size, num_classes)
    
    Retorna:
        float: Loss média do batch
    """
    n_samples = y_true.shape[0]
    
    # Clipping para evitar log(0) — sem isso, um único valor 0 em y_pred
    # faria a loss explodir para infinito
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Cross-entropy: só os termos onde y_true=1 contribuem (por causa da multiplicação)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples
    
    return loss


def cross_entropy_gradient(y_pred, y_true):
    """
    Gradiente combinado de Softmax + Cross-Entropy.
    
    Uma das coisas mais elegantes do combo softmax + cross-entropy é que
    o gradiente simplifica para uma expressão muito simples:
    
        dL/dz = y_pred - y_true
    
    Onde z são os logits (entrada do softmax, ou seja, a saída linear
    da última camada ANTES da ativação).
    
    Essa simplificação acontece porque as derivadas parciais do softmax
    e do log se cancelam de forma conveniente. O resultado é:
    - Se a classe correta é a classe k, o gradiente no neurônio k é (p_k - 1)
    - Para todas as outras classes j, o gradiente é p_j
    
    Isso significa que o gradiente "empurra" a probabilidade da classe correta
    para cima e as probabilidades das classes erradas para baixo.
    
    Dividimos por n_samples para obter o gradiente médio do batch,
    consistente com a loss média que calculamos acima.
    
    Parâmetros:
        y_pred (np.ndarray): Probabilidades previstas (saída do softmax), shape (batch_size, num_classes)
        y_true (np.ndarray): Rótulos one-hot, shape (batch_size, num_classes)
    
    Retorna:
        np.ndarray: Gradiente da loss em relação aos logits, shape (batch_size, num_classes)
    """
    n_samples = y_true.shape[0]
    return (y_pred - y_true) / n_samples
