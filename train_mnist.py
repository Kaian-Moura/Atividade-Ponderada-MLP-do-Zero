# -*- coding: utf-8 -*-
"""
Script de treinamento do MLP no dataset MNIST.

Carrega os dados, pré-processa, treina a rede e reporta a acurácia final.
O TensorFlow/Keras é usado APENAS para carregar o dataset MNIST.
Todo o treinamento é feito com NumPy puro.
"""

import numpy as np
import os
import json
import urllib.request

# Importa nossa implementação do MLP
from mlp.network import MLP


def load_and_preprocess_mnist():
    """
    Carrega o MNIST e faz o pré-processamento necessário.
    
    Tenta baixar o arquivo mnist.npz diretamente do Google Storage
    caso não esteja presente localmente, permitindo rodar sem TensorFlow.
    """
    mnist_path = "mnist.npz"
    if not os.path.exists(mnist_path):
        print(f"Baixando dataset MNIST de mirror público...")
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        try:
            urllib.request.urlretrieve(url, mnist_path)
            print("Download concluído com sucesso!")
        except Exception as e:
            print(f"Erro ao baixar: {e}")
            raise e
            
    # Carrega dados do arquivo NPZ
    with np.load(mnist_path) as data:
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']
    
    print(f"Dataset carregado:")
    print(f"  Treino: {X_train.shape[0]} amostras, imagens {X_train.shape[1]}x{X_train.shape[2]}")
    print(f"  Teste:  {X_test.shape[0]} amostras")
    
    # 1. Flatten: (60000, 28, 28) → (60000, 784)
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float64)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float64)
    
    # 2. Normalização: [0, 255] → [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # 3. One-hot encoding dos rótulos
    num_classes = 10
    y_train_onehot = np.zeros((y_train.shape[0], num_classes))
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
    
    y_test_onehot = np.zeros((y_test.shape[0], num_classes))
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
    
    return X_train, y_train_onehot, X_test, y_test_onehot, y_test



def main():
    """
    Fluxo principal de treinamento.
    """
    print("=" * 60)
    print("MLP do Zero — Treinamento no MNIST")
    print("=" * 60)
    print()
    
    # Carrega e pré-processa os dados
    X_train, y_train, X_test, y_test, y_test_labels = load_and_preprocess_mnist()
    print()
    
    # Define a arquitetura da rede
    # 784 (entrada) → 128 (oculta 1) → 64 (oculta 2) → 10 (saída)
    layer_sizes = [784, 128, 64, 10]
    
    print(f"Arquitetura: {' -> '.join(map(str, layer_sizes))}")
    print(f"Ativação ocultas: ReLU")
    print(f"Ativação saída: Softmax")
    print(f"Loss: Cross-Entropy")
    print(f"Otimizador: SGD")
    print()
    
    # Hiperparâmetros
    epochs = 30
    batch_size = 64
    learning_rate = 0.1
    
    print(f"Hiperparâmetros:")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print()
    
    # Fixa a seed para reprodutibilidade
    np.random.seed(42)
    
    # Cria e treina a rede
    mlp = MLP(layer_sizes)
    
    print("Iniciando treinamento...")
    print("-" * 60)
    
    history = mlp.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        X_val=X_test,
        y_val=y_test
    )
    
    print("-" * 60)
    print()
    
    # Avaliação final no conjunto de teste
    test_pred = mlp.predict(X_test)
    test_acc = np.mean(test_pred == y_test_labels)
    
    print(f"Acuracia final no teste: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    if test_acc >= 0.92:
        print("Meta de 92% atingida!")
    else:
        print("Meta de 92% NAO atingida. Considere ajustar hiperparametros.")
    
    # Salva o histórico para gerar plots depois
    os.makedirs("results", exist_ok=True)
    
    history_serializable = {
        key: [float(v) for v in values]
        for key, values in history.items()
    }
    history_serializable['test_accuracy'] = float(test_acc)
    history_serializable['config'] = {
        'layer_sizes': layer_sizes,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    
    with open("results/training_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)
    
    print(f"\nHistórico salvo em results/training_history.json")


if __name__ == "__main__":
    main()
