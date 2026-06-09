# -*- coding: utf-8 -*-
"""
Script para executar experimentos comparativos e gerar gráficos.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from train_mnist import load_and_preprocess_mnist
from mlp.network import MLP

def plot_history(history1, name1, history2=None, name2=None):
    """
    Gera gráficos de Loss e Acurácia ao longo das épocas.
    Salva as imagens na pasta results/.
    """
    epochs1 = range(1, len(history1['loss']) + 1)
    
    # Plot 1: Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs1, history1['loss'], 'b-', label=f'{name1} (Treino)')
    
    if history2:
        epochs2 = range(1, len(history2['loss']) + 1)
        plt.plot(epochs2, history2['loss'], 'r-', label=f'{name2} (Treino)')
        
    plt.title('Curva de Loss (Cross-Entropy)')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_comparison.png')
    plt.close()
    
    # Plot 2: Acurácia
    plt.figure(figsize=(10, 5))
    plt.plot(epochs1, history1['val_accuracy'], 'b-', label=f'{name1} (Validação)')
    plt.plot(epochs1, history1['accuracy'], 'b--', alpha=0.5, label=f'{name1} (Treino)')
    
    if history2:
        epochs2 = range(1, len(history2['accuracy']) + 1)
        plt.plot(epochs2, history2['val_accuracy'], 'r-', label=f'{name2} (Validação)')
        plt.plot(epochs2, history2['accuracy'], 'r--', alpha=0.5, label=f'{name2} (Treino)')
        
    plt.title('Curva de Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/accuracy_comparison.png')
    plt.close()
    
    print("Gráficos gerados com sucesso na pasta results/:")
    print("- results/loss_comparison.png")
    print("- results/accuracy_comparison.png")


def run_experiment_2():
    """
    Roda um segundo experimento com um Learning Rate menor (0.01 em vez de 0.1)
    para comparar com a execução original.
    """
    print("=" * 60)
    print("Iniciando Experimento 2 (Learning Rate = 0.01)")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test, y_test_labels = load_and_preprocess_mnist()
    
    layer_sizes = [784, 128, 64, 10]
    epochs = 30
    batch_size = 64
    learning_rate = 0.01  # Modificação aqui!
    
    np.random.seed(42)
    mlp = MLP(layer_sizes)
    
    history = mlp.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        X_val=X_test,
        y_val=y_test
    )
    
    test_pred = mlp.predict(X_test)
    test_acc = np.mean(test_pred == y_test_labels)
    
    print(f"\nAcuracia final Experimento 2: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    # Salva histórico
    history_serializable = {
        key: [float(v) for v in values]
        for key, values in history.items()
    }
    history_serializable['test_accuracy'] = float(test_acc)
    
    with open("results/history_exp2.json", "w") as f:
        json.dump(history_serializable, f, indent=2)
        
    return history_serializable

def main():
    # Carrega histórico original
    with open("results/training_history.json", "r") as f:
        history1 = json.load(f)
        
    print(f"Experimento 1 carregado. Acurácia: {history1.get('test_accuracy', 0):.4f}")
    
    # Verifica se já rodou o exp 2 para não rodar tudo de novo à toa
    if os.path.exists("results/history_exp2.json"):
        print("Experimento 2 já existe, carregando...")
        with open("results/history_exp2.json", "r") as f:
            history2 = json.load(f)
    else:
        history2 = run_experiment_2()
        
    # Gera os plots
    plot_history(history1, "Config 1 (LR=0.1)", history2, "Config 2 (LR=0.01)")

if __name__ == "__main__":
    main()
