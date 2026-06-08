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
    
    def backward(self, y_true):
        """
        Backpropagation — calcula os gradientes de todos os pesos e biases.
        
        O backprop usa a regra da cadeia (chain rule) para propagar o erro
        da saída de volta até a primeira camada. Para cada camada, calculamos:
        
        1. O "delta" (erro local da camada):
           - Última camada: delta = dL/dz = (y_pred - y_true) / N
             (gradiente combinado de softmax + cross-entropy)
           - Camadas ocultas: delta = (delta_próxima @ W_próxima.T) * relu'(z)
             (erro propagado da camada seguinte, modulado pela derivada da ativação)
        
        2. Os gradientes dos pesos e biases:
           - dL/dW = a_anterior.T @ delta
             (cada peso contribui proporcionalmente à ativação que ele recebeu
              e ao erro que ele propagou)
           - dL/db = soma dos deltas ao longo do batch
             (o bias contribui igualmente para todas as amostras)
        
        Parâmetros:
            y_true (np.ndarray): Rótulos one-hot, shape (batch_size, num_classes)
        
        Retorna:
            tuple: (grad_weights, grad_biases) — listas de gradientes para cada camada
        """
        grad_weights = [None] * self.num_layers
        grad_biases = [None] * self.num_layers
        
        # --- Última camada (softmax + cross-entropy) ---
        # O gradiente combinado é simplesmente (y_pred - y_true) / N
        # Isso é o "delta" da última camada
        delta = cross_entropy_gradient(self.a_values[-1], y_true)
        
        # Gradiente dos pesos: a_anterior.T @ delta
        # a_values[-2] é a ativação da penúltima camada (entrada da última)
        grad_weights[-1] = self.a_values[-2].T @ delta
        
        # Gradiente do bias: soma dos deltas (média já está embutida no delta)
        grad_biases[-1] = np.sum(delta, axis=0, keepdims=True)
        
        # --- Camadas ocultas (de trás para frente) ---
        # Percorremos da penúltima camada até a primeira
        for i in range(self.num_layers - 2, -1, -1):
            # Propaga o delta para a camada anterior:
            # delta_novo = delta_atual @ W_atual.T * relu'(z_atual)
            #
            # Intuitivamente: o erro de cada neurônio nesta camada é proporcional
            # a quanto ele contribuiu para o erro da camada seguinte (via pesos)
            # multiplicado por "quanto a ativação estava ligada" (derivada da ReLU)
            delta = (delta @ self.weights[i + 1].T) * relu_derivative(self.z_values[i])
            
            # Gradientes dos pesos e biases desta camada
            grad_weights[i] = self.a_values[i].T @ delta
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True)
        
        return grad_weights, grad_biases
    
    def train(self, X_train, y_train, epochs=20, batch_size=64, learning_rate=0.1,
              X_val=None, y_val=None):
        """
        Treina a rede usando mini-batch SGD.
        
        Em cada época:
        1. Embaralha os dados (para que os mini-batches sejam diferentes a cada época)
        2. Divide os dados em mini-batches
        3. Para cada mini-batch:
           a. Forward pass (calcula predições)
           b. Calcula a loss
           c. Backward pass (calcula gradientes)
           d. Atualiza pesos com SGD
        4. Registra loss e acurácia da época
        
        O uso de mini-batches em vez do dataset inteiro (batch gradient descent)
        tem duas vantagens:
        - É mais rápido (não precisa processar 60k amostras por atualização)
        - O ruído no gradiente pode ajudar a escapar de mínimos locais
        
        Parâmetros:
            X_train (np.ndarray): Dados de treino, shape (n_samples, n_features)
            y_train (np.ndarray): Rótulos one-hot, shape (n_samples, num_classes)
            epochs (int): Número de épocas de treinamento
            batch_size (int): Tamanho de cada mini-batch
            learning_rate (float): Taxa de aprendizado do SGD
            X_val (np.ndarray, optional): Dados de validação para acompanhar acurácia
            y_val (np.ndarray, optional): Rótulos de validação (one-hot)
        
        Retorna:
            dict: Histórico do treinamento com 'loss', 'accuracy' e opcionalmente
                  'val_accuracy' por época
        """
        optimizer = SGD(learning_rate=learning_rate)
        n_samples = X_train.shape[0]
        
        # Histórico para plots depois
        history = {
            'loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # 1. Embaralha os dados a cada época
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            # 2. Itera pelos mini-batches
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # 3a. Forward pass
                y_pred = self.forward(X_batch)
                
                # 3b. Calcula a loss do batch
                batch_loss = cross_entropy_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # 3c. Backward pass (calcula gradientes)
                grad_weights, grad_biases = self.backward(y_batch)
                
                # 3d. Atualiza pesos
                optimizer.update(self.weights, self.biases, grad_weights, grad_biases)
            
            # 4. Registra métricas da época
            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            
            # Calcula acurácia no treino
            train_pred = self.predict(X_train)
            train_true = np.argmax(y_train, axis=1)
            train_acc = np.mean(train_pred == train_true)
            history['accuracy'].append(train_acc)
            
            # Calcula acurácia na validação (se fornecida)
            val_acc_str = ""
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_true = np.argmax(y_val, axis=1)
                val_acc = np.mean(val_pred == val_true)
                history['val_accuracy'].append(val_acc)
                val_acc_str = f" | Val Acc: {val_acc:.4f}"
            
            print(f"Época {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f}{val_acc_str}")
        
        return history
