# MLP do Zero — Classificação de Dígitos MNIST

Este repositório contém a implementação de um Multi-Layer Perceptron (MLP) feito inteiramente do zero, usando apenas **NumPy** para operações matriciais. O objetivo é classificar os dígitos manuscritos do dataset MNIST (0 a 9) com uma acurácia superior a 92% no conjunto de teste.

## Estrutura do Projeto

```
.
├── README.md               ← Este arquivo
├── requirements.txt        ← Dependências do projeto
├── train_mnist.py          ← Script principal para treinar a configuração padrão
├── run_experiments.py      ← Script para treinar a segunda configuração e gerar os gráficos
├── mnist.npz               ← Dataset MNIST (baixado automaticamente caso não exista)
├── mlp/
│   ├── __init__.py
│   ├── network.py          ← Implementação da classe MLP
│   ├── activations.py      ← Funções ReLU, Softmax e suas derivadas
│   ├── losses.py           ← Cross-entropy loss com estabilidade numérica
│   └── optimizers.py       ← Otimizador SGD
└── results/
    ├── training_history.json
    ├── history_exp2.json
    ├── loss_comparison.png
    └── accuracy_comparison.png
```

## Como Rodar

Para executar este projeto localmente, siga os passos abaixo:

1. **Instale as dependências:**
   Você só precisará do `numpy` para a rede neural e `matplotlib` para gerar os gráficos. O TensorFlow **não** é necessário, já que adicionei um script customizado para baixar o dataset via web e carregá-lo direto como array.
   ```bash
   pip install -r requirements.txt
   ```

2. **Para treinar a rede com a configuração ideal (LR = 0.1):**
   ```bash
   python train_mnist.py
   ```
   *Este comando irá rodar as 30 épocas e imprimir a acurácia no terminal.*

3. **Para rodar o segundo experimento e gerar os gráficos de comparação:**
   ```bash
   python run_experiments.py
   ```
   *Isso executará o treinamento de uma nova configuração (LR = 0.01) e salvará os gráficos na pasta `results/`.*

## Arquitetura Escolhida

A arquitetura final selecionada foi:
- **Entrada**: 784 neurônios (imagens de 28x28 achatadas em vetor 1D).
- **Camada Oculta 1**: 128 neurônios, ativação **ReLU**.
- **Camada Oculta 2**: 64 neurônios, ativação **ReLU**.
- **Camada de Saída**: 10 neurônios, ativação **Softmax**.

**Por que essas escolhas?**
A arquitetura `784 -> 128 -> 64 -> 10` afunila progressivamente o espaço de features da imagem e possui parâmetros suficientes para modelar as curvas de um dígito desenhado à mão de maneira robusta. A **ReLU** foi escolhida por ser a padrão-ouro na indústria contra o "vanishing gradient", além de sua derivada ser rápida de calcular. A **Softmax** combinada com **Cross-Entropy** na camada de saída não apenas nos entrega probabilidades que somam $1.0$, mas também possui um gradiente muito elegante e fácil de computar `(ŷ - y)` que evita cálculo de matriz Jacobiana completa no retropropagador, acelerando e simplificando muito a backpropagation. A inicialização dos pesos foi a **He Initialization**, porque as ReLUs zeram metade da saída, logo elas precisam que os pesos sejam escalados por `sqrt(2/n)` para evitar a perda da variância entre as camadas (vanishing/exploding gradients na largada).

## Resultados

Realizamos dois experimentos alterando apenas a **Taxa de Aprendizado (Learning Rate)** com o Otimizador SGD ao longo de 30 épocas com batch de 64.

1. **Experimento 1 (LR = 0.1):**
   - Treinamento acelerado com rápida convergência.
   - **Acurácia Final no Teste:** **97.94%** (Bateu com folga a meta de 92%!).

2. **Experimento 2 (LR = 0.01):**
   - Como os "passos" de ajuste do SGD ficaram mais curtos, a rede aprendeu um pouco mais devagar.
   - **Acurácia Final no Teste:** **97.42%**.

Os gráficos que comprovam que a Loss (custo) diminuiu de maneira saudável sem grandes oscilações e que ambas as redes convergiram bem encontram-se na pasta `results/`:
- Curva de Loss
- Curva de Acurácia

## Decisões e Dificuldades

1. **Qual foi a decisão técnica mais difícil que você tomou? Por que fez essa escolha?**
   A decisão técnica mais difícil foi garantir a estabilidade numérica na função Loss e nas inicializações. No meio do desenvolvimento, quando fiz a `cross_entropy_loss`, os valores de loss podiam estourar para *Infinito* caso o modelo cuspisse zero no softmax. Tive a decisão de inserir um `np.clip(y_pred, 1e-15, 1 - 1e-15)` na ponta para nunca termos um verdadeiro `log(0)`. Apesar de existir a chance do modelo "sumir" com os pesos (exploding gradient), eu contornei isso com o "He Initialization", a escolha que permitiu o fluxo ser extremamente estável. 

2. **O que você tentou que não funcionou? O que aprendeu com isso?**
   Uma tentativa falha foi importar o dataset. Num primeiro momento, pensei em depender do `tensorflow.keras.datasets.mnist` apenas para carregar os dados. Isso não funcionou bem devido à versão do Python (3.14 não tinha build de TensorFlow na hora), inviabilizando totalmente que um avaliador o rodasse de forma leve. Aprendi que dependências pesadas para coisas banais são vilãs. Reescrevi a função com `urllib` para baixar diretamente do bucket GCS `mnist.npz` e desempacotar com `np.load()`. Com isso, a dependência se tornou 100% em Numpy e o código muito mais robusto.

3. **Se fosse refazer do zero, o que faria diferente?**
   Se eu tivesse que refazer do zero (e com mais tempo livre), eu optaria por implementar uma classe mais modular como as implementações de framework, herdando de uma classe `Layer` genérica, com `Linear`, `ReLU`, e encapsular o `forward()` e `backward()` dentro de cada camada isoladamente, em vez de um grande laço de repetição `for i in range(self.num_layers)` dentro da própria `MLP`. Isso tornaria muito mais fácil o uso do Polimorfismo no futuro caso eu quisesse testar outras funções de ativação como Sigmoid ou Tanh no meio.