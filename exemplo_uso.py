"""
Exemplo Simples: Como usar o Perceptron
"""

import numpy as np
from perceptron_simples import Perceptron


# Dados: função OR
X = np.array([
    [0, 0],  # entrada 1: 0 e 0
    [0, 1],  # entrada 2: 0 e 1
    [1, 0],  # entrada 3: 1 e 0
    [1, 1]   # entrada 4: 1 e 1
])

y = np.array([0, 1, 1, 1])  # saídas esperadas: 0, 1, 1, 1

# Passo 1: Criar o Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)

# Passo 2: Treinar
ppn.fit(X, y)

# Passo 3: Testar com dados que já conhecemos
print("Teste com dados de treinamento:")
print(f"Acurácia: {ppn.score(X, y):.0%}")

# Passo 4: Fazer predições em novos dados
print("\nPrevisões:")
novos_dados = np.array([[0, 0], [1, 0], [1, 1]])
for dado in novos_dados:
    predicao = ppn.predict(dado.reshape(1, -1))[0]
    print(f"  {dado} -> {predicao}")
