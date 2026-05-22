"""
Perceptron Simples - Implementação para Aula
Classificação binária com aprendizado por correção de erros
"""

import numpy as np


class Perceptron:
    """Perceptron simples para classificação binária."""
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        Args:
            eta: Taxa de aprendizado (velocidade de aprendizado)
            n_iter: Número de iterações de treinamento
            random_state: Seed para reprodutibilidade
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors = []
    
    def fit(self, X, y):
        """Treina o Perceptron com os dados X e rótulos y."""
        # Inicializa pesos com valores aleatórios pequenos
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = 0
        
        # Itera pelo número de épocas
        for _ in range(self.n_iter):
            errors = 0
            
            # Para cada amostra de treinamento
            for xi, target in zip(X, y):
                # Faz a predição
                output = self.predict(xi.reshape(1, -1))[0]
                
                # Calcula o erro
                error = target - output
                
                # Se errou, atualiza os pesos
                if error != 0:
                    self.weights += self.eta * error * xi
                    self.bias += self.eta * error
                    errors += 1
            
            self.errors.append(errors)
    
    def predict(self, X):
        """Faz predições: retorna 0 ou 1."""
        # Calcula: z = pesos·entrada + bias
        z = np.dot(X, self.weights) + self.bias
        # Retorna 1 se z >= 0, senão 0
        return np.where(z >= 0, 1, 0)
    
    def score(self, X, y):
        """Calcula a acurácia (porcentagem de acertos)."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
