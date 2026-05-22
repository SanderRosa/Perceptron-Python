import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron:
    """
    Implementação de um Perceptron simples para classificação binária.
    
    Attributes:
        eta (float): Taxa de aprendizado (entre 0.0 e 1.0)
        n_iter (int): Número de iterações de treinamento
        random_state (int): Seed para reprodutibilidade
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        Inicializa o Perceptron.
        
        Args:
            eta (float): Taxa de aprendizado
            n_iter (int): Número de iterações
            random_state (int): Seed para reprodutibilidade
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors = []
        self.is_trained = False
    
    def fit(self, X, y):
        """
        Treina o Perceptron.
        
        Args:
            X (np.ndarray): Dados de treinamento (n_samples, n_features)
            y (np.ndarray): Rótulos de treinamento (n_samples,)
            
        Returns:
            self: Retorna a instância do Perceptron
        """
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = 0.0
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Calcula a predição
                output = self.predict(xi.reshape(1, -1))[0]
                # Calcula o erro
                error = target - output
                # Atualiza pesos e bias se houver erro
                if error != 0:
                    self.weights += self.eta * error * xi
                    self.bias += self.eta * error
                    errors += 1
            self.errors.append(errors)
        
        self.is_trained = True
        return self
    
    def net_input(self, X):
        """
        Calcula a entrada de rede (z = w·x + b).
        
        Args:
            X (np.ndarray): Dados de entrada
            
        Returns:
            np.ndarray: Resultado da função linear
        """
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        Prediz a classe.
        
        Args:
            X (np.ndarray): Dados de entrada
            
        Returns:
            np.ndarray: Predições (0 ou 1)
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    def score(self, X, y):
        """
        Calcula a acurácia do modelo.
        
        Args:
            X (np.ndarray): Dados de teste
            y (np.ndarray): Rótulos de teste
            
        Returns:
            float: Acurácia (0.0 a 1.0)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def plot_decision_boundary(self, X, y, title="Fronteira de Decisão"):
        """
        Plota a fronteira de decisão do Perceptron.
        
        Args:
            X (np.ndarray): Dados (deve ter 2 features)
            y (np.ndarray): Rótulos
            title (str): Título do gráfico
        """
        if X.shape[1] != 2:
            print("A visualização é suportada apenas para dados com 2 features")
            return
        
        # Cria uma malha de pontos
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Prediz para todos os pontos da malha
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plota
        plt.figure(figsize=(8, 6))
        
        # Mapa de cores para a fronteira
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
        
        # Plota os pontos de dados
        colors = ['red' if label == 0 else 'blue' for label in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', s=100, alpha=0.8)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_training_errors(self, title="Erros durante o Treinamento"):
        """
        Plota o número de erros por iteração.
        
        Args:
            title (str): Título do gráfico
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('Iteração')
        plt.ylabel('Número de Erros')
        plt.title(title)
        plt.tight_layout()
        plt.show()


def test_and_function():
    """Testa o Perceptron com a função AND."""
    print("\n" + "="*50)
    print("TESTANDO FUNÇÃO AND")
    print("="*50)
    
    # Dados da função AND
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y = np.array([0, 0, 0, 1])  # AND: 1 apenas quando ambas são 1
    
    # Treina o Perceptron
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    # Exibe resultados
    print("\nPesos finais:", ppn.weights)
    print("Bias final:", ppn.bias)
    print("Acurácia:", ppn.score(X, y))
    print("\nPredições:")
    for xi, expected in zip(X, y):
        pred = ppn.predict(xi.reshape(1, -1))[0]
        print(f"  {xi} -> {pred} (esperado: {expected})")
    
    return ppn, X, y


def test_or_function():
    """Testa o Perceptron com a função OR."""
    print("\n" + "="*50)
    print("TESTANDO FUNÇÃO OR")
    print("="*50)
    
    # Dados da função OR
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y = np.array([0, 1, 1, 1])  # OR: 1 quando pelo menos um é 1
    
    # Treina o Perceptron
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    # Exibe resultados
    print("\nPesos finais:", ppn.weights)
    print("Bias final:", ppn.bias)
    print("Acurácia:", ppn.score(X, y))
    print("\nPredições:")
    for xi, expected in zip(X, y):
        pred = ppn.predict(xi.reshape(1, -1))[0]
        print(f"  {xi} -> {pred} (esperado: {expected})")
    
    return ppn, X, y


def test_nand_function():
    """Testa o Perceptron com a função NAND."""
    print("\n" + "="*50)
    print("TESTANDO FUNÇÃO NAND (NOT AND)")
    print("="*50)
    
    # Dados da função NAND
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y = np.array([1, 1, 1, 0])  # NAND: 0 apenas quando ambas são 1
    
    # Treina o Perceptron
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    # Exibe resultados
    print("\nPesos finais:", ppn.weights)
    print("Bias final:", ppn.bias)
    print("Acurácia:", ppn.score(X, y))
    print("\nPredições:")
    for xi, expected in zip(X, y):
        pred = ppn.predict(xi.reshape(1, -1))[0]
        print(f"  {xi} -> {pred} (esperado: {expected})")
    
    return ppn, X, y


if __name__ == "__main__":
    # Testa as três funções linearmente separáveis
    ppn_and, X_and, y_and = test_and_function()
    ppn_or, X_or, y_or = test_or_function()
    ppn_nand, X_nand, y_nand = test_nand_function()
    
    # Tenta visualizar as fronteiras de decisão
    print("\n" + "="*50)
    print("GERANDO VISUALIZAÇÕES")
    print("="*50)
    print("\nGráficos serão exibidos. Feche cada janela para continuar...")
    
    ppn_and.plot_decision_boundary(X_and, y_and, "Função AND - Fronteira de Decisão")
    ppn_or.plot_decision_boundary(X_or, y_or, "Função OR - Fronteira de Decisão")
    ppn_nand.plot_decision_boundary(X_nand, y_nand, "Função NAND - Fronteira de Decisão")
    
    ppn_and.plot_training_errors("AND - Erros durante Treinamento")
    ppn_or.plot_training_errors("OR - Erros durante Treinamento")
    ppn_nand.plot_training_errors("NAND - Erros durante Treinamento")
    
    print("\nTestes concluídos!")
