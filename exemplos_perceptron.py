"""
Exemplos práticos de uso do Perceptron
Este arquivo contém exemplos simples e diretos de como usar a classe Perceptron
"""

import numpy as np
from perceptron import Perceptron


def exemplo_1_funcao_and():
    """Exemplo 1: Treinando o Perceptron com a função AND"""
    
    print("\n" + "="*60)
    print("EXEMPLO 1: FUNÇÃO AND")
    print("="*60)
    
    # Dados da função AND
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    # Criar e treinar
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    # Testar
    print("\nResultados:")
    print(f"Pesos: {ppn.weights}")
    print(f"Bias: {ppn.bias}")
    print(f"Acurácia: {ppn.score(X, y):.1%}\n")
    
    # Fazer predições
    print("Predições:")
    for xi in X:
        pred = ppn.predict(xi.reshape(1, -1))[0]
        print(f"  {xi} -> {pred}")


def exemplo_2_funcao_or():
    """Exemplo 2: Treinando com a função OR"""
    
    print("\n" + "="*60)
    print("EXEMPLO 2: FUNÇÃO OR")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    print(f"\nAcurácia: {ppn.score(X, y):.1%}")
    print("\nPredições da função OR:")
    for xi in X:
        pred = ppn.predict(xi.reshape(1, -1))[0]
        print(f"  {xi} -> {pred}")


def exemplo_3_nova_predicao():
    """Exemplo 3: Treinar e fazer predição em novo dado"""
    
    print("\n" + "="*60)
    print("EXEMPLO 3: TREINAR E PREVER")
    print("="*60)
    
    # Dados
    X_treino = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_treino = np.array([0, 1, 1, 1])  # OR
    
    # Treinar
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X_treino, y_treino)
    
    # Novos dados para prever
    novos_dados = np.array([
        [0, 0],
        [1, 1],
        [0.5, 0.5]
    ])
    
    print("\nNovas predições:")
    for dado in novos_dados:
        pred = ppn.predict(dado.reshape(1, -1))[0]
        print(f"  {dado} -> {pred}")


def exemplo_4_comparar_taxas_aprendizado():
    """Exemplo 4: Comparar efeito da taxa de aprendizado"""
    
    print("\n" + "="*60)
    print("EXEMPLO 4: EFEITO DA TAXA DE APRENDIZADO")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # OR
    
    print("\nTreinando com diferentes valores de eta:")
    print(f"{'eta':<8} {'Acurácia':<15} {'Iterações até convergência':<30}")
    print("-" * 55)
    
    for eta in [0.01, 0.05, 0.1, 0.5]:
        ppn = Perceptron(eta=eta, n_iter=100)
        ppn.fit(X, y)
        
        accuracy = ppn.score(X, y)
        
        # Encontrar quando convergiu (erro = 0)
        conv_iter = None
        for i, errors in enumerate(ppn.errors):
            if errors == 0:
                conv_iter = i + 1
                break
        
        conv_text = str(conv_iter) if conv_iter else "Não convergiu"
        print(f"{eta:<8.2f} {accuracy:<15.1%} {conv_text:<30}")


def exemplo_5_numeros_maiores():
    """Exemplo 5: Teste com números contínuos"""
    
    print("\n" + "="*60)
    print("EXEMPLO 5: CLASSIFICAÇÃO DE NÚMEROS CONTÍNUOS")
    print("="*60)
    
    # Dados: classificar se x1 + x2 > 1
    np.random.seed(42)
    
    # Classe 0: x1 + x2 <= 1
    class_0 = np.random.rand(20, 2) * 0.7
    
    # Classe 1: x1 + x2 > 1
    class_1 = np.random.rand(20, 2) + [0.5, 0.5]
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(20), np.ones(20)])
    
    ppn = Perceptron(eta=0.1, n_iter=100)
    ppn.fit(X, y)
    
    print(f"\nProblema: Classificar se x1 + x2 > 1")
    print(f"Amostras: {len(X)}")
    print(f"Acurácia: {ppn.score(X, y):.1%}")
    
    # Testar alguns valores
    test_values = np.array([[0, 0], [0.3, 0.3], [0.6, 0.6], [1, 1]])
    print("\nTestando valores:")
    for val in test_values:
        pred = ppn.predict(val.reshape(1, -1))[0]
        soma = val[0] + val[1]
        esperado = 1 if soma > 1 else 0
        status = "✓" if pred == esperado else "✗"
        print(f"  {status} {val} (soma={soma:.1f}) -> {pred} (esperado: {esperado})")


def exemplo_6_monitorar_treinamento():
    """Exemplo 6: Monitorar o progresso durante o treinamento"""
    
    print("\n" + "="*60)
    print("EXEMPLO 6: MONITORAR PROGRESSO DO TREINAMENTO")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    ppn = Perceptron(eta=0.1, n_iter=20)
    ppn.fit(X, y)
    
    print("\nErros por iteração:")
    for i, errors in enumerate(ppn.errors, 1):
        bar = "█" * errors + "░" * (4 - errors)
        print(f"  Iter {i:2d}: {bar} ({errors} erros)")
    
    print(f"\nConvergência: {'✓ SIM' if ppn.errors[-1] == 0 else '✗ NÃO'}")
    print(f"Iterações para convergência: {next((i+1 for i, e in enumerate(ppn.errors) if e == 0), 'Não convergiu')}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXEMPLOS PRÁTICOS - USANDO O PERCEPTRON")
    print("="*60)
    
    # Executar todos os exemplos
    exemplo_1_funcao_and()
    exemplo_2_funcao_or()
    exemplo_3_nova_predicao()
    exemplo_4_comparar_taxas_aprendizado()
    exemplo_5_numeros_maiores()
    exemplo_6_monitorar_treinamento()
    
    print("\n" + "="*60)
    print("FIM DOS EXEMPLOS")
    print("="*60)
