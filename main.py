import numpy as np
from perceptron import Perceptron


def testar_funcao(nome, X, y):
    """Treina e testa o Perceptron com uma função."""
    print(f"\n{nome}")
    print("-" * 40)
    
    # Treina
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    # Testa
    acuracia = ppn.score(X, y)
    print(f"Acurácia: {acuracia:.0%}")
    
    # Mostra predições
    print("Predições:")
    for xi, esperado in zip(X, y):
        pred = ppn.predict(xi.reshape(1, -1))[0]
        status = "✓" if pred == esperado else "✗"
        print(f"  {status} {xi} -> {pred} (esperado: {esperado})")
    
    return acuracia == 1.0


def main():
    """Função principal."""
    # Dados de teste
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Testes
    print("="*40)
    print("TESTANDO PERCEPTRON")
    print("="*40)
    
    y_and = np.array([0, 0, 0, 1])  # AND
    testar_funcao("Função AND", X, y_and)
    
    y_or = np.array([0, 1, 1, 1])   # OR
    testar_funcao("Função OR", X, y_or)
    
    y_nand = np.array([1, 1, 1, 0]) # NAND
    testar_funcao("Função NAND", X, y_nand)
    
    print("\n" + "="*40)
    print("✓ TESTES CONCLUÍDOS")
    print("="*40)


if __name__ == "__main__":
    main()
