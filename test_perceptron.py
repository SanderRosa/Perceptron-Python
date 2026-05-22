import numpy as np
from perceptron import Perceptron
import sys


def test_perceptron_convergence():
    """Testa se o Perceptron converge para diferentes funções linearmente separáveis."""
    
    print("\n" + "="*60)
    print("TESTES DE CONVERGÊNCIA DO PERCEPTRON")
    print("="*60)
    
    test_cases = [
        {
            "name": "Função AND",
            "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            "y": np.array([0, 0, 0, 1]),
            "description": "Saída 1 apenas quando ambas entradas são 1"
        },
        {
            "name": "Função OR",
            "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            "y": np.array([0, 1, 1, 1]),
            "description": "Saída 1 quando pelo menos uma entrada é 1"
        },
        {
            "name": "Função NAND",
            "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            "y": np.array([1, 1, 1, 0]),
            "description": "Saída 0 apenas quando ambas entradas são 1"
        },
        {
            "name": "Função NOR",
            "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            "y": np.array([1, 0, 0, 0]),
            "description": "Saída 1 apenas quando ambas entradas são 0"
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n{'-'*60}")
        print(f"Teste: {test_case['name']}")
        print(f"Descrição: {test_case['description']}")
        print(f"{'-'*60}")
        
        X = test_case["X"]
        y = test_case["y"]
        
        # Testa com diferentes configurações
        configs = [
            {"eta": 0.01, "n_iter": 50},
            {"eta": 0.1, "n_iter": 50},
            {"eta": 0.5, "n_iter": 50},
        ]
        
        for config in configs:
            ppn = Perceptron(eta=config["eta"], n_iter=config["n_iter"])
            ppn.fit(X, y)
            
            accuracy = ppn.score(X, y)
            converged = accuracy == 1.0
            
            status = "✓ PASSOU" if converged else "✗ FALHOU"
            print(f"\n  eta={config['eta']}, n_iter={config['n_iter']}")
            print(f"  Acurácia: {accuracy:.1%}")
            print(f"  Erros finais: {ppn.errors[-1]}")
            print(f"  Status: {status}")
            
            if not converged:
                all_passed = False
                print(f"  Predições:")
                for xi, expected in zip(X, y):
                    pred = ppn.predict(xi.reshape(1, -1))[0]
                    match = "✓" if pred == expected else "✗"
                    print(f"    {match} {xi} -> {pred} (esperado: {expected})")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ TODOS OS TESTES PASSARAM!")
    else:
        print("✗ ALGUNS TESTES FALHARAM")
    print("="*60)
    
    return all_passed


def test_perceptron_with_linear_data():
    """Testa com dados linearmente separáveis mais complexos."""
    
    print("\n" + "="*60)
    print("TESTE COM DADOS LINEARMENTE SEPARÁVEIS CUSTOMIZADOS")
    print("="*60)
    
    # Cria dados sintéticos linearmente separáveis
    np.random.seed(42)
    
    # Classe 0: pontos bem separados (abaixo da linha y = -x + 0.5)
    class_0 = np.random.randn(30, 2) * 0.5 + [-1.5, -1.5]
    
    # Classe 1: pontos bem separados (acima da linha y = -x + 0.5)
    class_1 = np.random.randn(30, 2) * 0.5 + [1.5, 1.5]
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
    
    # Treina com mais iterações
    ppn = Perceptron(eta=0.1, n_iter=200)
    ppn.fit(X, y)
    
    accuracy = ppn.score(X, y)
    
    print(f"\nDados customizados linearmente separáveis")
    print(f"Número de amostras: {len(X)}")
    print(f"Número de features: {X.shape[1]}")
    print(f"\nPesos aprendidos: {ppn.weights}")
    print(f"Bias aprendido: {ppn.bias}")
    print(f"Acurácia: {accuracy:.1%}")
    print(f"Erros finais: {ppn.errors[-1]}")
    
    status = "✓ PASSOU" if accuracy >= 0.95 else "✗ FALHOU"
    print(f"Status: {status}")
    
    return accuracy >= 0.95


def test_learning_rate_effect():
    """Analisa o efeito da taxa de aprendizado."""
    
    print("\n" + "="*60)
    print("ANÁLISE: EFEITO DA TAXA DE APRENDIZADO")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # Função OR
    
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    print("\nFunção OR - Testando diferentes taxa de aprendizado (eta)")
    print(f"{'eta':<8} {'Acurácia':<12} {'Iterações para convergência':<30} {'Erros finais':<15}")
    print("-" * 70)
    
    for eta in learning_rates:
        ppn = Perceptron(eta=eta, n_iter=100)
        ppn.fit(X, y)
        
        accuracy = ppn.score(X, y)
        
        # Encontra em qual iteração convergiu (erro = 0)
        convergence_iter = None
        for i, errors in enumerate(ppn.errors):
            if errors == 0:
                convergence_iter = i + 1
                break
        
        convergence_text = str(convergence_iter) if convergence_iter else "Não convergiu"
        
        print(f"{eta:<8.3f} {accuracy:<12.1%} {convergence_text:<30} {ppn.errors[-1]:<15}")
    
    print()


def main():
    """Executa todos os testes."""
    
    print("\n" + "="*60)
    print("SUITE DE TESTES DO PERCEPTRON")
    print("="*60)
    
    try:
        # Executa os testes
        test1 = test_perceptron_convergence()
        test2 = test_perceptron_with_linear_data()
        test_learning_rate_effect()
        
        # Resumo final
        print("\n" + "="*60)
        print("RESUMO DOS TESTES")
        print("="*60)
        print(f"✓ Teste de Convergência: {'PASSOU' if test1 else 'FALHOU'}")
        print(f"✓ Teste com Dados Customizados: {'PASSOU' if test2 else 'FALHOU'}")
        print(f"✓ Análise de Taxa de Aprendizado: CONCLUÍDA")
        print("="*60)
        
        return 0 if (test1 and test2) else 1
        
    except Exception as e:
        print(f"\n✗ ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
