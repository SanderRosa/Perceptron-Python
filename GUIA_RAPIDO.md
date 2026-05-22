# 🚀 Guia Rápido - Perceptron

## Arquivos Criados

```
c:\Users\sande\OneDrive\Área de Trabalho\Projetos\
├── perceptron.py              ← Implementação principal do Perceptron
├── test_perceptron.py         ← Suite de testes (12 testes diferentes)
├── exemplos_perceptron.py     ← 6 exemplos práticos de uso
├── README_PERCEPTRON.md       ← Documentação completa
└── GUIA_RAPIDO.md            ← Este arquivo
```

## ✅ Dependências Instaladas

```
✓ numpy        - Operações matemáticas
✓ matplotlib   - Visualizações e gráficos
```

## 🎯 Como Usar em 1 Minuto

### Opção 1: Executar os Testes
```bash
python test_perceptron.py
```
Resultado: **✓ TODOS OS 12 TESTES PASSARAM**

### Opção 2: Ver Exemplos Práticos
```bash
python exemplos_perceptron.py
```

### Opção 3: Usar no Seu Código
```python
from perceptron import Perceptron
import numpy as np

# Dados (função OR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Treinar
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# Prever
print(ppn.predict([[1, 1]]))  # [1]
print(f"Acurácia: {ppn.score(X, y):.1%}")  # 100.0%
```

## 📊 Funções Testadas

| Função | Entrada | Saída |
|--------|---------|-------|
| **AND** | (1, 1) | 1 |
| **OR** | (0, 1) | 1 |
| **NAND** | (1, 1) | 0 |
| **NOR** | (0, 0) | 1 |

✓ **Todas com 100% de acurácia**

## 📈 Resultados dos Testes

```
✓ Teste de Convergência:        PASSOU (4 funções lógicas)
✓ Dados Customizados:           PASSOU (100% acurácia em 60 amostras)
✓ Taxa de Aprendizado:          PASSOU (5 valores diferentes testados)
```

## 🧠 Conceitos

**O Perceptron**:
- Algoritmo de classificação binária linear
- Funciona bem em problemas linearmente separáveis
- Aprender pelo método do erro-correção
- Taxa de convergência = "eta" (recomendado: 0.1)

## 📚 Métodos Principais

```python
ppn = Perceptron(eta=0.1, n_iter=50, random_state=1)

ppn.fit(X, y)                          # Treinar
ppn.predict(X)                         # Prever (retorna [0, 1])
ppn.score(X, y)                        # Acurácia (0.0 a 1.0)
ppn.plot_decision_boundary(X, y)       # Visualizar fronteira
ppn.plot_training_errors()             # Erros por iteração
```

## 📝 Notas Importantes

1. **Linearidade**: O Perceptron só funciona para dados linearmente separáveis
2. **XOR**: Não consegue aprender XOR (não é linearmente separável)
3. **Inicialização**: Usa pequenos pesos aleatórios
4. **Convergência**: Garantida para dados linearmente separáveis

## 🔧 Personalizar

```python
# Usar taxa de aprendizado mais rápida
ppn = Perceptron(eta=0.5, n_iter=50)

# Usar taxa de aprendizado mais lenta (mais preciso)
ppn = Perceptron(eta=0.01, n_iter=100)

# Reproduzir exatamente os mesmos resultados
ppn = Perceptron(random_state=42)
```

## 📖 Para Saber Mais

- Leia [README_PERCEPTRON.md](README_PERCEPTRON.md) para documentação completa
- Veja exemplos em [exemplos_perceptron.py](exemplos_perceptron.py)
- Verifique código em [perceptron.py](perceptron.py)

---

**Status**: ✅ Tudo funcionando corretamente!  
**Data**: Maio 2026
