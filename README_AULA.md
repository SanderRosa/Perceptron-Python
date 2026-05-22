# Perceptron - Implementação Simples para Aula

## 📁 Arquivos

```
├── perceptron_simples.py   ← Implementação (50 linhas de código)
├── teste_simples.py        ← Testes com AND, OR, NAND
├── exemplo_uso.py          ← Exemplo de como usar
└── README_AULA.md          ← Este arquivo
```

## 🚀 Como Usar

### 1. Testar
```bash
python teste_simples.py
```

### 2. Ver Exemplo
```bash
python exemplo_uso.py
```

### 3. Usar no Seu Código
```python
from perceptron_simples import Perceptron
import numpy as np

# Dados
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])  # função OR

# Treinar
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# Testar
print(f"Acurácia: {ppn.score(X, y):.0%}")

# Prever
print(ppn.predict([[1, 1]]))  # [1]
```

## 📚 O que o Perceptron Faz

1. **Aprende** a classificar dados em 2 categorias (0 ou 1)
2. **Funciona** bem com problemas linearmente separáveis
3. **Usa** correção de erros para ajustar os pesos

## 🧠 Como Funciona

```
1. Inicializa pesos aleatórios
2. Para cada amostra:
   - Calcula: z = pesos·entrada + bias
   - Prediz: 1 se z≥0, senão 0
   - Se errou: atualiza pesos
3. Repete N vezes (n_iter)
```

## 📊 Funções Testadas

- ✓ **AND**: saída 1 quando ambas entradas são 1
- ✓ **OR**: saída 1 quando pelo menos uma entrada é 1
- ✓ **NAND**: saída 0 quando ambas entradas são 1

**Resultado**: 100% de acurácia em todas

## 🔧 Parâmetros

```python
Perceptron(eta=0.1, n_iter=10)
```

- **eta**: taxa de aprendizado (0.01 a 1.0)
  - Menor = mais lento mas mais preciso
  - Maior = mais rápido mas pode oscilar

- **n_iter**: quantas vezes treina com os dados

## 📖 Métodos

```python
ppn.fit(X, y)           # Treina
ppn.predict(X)          # Faz predições
ppn.score(X, y)         # Calcula acurácia (0 a 1)
ppn.errors              # Erros por iteração
```

## ✅ Resultado dos Testes

```
Função AND   → 100% ✓
Função OR    → 100% ✓
Função NAND  → 100% ✓
```

---

**Versão**: 1.0 - Simplificada para Aula  
**Data**: Maio 2026
