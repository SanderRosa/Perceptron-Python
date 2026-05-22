# Perceptron - Implementação e Testes

## 📋 Descrição

Este projeto implementa um **Perceptron simples** para classificação binária em Python, com testes e visualizações de funções linearmente separáveis.

## 🎯 Objetivo

- ✓ Implementar um Perceptron do zero
- ✓ Testar com funções linearmente separáveis (AND, OR, NAND, NOR)
- ✓ Visualizar fronteiras de decisão
- ✓ Analisar efeitos de parâmetros (taxa de aprendizado, iterações)

## 📁 Estrutura de Arquivos

```
projetos/
├── perceptron.py          # Implementação da classe Perceptron
├── test_perceptron.py     # Suite de testes
└── README.md              # Este arquivo
```

## 🔧 Dependências Instaladas

- **numpy**: Operações matemáticas e manipulação de arrays
- **matplotlib**: Visualização de gráficos e fronteiras de decisão

```bash
pip install numpy matplotlib
```

## 📚 Como Usar

### 1. Executar Testes Completos

```bash
python test_perceptron.py
```

Este comando executa:
- Testes de convergência para AND, OR, NAND, NOR
- Teste com dados customizados linearmente separáveis
- Análise do efeito da taxa de aprendizado

### 2. Executar Demonstração com Visualizações

```bash
python perceptron.py
```

Este comando:
- Treina o Perceptron para AND, OR e NAND
- Gera gráficos das fronteiras de decisão
- Plota erros durante o treinamento

### 3. Usar em Seu Próprio Código

```python
from perceptron import Perceptron
import numpy as np

# Dados (função OR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Criar e treinar Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# Fazer predições
predictions = ppn.predict(X)
print(f"Acurácia: {ppn.score(X, y):.1%}")

# Visualizar
ppn.plot_decision_boundary(X, y, "Minha Função")
```

## 📖 Classe Perceptron

### Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `eta` | 0.01 | Taxa de aprendizado (0.0 - 1.0) |
| `n_iter` | 50 | Número de iterações de treinamento |
| `random_state` | 1 | Seed para reprodutibilidade |

### Métodos

- **`fit(X, y)`**: Treina o Perceptron
- **`predict(X)`**: Faz predições
- **`score(X, y)`**: Calcula acurácia
- **`plot_decision_boundary(X, y, title)`**: Visualiza a fronteira de decisão
- **`plot_training_errors(title)`**: Plota erros por iteração

## 🧪 Funções Testadas

### 1. Função AND
```
Entrada  Saída
(0, 0) →  0
(0, 1) →  0
(1, 0) →  0
(1, 1) →  1
```
Descrição: Saída 1 apenas quando **ambas** entradas são 1.

### 2. Função OR
```
Entrada  Saída
(0, 0) →  0
(0, 1) →  1
(1, 0) →  1
(1, 1) →  1
```
Descrição: Saída 1 quando **pelo menos uma** entrada é 1.

### 3. Função NAND
```
Entrada  Saída
(0, 0) →  1
(0, 1) →  1
(1, 0) →  1
(1, 1) →  0
```
Descrição: Saída 0 apenas quando **ambas** entradas são 1.

### 4. Função NOR
```
Entrada  Saída
(0, 0) →  1
(0, 1) →  0
(1, 0) →  0
(1, 1) →  0
```
Descrição: Saída 1 apenas quando **ambas** entradas são 0.

## 🧠 Conceitos do Perceptron

### Funcionamento

1. **Inicialização**: Pesos aleatórios pequenos
2. **Forward Pass**: z = w·x + b
3. **Decisão**: y_pred = 1 se z ≥ 0, senão 0
4. **Atualização**: Se erro ≠ 0, atualiza w e b
5. **Iteração**: Repete até convergência ou n_iter

### Fórmulas

**Entrada de rede:**
$$z = \sum_{i=1}^{n} w_i x_i + b$$

**Ativação (função degrau):**
$$y = \begin{cases} 1 & \text{se } z \geq 0 \\ 0 & \text{senão} \end{cases}$$

**Atualização de pesos:**
$$w_i := w_i + \eta (y_{true} - y_{pred}) x_i$$
$$b := b + \eta (y_{true} - y_{pred})$$

Onde:
- $\eta$ = taxa de aprendizado
- $y_{true}$ = rótulo verdadeiro
- $y_{pred}$ = predição
- $x_i$ = features de entrada

## 📊 Resultados Esperados

### Taxa de Aprendizado
- **eta = 0.001**: Convergência lenta
- **eta = 0.1**: Convergência ideal
- **eta = 1.0**: Convergência rápida (pode oscilar)

### Acurácia
Para funções linearmente separáveis, o Perceptron deve alcançar **100% de acurácia**.

## 🔍 Análise e Observações

1. **Linearidade**: O Perceptron só funciona bem em problemas linearmente separáveis
2. **Não consegue aprender XOR**: Esta função não é linearmente separável
3. **Taxa de aprendizado**: Afeta velocidade de convergência e estabilidade
4. **Inicialização**: Pesos aleatórios podem afetar convergência

## 📈 Visualizações

Os gráficos mostram:
- **Fronteira de decisão**: Linha que separa as duas classes
- **Pontos de dados**: Vermelhos (classe 0) e Azuis (classe 1)
- **Erros**: Número de erros de classificação por iteração

## ✅ Testes Inclusos

O arquivo `test_perceptron.py` contém:
- ✓ Teste de convergência para 4 funções
- ✓ Teste com dados customizados
- ✓ Análise de taxa de aprendizado
- ✓ Verificação de acurácia

## 📝 Exemplo de Saída

```
============================================================
TESTES DE CONVERGÊNCIA DO PERCEPTRON
============================================================

------------------------------------------------------------
Teste: Função AND
Descrição: Saída 1 apenas quando ambas entradas são 1
------------------------------------------------------------

  eta=0.01, n_iter=50
  Acurácia: 100.0%
  Erros finais: 0
  Status: ✓ PASSOU

  eta=0.1, n_iter=50
  Acurácia: 100.0%
  Erros finais: 0
  Status: ✓ PASSOU

...

============================================================
✓ TODOS OS TESTES PASSARAM!
============================================================
```

## 🚀 Próximas Melhorias

- [ ] Suporte para multi-classe
- [ ] Versão regularizada (L1/L2)
- [ ] SGD (Stochastic Gradient Descent)
- [ ] Kernel Perceptron para dados não-lineares
- [ ] Testes unitários com pytest

## 📚 Referências

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"

## 📄 Licença

Código de educação - Uso livre

---

**Criado em**: Maio 2026  
**Versão**: 1.0
