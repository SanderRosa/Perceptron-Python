## 🔧 Configuração para VS Code

### Como Rodar o Projeto

1. **Abra a pasta `Perceptron` no VS Code**
   - `File` → `Open Folder` → selecione a pasta `Perceptron`

2. **Clique no botão "Run" (Play)**
   - Ou use: `Ctrl + F5` (Run Without Debugging)
   - Ou use: `F5` (Run with Debugging)

3. **Ou use o Terminal Integrado**
   - `Ctrl + Shift + '` para abrir o terminal
   - Digite: `python main.py`

### ✅ Estrutura de Arquivo

```
Perceptron/
├── .vscode/
│   ├── launch.json      ← Configuração para Run
│   ├── tasks.json       ← Tarefas
│   └── settings.json    ← Configurações do editor
├── perceptron.py        ← Classe Perceptron
├── main.py              ← Arquivo principal
├── README.md            ← Documentação
└── .gitignore           ← Arquivos ignorados
```

### 📝 Notas

- O arquivo `launch.json` configura o VS Code para rodar como Python
- Ele usa o interpretador Python da pasta pai (`Projetos/.venv`)
- O terminal integrado abrirá automaticamente ao rodar

Se tiver problemas:
1. Feche e abra novamente o VS Code
2. Certifique-se de estar na pasta correta
3. Verifique se Python está instalado: `python --version`
