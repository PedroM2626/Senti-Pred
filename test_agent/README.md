# Agente de Geração de Testes Unitários

Este diretório contém um agente de IA que gera automaticamente testes unitários para código Python usando a biblioteca `pytest` e o Azure OpenAI.

## Visão Geral

O agente analisa o código Python fornecido e gera testes unitários abrangentes que cobrem casos de sucesso, casos de erro e validações de tipos. É uma ferramenta poderosa para garantir a qualidade do código e aumentar a cobertura de testes automaticamente.

## Instalação

1. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

## Configuração

1. Configure as variáveis de ambiente do Azure OpenAI:
   - Copie o arquivo `.env.example` para `.env`
   - Preencha com suas credenciais do Azure OpenAI

```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

## Como Usar

### Método 1: Usando o código de exemplo

1. Coloque o código Python que deseja testar no arquivo `example_code.py`
2. Execute o agente:
```bash
python agent.py
```
3. Os testes serão gerados no arquivo `test_example_code.py`
4. Execute os testes gerados:
```bash
pytest test_example_code.py
```

### Método 2: Usando seu próprio arquivo

1. Crie um arquivo Python com o código que deseja testar
2. Modifique a variável `CODE_FILE` no arquivo `agent.py` para apontar para seu arquivo:
```python
CODE_FILE = "seu_arquivo.py"
```
3. Execute o agente:
```bash
python agent.py
```
4. Os testes serão gerados com o prefixo `test_` no nome do arquivo original

## Exemplo de Uso

### Código de entrada (example_code.py):
```python
def soma(a, b):
    """Soma dois números."""
    return a + b

def divide(a, b):
    """Divide dois números."""
    if b == 0:
        raise ValueError("Divisão por zero não permitida")
    return a / b
```

### Testes gerados (test_example_code.py):
```python
import pytest
from example_code import soma, divide

class TestSoma:
    def test_soma_positivos(self):
        assert soma(2, 3) == 5
    
    def test_soma_negativos(self):
        assert soma(-1, -1) == -2
    
    def test_soma_zero(self):
        assert soma(0, 5) == 5

class TestDivide:
    def test_divide_normal(self):
        assert divide(10, 2) == 5
    
    def test_divide_por_zero(self):
        with pytest.raises(ValueError, match="Divisão por zero não permitida"):
            divide(10, 0)
```

## Características do Agente

- **Análise inteligente**: Identifica funções, classes e métodos automaticamente
- **Cobertura abrangente**: Gera testes para casos de sucesso, erro e edge cases
- **Validação de tipos**: Inclui testes para verificar tipos de dados
- **Tratamento de exceções**: Gera testes para verificar erros e exceções
- **Documentação**: Inclui docstrings nos testes gerados
- **Organização**: Estrutura os testes em classes e métodos bem organizados

## Personalização

Você pode personalizar o comportamento do agente modificando:

- **Prompts**: Ajuste os prompts no arquivo `agent.py` para mudar o estilo ou foco dos testes
- **Configurações**: Modifique parâmetros como temperatura do modelo, tokens máximos, etc.
- **Templates**: Adicione templates personalizados para diferentes tipos de testes

## Solução de Problemas

### Erro de autenticação
- Verifique se as credenciais do Azure OpenAI estão corretas no arquivo `.env`
- Certifique-se de que o endpoint e deployment name estão configurados corretamente

### Testes não gerados
- Verifique se o código Python é válido e não contém erros de sintaxe
- Certifique-se de que o arquivo de entrada existe e tem conteúdo

### Testes falhando
- Revise os testes gerados para garantir que correspondem ao comportamento esperado
- Ajuste o código ou os testes conforme necessário

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorar o agente.

## Licença

Este agente faz parte do projeto Senti-Pred e está licenciado sob a mesma licença MIT.