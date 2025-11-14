import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
import pytest

load_dotenv()

# Configurações do Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
    raise ValueError("Por favor, configure as variáveis de ambiente AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY e AZURE_OPENAI_DEPLOYMENT_NAME.")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    temperature=0.7,
)

prompt_template = PromptTemplate(
    input_variables=["code"],
    template="""Você é um agente de IA especializado em gerar testes unitários para código Python usando a biblioteca pytest.
Gere testes unitários para o seguinte código Python. Inclua casos de sucesso e falha.
O arquivo de teste deve começar com 'import pytest' e as funções de teste devem seguir o padrão 'def test_*'.

Código Python:
{code}

Testes unitários (formato puro de Python):
"""
)

test_chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_tests(python_code: str, output_filename: str = "test_generated.py"):
    """
    Gera testes unitários para um dado código Python e salva em um arquivo.
    """
    try:
        print("Gerando testes unitários...")
        generated_tests = test_chain.run(code=python_code)
        
        with open(output_filename, "w") as f:
            f.write(generated_tests)
        print(f"Testes gerados e salvos em {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Erro ao gerar testes: {e}")
        return None

if __name__ == "__main__":
    example_code_path = "example_code.py"
    if os.path.exists(example_code_path):
        with open(example_code_path, "r") as f:
            code_to_test = f.read()
        
        generated_file = generate_tests(code_to_test, output_filename="test_example_code.py")
        if generated_file:
            print(f"Para rodar os testes, execute: pytest {generated_file}")
    else:
        print(f"Arquivo de exemplo '{example_code_path}' não encontrado. Por favor, crie-o ou forneça o código Python para gerar testes.")
    print("Agente de geração de testes inicializado. Por favor, forneça o código Python para gerar testes.")